"""
committor_control.py
======================
Committor-based optimal control for stochastic saddle-point systems.
Evaluates static and adaptive (closed-loop) CCA tracking procedures.
"""

import os
import copy
import csv
import warnings
import logging
from functools import partial

import numpy as np
import scipy.linalg
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D
from matplotlib.collections import LineCollection
from scipy.special import erf
import seaborn as sns

# ──────────────────────────────────────────────────────────────────────────────
# AGGRESSIVE WARNING & LOGGER SUPPRESSION
# ──────────────────────────────────────────────────────────────────────────────
warnings.simplefilter("ignore")
logging.getLogger('PIL').setLevel(logging.CRITICAL)
logging.getLogger('PIL.PdfImagePlugin').setLevel(logging.CRITICAL)
logging.getLogger('PIL.PngImagePlugin').setLevel(logging.CRITICAL)
logging.getLogger('matplotlib').setLevel(logging.CRITICAL)

os.makedirs("figures", exist_ok=True)

# ──────────────────────────────────────────────────────────────────────────────
# GLOBAL PLOT CONFIGURATION & IEEE STYLE
# ──────────────────────────────────────────────────────────────────────────────
import matplotlib.colors as mcolors
import matplotlib.lines as mlines

PLOT_CONFIG = {
    "COLORS": {
        "Autonomous": "#444444",
        "Optimal": "#1a6faf",
        "CCA": "#2d307b",
        "CCA (Whitened)": "#005f38",
        "CCA (Algo 1)": "#2d307b",
        "CCA (Adaptive)": "#7b2d7b"
    },
    "LABELS": {
        "Autonomous": "Autonomous",
        "Optimal": "Model-based",
        "CCA": "Data-enabled",
        "CCA (Whitened)": "Data-enabled",
        "CCA (Algo 1)": "Data-enabled",
        "CCA (Adaptive)": "Closed-loop"
    },
    "LINESTYLES": {
        "Optimal": "-",
        "CCA": "--",
        "CCA (Whitened)": "--",
        "CCA (Algo 1)": ":",
        "CCA (Adaptive)": "-."
    }
}

def set_ieee_style():
    """Applies IEEE CDC-compliant styling."""
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "axes.labelsize": 10,
        "axes.titlesize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 8,
        "axes.linewidth": 0.8,
        "grid.linewidth": 0.5,
        "grid.alpha": 0.5,
        "lines.linewidth": 1.5,
        "figure.dpi": 300
    })

# ──────────────────────────────────────────────────────────────────────────────
# 1. Math helpers
# ──────────────────────────────────────────────────────────────────────────────

def _sign_align(v, reference):
    return v if np.dot(v, reference) >= 0 else -v

def committor_score_1d(x_u, a, r):
    x_u = np.asarray(x_u, dtype=float)
    r = float(np.clip(r, 0.0, 1e4))
    if r > 1e-10:
        sr = np.sqrt(r)
        erf_diff = erf(sr * a) - erf(sr * x_u)
        with np.errstate(invalid='ignore', divide='ignore'):
            num = -np.exp(-r * x_u**2)
            den = np.sqrt(np.pi / (4.0 * r)) * erf_diff
            den_safe = np.where(np.abs(den) < 1e-300, 1e-300 * np.sign(den + 1e-300), den)
            score = num / den_safe
            score_asym = -2.0 * r * x_u
        return np.where(np.abs(erf_diff) < 1e-12, score_asym, score)
    return -1.0 / (a - x_u + 1e-15) * np.ones_like(x_u)

def optimal_control_scalar(x_u, a, lambda_u, D):
    r = lambda_u / (2.0 * D) if D > 0 else 1e12
    return 2.0 * D * committor_score_1d(x_u, a, r)

def apply_underactuated_constraint(u_ideal, B):
    if np.linalg.norm(u_ideal) < 1e-12:
        return np.zeros_like(u_ideal)
    return B @ (np.linalg.pinv(B) @ u_ideal)

def _sym_inv_sqrt(mat, reg=1e-6):
    vals, vecs = np.linalg.eigh(mat)
    floor = max(np.max(vals) * reg, 1e-12)
    vals = np.maximum(vals, floor)
    return vecs @ np.diag(1.0 / np.sqrt(vals)) @ vecs.T

# ──────────────────────────────────────────────────────────────────────────────
# 2. Controllers
# ──────────────────────────────────────────────────────────────────────────────

def _unstable_proj(x, w_u, x_fp):
    return float(np.dot(w_u, x - x_fp))

def erf_feedback(x, w_u, lambda_u, D, x_fp, a, r_neigh=np.inf):
    u = np.zeros_like(x, dtype=float)
    x_u = _unstable_proj(x, w_u, x_fp)
    if abs(x_u) >= a: return u
    if r_neigh < np.inf and np.linalg.norm(x - x_fp) > r_neigh: return u
    return float(optimal_control_scalar(x_u, a, lambda_u, D)) * w_u

def relu_feedback(x, w_u, lambda_u, D, x_fp, a=None, r_neigh=np.inf):
    u = np.zeros_like(x, dtype=float)
    x_u = _unstable_proj(x, w_u, x_fp)
    if r_neigh < np.inf and np.linalg.norm(x - x_fp) > r_neigh: return u
    if x_u >= 0.0: u = -2.0 * lambda_u * x_u * w_u
    return u

def lqr_feedback(x, w_u, lambda_u, D, x_fp, a=None, r_neigh=np.inf):
    """Full LQR-style feedback: u = -2*lambda_u*x_u*w_u on BOTH sides."""
    u = np.zeros_like(x, dtype=float)
    x_u = _unstable_proj(x, w_u, x_fp)
    if r_neigh < np.inf and np.linalg.norm(x - x_fp) > r_neigh: return u
    u = -2.0 * lambda_u * x_u * w_u
    return u

def sign_gated_feedback(x, w_u, lambda_u, D, x_fp, a=None, r_neigh=np.inf, gain=None):
    """Sign-gated linear: u = -k*max(0, x_u)*w_u with fixed gain k (default 2*lambda_u)."""
    u = np.zeros_like(x, dtype=float)
    x_u = _unstable_proj(x, w_u, x_fp)
    if r_neigh < np.inf and np.linalg.norm(x - x_fp) > r_neigh: return u
    k = gain if gain is not None else 2.0 * lambda_u
    if x_u >= 0.0: u = -k * x_u * w_u
    return u

# ──────────────────────────────────────────────────────────────────────────────
# 3. System factories
# ──────────────────────────────────────────────────────────────────────────────

class SystemSpec:
    def __init__(self, name, drift, x_star, J, w_u_true, lambda_u_true, lambda_s_true, plot_xlim, plot_ylim):
        self.name = name
        self.drift = drift
        self.x_star = np.asarray(x_star, dtype=float)
        self.J = np.asarray(J, dtype=float)
        self.w_u_true = np.asarray(w_u_true, dtype=float)
        self.w_u_true /= np.linalg.norm(self.w_u_true)
        self.lambda_u_true = float(lambda_u_true)
        self.lambda_s_true = float(lambda_s_true)
        self.plot_xlim = plot_xlim
        self.plot_ylim = plot_ylim

def make_double_well_curved(lambda_u=2.0, lambda_s=8.0, c=0.5):
    def drift(t, x):
        X, Y = x[0], x[1]
        dx = -lambda_u * X * (X**2 - 1) + 2*c*lambda_s * X * (Y - c*X**2)
        dy = -lambda_s * (Y - c * X**2)
        return np.array([dx, dy])
    J = np.array([[lambda_u, 0.0], [0.0, -lambda_s]])
    w_u = np.array([1.0, 0.0])
    return SystemSpec("Curved_Double_Well", drift, np.zeros(2), J, w_u, lambda_u, -lambda_s,
                      plot_xlim=(-0.6, 0.6), plot_ylim=(-0.4, 0.4))

def make_linear_system(lambda_u=2.0, lambda_s=-8.0):
    A_ref = np.array([[0.78830835, 0.16234692], [2.3318579, -0.78830835]])
    _, V_right = np.linalg.eig(A_ref)
    order = np.argsort(np.linalg.eig(A_ref)[0].real)[::-1]
    V_right = V_right[:, order].real
    A = V_right @ np.diag([lambda_u, lambda_s]) @ np.linalg.inv(V_right)
    eigvals_l, V_left = np.linalg.eig(A.T)
    order_l = np.argsort(eigvals_l.real)[::-1]
    V_left = V_left[:, order_l].real
    w_u = V_left[:, 0].real
    w_u /= np.linalg.norm(w_u)
    if np.dot(w_u, V_right[:, 0]) < 0: w_u = -w_u
    def drift(t, x): return A @ x
    return SystemSpec("Linear_2D", drift, np.zeros(2), A, w_u, lambda_u, lambda_s,
                      plot_xlim=(-1.5, 1.5), plot_ylim=(-1.5, 1.5))

def make_pendulum(g=5.0, length=1.0, damping=0.0):
    def drift(t, x): return np.array([x[1], -(g / length) * np.sin(x[0]) - damping * x[1]])
    J = np.array([[0.0, 1.0], [g / length, -damping]])
    eigvals, V_right = np.linalg.eig(J)
    order = np.argsort(eigvals.real)[::-1]
    eigvals = eigvals[order].real
    V_right = V_right[:, order].real
    eigvals_l, V_left = np.linalg.eig(J.T)
    order_l = np.argsort(eigvals_l.real)[::-1]
    V_left = V_left[:, order_l].real
    w_u = V_left[:, 0].real
    w_u /= np.linalg.norm(w_u)
    if np.dot(w_u, V_right[:, 0]) < 0: w_u = -w_u
    sq2 = np.sqrt(2.0)
    return SystemSpec("Pendulum", drift, np.array([np.pi, 0.0]), J, w_u, float(eigvals[0]), float(eigvals[1]),
                      plot_xlim=(np.pi - sq2, np.pi + sq2), plot_ylim=(-sq2, sq2))

def make_random_linear_system(dim, lambda_u, lambda_s_max=-1.0):
    eigenvalues = [lambda_u] + list(np.random.uniform(-5.0, lambda_s_max, size=dim - 1))
    cond = np.inf
    while cond > 10.0:
        Q = np.random.randn(dim, dim)
        cond = np.linalg.cond(Q)
    A = Q @ np.diag(eigenvalues) @ np.linalg.inv(Q)
    eigvals, V_right = np.linalg.eig(A)
    order = np.argsort(eigvals.real)[::-1]
    eigvals = eigvals[order].real
    V_right = V_right[:, order].real
    eigvals_l, V_left = np.linalg.eig(A.T)
    order_l = np.argsort(eigvals_l.real)[::-1]
    V_left = V_left[:, order_l].real
    w_u = V_left[:, 0].real
    w_u /= np.linalg.norm(w_u)
    if np.dot(w_u, V_right[:, 0]) < 0: w_u = -w_u
    def drift(t, x): return A @ x
    return SystemSpec(f"Random_Linear_{dim}D", drift, np.zeros(dim), A, w_u, float(eigvals[0]), float(eigvals[1]),
                      plot_xlim=(-1, 1), plot_ylim=(-1, 1))

# ──────────────────────────────────────────────────────────────────────────────
# 4. CCA Trackers
# ──────────────────────────────────────────────────────────────────────────────

class CCATracker:
    def __init__(self, dim=2, dt=0.01, tau=10, rng_seed=0, use_whitened=True):
        self.dim = dim
        self.dt = dt
        self.tau = tau
        self.lambda_u = 0.0
        self.D = 1e-6
        self.Delta = 0.0
        self.x_star = np.zeros(dim)
        self.v0 = np.random.default_rng(rng_seed).standard_normal(dim)
        self.v0 /= np.linalg.norm(self.v0) + 1e-15
        self.use_whitened = use_whitened

    def reset(self):
        pass

    def update(self, x_t, x_next_pseudo, sign_ref):
        pass

    def train(self, drift_func, T, dt, D, seed, a, sign_ref, init_center, r_neigh=np.inf):
        noise_std = np.sqrt(2.0 * D)
        rng = np.random.default_rng(seed)
        x = init_center + rng.uniform(-0.1, 0.1, self.dim)
        limit = r_neigh if r_neigh < np.inf else max(2.0, a * 2.0)
        segments = []
        current_seg = [x.copy()]
        for t in range(T):
            x_next = x + drift_func(t * dt, x) * dt + noise_std * rng.standard_normal(self.dim) * np.sqrt(dt)
            out_of_bounds = np.linalg.norm(x_next - init_center) > limit
            if out_of_bounds:
                if len(current_seg) > self.tau + 2:
                    segments.append(np.array(current_seg))
                x = init_center + rng.normal(0, 0.05, self.dim)
                current_seg = [x.copy()]
                continue
            current_seg.append(x_next.copy())
            x = x_next
        if len(current_seg) > self.tau + 2:
            segments.append(np.array(current_seg))
        self._fit(segments, sign_ref)

    def _fit(self, segments, sign_ref):
        all_p, all_f, all_xp, all_xf = [], [], [], []
        for seg in segments:
            if len(seg) < self.tau + 2: continue
            for i in range(self.tau, len(seg) - 1):
                all_p.append(seg[i] - seg[i - self.tau])
                all_f.append(seg[i + 1] - seg[i + 1 - self.tau])
                all_xp.append(seg[i])
                all_xf.append(seg[i + 1])
        if len(all_p) < self.dim + 2: return
        P, F = np.array(all_p), np.array(all_f)
        Xp, Xf = np.array(all_xp), np.array(all_xf)
        n = len(P)
        C_pp = P.T @ P / n
        C_ff = F.T @ F / n
        C_pf = P.T @ F / n
        C_fp = F.T @ P / n
        try:
            if self.use_whitened:
                Cpp_inv_half = _sym_inv_sqrt(C_pp, reg=1e-6)
                M_cca = Cpp_inv_half @ C_fp @ Cpp_inv_half
                ev, ec = np.linalg.eig(M_cca.T)
                idx = np.argmax(ev.real)
                psi = ec[:, idx].real
                alpha_0 = Cpp_inv_half @ psi
            else:
                reg_mat = np.eye(self.dim) * 1e-6
                left_matrix = C_pf @ np.linalg.pinv(C_ff + reg_mat) @ C_fp
                ev, ec = scipy.linalg.eig(left_matrix, C_pp + reg_mat)
                idx = np.argmax(ev.real)
                alpha_0 = ec[:, idx].real
            alpha_0 /= np.linalg.norm(alpha_0) + 1e-15
            w_u = _sign_align(alpha_0, sign_ref)
            P_c = P - np.mean(P, axis=0)
            Xp_c = Xp - np.mean(Xp, axis=0)
            Xf_c = Xf - np.mean(Xf, axis=0)
            Cov_pc = (P_c.T @ Xp_c) / n
            Cov_pn = (P_c.T @ Xf_c) / n
            var_iv = float(w_u @ Cov_pc @ w_u)
            cov_iv = float(w_u @ Cov_pn @ w_u)
            lam0_disc = cov_iv / max(var_iv, 1e-15)
            self.lambda_u = max(0.0, np.log(max(lam0_disc, 1e-12)) / self.dt)
            mu_p_val = float(w_u @ np.mean(Xp, axis=0))
            mu_f_val = float(w_u @ np.mean(Xf, axis=0))
            if abs(1.0 - lam0_disc) > 1e-10:
                self.Delta = (mu_f_val - lam0_disc * mu_p_val) / (1.0 - lam0_disc)
            else:
                self.Delta = mu_p_val
            s_t = Xp @ w_u - self.Delta
            s_next = Xf @ w_u - self.Delta
            D_discrete = float(np.mean((s_next - lam0_disc * s_t)**2)) / 2.0
            self.v0 = w_u
            self.D = max(D_discrete / self.dt, 1e-15)
            mean_x = np.mean(Xp, axis=0)
            self.x_star = mean_x + (self.Delta - float(w_u @ mean_x)) * w_u
        except np.linalg.LinAlgError: return

    def get_params(self, sign_ref):
        return dict(w_u=_sign_align(self.v0.copy(), sign_ref), lambda_u=self.lambda_u, D=self.D, x_star=self.x_star.copy())

class OnlineCCATracker:
    def __init__(self, dim=2, dt=0.01, gamma=0.01, tau=10, min_steps=20, a=0.5, rng_seed=0, use_whitened=True):
        self.dim = dim
        self.dt = dt
        self.gamma = gamma
        self.tau = tau
        self.min_steps = min_steps
        self.a = a
        self.lambda_u = 0.0
        self.D = 1e-6
        self.Delta = 0.0
        self.x_star = np.zeros(dim)
        self.v0 = np.random.default_rng(rng_seed).standard_normal(dim)
        self.v0 /= np.linalg.norm(self.v0) + 1e-15
        self.use_whitened = use_whitened
        eps = 1e-6
        self.C_pp = np.eye(dim) * eps
        self.C_ff = np.eye(dim) * eps
        self.C_pf = np.zeros((dim, dim))
        self.C_fp = np.zeros((dim, dim))
        self.mu_p_lag = np.zeros(dim)
        self.mu_x_curr = np.zeros(dim)
        self.mu_x_next = np.zeros(dim)
        self.M_p_c = np.zeros((dim, dim))
        self.M_p_n = np.zeros((dim, dim))
        self.M_c_c = np.eye(dim) * eps
        self.M_n_n = np.eye(dim) * eps
        self.M_c_n = np.zeros((dim, dim))
        self.history = []
        self.n_updates = 0

    def reset(self):
        self.history = []

    def update(self, x_t, x_next_pseudo, sign_ref):
        self.history.append((x_t, x_next_pseudo))
        if len(self.history) > self.tau + 1:
            self.history.pop(0)
        if len(self.history) == self.tau + 1:
            x_t_minus_tau, x_next_pseudo_t_minus_tau = self.history[0]
            x_t_curr, x_next_pseudo_t_curr = self.history[-1]
            p_t = x_t_curr - x_t_minus_tau
            f_t = x_next_pseudo_t_curr - x_next_pseudo_t_minus_tau
            g = self.gamma
            self.C_pp = (1 - g) * self.C_pp + g * np.outer(p_t, p_t)
            self.C_ff = (1 - g) * self.C_ff + g * np.outer(f_t, f_t)
            self.C_pf = (1 - g) * self.C_pf + g * np.outer(p_t, f_t)
            self.C_fp = (1 - g) * self.C_fp + g * np.outer(f_t, p_t)
            self.mu_p_lag = (1 - g) * self.mu_p_lag + g * p_t
            self.mu_x_curr = (1 - g) * self.mu_x_curr + g * x_t_curr
            self.mu_x_next = (1 - g) * self.mu_x_next + g * x_next_pseudo_t_curr
            self.M_p_c = (1 - g) * self.M_p_c + g * np.outer(p_t, x_t_curr)
            self.M_p_n = (1 - g) * self.M_p_n + g * np.outer(p_t, x_next_pseudo_t_curr)
            self.M_c_c = (1 - g) * self.M_c_c + g * np.outer(x_t_curr, x_t_curr)
            self.M_n_n = (1 - g) * self.M_n_n + g * np.outer(x_next_pseudo_t_curr, x_next_pseudo_t_curr)
            self.M_c_n = (1 - g) * self.M_c_n + g * np.outer(x_t_curr, x_next_pseudo_t_curr)
            self.n_updates += 1
            if self.n_updates < self.min_steps: return
            try:
                if self.use_whitened:
                    Cpp_inv_half = _sym_inv_sqrt(self.C_pp, reg=1e-6)
                    M_cca = Cpp_inv_half @ self.C_fp @ Cpp_inv_half
                    ev, ec = np.linalg.eig(M_cca.T)
                    idx = np.argmax(ev.real)
                    psi = ec[:, idx].real
                    alpha_0 = Cpp_inv_half @ psi
                else:
                    reg_mat = np.eye(self.dim) * 1e-6
                    left_matrix = self.C_pf @ np.linalg.pinv(self.C_ff + reg_mat) @ self.C_fp
                    ev, ec = scipy.linalg.eig(left_matrix, self.C_pp + reg_mat)
                    idx = np.argmax(ev.real)
                    alpha_0 = ec[:, idx].real
                alpha_0 /= np.linalg.norm(alpha_0) + 1e-15
                w_u = _sign_align(alpha_0, sign_ref)
                Cov_pc = self.M_p_c - np.outer(self.mu_p_lag, self.mu_x_curr)
                Cov_pn = self.M_p_n - np.outer(self.mu_p_lag, self.mu_x_next)
                var_iv = float(w_u @ Cov_pc @ w_u)
                cov_iv = float(w_u @ Cov_pn @ w_u)
                lam0_disc = cov_iv / max(var_iv, 1e-15)
                mu_c_val = float(w_u @ self.mu_x_curr)
                mu_n_val = float(w_u @ self.mu_x_next)
                if abs(1.0 - lam0_disc) > 1e-10:
                    self.Delta = (mu_n_val - lam0_disc * mu_c_val) / (1.0 - lam0_disc)
                else:
                    self.Delta = mu_c_val
                Cov_cc = self.M_c_c - np.outer(self.mu_x_curr, self.mu_x_curr)
                Cov_nn = self.M_n_n - np.outer(self.mu_x_next, self.mu_x_next)
                Cov_cn = self.M_c_n - np.outer(self.mu_x_curr, self.mu_x_next)
                var_c = float(w_u @ Cov_cc @ w_u)
                var_n = float(w_u @ Cov_nn @ w_u)
                cov_cn_val = float(w_u @ Cov_cn @ w_u)
                D_discrete = (var_n - 2 * lam0_disc * cov_cn_val + (lam0_disc**2) * var_c) / 2.0
                self.v0 = w_u
                self.lambda_u = max(0.0, np.log(max(lam0_disc, 1e-12)) / self.dt)
                self.D = max(D_discrete / self.dt, 1e-15)
                self.x_star = self.mu_x_curr + (self.Delta - mu_c_val) * self.v0
            except np.linalg.LinAlgError: pass

    def train(self, drift_func, T, dt, D, seed, sign_ref, init_center, r_neigh=np.inf):
        noise_std = np.sqrt(2.0 * D)
        rng = np.random.default_rng(seed)
        x = init_center + rng.uniform(-0.1, 0.1, self.dim)
        limit = r_neigh if r_neigh < np.inf else max(2.0, self.a * 2.0)
        self.reset()
        for t in range(T):
            x_next = x + drift_func(t * dt, x) * dt + noise_std * rng.standard_normal(self.dim) * np.sqrt(dt)
            out_of_bounds = np.linalg.norm(x_next - init_center) > limit
            if out_of_bounds:
                x = init_center + rng.normal(0, 0.05, self.dim)
                self.reset()
                continue
            self.update(x, x_next, sign_ref)
            x = x_next

    def get_params(self, sign_ref):
        return dict(w_u=_sign_align(self.v0.copy(), sign_ref), lambda_u=self.lambda_u, D=self.D, x_star=self.x_star.copy())

class DirectVARTracker:
    """Direct local VAR(1) estimator: least-squares on (x_{t+1}, x_t) pairs
    without whitening or temporal differencing. Baseline for comparison."""
    def __init__(self, dim=2, dt=0.01, rng_seed=0):
        self.dim = dim; self.dt = dt
        self.lambda_u = 0.0; self.D = 1e-6
        self.x_star = np.zeros(dim)
        self.v0 = np.random.default_rng(rng_seed).standard_normal(dim)
        self.v0 /= np.linalg.norm(self.v0) + 1e-15

    def reset(self): pass
    def update(self, x_t, x_next_pseudo, sign_ref): pass

    def train(self, drift_func, T, dt, D, seed, a, sign_ref, init_center, r_neigh=np.inf):
        noise_std = np.sqrt(2.0 * D)
        rng = np.random.default_rng(seed)
        x = init_center + rng.uniform(-0.1, 0.1, self.dim)
        limit = r_neigh if r_neigh < np.inf else max(2.0, a * 2.0)
        xs_curr, xs_next = [], []
        for t in range(T):
            x_next = x + drift_func(t * dt, x) * dt + noise_std * rng.standard_normal(self.dim) * np.sqrt(dt)
            if np.linalg.norm(x_next - init_center) > limit:
                x = init_center + rng.normal(0, 0.05, self.dim); continue
            xs_curr.append(x.copy()); xs_next.append(x_next.copy()); x = x_next
        if len(xs_curr) < self.dim + 2: return
        Xc = np.array(xs_curr); Xn = np.array(xs_next)
        # Estimate fixed point as mean
        self.x_star = np.mean(Xc, axis=0)
        # Center
        Xc_c = Xc - self.x_star; Xn_c = Xn - self.x_star
        # Least-squares: A_hat = (Xn_c^T Xc_c) (Xc_c^T Xc_c)^{-1}
        try:
            C_xx = Xc_c.T @ Xc_c / len(Xc_c)
            C_yx = Xn_c.T @ Xc_c / len(Xc_c)
            A_hat = C_yx @ np.linalg.pinv(C_xx)
            eigvals, V_left = np.linalg.eig(A_hat.T)
            order = np.argsort(np.abs(eigvals))[::-1]
            mu_u = eigvals[order[0]].real
            w_u = V_left[:, order[0]].real
            w_u /= np.linalg.norm(w_u) + 1e-15
            w_u = _sign_align(w_u, sign_ref)
            self.v0 = w_u
            self.lambda_u = max(0.0, np.log(max(abs(mu_u), 1e-12)) / self.dt)
            # Noise from residuals
            residuals = Xn_c - (A_hat @ Xc_c.T).T
            proj_resid = residuals @ w_u
            self.D = max(np.var(proj_resid) / (2.0 * self.dt), 1e-15)
        except np.linalg.LinAlgError: pass

    def get_params(self, sign_ref):
        return dict(w_u=_sign_align(self.v0.copy(), sign_ref), lambda_u=self.lambda_u, D=self.D, x_star=self.x_star.copy())

# ──────────────────────────────────────────────────────────────────────────────
# 5. Simulation
# ──────────────────────────────────────────────────────────────────────────────

def simulate_trajectory(x0, T, dt, drift_func, D, control_func, sys, a, rng):
    dim = len(x0)
    noise_std = np.sqrt(2.0 * D) if D > 0 else 0.0
    X = np.zeros((T, dim), dtype=float); x = x0.copy()
    energy = 0.0
    for t in range(T):
        X[t] = x
        x_u = float(np.dot(sys.w_u_true, x - sys.x_star))
        if x_u >= a: return X[:t + 1], "hit_B", energy
        if x_u <= -a: return X[:t + 1], "reached_A", energy
        u = control_func(x) if control_func is not None else np.zeros(dim)
        energy += np.sum(u**2) * dt
        x = x + (drift_func(t * dt, x) + u) * dt + (noise_std * rng.standard_normal(dim) * np.sqrt(dt) if D > 0 else np.zeros(dim))
    return X, "timeout", energy

def simulate_adaptive_trajectory(x0, T, dt, drift_func, D, tracker, sys, a, r_neigh, B_underactuated, rng, is_adaptive=False):
    dim = len(x0)
    noise_std = np.sqrt(2.0 * D) if D > 0 else 0.0
    X = np.zeros((T, dim), dtype=float); x = x0.copy()
    tracker.reset()
    energy = 0.0
    for t in range(T):
        X[t] = x
        x_u = float(np.dot(sys.w_u_true, x - sys.x_star))
        if x_u >= a: return X[:t + 1], "hit_B", energy
        if x_u <= -a: return X[:t + 1], "reached_A", energy
        params = tracker.get_params(sign_ref=sys.w_u_true)
        if D < 1e-8:
            u_ideal = relu_feedback(x, params['w_u'], params['lambda_u'], D, params['x_star'], a, r_neigh)
        else:
            u_ideal = erf_feedback(x, params['w_u'], params['lambda_u'], params['D'], params['x_star'], a, r_neigh)
        u_applied = apply_underactuated_constraint(u_ideal, B_underactuated) if B_underactuated is not None else u_ideal
        u_applied = np.clip(u_applied, -50.0, 50.0)
        energy += np.sum(u_applied**2) * dt
        drift_val = drift_func(t * dt, x)
        noise_val = noise_std * rng.standard_normal(dim) * np.sqrt(dt) if D > 0 else np.zeros(dim)
        x_next = x + (drift_val + u_applied) * dt + noise_val
        x_next_pseudo = x_next - u_applied * dt
        if is_adaptive:
            if r_neigh >= np.inf or np.linalg.norm(x - sys.x_star) <= r_neigh:
                tracker.update(x, x_next_pseudo, sys.w_u_true)
            else:
                tracker.reset()
        x = x_next
    return X, "timeout", energy

# ──────────────────────────────────────────────────────────────────────────────
# 6. Plotting Utilities
# ──────────────────────────────────────────────────────────────────────────────

def draw_invariant_manifolds(ax, sys):
    try:
        eigvals, V_right = np.linalg.eig(sys.J)
        order = np.argsort(eigvals.real)[::-1]
        v_u = V_right[:, order[0]].real; v_u /= np.linalg.norm(v_u)
        v_s = V_right[:, order[1]].real; v_s /= np.linalg.norm(v_s)
        is_pendulum = (sys.name == "Pendulum")
        eps = 1e-2; dt_int = 0.02; steps = 150
        c_s = 'black'; c_u = 'black'; ls_s = '-'; ls_u = '-'; lw_str = 1.5
        arrow_kwargs = dict(arrowstyle="-|>", color="black", lw=1.5, mutation_scale=15)
        def rk4(f, x, h):
            k1 = f(0, x); k2 = f(0, x+k1*h/2); k3 = f(0, x+k2*h/2); k4 = f(0, x+k3*h)
            return x + (k1+2*k2+2*k3+k4)*h/6.0
        if is_pendulum and sys.J[1, 1] == 0.0:
            th = np.linspace(sys.plot_xlim[0] - 1.0, sys.plot_xlim[1] + 1.0, 500)
            gl = sys.J[1, 0]
            omega_upper = 2.0 * np.sqrt(gl) * np.cos(th / 2.0)
            omega_lower = -2.0 * np.sqrt(gl) * np.cos(th / 2.0)
            ax.plot(th, omega_upper, color='black', lw=1.8, alpha=0.85, zorder=2)
            ax.plot(th, omega_lower, color='black', lw=1.8, alpha=0.85, zorder=2)
        else:
            for sign in [1, -1]:
                x = sys.x_star + sign * eps * v_s; traj = [x.copy()]
                for _ in range(steps): x = rk4(sys.drift, x, -dt_int); traj.append(x.copy())
                traj = np.array(traj)
                ax.plot(traj[:, 0], traj[:, 1], color=c_s, ls=ls_s, lw=lw_str, alpha=0.8, zorder=2)
                if len(traj) > 25:
                    ax.annotate("", xy=traj[10][:2], xytext=traj[25][:2], arrowprops=arrow_kwargs, zorder=3)
            for sign in [1, -1]:
                x = sys.x_star + sign * eps * v_u; traj = [x.copy()]
                for _ in range(steps): x = rk4(sys.drift, x, dt_int); traj.append(x.copy())
                traj = np.array(traj)
                ax.plot(traj[:, 0], traj[:, 1], color=c_u, ls=ls_u, lw=lw_str, alpha=0.8, zorder=2)
                if len(traj) > 25:
                    ax.annotate("", xy=traj[25][:2], xytext=traj[10][:2], arrowprops=arrow_kwargs, zorder=3)
    except Exception as e:
        print(f"Error drawing manifolds: {e}")

def draw_background(ax, sys, a):
    c_desired = "#176533"; c_undesired = "#ed2024"
    e_true = sys.w_u_true[:2] / np.linalg.norm(sys.w_u_true[:2])
    perp_dir = np.array([-e_true[1], e_true[0]])
    scale = 5.0; x0 = sys.x_star[:2]
    pt_desired = x0 + e_true * a
    ax.plot([pt_desired[0] - perp_dir[0]*scale, pt_desired[0] + perp_dir[0]*scale],
            [pt_desired[1] - perp_dir[1]*scale, pt_desired[1] + perp_dir[1]*scale],
            color=c_undesired, linestyle="--", lw=1.5, zorder=1)
    pt_undesired = x0 - e_true * a
    ax.plot([pt_undesired[0] - perp_dir[0]*scale, pt_undesired[0] + perp_dir[0]*scale],
            [pt_undesired[1] - perp_dir[1]*scale, pt_undesired[1] + perp_dir[1]*scale],
            color=c_desired, linestyle="--", lw=1.5, zorder=1)

def get_gradient_cmap(base_hex):
    base_rgb = mcolors.hex2color(base_hex)
    light_rgb = tuple(c + 0.5 * (1.0 - c) for c in base_rgb)
    dark_rgb = tuple(c * 0.85 for c in base_rgb)
    return mcolors.LinearSegmentedColormap.from_list("custom_grad", [light_rgb, base_rgb, dark_rgb])

def plot_trajectory_panels(panels, sys, a, is_single=False, combined=False, filepath=None):
    n = len(panels)
    if combined:
        fig, ax = plt.subplots(figsize=(6.0, 6.0))
        axes_flat = [ax] * n
        draw_background(ax, sys, a)
        draw_invariant_manifolds(ax, sys)
        e_true = sys.w_u_true[:2].copy(); e_true /= np.linalg.norm(e_true) + 1e-15
        ax.annotate("", xy=sys.x_star[:2] + (a * 0.6) * e_true, xytext=sys.x_star[:2],
                    arrowprops=dict(arrowstyle="-|>", color="#2b3c96", lw=2.5), zorder=8)
        legend_handles = []
    else:
        cols = min(n, 3); rows = (n + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(4.0*cols, 4.0*rows))
        axes_flat = np.atleast_1d(axes).flatten()
        fig.subplots_adjust(wspace=0.1, hspace=0.2)
    for i, panel in enumerate(panels):
        if not combined and i >= n: axes_flat[i].axis('off'); continue
        ax = axes_flat[i]
        display_label, col, est_dir, trajs, stats_str = panel
        if not combined:
            draw_background(ax, sys, a); draw_invariant_manifolds(ax, sys)
            e_true = sys.w_u_true[:2].copy(); e_true /= np.linalg.norm(e_true) + 1e-15
            ax.annotate("", xy=sys.x_star[:2] + (a * 0.6) * e_true, xytext=sys.x_star[:2],
                        arrowprops=dict(arrowstyle="-|>", color="black", lw=2.0), zorder=8)
        if est_dir is not None:
            e_est = est_dir[:2].copy(); e_est /= np.linalg.norm(e_est) + 1e-15
            ax.annotate("", xy=sys.x_star[:2] + (a * 0.6) * e_est, xytext=sys.x_star[:2],
                        arrowprops=dict(arrowstyle="-|>", color=col, lw=2.0, ls="dashed"), zorder=9)
        cmap = get_gradient_cmap(col); dark_col = cmap(1.0)
        alpha = 1.0 if is_single else max(0.4, min(0.8, 3.0/len(trajs)))
        line_weight = 2.0 if is_single else 1.2; scatter_size = 40 if is_single else 25
        for X, hs, _ in trajs:
            pts = X[:, :2]
            segs = np.concatenate([pts[:-1,None,:], pts[1:,None,:]], axis=1)
            lc = LineCollection(segs, cmap=cmap, norm=plt.Normalize(0, len(X)), lw=line_weight, alpha=alpha, zorder=4)
            lc.set_array(np.arange(len(X)-1)); ax.add_collection(lc)
            ax.scatter(*pts[-1], s=scatter_size, color=dark_col, edgecolors="white", lw=0.6, zorder=6)
        if not combined:
            ax.set_title(f"{display_label}\n{stats_str}", fontsize=9, pad=8, color=col if "Autonomous" not in display_label else "#333")
            ax.set_xlim(sys.plot_xlim); ax.set_ylim(sys.plot_ylim); ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect("equal")
        else:
            legend_handles.append(mlines.Line2D([], [], color=dark_col, lw=2.0, label=display_label))
    if combined:
        ax.set_xlim(sys.plot_xlim); ax.set_ylim(sys.plot_ylim); ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect("equal")
        ax.legend(handles=legend_handles, loc='upper center', ncol=3, fontsize=9, framealpha=0.9, bbox_to_anchor=(0.5, 1.0))
    if filepath:
        fig.savefig(filepath, bbox_inches="tight")
        fig.savefig(filepath.replace('.png','.pdf'), bbox_inches="tight")
        mode = "Combined" if combined else "Grid"; traj_type = "single" if is_single else "multi"
        print(f"Saved {mode} {traj_type} traj -> {filepath}")
    plt.close(fig)

# ──────────────────────────────────────────────────────────────────────────────
# 6b. Convergence data generation & plotting
# ──────────────────────────────────────────────────────────────────────────────

def generate_convergence_data(sys, a=0.5, D_train=0.01, dt=0.01, T_train=500, r_neigh=np.inf, num_trajs=20, tau=10, use_whitened=True, csv_filepath="figures/convergence_data.csv"):
    print(f"  -> Generating convergence data (saved to {csv_filepath})...")
    os.makedirs(os.path.dirname(csv_filepath), exist_ok=True)
    log_interval = 1
    with open(csv_filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Seed", "Step", "Cos_Sim", "Lambda_u", "D_val"])
        for seed in range(num_trajs):
            tracker = OnlineCCATracker(dim=sys.x_star.size, dt=dt, gamma=0.01, tau=tau, min_steps=20, a=a, rng_seed=seed, use_whitened=use_whitened)
            noise_std = np.sqrt(2.0 * D_train)
            rng = np.random.default_rng(seed)
            x = sys.x_star + rng.uniform(-0.1, 0.1, sys.x_star.size)
            tracker.reset()
            limit = r_neigh if r_neigh < np.inf else max(2.0, a * 2.0)
            for t in range(T_train):
                x_next = (x + sys.drift(t * dt, x) * dt + noise_std * rng.standard_normal(sys.x_star.size) * np.sqrt(dt))
                if np.linalg.norm(x_next - sys.x_star) > limit:
                    x = sys.x_star + rng.normal(0, 0.05, sys.x_star.size)
                    tracker.reset()
                    continue
                tracker.update(x, x_next, sys.w_u_true)
                x = x_next
                if t % log_interval == 0 and tracker.n_updates >= tracker.min_steps:
                    p = tracker.get_params(sys.w_u_true)
                    writer.writerow([seed, t, abs(float(np.dot(p['w_u'], sys.w_u_true))), p['lambda_u'], p['D']])

def plot_convergence_from_data(csv_filepath, sys_lambda_true=2.0, sys_D_true=0.01, filepath="figures/convergence.pdf",
                               layout="stacked", label_fontsize=12, tick_fontsize=11, legend_fontsize=10,
                               burn_in=30, direction_ylim=(0.8, 1.0), lambda_ylim=(0.5, 2.5), D_ylim=(0.005, 0.015)):
    set_ieee_style()
    print(f"  -> Plotting {layout} convergence from {csv_filepath}...")
    if not os.path.exists(csv_filepath):
        print(f"     [ERROR] Could not find data at: {csv_filepath}"); return
    data_by_step = {}
    with open(csv_filepath, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            step = int(row['Step'])
            if step not in data_by_step:
                data_by_step[step] = {'cos': [], 'lam': [], 'D': []}
            data_by_step[step]['cos'].append(float(row['Cos_Sim']))
            data_by_step[step]['lam'].append(float(row['Lambda_u']))
            data_by_step[step]['D'].append(float(row['D_val']))
    steps = sorted(list(data_by_step.keys()))
    if not steps: return
    burn_in_step = steps[min(burn_in, max(0, len(steps)-10))]
    valid_steps = [s for s in steps if s >= burn_in_step]
    cos_med, cos_q1, cos_q3 = [], [], []
    lam_med, lam_q1, lam_q3 = [], [], []
    D_med, D_q1, D_q3 = [], [], []
    for s in valid_steps:
        c_arr = np.array(data_by_step[s]['cos'])
        cos_med.append(np.median(c_arr)); cos_q1.append(np.percentile(c_arr, 25)); cos_q3.append(np.percentile(c_arr, 75))
        l_arr = np.clip(data_by_step[s]['lam'], lambda_ylim[0] - 1.0, lambda_ylim[1] + 1.0)
        lam_med.append(np.median(l_arr)); lam_q1.append(np.percentile(l_arr, 25)); lam_q3.append(np.percentile(l_arr, 75))
        d_arr = np.clip(data_by_step[s]['D'], D_ylim[0] - 0.005, D_ylim[1] + 0.005)
        D_med.append(np.median(d_arr)); D_q1.append(np.percentile(d_arr, 25)); D_q3.append(np.percentile(d_arr, 75))
    c_dir = "#2d307b"; c_eig = "#19421e"; c_D = "#a01b2a"
    if layout == "stacked":
        fig, axes = plt.subplots(3, 1, figsize=(5.5, 6), sharex=True)
        ax1, ax2, ax3 = axes; fig.subplots_adjust(hspace=0.1)
        ax1.plot(valid_steps, cos_med, color=c_dir, lw=2.0)
        ax1.fill_between(valid_steps, cos_q1, cos_q3, color=c_dir, alpha=0.2)
        ax1.axhline(1.0, color=c_dir, ls='--', lw=1.2, alpha=0.6)
        ax1.set_ylabel(r"$|\cos(\hat{w}_u, w_u)|$", color=c_dir, fontsize=label_fontsize)
        ax1.tick_params(axis='y', labelcolor=c_dir, labelsize=tick_fontsize); ax1.set_ylim(direction_ylim)
        ax1.grid(True, linestyle='-', alpha=0.3, color='gray')
        ax2.plot(valid_steps, lam_med, color=c_eig, lw=2.0)
        ax2.fill_between(valid_steps, lam_q1, lam_q3, color=c_eig, alpha=0.2)
        ax2.axhline(sys_lambda_true, color=c_eig, ls='--', lw=1.2, alpha=0.6)
        ax2.set_ylabel(r"$\hat{\lambda}_u$", color=c_eig, fontsize=label_fontsize)
        ax2.tick_params(axis='y', labelcolor=c_eig, labelsize=tick_fontsize); ax2.set_ylim(lambda_ylim)
        ax2.grid(True, linestyle='-', alpha=0.3, color='gray')
        ax3.plot(valid_steps, D_med, color=c_D, lw=2.0)
        ax3.fill_between(valid_steps, D_q1, D_q3, color=c_D, alpha=0.2)
        ax3.axhline(sys_D_true, color=c_D, ls='--', lw=1.2, alpha=0.6)
        ax3.set_ylabel(r"$\hat{D}$", color=c_D, fontsize=label_fontsize)
        ax3.tick_params(axis='y', labelcolor=c_D, labelsize=tick_fontsize); ax3.set_ylim(D_ylim)
        ax3.grid(True, linestyle='-', alpha=0.3, color='gray')
        ax3.set_xlabel("Training step", fontsize=label_fontsize)
        ax3.tick_params(axis='x', labelsize=tick_fontsize); ax3.set_xlim([valid_steps[0], valid_steps[-1]])
        fig.align_ylabels(axes)
    elif layout == "combined":
        fig, ax1 = plt.subplots(figsize=(6.5, 3.5)); fig.subplots_adjust(right=0.78)
        ax2 = ax1.twinx(); ax3 = ax1.twinx(); ax3.spines["right"].set_position(("axes", 1.15))
        ax1.plot(valid_steps, cos_med, color=c_dir, lw=2.0, label=r"Direction $|\cos(\hat{w}_u, w_u)|$")
        ax1.fill_between(valid_steps, cos_q1, cos_q3, color=c_dir, alpha=0.2)
        ax1.axhline(1.0, color=c_dir, ls='--', lw=1.2, alpha=0.6)
        ax2.plot(valid_steps, lam_med, color=c_eig, lw=2.0, label=r"Eigenvalue $\hat{\lambda}_u$")
        ax2.fill_between(valid_steps, lam_q1, lam_q3, color=c_eig, alpha=0.2)
        ax2.axhline(sys_lambda_true, color=c_eig, ls='--', lw=1.2, alpha=0.6)
        ax3.plot(valid_steps, D_med, color=c_D, lw=2.0, label=r"Noise $\hat{D}$")
        ax3.fill_between(valid_steps, D_q1, D_q3, color=c_D, alpha=0.2)
        ax3.axhline(sys_D_true, color=c_D, ls='--', lw=1.2, alpha=0.6)
        ax1.set_ylim(direction_ylim); ax1.set_yticks(np.linspace(direction_ylim[0], direction_ylim[1], 5))
        ax2.set_ylim(lambda_ylim); ax2.set_yticks(np.linspace(lambda_ylim[0], lambda_ylim[1], 5))
        ax3.set_ylim(D_ylim); ax3.set_yticks(np.linspace(D_ylim[0], D_ylim[1], 5))
        ax1.grid(True, linestyle='-', alpha=0.3, color='gray')
        ax1.set_xlabel("Training step", fontsize=label_fontsize)
        ax1.set_ylabel(r"$|\cos(\hat{w}_u, w_u)|$", color=c_dir, fontsize=label_fontsize)
        ax2.set_ylabel(r"$\hat{\lambda}_u$", color=c_eig, fontsize=label_fontsize)
        ax3.set_ylabel(r"$\hat{D}$", color=c_D, fontsize=label_fontsize)
        ax1.tick_params(axis='y', labelcolor=c_dir, labelsize=tick_fontsize)
        ax2.tick_params(axis='y', labelcolor=c_eig, labelsize=tick_fontsize)
        ax3.tick_params(axis='y', labelcolor=c_D, labelsize=tick_fontsize)
        ax1.tick_params(axis='x', labelsize=tick_fontsize); ax1.set_xlim([valid_steps[0], valid_steps[-1]])
        lines_1, labels_1 = ax1.get_legend_handles_labels()
        lines_2, labels_2 = ax2.get_legend_handles_labels()
        lines_3, labels_3 = ax3.get_legend_handles_labels()
        ax1.legend(lines_1 + lines_2 + lines_3, labels_1 + labels_2 + labels_3, loc='lower right', fontsize=legend_fontsize, framealpha=0.95)
    fig.savefig(filepath, bbox_inches="tight")
    fig.savefig(filepath.replace('.pdf', '.png'), bbox_inches="tight")
    print(f"     Saved {layout} convergence plot -> {filepath}")
    plt.close(fig)

# ──────────────────────────────────────────────────────────────────────────────
# 6c. Sweep & Scaling Helpers 
# ──────────────────────────────────────────────────────────────────────────────

def _plot_energy_panel(ax, data_en, noises, methods_in_data, pseudo_zero, map_x, label_fontsize, legend_fontsize, energy_ylim=None, energy_yticks=None):
    """Shared helper to plot the Mean Energy panel with SEM shading."""
    c_opt = PLOT_CONFIG["COLORS"].get("Optimal"); c_cca = PLOT_CONFIG["COLORS"].get("CCA"); c_adapt = PLOT_CONFIG["COLORS"].get("CCA (Adaptive)")
    lbl_map = {'Optimal': 'Model-based', 'CCA': 'Data-enabled', 'CCA (Adaptive)': 'Closed-loop'}
    color_map = {'Optimal': c_opt, 'CCA': c_cca, 'CCA (Adaptive)': c_adapt}
    ls_map = {'Optimal': '-', 'CCA': '--', 'CCA (Adaptive)': '-.'}
    marker_map = {'Optimal': 's', 'CCA': 'o', 'CCA (Adaptive)': 'd'}

    for method in methods_in_data:
        xs_zero, ys_zero, sem_zero = [], [], []
        xs_pos, ys_pos, sem_pos = [], [], []
        for n in noises:
            if method == "Optimal" and n == 0.0: continue
            row = next((r for r in data_en if r['Noise_D'] == n and r['Method'] == method), None)
            if row:
                sem = row['Std_Energy'] / np.sqrt(max(row['N_Success'], 1)) if row.get('N_Success', 0) > 0 else 0.0
                if n == 0.0:
                    xs_zero.append(map_x(n)); ys_zero.append(row['Mean_Energy']); sem_zero.append(sem)
                else:
                    xs_pos.append(map_x(n)); ys_pos.append(row['Mean_Energy']); sem_pos.append(sem)

        col = color_map.get(method, "#000"); ls = ls_map.get(method, "-"); lbl = lbl_map.get(method, method); mrk = marker_map.get(method, "s")
        
        if xs_zero: ax.plot(xs_zero, ys_zero, marker=mrk, color=col, ls='none', lw=1.5, markersize=5, label=lbl if not xs_pos else "")
        if xs_pos: ax.plot(xs_pos, ys_pos, marker=mrk, color=col, ls=ls, lw=1.5, markersize=5, label=lbl)

        ys_pos, sem_pos = np.array(ys_pos), np.array(sem_pos)
        ys_zero, sem_zero = np.array(ys_zero), np.array(sem_zero)

        if len(xs_pos) > 0 and np.any(sem_pos > 0):
            ax.fill_between(xs_pos, ys_pos - 1.96*sem_pos, ys_pos + 1.96*sem_pos, color=col, alpha=0.15)
        if len(xs_zero) > 0 and np.any(sem_zero > 0):
            ax.fill_between(xs_zero, ys_zero - 1.96*sem_zero, ys_zero + 1.96*sem_zero, color=col, alpha=0.15)

    all_en = [d['Mean_Energy'] for d in data_en if d['Method'] in methods_in_data and d['Mean_Energy'] > 0]
    if energy_ylim: ax.set_ylim(energy_ylim)
    elif all_en: ax.set_ylim(min(all_en)*0.8, max(all_en)*1.2)
    
    if energy_yticks: ax.set_yticks(energy_yticks)
    if pseudo_zero > 0: ax.axvline(x=pseudo_zero * 1.5, color='gray', linestyle=':', alpha=0.5)

    ax.set_ylabel("Mean control energy", fontsize=label_fontsize)
    ax.set_xscale('log')
    if noises:
        ax.set_xticks([map_x(n) for n in noises])
        ax.set_xticklabels(['0' if n == 0.0 else f"{n:.3g}" for n in noises], rotation=45, ha='right')
    ax.set_xlabel("Noise strength ($D$)", fontsize=label_fontsize)
    ax.legend(loc='upper left', fontsize=legend_fontsize)


def plot_noise_sweep(csv_filepath="figures/dimensionality_scaling_stats.csv", plot_grid=False, log_x=True, fontsize=10):
    set_ieee_style()
    print("\n" + "-"*80 + "\n Plotting Noise Sweep\n" + "-"*80)
    if not os.path.exists(csv_filepath):
        print(f"  [!] Data file {csv_filepath} not found."); return
    data = []
    with open(csv_filepath, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            data.append({'Dimension': int(row['Dimension']), 'Lambda_u': float(row['Lambda_u']),
                         'Noise_D': float(row['Noise_D']), 'Method': row['Method'],
                         'Success_Rate': float(row['Success_Rate']), 'Timeout_Rate': float(row['Timeout_Rate'])})
    dims = sorted(list(set(d['Dimension'] for d in data)))
    lams = sorted(list(set(d['Lambda_u'] for d in data)))
    noises = sorted(list(set(d['Noise_D'] for d in data)))
    non_zero_noises = [n for n in noises if n > 0]
    pseudo_zero = min(non_zero_noises) / 3.0 if non_zero_noises else 0.0001
    def map_x(n): return pseudo_zero if (log_x and n == 0.0) else n
    if not plot_grid:
        fig, ax = plt.subplots(1, 1, figsize=(5.5, 4.0))
        lambda_color_palette = sns.color_palette("rocket", n_colors=len(lams))
        dim_marker_list = ['-', '--', '-.', ':']
        for i, lam in enumerate(lams):
            for j, dim in enumerate(dims):
                xs_zero, ys_zero, xs_pos, ys_pos = [], [], [], []
                for noise in noises:
                    row_data = next((r for r in data if r['Dimension'] == dim and r['Lambda_u'] == lam and r['Noise_D'] == noise and r['Method'] == "CCA"), None)
                    if row_data is not None:
                        val = row_data['Success_Rate'] + row_data['Timeout_Rate']
                        if noise == 0.0: xs_zero.append(map_x(noise)); ys_zero.append(val)
                        else: xs_pos.append(map_x(noise)); ys_pos.append(val)
                if xs_zero: ax.plot(xs_zero, ys_zero, marker='o', markersize=4, color=lambda_color_palette[i], ls='none', lw=1.5)
                if xs_pos: ax.plot(xs_pos, ys_pos, marker='o', markersize=4, color=lambda_color_palette[i], ls=dim_marker_list[j], lw=1.5)
        ax.set_ylim(-0.05, 1.05)
        if log_x:
            ax.set_xscale('log'); ax.set_xticks([map_x(n) for n in noises])
            ax.set_xticklabels(['0' if n == 0 else f"{n:.3g}" for n in noises], rotation=45, ha='right')
            if non_zero_noises: ax.axvline(x=min(non_zero_noises)*0.6, color='gray', linestyle=':', alpha=0.5)
        ax.set_yticks([0, 0.5, 1]); ax.set_ylabel("Success rate", fontsize=fontsize); ax.set_xlabel("Noise strength ($D$)", fontsize=fontsize)
        col1 = [Line2D([0], [0], color='none', label=r'$\lambda_u$')]
        for i, lam in enumerate(lams): col1.append(Line2D([0], [0], color=lambda_color_palette[i], lw=2.0, label=f" {int(lam)}"))
        col2 = [Line2D([0], [0], color='none', label='Dim.')]
        for j, dim in enumerate(dims): col2.append(Line2D([0], [0], color='gray', marker='o', ls=dim_marker_list[j], lw=1.5, label=f" {dim}"))
        max_rows = max(len(col1), len(col2))
        while len(col1) < max_rows: col1.append(Line2D([0], [0], color='none', label=''))
        while len(col2) < max_rows: col2.append(Line2D([0], [0], color='none', label=''))
        ax.legend(handles=col1 + col2, loc="lower left", fontsize=8, ncol=2)
    fig.tight_layout()
    out_file = csv_filepath[:-4]
    if plot_grid: out_file += "_grid"
    if log_x: out_file += "_log"
    out_file += ".png"
    fig.savefig(out_file, bbox_inches="tight"); fig.savefig(out_file.replace(".png", ".pdf"), bbox_inches="tight")
    print(f"  Saved Noise Sweep Plot -> {out_file} (and .pdf)")

def simulate_and_save_noise_energy_sweep(sys, a=0.5, dt=0.01, r_neigh=np.inf, num_trajs=100, T_sim=2000, T_train=2500, tau=10, use_whitened=True, noise_levels=None, csv_filepath="figures/noise_energy_sweep.csv"):
    if noise_levels is None:
        noise_levels = [0.0, 0.001, 0.005, 0.01, 0.02, 0.05, 0.1]
    w_u_norm = sys.w_u_true / np.linalg.norm(sys.w_u_true)
    w_s = np.random.default_rng(42).standard_normal(sys.x_star.size)
    w_s -= np.dot(w_s, w_u_norm) * w_u_norm; w_s /= np.linalg.norm(w_s) + 1e-15
    os.makedirs(os.path.dirname(csv_filepath), exist_ok=True)
    with open(csv_filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["System", "Method", "Noise_D", "Success_Rate", "Mean_Energy", "Std_Energy", "N_Success"])
        for D_test in noise_levels:
            tracker = CCATracker(dim=sys.x_star.size, dt=dt, tau=tau, rng_seed=0, use_whitened=use_whitened)
            tracker.train(sys.drift, T_train, dt, max(D_test, 0.005), 0, a, sys.w_u_true, init_center=sys.x_star, r_neigh=r_neigh)
            tracker_online = OnlineCCATracker(dim=sys.x_star.size, dt=dt, gamma=0.01, tau=tau, min_steps=20, a=a, rng_seed=0, use_whitened=use_whitened)
            tracker_online.train(sys.drift, T_train, dt, max(D_test, 0.005), 0, sys.w_u_true, sys.x_star, r_neigh=r_neigh)
            kw = dict(a=a, r_neigh=r_neigh)
            if D_test < 1e-8: c_opt = partial(relu_feedback, w_u=sys.w_u_true, lambda_u=sys.lambda_u_true, D=D_test, x_fp=sys.x_star, **kw)
            else: c_opt = partial(erf_feedback, w_u=sys.w_u_true, lambda_u=sys.lambda_u_true, D=D_test, x_fp=sys.x_star, **kw)
            rng_ic = np.random.default_rng(11)
            ics = [sys.x_star + rng_ic.uniform(0.02, a * 0.9) * w_u_norm + rng_ic.uniform(-a * 0.4, a * 0.4) * w_s for _ in range(num_trajs)]
            methods_config = [
                ('Optimal', c_opt, False, False, None),
                ('CCA', None, True, False, tracker),
                ('CCA (Adaptive)', None, True, True, tracker_online)
            ]
            for method_name, ctrl, use_tracker, is_adaptive, trk in methods_config:
                successes, energies = 0, []
                for x0 in ics:
                    run_rng = np.random.default_rng(int(rng_ic.integers(0, 10**6)))
                    if use_tracker:
                        tr_copy = copy.deepcopy(trk)
                        traj, hit_side, energy = simulate_adaptive_trajectory(x0, T_sim, dt, sys.drift, D_test, tr_copy, sys, a, r_neigh, None, run_rng, is_adaptive=is_adaptive)
                    else:
                        traj, hit_side, energy = simulate_trajectory(x0, T_sim, dt, sys.drift, D_test, ctrl, sys, a, run_rng)
                    if hit_side == "reached_A" or (D_test < 1e-8 and hit_side == "timeout"):
                        successes += 1; energies.append(energy)
                succ_rate = successes / num_trajs
                mean_energy = np.mean(energies) if energies else 0.0
                std_energy = np.std(energies) if energies else 0.0
                writer.writerow([sys.name, method_name, D_test, succ_rate, mean_energy, std_energy, len(energies)])
                print(f"  {sys.name} | {method_name:<20} | D={D_test:.4f} | Success={succ_rate:.2f} | E={mean_energy:.3f} ± {std_energy:.3f}")

def plot_noise_energy_sweep_data(csv_filepath="figures/noise_energy_sweep.csv", sys_name=None, filepath="figures/noise_energy_sweep.pdf", fontsize=10):
    set_ieee_style()
    if not os.path.exists(csv_filepath): return
    
    data = []
    with open(csv_filepath, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            entry = {'System': row['System'], 'Method': row['Method'], 'Noise_D': float(row['Noise_D']),
                     'Success_Rate': float(row['Success_Rate']), 'Mean_Energy': float(row['Mean_Energy'])}
            if 'Std_Energy' in row and 'N_Success' in row:
                entry['Std_Energy'] = float(row['Std_Energy'])
                entry['N_Success'] = int(row['N_Success'])
            else:
                entry['Std_Energy'] = 0.0
                entry['N_Success'] = 0
            data.append(entry)
            
    if sys_name and any(d['System'] == sys_name for d in data):
        data = [d for d in data if d['System'] == sys_name]
        
    if not data: return

    noises = sorted(list(set(d['Noise_D'] for d in data)))
    methods_in_data = ["Optimal", "CCA", "CCA (Adaptive)"]
    non_zero_noises = [n for n in noises if n > 0]
    pseudo_zero = min(non_zero_noises) / 3.0 if non_zero_noises else 0.0001
    def map_x(n): return pseudo_zero if n == 0.0 else n

    fig, ax = plt.subplots(1, 1, figsize=(4.5, 3.5))
    _plot_energy_panel(ax, data, noises, methods_in_data, pseudo_zero, map_x, 
                       label_fontsize=fontsize, legend_fontsize=fontsize-1)

    ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    fig.tight_layout()
    fig.savefig(filepath, bbox_inches="tight")
    fig.savefig(filepath.replace('.pdf', '.png'), bbox_inches="tight")
    plt.close(fig)
    print(f"Saved energy sweep plot -> {filepath}")

def plot_noise_energy_sweep_single_sys(sys, a=0.5, dt=0.01, r_neigh=np.inf, num_trajs=100, T_sim=2000, T_train=2500, tau=10, use_whitened=True, noise_levels=None, filepath="figures/noise_energy_sweep.pdf"):
    csv_path = filepath.replace(".pdf", ".csv")
    simulate_and_save_noise_energy_sweep(sys, a, dt, r_neigh, num_trajs, T_sim, T_train, tau, use_whitened, noise_levels, csv_path)
    plot_noise_energy_sweep_data(csv_path, sys.name, filepath)

def plot_success_and_energy_vs_noise(csv_filepath, sys_name, filepath, layout="stacked", include_adaptive=False, label_fontsize=11, legend_fontsize=9):
    """Plots Success Rate and Mean Energy (with SEM error bars) vs Noise (D)."""
    set_ieee_style()
    if not os.path.exists(csv_filepath): print(f"     [ERROR] {csv_filepath} not found"); return
    data = []
    with open(csv_filepath, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['System'] == sys_name:
                entry = {'Method': row['Method'], 'Noise_D': float(row['Noise_D']),
                         'Success_Rate': float(row['Success_Rate']), 'Mean_Energy': float(row['Mean_Energy'])}
                if 'Std_Energy' in row and 'N_Success' in row:
                    entry['Std_Energy'] = float(row['Std_Energy']); entry['N_Success'] = int(row['N_Success'])
                else:
                    entry['Std_Energy'] = 0.0; entry['N_Success'] = 0
                data.append(entry)
                
    if not data: print(f"     [ERROR] No data for {sys_name}"); return
    
    noises = sorted(set(d['Noise_D'] for d in data))
    non_zero_noises = [n for n in noises if n > 0]
    pseudo_zero = min(non_zero_noises) / 3.0 if non_zero_noises else 0.0001
    def map_x(n): return pseudo_zero if n == 0.0 else n
    
    methods = ["Optimal", "CCA"]
    if include_adaptive: methods.append("CCA (Adaptive)")
    color_map = {'Optimal': PLOT_CONFIG["COLORS"]["Optimal"], 'CCA': PLOT_CONFIG["COLORS"]["CCA"], 'CCA (Adaptive)': PLOT_CONFIG["COLORS"]["CCA (Adaptive)"]}
    lbl_map = {'Optimal': 'Model-based', 'CCA': 'Data-enabled', 'CCA (Adaptive)': 'Closed-loop'}
    
    if layout == "stacked":
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(5.0, 5.5), sharex=True); fig.subplots_adjust(hspace=0.1)
        for method in methods:
            xs_zero, ys1_zero = [], []
            xs_pos, ys1_pos = [], []
            for n in noises:
                row = next((r for r in data if r['Noise_D'] == n and r['Method'] == method), None)
                if row:
                    if n == 0.0: xs_zero.append(map_x(n)); ys1_zero.append(row['Success_Rate'])
                    else: xs_pos.append(map_x(n)); ys1_pos.append(row['Success_Rate'])
            
            col = color_map.get(method, "#000"); lbl = lbl_map.get(method, method)
            if xs_zero: ax1.plot(xs_zero, ys1_zero, marker='o', color=col, ls='none', lw=1.5, markersize=5)
            if xs_pos: ax1.plot(xs_pos, ys1_pos, marker='o', color=col, ls='-', lw=1.5, markersize=5, label=lbl)
            elif xs_zero: ax1.plot([], [], marker='o', color=col, ls='-', lw=1.5, markersize=5, label=lbl)
            
        ax1.set_ylim(-0.05, 1.05); ax1.set_ylabel("Success rate", fontsize=label_fontsize)
        ax1.legend(loc='lower left', fontsize=legend_fontsize)
        if non_zero_noises: ax1.axvline(x=pseudo_zero * 1.5, color='gray', linestyle=':', alpha=0.5)

        _plot_energy_panel(ax2, data, noises, methods, pseudo_zero, map_x, label_fontsize, legend_fontsize)
        fig.align_ylabels([ax1, ax2])
        
    fig.savefig(filepath, bbox_inches="tight"); fig.savefig(filepath.replace('.pdf', '.png'), bbox_inches="tight")
    print(f"     Saved {layout} success/energy plot -> {filepath}"); plt.close(fig)


def plot_scaling_and_energy(dim_csv_filepath, energy_csv_filepath, filepath="figures/combined_scaling_energy.pdf",
                            include_adaptive=False, line_alpha=0.8, success_ylim=(0.45, 1.05), success_yticks=(0.5, 1.0),
                            energy_ylim=None, energy_yticks=None, legend_fontsize=12, label_fontsize=12, title_fontsize=13):
    set_ieee_style()
    fig, (ax_scale, ax_energy) = plt.subplots(2, 1, figsize=(6, 6.5), sharex=True)
    fig.subplots_adjust(hspace=0.1)
    print(f"  -> Plotting stacked combined figure...")
    
    noises, non_zero_noises = [], []
    pseudo_zero = 0.0001
    def map_x(n): return pseudo_zero if n == 0.0 else n

    # --- PANEL A: DIMENSIONALITY SCALING ---
    if os.path.exists(dim_csv_filepath):
        data_dim = []; methods_scale = ["CCA"]
        if include_adaptive: methods_scale.append("CCA (Adaptive)")
        with open(dim_csv_filepath, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row['Method'] not in methods_scale: continue
                data_dim.append({'Method': row['Method'], 'Dimension': int(row['Dimension']),
                                 'Lambda_u': float(row['Lambda_u']), 'Noise_D': float(row['Noise_D']),
                                 'Value': float(row['Success_Rate']) + float(row.get('Timeout_Rate', 0.0))})
        if data_dim:
            dims = sorted(list(set(d['Dimension'] for d in data_dim)))
            lams = sorted(list(set(d['Lambda_u'] for d in data_dim)))
            noises = sorted(list(set(d['Noise_D'] for d in data_dim)))
            non_zero_noises = [n for n in noises if n > 0]
            pseudo_zero = min(non_zero_noises) / 2.0 if non_zero_noises else 0.0001
            
            palettes = {
                lams[0]: [plt.cm.Purples(x) for x in np.linspace(0.85, 0.4, len(dims))],
                lams[1]: [plt.cm.Oranges(x) for x in np.linspace(0.85, 0.4, len(dims))]
            }
            dim_markers = ['o', 'D', 'X', '^', 's']
            for method in methods_scale:
                ls_val = '-' if method == "CCA" else '--'
                alpha_val = line_alpha if method == "CCA" else line_alpha * 0.75
                for i, lam in enumerate(lams):
                    colors = palettes[lam]
                    for j, dim in enumerate(dims):
                        xs_zero, ys_zero, xs_pos, ys_pos = [], [], [], []
                        for noise in noises:
                            row = next((r for r in data_dim if r['Method'] == method and r['Dimension'] == dim and r['Lambda_u'] == lam and r['Noise_D'] == noise), None)
                            if row:
                                if noise == 0.0: xs_zero.append(map_x(noise)); ys_zero.append(row['Value'])
                                else: xs_pos.append(map_x(noise)); ys_pos.append(row['Value'])
                        col = colors[j]; mrk = dim_markers[j % len(dim_markers)]
                        if xs_zero: ax_scale.plot(xs_zero, ys_zero, marker=mrk, markersize=5, color=col, ls='none', alpha=alpha_val)
                        if xs_pos: ax_scale.plot(xs_pos, ys_pos, marker=mrk, markersize=5, color=col, ls=ls_val, lw=1.5, alpha=alpha_val)
                        
            ax_scale.set_ylim(success_ylim); ax_scale.set_yticks(success_yticks)
            if non_zero_noises: ax_scale.axvline(x=pseudo_zero * 1.5, color='gray', linestyle=':', alpha=0.5)
            ax_scale.set_ylabel("Success rate", fontsize=label_fontsize)
            
            legend_elements = [Line2D([0], [0], color='none', label=r'$\lambda_u$')]
            for lam in lams:
                mid_color = palettes[lam][len(dims)//2]
                legend_elements.append(Line2D([0], [0], color=mid_color, lw=2.0, label=f" {int(lam)}"))
            legend_elements.append(Line2D([0], [0], color='none', label='Dim.'))
            for j, dim in enumerate(dims):
                gray_val = str(0.2 + (j / len(dims)) * 0.5)
                mrk = dim_markers[j % len(dim_markers)]
                legend_elements.append(Line2D([0], [0], color=gray_val, marker=mrk, ls='-', lw=1.5, label=f" {dim}"))
            if include_adaptive:
                legend_elements.append(Line2D([0], [0], color='none', label='Mode'))
                legend_elements.append(Line2D([0], [0], color='gray', ls='-', lw=1.5, label=' Static'))
                legend_elements.append(Line2D([0], [0], color='gray', ls='--', lw=1.5, label=' Adapt'))
            ax_scale.legend(handles=legend_elements, loc="lower left", fontsize=legend_fontsize, ncol=3 if include_adaptive else 2, framealpha=0.95)
    else: print(f"     [ERROR] Could not find Dimensionality CSV at: {dim_csv_filepath}")

    # --- PANEL B: ENERGY SWEEP ---
    if os.path.exists(energy_csv_filepath):
        data_en = []
        with open(energy_csv_filepath, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                entry = {'Method': row['Method'], 'Noise_D': float(row['Noise_D']), 'Mean_Energy': float(row['Mean_Energy'])}
                if 'Std_Energy' in row and 'N_Success' in row:
                    entry['Std_Energy'] = float(row['Std_Energy']); entry['N_Success'] = int(row['N_Success'])
                else:
                    entry['Std_Energy'] = 0.0; entry['N_Success'] = 0
                data_en.append(entry)
                
        if data_en:
            methods_in_data = ["Optimal", "CCA"]
            if include_adaptive: methods_in_data.append("CCA (Adaptive)")
            
            if not noises:
                noises = sorted(list(set(d['Noise_D'] for d in data_en)))
                non_zero_noises = [n for n in noises if n > 0]
                pseudo_zero = min(non_zero_noises) / 2.0 if non_zero_noises else 0.0001
                
            _plot_energy_panel(ax_energy, data_en, noises, methods_in_data, pseudo_zero, map_x, label_fontsize, legend_fontsize, energy_ylim, energy_yticks)
    else: print(f"     [ERROR] Could not find Energy CSV at: {energy_csv_filepath}")

    ax_scale.tick_params(axis='y', labelsize=label_fontsize); ax_energy.tick_params(axis='y', labelsize=label_fontsize)
    fig.align_ylabels([ax_scale, ax_energy])
    fig.savefig(filepath, bbox_inches="tight"); fig.savefig(filepath.replace('.pdf', '.png'), bbox_inches="tight")
    print(f"     Saved stacked scaling & energy plot -> {filepath}")
    plt.close(fig)

# ──────────────────────────────────────────────────────────────────────────────
# 7. Base Evaluation Suite
# ──────────────────────────────────────────────────────────────────────────────

def run_evaluation_suite(sys, B_restrictive=None, a=0.5, r_neigh=np.inf, D_eval=0.01, dt=0.01, num_trajs=1000, plot_trajs=3, spread_s=0.4, tau=10, min_steps=20, methods_to_run=None, use_whitened=True, file_suffix=""):
    print("\n" + "="*80)
    if "Ablation" in file_suffix: mode_str = "ABLATION STUDY"
    else: mode_str = "WHITENED CCA" if use_whitened else "STANDARD CCA (Algo 1)"
    print(f" Evaluating: {sys.name}  (a={a}, D={D_eval}, r_neigh={r_neigh}) | Mode: {mode_str}")
    print("="*80)
    T_sim, T_budget, seed = 5000, 2500, 11
    if methods_to_run is None: methods_to_run = ["Autonomous", "CCA"]
    D_train = max(D_eval, 0.005)
    
    alg_algo1 = CCATracker(dim=sys.x_star.size, dt=dt, tau=tau, rng_seed=0, use_whitened=False)
    alg_algo1.train(sys.drift, T_budget, dt, D_train, 0, a, sys.w_u_true, init_center=sys.x_star, r_neigh=r_neigh)
    p_algo1 = alg_algo1.get_params(sys.w_u_true)
    
    alg_whitened = CCATracker(dim=sys.x_star.size, dt=dt, tau=tau, rng_seed=0, use_whitened=True)
    alg_whitened.train(sys.drift, T_budget, dt, D_train, 0, a, sys.w_u_true, init_center=sys.x_star, r_neigh=r_neigh)
    p_whitened = alg_whitened.get_params(sys.w_u_true)
    
    alg_online = OnlineCCATracker(dim=sys.x_star.size, dt=dt, gamma=0.01, tau=tau, min_steps=20, a=a, rng_seed=0, use_whitened=use_whitened)
    alg_online.train(sys.drift, T_budget, dt, D_train, 0, sys.w_u_true, sys.x_star, r_neigh=r_neigh)
    p_online = alg_online.get_params(sys.w_u_true)
    
    print(f"  True:               lambda_u={sys.lambda_u_true:.3f}, w_u={sys.w_u_true}")
    if "CCA (Algo 1)" in methods_to_run: print(f"  CCA (Algo 1):   lambda_u={p_algo1['lambda_u']:.3f}, D={p_algo1['D']:.5f}, w_u={p_algo1['w_u']}")
    if "CCA (Whitened)" in methods_to_run: print(f"  CCA (Whitened): lambda_u={p_whitened['lambda_u']:.3f}, D={p_whitened['D']:.5f}, w_u={p_whitened['w_u']}")
    if "CCA (Adaptive)" in methods_to_run: print(f"  CCA (Adaptive): lambda_u={p_online['lambda_u']:.3f}, D={p_online['D']:.5f}, w_u={p_online['w_u']}")
    
    kw = dict(a=a, r_neigh=r_neigh)
    c_opt = partial(erf_feedback, w_u=sys.w_u_true, lambda_u=sys.lambda_u_true, D=D_eval, x_fp=sys.x_star, **kw)
    alg_default = alg_whitened if use_whitened else alg_algo1
    p_default = p_whitened if use_whitened else p_algo1
    
    all_methods = {
        "Autonomous":      (PLOT_CONFIG["LABELS"]["Autonomous"],      None,  PLOT_CONFIG["COLORS"]["Autonomous"],      None,               None,         False),
        "Optimal":         (PLOT_CONFIG["LABELS"]["Optimal"],         c_opt, PLOT_CONFIG["COLORS"]["Optimal"],         None,               None,         False),
        "CCA (Algo 1)":    (PLOT_CONFIG["LABELS"]["CCA (Algo 1)"],    None,  PLOT_CONFIG["COLORS"]["CCA (Algo 1)"],    p_algo1['w_u'],     alg_algo1,    False),
        "CCA (Whitened)":  (PLOT_CONFIG["LABELS"]["CCA (Whitened)"],  None,  PLOT_CONFIG["COLORS"]["CCA (Whitened)"],  p_whitened['w_u'],  alg_whitened, False),
        "CCA":             (PLOT_CONFIG["LABELS"]["CCA"],             None,  PLOT_CONFIG["COLORS"]["CCA"],             p_default['w_u'],   alg_default,  False),
        "CCA (Adaptive)":  (PLOT_CONFIG["LABELS"]["CCA (Adaptive)"],  None,  PLOT_CONFIG["COLORS"]["CCA (Adaptive)"],  p_online['w_u'],    alg_online,   True),
    }
    methods = [(m,) + all_methods[m] for m in methods_to_run if m in all_methods]
    
    w_u_norm = sys.w_u_true / np.linalg.norm(sys.w_u_true)
    w_s = np.random.default_rng(42).standard_normal(sys.x_star.size)
    w_s -= np.dot(w_s, w_u_norm) * w_u_norm; w_s /= np.linalg.norm(w_s) + 1e-15
    rng_ic = np.random.default_rng(seed)
    ics = [sys.x_star + rng_ic.uniform(0.02, a*0.9)*w_u_norm + rng_ic.uniform(-a*spread_s, a*spread_s)*w_s for _ in range(num_trajs)]
    
    single_trajs, multi_trajs = [], []
    for raw_name, display_label, ctrl, col, est_dir, tracker, is_adaptive in methods:
        t_list = []
        for x0 in ics:
            run_rng = np.random.default_rng(int(rng_ic.integers(0, 10**6)))
            if tracker is not None:
                X, hs, en = simulate_adaptive_trajectory(x0, T_sim, dt, sys.drift, D_eval, copy.deepcopy(tracker), sys, a, r_neigh, None, run_rng, is_adaptive=is_adaptive)
            else:
                X, hs, en = simulate_trajectory(x0, T_sim, dt, sys.drift, D_eval, ctrl, sys, a, run_rng)
            t_list.append((X, hs, en))
            
        n_success = sum(1 for _, hs, _ in t_list if hs == "reached_A" or (D_eval < 1e-8 and hs == "timeout"))
        n_time = sum(1 for _, hs, _ in t_list if hs == "timeout")
        n_fail = sum(1 for _, hs, _ in t_list if hs == "hit_B")
        energies_succ = [en for _, hs, en in t_list if hs == "reached_A" or (D_eval < 1e-8 and hs == "timeout")]
        energies_all = [en for _, _, en in t_list]
        avg_en_succ = np.mean(energies_succ) if energies_succ else 0.0
        avg_en_all = np.mean(energies_all) if energies_all else 0.0
        
        print(f"  {raw_name:<20} Reach A = {n_success}/{num_trajs} | Time = {n_time}/{num_trajs} | Fail = {n_fail}/{num_trajs} | E(succ) = {avg_en_succ:.3f} | E(all) = {avg_en_all:.3f}")
        stats_str = f"Reach A: {n_success}/{num_trajs} | Fail: {n_fail} | Time: {n_time}\nE(succ): {avg_en_succ:.2f} | E(all): {avg_en_all:.2f}"
        
        if raw_name in ["CCA", "CCA (Whitened)"]: stats_str += f"\n$\\hat{{\\lambda}}_u$={p_whitened['lambda_u']:.2f}, $\\hat{{D}}$={p_whitened['D']:.4f}"
        elif raw_name == "CCA (Algo 1)": stats_str += f"\n$\\hat{{\\lambda}}_u$={p_algo1['lambda_u']:.2f}, $\\hat{{D}}$={p_algo1['D']:.4f}"
        elif raw_name == "CCA (Adaptive)": stats_str += f"\n$\\hat{{\\lambda}}_u$={p_online['lambda_u']:.2f}, $\\hat{{D}}$={p_online['D']:.4f}"
        
        single_trajs.append((display_label, col, est_dir, [t_list[0]], stats_str))
        multi_trajs.append((display_label, col, est_dir, t_list[:plot_trajs], stats_str))
        
    clean_mode_str = mode_str.replace(" ", "_").replace("(", "").replace(")", "").replace(".", "")
    suffix = f"_{file_suffix}_{clean_mode_str}" if file_suffix else f"_{clean_mode_str}"
    plot_trajectory_panels(multi_trajs, sys, a, is_single=False, filepath=f"figures/{sys.name}{suffix}_multi_traj.png")
    plot_trajectory_panels(single_trajs, sys, a, is_single=True, combined=True, filepath=f"figures/{sys.name}{suffix}_single_traj_combined.png")

def run_dimensionality_scaling_experiment(dt=0.01, num_trajs=1000, T_sim=5000, use_whitened=True, dimensions=[2, 5, 10, 20], lambdas=[2.0, 5.0], noise_levels=[0.0, 0.001, 0.005, 0.01, 0.02, 0.05, 0.075, 0.1]):
    print("\n" + "="*80)
    mode_str = "WHITENED CCA" if use_whitened else "STANDARD CCA (Algo 1)"
    print(f" Dimensionality Scaling Experiment | Mode: {mode_str}")
    print("="*80)
    
    a = 0.5
    r_neigh = a
    clean_mode_str = mode_str.replace(" ", "_").replace("(", "").replace(")", "").replace(".", "")
    filepath = f"figures/dimensionality_scaling_stats_{clean_mode_str}.csv"
    
    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Dimension", "Lambda_u", "Noise_D", "Method", "Success_Rate", "Timeout_Rate", "Fail_Rate"])
        for dim in dimensions:
            for lam_u in lambdas:
                sys = make_random_linear_system(dim, lam_u)
                w_u_norm = sys.w_u_true / np.linalg.norm(sys.w_u_true)
                w_s = np.random.default_rng(42).standard_normal(dim)
                w_s -= np.dot(w_s, w_u_norm) * w_u_norm; w_s /= np.linalg.norm(w_s)+1e-15
                
                for D_test in noise_levels:
                    rng_ic = np.random.default_rng(11)
                    ics = [sys.x_star + rng_ic.uniform(0.02, a*0.9)*w_u_norm + rng_ic.uniform(-a*0.4, a*0.4)*w_s for _ in range(num_trajs)]
                    kw = dict(a=a, r_neigh=r_neigh)
                    
                    if D_test < 1e-8: c_opt = partial(relu_feedback, w_u=sys.w_u_true, lambda_u=sys.lambda_u_true, D=D_test, x_fp=sys.x_star, **kw)
                    else: c_opt = partial(erf_feedback, w_u=sys.w_u_true, lambda_u=sys.lambda_u_true, D=D_test, x_fp=sys.x_star, **kw)
                    
                    opt_results = [simulate_trajectory(x0, T_sim, dt, sys.drift, D_test, c_opt, sys, a, np.random.default_rng())[1] for x0 in ics]
                    hits_opt = sum(1 for hs in opt_results if hs == "reached_A" or (D_test < 1e-8 and hs == "timeout"))
                    time_opt = sum(1 for hs in opt_results if hs == "timeout")
                    fail_opt = sum(1 for hs in opt_results if hs == "hit_B")
                    
                    D_train = max(D_test, 0.005)
                    alg = CCATracker(dim=dim, dt=dt, tau=10, rng_seed=0, use_whitened=use_whitened)
                    alg.train(sys.drift, 2500, dt, D_train, 0, a, sys.w_u_true, init_center=sys.x_star)
                    alg_online = OnlineCCATracker(dim=dim, dt=dt, gamma=0.01, tau=10, min_steps=20, a=a, rng_seed=0, use_whitened=use_whitened)
                    alg_online.train(sys.drift, 2500, dt, D_train, 0, sys.w_u_true, sys.x_star)
                    
                    cca_results = [simulate_adaptive_trajectory(x0, T_sim, dt, sys.drift, D_test, copy.deepcopy(alg), sys, a, r_neigh, None, np.random.default_rng(), is_adaptive=False)[1] for x0 in ics]
                    hits_cca = sum(1 for hs in cca_results if hs == "reached_A" or (D_test < 1e-8 and hs == "timeout"))
                    time_cca = sum(1 for hs in cca_results if hs == "timeout")
                    fail_cca = sum(1 for hs in cca_results if hs == "hit_B")
                    
                    cca_adapt_results = [simulate_adaptive_trajectory(x0, T_sim, dt, sys.drift, D_test, copy.deepcopy(alg_online), sys, a, r_neigh, None, np.random.default_rng(), is_adaptive=True)[1] for x0 in ics]
                    hits_adapt = sum(1 for hs in cca_adapt_results if hs == "reached_A" or (D_test < 1e-8 and hs == "timeout"))
                    time_adapt = sum(1 for hs in cca_adapt_results if hs == "timeout")
                    fail_adapt = sum(1 for hs in cca_adapt_results if hs == "hit_B")
                    
                    writer.writerow([dim, lam_u, D_test, "Optimal", hits_opt/num_trajs, time_opt/num_trajs, fail_opt/num_trajs])
                    writer.writerow([dim, lam_u, D_test, "CCA", hits_cca/num_trajs, time_cca/num_trajs, fail_cca/num_trajs])
                    writer.writerow([dim, lam_u, D_test, "CCA (Adaptive)", hits_adapt/num_trajs, time_adapt/num_trajs, fail_adapt/num_trajs])
                    print(f"  Dim {dim:2}, lam_u {lam_u:3}, D {D_test:5.3f} | Opt [A: {hits_opt/num_trajs:.2f}] | CCA [A: {hits_cca/num_trajs:.2f}] | Adapt [A: {hits_adapt/num_trajs:.2f}]")
                    
    print(f"Saved -> {filepath}")
    plot_noise_sweep(filepath, plot_grid=False, log_x=True)


# ──────────────────────────────────────────────────────────────────────────────
# 8. Baseline & Ablation Studies (For Reviewers)
# ──────────────────────────────────────────────────────────────────────────────

def run_baseline_comparison(sys, a=0.5, r_neigh=np.inf, dt=0.01, num_trajs=100, T_sim=2000, T_train=2500, tau=10, noise_levels=None, csv_filepath="figures/baselines.csv"):
    """Compares the Committor formulation against LQR and Sign-Gated (ReLU) baselines using BOTH Oracle and Estimated parameters."""
    print(f"\n -> Running Baseline Comparison for {sys.name}...")
    if noise_levels is None: noise_levels = [0.0, 0.005, 0.01, 0.05]
    
    w_u_norm = sys.w_u_true / np.linalg.norm(sys.w_u_true)
    w_s = np.random.default_rng(42).standard_normal(sys.x_star.size)
    w_s -= np.dot(w_s, w_u_norm) * w_u_norm; w_s /= np.linalg.norm(w_s) + 1e-15
    os.makedirs(os.path.dirname(csv_filepath), exist_ok=True)
    
    with open(csv_filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["System", "Method", "Noise_D", "Success_Rate", "Timeout_Rate", "Fail_Rate", "Mean_Energy", "Std_Energy"])
        
        for D_test in noise_levels:
            D_train = max(D_test, 0.005)
            kw = dict(a=a, r_neigh=r_neigh)
            
            # --- 1. Oracle Controllers (True System Parameters) ---
            if D_test < 1e-8:
                c_oracle_commit = partial(relu_feedback, w_u=sys.w_u_true, lambda_u=sys.lambda_u_true, D=D_test, x_fp=sys.x_star, **kw)
            else:
                c_oracle_commit = partial(erf_feedback, w_u=sys.w_u_true, lambda_u=sys.lambda_u_true, D=D_test, x_fp=sys.x_star, **kw)
            
            c_oracle_lqr = partial(lqr_feedback, w_u=sys.w_u_true, lambda_u=sys.lambda_u_true, D=D_test, x_fp=sys.x_star, **kw)
            c_oracle_relu = partial(relu_feedback, w_u=sys.w_u_true, lambda_u=sys.lambda_u_true, D=D_test, x_fp=sys.x_star, **kw)

            # --- 2. Train Trackers (Data-Driven Parameters) ---
            alg_wsvd = CCATracker(dim=sys.x_star.size, dt=dt, tau=tau, rng_seed=0, use_whitened=True)
            alg_wsvd.train(sys.drift, T_train, dt, D_train, 0, a, sys.w_u_true, init_center=sys.x_star, r_neigh=r_neigh)
            p_wsvd = alg_wsvd.get_params(sys.w_u_true)

            alg_var = DirectVARTracker(dim=sys.x_star.size, dt=dt, rng_seed=0)
            alg_var.train(sys.drift, T_train, dt, D_train, 0, a, sys.w_u_true, sys.x_star, r_neigh=r_neigh)
            p_var = alg_var.get_params(sys.w_u_true)

            # --- 3. Estimated Controllers (Using CCA/WSVD Parameters) ---
            if D_test < 1e-8:
                c_cca_commit = partial(relu_feedback, w_u=p_wsvd['w_u'], lambda_u=p_wsvd['lambda_u'], D=p_wsvd['D'], x_fp=p_wsvd['x_star'], **kw)
                c_var_commit = partial(relu_feedback, w_u=p_var['w_u'], lambda_u=p_var['lambda_u'], D=p_var['D'], x_fp=p_var['x_star'], **kw)
            else:
                c_cca_commit = partial(erf_feedback, w_u=p_wsvd['w_u'], lambda_u=p_wsvd['lambda_u'], D=p_wsvd['D'], x_fp=p_wsvd['x_star'], **kw)
                c_var_commit = partial(erf_feedback, w_u=p_var['w_u'], lambda_u=p_var['lambda_u'], D=p_var['D'], x_fp=p_var['x_star'], **kw)

            c_cca_lqr = partial(lqr_feedback, w_u=p_wsvd['w_u'], lambda_u=p_wsvd['lambda_u'], D=p_wsvd['D'], x_fp=p_wsvd['x_star'], **kw)
            c_cca_relu = partial(relu_feedback, w_u=p_wsvd['w_u'], lambda_u=p_wsvd['lambda_u'], D=p_wsvd['D'], x_fp=p_wsvd['x_star'], **kw)

            # --- Generate Initial Conditions ---
            rng_ic = np.random.default_rng(11)
            ics = [sys.x_star + rng_ic.uniform(0.02, a * 0.9) * w_u_norm + rng_ic.uniform(-a * 0.4, a * 0.4) * w_s for _ in range(num_trajs)]

            # Compare apples to apples
            methods_config = [
                ('Oracle Committor', c_oracle_commit),
                ('Oracle LQR', c_oracle_lqr),
                ('Oracle ReLU', c_oracle_relu),
                ('Est CCA Committor', c_cca_commit),
                ('Est CCA LQR', c_cca_lqr),
                ('Est CCA ReLU', c_cca_relu),
                ('Est VAR Committor', c_var_commit)
            ]

            for method_name, ctrl in methods_config:
                successes, timeouts, failures, energies = 0, 0, 0, []
                for x0 in ics:
                    run_rng = np.random.default_rng(int(rng_ic.integers(0, 10**6)))
                    traj, hit_side, energy = simulate_trajectory(x0, T_sim, dt, sys.drift, D_test, ctrl, sys, a, run_rng)
                    
                    if hit_side == "timeout": timeouts += 1
                    elif hit_side == "hit_B": failures += 1
                    
                    if hit_side == "reached_A" or (D_test < 1e-8 and hit_side == "timeout"):
                        successes += 1
                        energies.append(energy)
                
                succ_rate = successes / num_trajs
                to_rate = timeouts / num_trajs
                fail_rate = failures / num_trajs
                mean_energy = np.mean(energies) if energies else 0.0
                std_energy = np.std(energies) if energies else 0.0
                writer.writerow([sys.name, method_name, D_test, succ_rate, to_rate, fail_rate, mean_energy, std_energy])
                print(f"  {method_name:<20} | D={D_test:.4f} | SR={succ_rate:.2f} | TO={to_rate:.2f} | Fail={fail_rate:.2f} | E={mean_energy:.3f}")

def plot_baseline_comparison(csv_filepath="figures/baselines.csv", filepath="figures/baseline_comparison.pdf"):
    """Plots Success Rate and Mean Control Energy for the baseline comparisons."""
    set_ieee_style()
    if not os.path.exists(csv_filepath): 
        print(f" [!] Data file {csv_filepath} not found.")
        return
        
    data = []
    with open(csv_filepath, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader: 
            data.append({
                'Method': row['Method'], 
                'Noise_D': float(row['Noise_D']), 
                'Success_Rate': float(row['Success_Rate']),
                'Mean_Energy': float(row['Mean_Energy']),
                'Std_Energy': float(row.get('Std_Energy', 0.0))
            })
            
    noises = sorted(list(set(d['Noise_D'] for d in data)))
    non_zero_noises = [n for n in noises if n > 0]
    pseudo_zero = min(non_zero_noises) / 3.0 if non_zero_noises else 0.0001
    def map_x(n): return pseudo_zero if n == 0.0 else n

    # Create a 2-panel stacked plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6.5, 6.5), sharex=True)
    fig.subplots_adjust(hspace=0.1)
    
    colors = {
        "Oracle Committor": "#1a6faf", 
        "Oracle LQR": "#ff7f0e", 
        "Oracle ReLU": "#2ca02c", 
        "Est CCA Committor": "#005f38", 
        "Est CCA LQR": "#e377c2",
        "Est CCA ReLU": "#8c564b",
        "Est VAR Committor": "#d62728"
    }
    markers = {
        "Oracle Committor": "s", "Oracle LQR": "v", "Oracle ReLU": "^", 
        "Est CCA Committor": "o", "Est CCA LQR": "p", "Est CCA ReLU": "*", "Est VAR Committor": "D"
    }
    linestyles = {
        "Oracle Committor": "-", "Oracle LQR": ":", "Oracle ReLU": "--", 
        "Est CCA Committor": "-", "Est CCA LQR": ":", "Est CCA ReLU": "--", "Est VAR Committor": "-."
    }
    
    for method in colors.keys():
        xs_zero, ys_sr_zero, ys_en_zero, sem_en_zero = [], [], [], []
        xs_pos, ys_sr_pos, ys_en_pos, sem_en_pos = [], [], [], []
        
        for n in noises:
            row = next((r for r in data if r['Method'] == method and r['Noise_D'] == n), None)
            if row:
                # Approximate SEM (Assuming 100 trajectories per condition as set in the script)
                sem = row['Std_Energy'] / np.sqrt(100)
                if n == 0.0: 
                    xs_zero.append(map_x(n))
                    ys_sr_zero.append(row['Success_Rate'])
                    ys_en_zero.append(row['Mean_Energy'])
                    sem_en_zero.append(sem)
                else: 
                    xs_pos.append(map_x(n))
                    ys_sr_pos.append(row['Success_Rate'])
                    ys_en_pos.append(row['Mean_Energy'])
                    sem_en_pos.append(sem)
                    
        col, mrk, ls = colors[method], markers[method], linestyles[method]
        
        # --- TOP PANEL: Success Rate ---
        if xs_zero: ax1.plot(xs_zero, ys_sr_zero, marker=mrk, color=col, ls='none', lw=1.5, markersize=6)
        if xs_pos: ax1.plot(xs_pos, ys_sr_pos, marker=mrk, color=col, ls=ls, lw=1.5, markersize=6, label=method)
        
        # --- BOTTOM PANEL: Control Energy ---
        if xs_zero: 
            ax2.plot(xs_zero, ys_en_zero, marker=mrk, color=col, ls='none', lw=1.5, markersize=6)
            # Add subtle error bars for zero noise if energy > 0
            if any(e > 0 for e in ys_en_zero):
                ax2.errorbar(xs_zero, ys_en_zero, yerr=1.96*np.array(sem_en_zero), color=col, fmt='none', alpha=0.5)

        if xs_pos: 
            ax2.plot(xs_pos, ys_en_pos, marker=mrk, color=col, ls=ls, lw=1.5, markersize=6)
            # Add SEM shading for energy
            ys_en_pos_arr = np.array(ys_en_pos)
            sem_en_pos_arr = np.array(sem_en_pos)
            ax2.fill_between(xs_pos, ys_en_pos_arr - 1.96*sem_en_pos_arr, ys_en_pos_arr + 1.96*sem_en_pos_arr, color=col, alpha=0.15)
        
    # Formatting Top Panel (Success Rate)
    ax1.set_ylim(-0.05, 1.05)
    ax1.set_ylabel("Success rate", fontsize=10)
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9, framealpha=0.95)
    ax1.grid(True, linestyle='-', alpha=0.3, color='gray')
    
    # Formatting Bottom Panel (Mean Energy)
    ax2.set_xscale('log')
    ax2.set_xticks([map_x(n) for n in noises])
    ax2.set_xticklabels(['0' if n == 0.0 else f"{n:.3g}" for n in noises], rotation=45, ha='right')
    ax2.set_ylabel("Mean control energy", fontsize=10)
    ax2.set_xlabel("Noise strength ($D$)", fontsize=10)
    ax2.grid(True, linestyle='-', alpha=0.3, color='gray')
    
    if non_zero_noises: 
        ax1.axvline(x=pseudo_zero * 1.5, color='gray', linestyle=':', alpha=0.5)
        ax2.axvline(x=pseudo_zero * 1.5, color='gray', linestyle=':', alpha=0.5)
    
    fig.align_ylabels([ax1, ax2])
    fig.tight_layout()
    fig.savefig(filepath, bbox_inches="tight")
    fig.savefig(filepath.replace('.pdf', '.png'), bbox_inches="tight")
    print(f" Saved 2-Panel Baseline Plot -> {filepath}")
    plt.close(fig)



def run_gating_weighting_ablation(sys, a=0.5, dt=0.01, D_eval=0.01, num_trajs=100, T_sim=2000, T_train=2500, csv_filepath="figures/ablation_gating_weighting.csv"):
    """
    Isolates the practical weakness of discrete-time tracking bias far from the saddle.
    Compares:
      1. No gating
      2. Coordinate gating by estimated |x_u| and |x_s|
      3. Weighted estimation (Gaussian kernel modulating gamma near saddle)
    Tracks changes in \hat\lambda_u, success rate, and control energy.
    """
    print(f"\n -> Running Gating & Weighting Ablation for {sys.name} at D={D_eval}...")
    os.makedirs(os.path.dirname(csv_filepath), exist_ok=True)
    
    w_u_norm = sys.w_u_true / np.linalg.norm(sys.w_u_true)
    w_s = np.random.default_rng(42).standard_normal(sys.x_star.size)
    w_s -= np.dot(w_s, w_u_norm) * w_u_norm
    w_s /= np.linalg.norm(w_s) + 1e-15
    rng_ic = np.random.default_rng(11)
    
    # BUG FIX 1: Ensure initial conditions start INSIDE the coordinate gating box (max 0.8a and 0.4a)
    ics = [sys.x_star + rng_ic.uniform(0.1, a * 0.8) * w_u_norm + rng_ic.uniform(-a * 0.4, a * 0.4) * w_s for _ in range(num_trajs)]

    conditions = [
        {"name": "No Gating", "type": "none", "xu": np.inf, "xs": np.inf, "sigma": np.inf},
        {"name": "Coordinate Gating", "type": "coordinate", "xu": a * 0.9, "xs": a * 0.5, "sigma": np.inf},
        {"name": "Weighted Estimation", "type": "weighting", "xu": np.inf, "xs": np.inf, "sigma": 0.3}
    ]

    with open(csv_filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["System", "Condition", "Success_Rate", "Timeout_Rate", "Fail_Rate", "Mean_Energy", "Mean_Lambda_u"])
        
        for cond in conditions:
            base_gamma = 0.01
            tracker = OnlineCCATracker(dim=sys.x_star.size, dt=dt, gamma=base_gamma, tau=10, min_steps=20, a=a, rng_seed=0, use_whitened=True)
            
            # BUG FIX 2: Pre-train locally (r_neigh=a) so nonlinear systems don't poison the initial estimate
            tracker.train(sys.drift, T_train, dt, D_eval, 0, sys.w_u_true, sys.x_star, r_neigh=a)
            
            successes, timeouts, failures, energies, lambda_estimates = 0, 0, 0, [], []
            
            for x0 in ics:
                run_rng = np.random.default_rng(int(rng_ic.integers(0, 10**6)))
                tr_copy = copy.deepcopy(tracker)
                x = x0.copy()
                energy = 0.0
                hit_side = "timeout"
                
                for t in range(T_sim):
                    x_u_true = float(np.dot(sys.w_u_true, x - sys.x_star))
                    if x_u_true >= a: hit_side = "hit_B"; break
                    if x_u_true <= -a: hit_side = "reached_A"; break
                    
                    params = tr_copy.get_params(sign_ref=sys.w_u_true)
                    
                    # 1. Check Coordinate Gating
                    in_trust_region = True
                    if cond["type"] == "coordinate":
                        dx = x - params['x_star']
                        x_u_mag = abs(float(np.dot(params['w_u'], dx)))
                        x_s_mag = np.linalg.norm(dx - float(np.dot(params['w_u'], dx)) * params['w_u'])
                        in_trust_region = (x_u_mag <= cond["xu"]) and (x_s_mag <= cond["xs"])

                    # 2. Compute Control (cut if gated)
                    if in_trust_region:
                        if D_eval < 1e-8:
                            u_ideal = relu_feedback(x, params['w_u'], params['lambda_u'], D_eval, params['x_star'], a, np.inf)
                        else:
                            u_ideal = erf_feedback(x, params['w_u'], params['lambda_u'], params['D'], params['x_star'], a, np.inf)
                    else:
                        u_ideal = np.zeros_like(x)
                        tr_copy.reset()

                    u_applied = np.clip(u_ideal, -50.0, 50.0)
                    energy += np.sum(u_applied**2) * dt
                    
                    # 3. Environment Step
                    drift_val = sys.drift(t * dt, x)
                    noise_val = np.sqrt(2.0 * D_eval) * run_rng.standard_normal(len(x)) * np.sqrt(dt) if D_eval > 0 else 0.0
                    x_next = x + (drift_val + u_applied) * dt + noise_val
                    x_next_pseudo = x_next - u_applied * dt
                    
                    # 4. Tracker Update & Weighting
                    if in_trust_region:
                        if cond["type"] == "weighting":
                            dist_sq = np.sum((x - params['x_star'])**2)
                            weight = np.exp(-dist_sq / (2.0 * cond["sigma"]**2))
                            tr_copy.gamma = base_gamma * weight
                        else:
                            tr_copy.gamma = base_gamma
                            
                        tr_copy.update(x, x_next_pseudo, sys.w_u_true)
                        
                    x = x_next
                
                # Log outcomes
                if hit_side == "timeout": timeouts += 1
                elif hit_side == "hit_B": failures += 1
                
                if hit_side == "reached_A" or (D_eval < 1e-8 and hit_side == "timeout"):
                    successes += 1
                    energies.append(energy)
                    lambda_estimates.append(tr_copy.lambda_u)
            
            succ_rate = successes / num_trajs
            to_rate = timeouts / num_trajs
            fail_rate = failures / num_trajs
            mean_energy = np.mean(energies) if energies else 0.0
            mean_lambda = np.mean(lambda_estimates) if lambda_estimates else 0.0
            
            writer.writerow([sys.name, cond["name"], succ_rate, to_rate, fail_rate, mean_energy, mean_lambda])
            print(f"  {cond['name']:<20} | SR={succ_rate:.2f} | Fail={fail_rate:.2f} | E={mean_energy:.3f} | Lam_u_est={mean_lambda:.3f} (True={sys.lambda_u_true:.3f})")


def run_hyperparameter_ablation(sys, a=0.5, dt=0.01, D_eval=0.01, num_trajs=100, T_sim=2000, T_train=2500, csv_filepath="figures/ablation_hyperparameters.csv"):
    """
    Evaluates robustness to the central hyperparameters in Algorithm 1:
    Lag (\tau) and EMA discount (\gamma).
    """
    print(f"\n -> Running Hyperparameter Ablation (Tau and Gamma) for {sys.name}...")
    os.makedirs(os.path.dirname(csv_filepath), exist_ok=True)
    
    w_u_norm = sys.w_u_true / np.linalg.norm(sys.w_u_true)
    w_s = np.random.default_rng(42).standard_normal(sys.x_star.size)
    w_s -= np.dot(w_s, w_u_norm) * w_u_norm
    w_s /= np.linalg.norm(w_s) + 1e-15
    rng_ic = np.random.default_rng(11)
    
    # Safe ICs to avoid early boundary clipping
    ics = [sys.x_star + rng_ic.uniform(0.02, a * 0.8) * w_u_norm + rng_ic.uniform(-a * 0.4, a * 0.4) * w_s for _ in range(num_trajs)]

    # Test a grid of values
    tau_list = [1, 5, 10, 20]
    gamma_list = [0.001, 0.01, 0.05, 0.1]

    with open(csv_filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["System", "Tau", "Gamma", "Success_Rate", "Fail_Rate", "Mean_Energy"])
        
        # 1. Sweep Tau (fixing Gamma = 0.01)
        for tau in tau_list:
            tracker = OnlineCCATracker(dim=sys.x_star.size, dt=dt, gamma=0.01, tau=tau, min_steps=20, a=a, rng_seed=0, use_whitened=True)
            
            # BUG FIX 2: Pre-train locally (r_neigh=a)
            tracker.train(sys.drift, T_train, dt, D_eval, 0, sys.w_u_true, sys.x_star, r_neigh=a)
            
            successes, failures, energies = 0, 0, []
            for x0 in ics:
                run_rng = np.random.default_rng(int(rng_ic.integers(0, 10**6)))
                tr_copy = copy.deepcopy(tracker)
                traj, hit_side, energy = simulate_adaptive_trajectory(x0, T_sim, dt, sys.drift, D_eval, tr_copy, sys, a, np.inf, None, run_rng, is_adaptive=True)
                
                if hit_side == "hit_B": failures += 1
                if hit_side == "reached_A" or (D_eval < 1e-8 and hit_side == "timeout"):
                    successes += 1; energies.append(energy)
            
            sr, fr, me = successes/num_trajs, failures/num_trajs, np.mean(energies) if energies else 0.0
            writer.writerow([sys.name, tau, 0.01, sr, fr, me])
            print(f"  Tau={tau:<3} | Gamma=0.01  | SR={sr:.2f} | Fail={fr:.2f} | E={me:.3f}")

        # 2. Sweep Gamma (fixing Tau = 10)
        for gamma in gamma_list:
            if gamma == 0.01: continue # Already tested above
            tracker = OnlineCCATracker(dim=sys.x_star.size, dt=dt, gamma=gamma, tau=10, min_steps=20, a=a, rng_seed=0, use_whitened=True)
            
            # BUG FIX 2: Pre-train locally (r_neigh=a)
            tracker.train(sys.drift, T_train, dt, D_eval, 0, sys.w_u_true, sys.x_star, r_neigh=a)
            
            successes, failures, energies = 0, 0, []
            for x0 in ics:
                run_rng = np.random.default_rng(int(rng_ic.integers(0, 10**6)))
                tr_copy = copy.deepcopy(tracker)
                traj, hit_side, energy = simulate_adaptive_trajectory(x0, T_sim, dt, sys.drift, D_eval, tr_copy, sys, a, np.inf, None, run_rng, is_adaptive=True)
                
                if hit_side == "hit_B": failures += 1
                if hit_side == "reached_A" or (D_eval < 1e-8 and hit_side == "timeout"):
                    successes += 1; energies.append(energy)
            
            sr, fr, me = successes/num_trajs, failures/num_trajs, np.mean(energies) if energies else 0.0
            writer.writerow([sys.name, 10, gamma, sr, fr, me])
            print(f"  Tau=10  | Gamma={gamma:<5} | SR={sr:.2f} | Fail={fr:.2f} | E={me:.3f}")


def plot_gating_weighting_ablation(csv_filepath, true_lambda=2.0, filepath="figures/ablation_fixes.pdf"):
    """Plots the results of the gating/weighting ablation: Success Rate and Eigenvalue Bias."""
    set_ieee_style()
    if not os.path.exists(csv_filepath): return
    
    conditions, sr, lam = [], [], []
    with open(csv_filepath, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Shorten names for the plot labels
            cond = row['Condition'].replace(" Estimation", "").replace(" Gating", "")
            conditions.append(cond)
            sr.append(float(row['Success_Rate']))
            lam.append(float(row['Mean_Lambda_u']))
            
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 3.5))
    
    # Left Panel: Success Rate
    ax1.bar(conditions, sr, color="#2d307b", alpha=0.8, width=0.5, edgecolor="black", linewidth=1.2)
    ax1.set_ylim(0, 1.05)
    ax1.set_ylabel("Success Rate", fontsize=10)
    ax1.set_xticklabels(conditions, rotation=20, ha="right", fontsize=9)
    ax1.grid(True, axis='y', linestyle='--', alpha=0.5)
    ax1.set_title("Tracking Robustness", fontsize=11)

    # Right Panel: Eigenvalue Estimation Bias
    ax2.bar(conditions, lam, color="#005f38", alpha=0.8, width=0.5, edgecolor="black", linewidth=1.2)
    ax2.axhline(true_lambda, color="#ed2024", linestyle="--", linewidth=2.0, label="True $\lambda_u$")
    ax2.set_ylabel("Estimated $\hat{\lambda}_u$", fontsize=10)
    ax2.set_xticklabels(conditions, rotation=20, ha="right", fontsize=9)
    ax2.legend(loc="upper left")
    ax2.grid(True, axis='y', linestyle='--', alpha=0.5)
    ax2.set_title("Estimation Bias Far From Saddle", fontsize=11)

    fig.tight_layout()
    fig.savefig(filepath, bbox_inches="tight")
    fig.savefig(filepath.replace('.pdf', '.png'), bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved Gating/Weighting Ablation Plot -> {filepath}")

def plot_hyperparameter_ablation(csv_filepath, filepath="figures/ablation_hyperparameters.pdf"):
    """Plots the Success Rate robustness against Tau and Gamma."""
    set_ieee_style()
    if not os.path.exists(csv_filepath): return
    
    tau_vals, tau_sr = [], []
    gamma_vals, gamma_sr = [], []
    
    with open(csv_filepath, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            t = float(row['Tau'])
            g = float(row['Gamma'])
            sr = float(row['Success_Rate'])
            
            # Extract the Tau sweep (where Gamma was fixed at 0.01)
            if abs(g - 0.01) < 1e-5:
                tau_vals.append(t)
                tau_sr.append(sr)
                
            # Extract the Gamma sweep (where Tau was fixed at 10)
            if abs(t - 10.0) < 1e-5:
                # To prevent duplicating the intersection point in the arrays if logged twice
                if g not in gamma_vals: 
                    gamma_vals.append(g)
                    gamma_sr.append(sr)

    # Sort just in case CSV wrote out of order
    tau_vals, tau_sr = zip(*sorted(zip(tau_vals, tau_sr)))
    gamma_vals, gamma_sr = zip(*sorted(zip(gamma_vals, gamma_sr)))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 3.5), sharey=True)
    
    # Left Panel: Tau Sweep
    ax1.plot(tau_vals, tau_sr, marker='s', color="#1a6faf", lw=2.0, markersize=7)
    ax1.set_ylim(-0.05, 1.05)
    ax1.set_ylabel("Success Rate", fontsize=10)
    ax1.set_xlabel("Lag Window ($\\tau$)", fontsize=10)
    ax1.set_xticks(tau_vals)
    ax1.grid(True, linestyle='--', alpha=0.5)
    ax1.set_title("Robustness to Lag ($\\gamma=0.01$)", fontsize=11)

    # Right Panel: Gamma Sweep
    ax2.plot(gamma_vals, gamma_sr, marker='o', color="#7b2d7b", lw=2.0, markersize=7)
    ax2.set_xscale('log')
    ax2.set_xlabel("EMA Discount ($\\gamma$)", fontsize=10)
    ax2.set_xticks(gamma_vals)
    ax2.set_xticklabels([f"{g:g}" for g in gamma_vals])
    ax2.grid(True, linestyle='--', alpha=0.5)
    ax2.set_title("Robustness to Discount ($\\tau=10$)", fontsize=11)

    fig.tight_layout()
    fig.savefig(filepath, bbox_inches="tight")
    fig.savefig(filepath.replace('.pdf', '.png'), bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved Hyperparameter Ablation Plot -> {filepath}")

# ──────────────────────────────────────────────────────────────────────────────
# 9. MAIN EXECUTION SCRIPT
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    dt = 0.01
    num_trajs = 1000

    sys_dw   = make_double_well_curved(lambda_u=2.0, lambda_s=8.0, c=0.5)
    sys_lin  = make_linear_system(lambda_u=2.0, lambda_s=-8.0)
    sys_pend = make_pendulum(g=1.5, length=1.0, damping=0.0)

    use_whitened = True

    systems = {
        "linear_2d":   (sys_lin,  1.0, np.inf, 10, 0.01),
        "double_well": (sys_dw,   0.5, 0.5,    10, 0.01),
        "pendulum":    (sys_pend, 1.0, 1.0,    30, 0.01),
    }

    # ══════════════════════════════════════════════════════════════════════
    # FIGURE 1: Single trajectory examples (all 3 systems)
    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "#"*80 + "\n FIGURE 1: Trajectory examples\n" + "#"*80)
    traj_methods = ["Autonomous", "CCA (Whitened)", "CCA (Adaptive)"]
    for tag, (sys, a, r_neigh, tau, D_eval) in systems.items():
        run_evaluation_suite(sys, a=a, r_neigh=r_neigh, D_eval=D_eval, dt=dt,
                             num_trajs=100, plot_trajs=3, spread_s=0.75, tau=tau,
                             methods_to_run=traj_methods, use_whitened=use_whitened,
                             file_suffix="trajs")

    # ══════════════════════════════════════════════════════════════════════
    # FIGURE 2: Linear system diagnostics
    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "#"*80 + "\n FIGURE 2: Linear diagnostics\n" + "#"*80)
    sys_lin2d, a_lin, r_neigh_lin, tau_lin, D_eval_lin = systems["linear_2d"]

    # 2a. Convergence
    conv_csv = "figures/linear_2d_convergence.csv"
    generate_convergence_data(sys_lin2d, a=a_lin, D_train=D_eval_lin, dt=dt, T_train=500,
                              r_neigh=r_neigh_lin, num_trajs=100, tau=tau_lin,
                              use_whitened=use_whitened, csv_filepath=conv_csv)
    plot_convergence_from_data(csv_filepath=conv_csv, sys_lambda_true=sys_lin2d.lambda_u_true,
                               sys_D_true=D_eval_lin, filepath="figures/linear_2d_convergence_stacked.pdf",
                               layout="stacked", label_fontsize=13, tick_fontsize=12, legend_fontsize=10)
    plot_convergence_from_data(csv_filepath=conv_csv, sys_lambda_true=sys_lin2d.lambda_u_true,
                               sys_D_true=D_eval_lin, filepath="figures/linear_2d_convergence_combined.pdf",
                               layout="combined", label_fontsize=13, tick_fontsize=12, legend_fontsize=10)

    # 2b. Noise sweep (D)
    energy_csv = "figures/linear_2d_noise_sweep.csv"
    simulate_and_save_noise_energy_sweep(sys_lin2d, a=a_lin, dt=dt, r_neigh=r_neigh_lin,
                                         num_trajs=num_trajs, T_sim=2000, tau=tau_lin,
                                         use_whitened=use_whitened, csv_filepath=energy_csv)
    plot_success_and_energy_vs_noise(csv_filepath=energy_csv, sys_name=sys_lin2d.name,
                                     filepath="figures/linear_2d_noise_sweep.pdf",
                                     layout="stacked", include_adaptive=False)

    # 2c. Dimensionality & lambda scaling
    run_dimensionality_scaling_experiment(dt=dt, num_trajs=num_trajs, T_sim=5000, use_whitened=use_whitened)

    # 2d. Combined scaling + energy
    plot_scaling_and_energy(dim_csv_filepath="figures/dimensionality_scaling_stats_WHITENED_CCA.csv",
                            energy_csv_filepath=energy_csv,
                            include_adaptive=False, line_alpha=0.7,
                            filepath="figures/linear_2d_combined_scaling_energy.pdf")

    # ══════════════════════════════════════════════════════════════════════
    # FIGURE 3: Nonlinear systems — noise sweep (D) only
    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "#"*80 + "\n FIGURE 3: Nonlinear noise sweeps\n" + "#"*80)
    for tag in ["double_well", "pendulum"]:
        sys_nl, a_nl, r_neigh_nl, tau_nl, D_eval_nl = systems[tag]
        csv_path = f"figures/{tag}_noise_sweep.csv"
        simulate_and_save_noise_energy_sweep(sys_nl, a=a_nl, dt=dt, r_neigh=r_neigh_nl,
                                              num_trajs=num_trajs, T_sim=2000, tau=tau_nl,
                                              use_whitened=use_whitened, csv_filepath=csv_path)
        plot_success_and_energy_vs_noise(csv_filepath=csv_path, sys_name=sys_nl.name,
                                          filepath=f"figures/{tag}_noise_sweep.pdf",
                                          layout="stacked", include_adaptive=False)

    # ══════════════════════════════════════════════════════════════════════
    # FIGURE 4: Baseline & Ablation Studies 
    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "#"*80 + "\n FIGURE 4: Baselines & Ablations\n" + "#"*80)
    
    # We will test the ablations on the nonlinear Double Well system as requested
    sys_eval, a_eval, r_neigh_eval, tau_eval, D_eval = systems["double_well"]
    
    # 4a. Baselines (LQR vs Sign-Gated vs VAR vs CCA)
    baseline_csv = f"figures/baselines_{sys_eval.name}.csv"
    run_baseline_comparison(sys_eval, a=a_eval, r_neigh=r_neigh_eval, dt=dt, num_trajs=100, 
                            noise_levels=[0.0, 0.005, 0.01, 0.05], csv_filepath=baseline_csv)
    plot_baseline_comparison(csv_filepath=baseline_csv, filepath=f"figures/baseline_comparison_{sys_eval.name}.pdf")

    # 4b. The "Fixes" Ablation (Gating & Weighting)
    run_gating_weighting_ablation(sys_eval, a=a_eval, dt=dt, D_eval=D_eval, num_trajs=100, 
                                  csv_filepath=f"figures/ablation_fixes_{sys_eval.name}.csv")
    
    # 4c. Hyperparameter Robustness (Tau & Gamma)
    run_hyperparameter_ablation(sys_eval, a=a_eval, dt=dt, D_eval=D_eval, num_trajs=100, 
                                csv_filepath=f"figures/ablation_hyperparameters_{sys_eval.name}.csv")

    print("\n" + "="*80)
    print(" Execution completed successfully for all systems. Check the 'figures/' directory.")
    print("="*80)