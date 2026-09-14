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
from collections import defaultdict

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
    """Applies IEEE CDC-compliant styling with high-quality output."""
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.labelsize": 11,
        "axes.titlesize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 8,
        "axes.linewidth": 0.8,
        "grid.linewidth": 0.4,
        "grid.alpha": 0.3,
        "lines.linewidth": 1.5,
        "lines.markersize": 5,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
    })

# Metric colors: blue family = success rate, warm family = energy
C_SUCCESS = "#1a3a5c"
C_ENERGY  = "#b5432a"

# Method visual encoding: marker + linestyle (color comes from metric)
METHOD_STYLE = {
    "Optimal":            dict(marker="s", ls="-",  lw=1.8, label="Model-based"),
    "CCA":                dict(marker="o", ls="--", lw=1.5, label="Data-enabled"),
    "CCA (Adaptive)":     dict(marker="D", ls="-.", lw=1.5, label="Closed-loop"),
    "CCA (Whitened)":     dict(marker="o", ls="--", lw=1.5, label="Data-enabled"),
    "Oracle Committor":   dict(marker="s", ls="-",  lw=1.8, label="Committor (oracle)"),
    "Oracle ReLU":        dict(marker="v", ls="-",  lw=1.5, label="ReLU (oracle)"),
    "Est CCA Committor":  dict(marker="o", ls="--", lw=1.5, label="Data-enabled"),
    "Est CCA ReLU":       dict(marker="v", ls="--", lw=1.5, label="ReLU (wSVD)"),
    "Est VAR Committor":  dict(marker="p", ls=":",  lw=1.5, label="Committor (VAR)"),
}

def _get_method_style(method):
    return METHOD_STYLE.get(method, dict(marker="x", ls="-", lw=1.2, label=method))

def _save_fig(fig, filepath):
    """Save as both PDF and PNG."""
    d = os.path.dirname(filepath)
    if d: os.makedirs(d, exist_ok=True)
    fig.savefig(filepath, bbox_inches="tight")
    alt = filepath.replace('.pdf', '.png') if filepath.endswith('.pdf') else filepath.replace('.png', '.pdf')
    fig.savefig(alt, bbox_inches="tight")
    print(f"  Saved -> {filepath}")
    plt.close(fig)

def _make_log_axis(noises):
    """Map noise levels to pseudo-log x positions (D=0 gets a spot left of min)."""
    non_zero = sorted([n for n in noises if n > 0])
    pseudo_zero = non_zero[0] / 3.0 if non_zero else 0.0001
    def map_x(n): return pseudo_zero if n == 0.0 else n
    return pseudo_zero, map_x, non_zero

def _read_sweep_csv(csv_filepath, sys_name=None):
    """Read sweep CSV, backward-compatible with old and new formats."""
    if not os.path.exists(csv_filepath): return []
    data = []
    with open(csv_filepath, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if sys_name and row.get('System', '') != sys_name: continue
            entry = {'Method': row['Method'], 'Noise_D': float(row['Noise_D']),
                     'Success_Rate': float(row['Success_Rate'])}
            if 'Mean_Energy_Succ' in row:
                entry['Mean_Energy_Succ'] = float(row['Mean_Energy_Succ'])
                entry['Std_Energy_Succ']  = float(row['Std_Energy_Succ'])
                entry['N_Success']        = int(row['N_Success'])
                entry['Mean_Energy_All']  = float(row['Mean_Energy_All'])
                entry['Std_Energy_All']   = float(row['Std_Energy_All'])
            elif 'Mean_Energy' in row:
                me = float(row['Mean_Energy']); se = float(row.get('Std_Energy', 0.0)); ns = int(row.get('N_Success', 0))
                entry['Mean_Energy_Succ'] = me; entry['Std_Energy_Succ'] = se; entry['N_Success'] = ns
                entry['Mean_Energy_All'] = me; entry['Std_Energy_All'] = se
            else:
                entry['Mean_Energy_Succ'] = 0.0; entry['Std_Energy_Succ'] = 0.0; entry['N_Success'] = 0
                entry['Mean_Energy_All'] = 0.0; entry['Std_Energy_All'] = 0.0
            data.append(entry)
    return data

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
# 6. Plotting Utilities (Compact & Unified)
# ──────────────────────────────────────────────────────────────────────────────

def draw_invariant_manifolds(ax, sys):
    try:
        eigvals, V = np.linalg.eig(sys.J)
        v_u, v_s = V[:, np.argsort(eigvals.real)[::-1][:2]].real
        v_u /= np.linalg.norm(v_u); v_s /= np.linalg.norm(v_s)
        def rk4(f, x, h):
            k1 = f(0, x); k2 = f(0, x+k1*h/2); k3 = f(0, x+k2*h/2); k4 = f(0, x+k3*h)
            return x + (k1+2*k2+2*k3+k4)*h/6.0
        if sys.name == "Pendulum" and sys.J[1, 1] == 0.0:
            th = np.linspace(sys.plot_xlim[0]-1, sys.plot_xlim[1]+1, 500)
            w = 2.0 * np.sqrt(sys.J[1, 0]) * np.cos(th / 2.0)
            ax.plot(th, w, color='k', lw=1.8, alpha=0.85, zorder=2)
            ax.plot(th, -w, color='k', lw=1.8, alpha=0.85, zorder=2)
        else:
            akw = dict(arrowstyle="-|>", color="k", lw=1.5, mutation_scale=15)
            for v, step, c in [(v_s, -0.02, 'k'), (v_u, 0.02, 'k')]:
                for sign in [1, -1]:
                    x = sys.x_star + sign * 1e-2 * v; traj = [x.copy()]
                    for _ in range(150): x = rk4(sys.drift, x, step); traj.append(x.copy())
                    traj = np.array(traj)
                    ax.plot(traj[:,0], traj[:,1], color=c, ls='-', lw=1.5, alpha=0.8, zorder=2)
                    if len(traj) > 25:
                        i1, i2 = (10, 25) if step < 0 else (25, 10)
                        ax.annotate("", xy=traj[i1][:2], xytext=traj[i2][:2], arrowprops=akw, zorder=3)
    except Exception as e: print(f"Error drawing manifolds: {e}")

def draw_background(ax, sys, a):
    e_true = sys.w_u_true[:2] / np.linalg.norm(sys.w_u_true[:2])
    perp = np.array([-e_true[1], e_true[0]]) * 5.0; x0 = sys.x_star[:2]
    for sign, c in [(1, "#ed2024"), (-1, "#176533")]:  # Red (undesired), Green (desired)
        pt = x0 + sign * e_true * a
        ax.plot([pt[0]-perp[0], pt[0]+perp[0]], [pt[1]-perp[1], pt[1]+perp[1]], color=c, ls="--", lw=1.5, zorder=1)

def get_gradient_cmap(base_hex):
    base = mcolors.hex2color(base_hex)
    return mcolors.LinearSegmentedColormap.from_list("grad", [
        tuple(c + 0.5*(1-c) for c in base), base, tuple(c*0.85 for c in base)])

def plot_trajectory_panels(panels, sys, a, is_single=False, combined=False, filepath=None):
    n = len(panels); legend_handles = []
    if combined:
        fig, ax = plt.subplots(figsize=(6, 6)); axes_flat = [ax] * n
    else:
        cols = min(n, 3); rows = (n + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 4*rows))
        axes_flat = np.atleast_1d(axes).flatten()
        fig.subplots_adjust(wspace=0.1, hspace=0.2)

    for i, (label, col, est_dir, trajs, stats) in enumerate(panels):
        if not combined and i >= n: axes_flat[i].axis('off'); continue
        ax = axes_flat[i]
        if not combined:
            draw_background(ax, sys, a); draw_invariant_manifolds(ax, sys)
            e_true = sys.w_u_true[:2] / (np.linalg.norm(sys.w_u_true[:2]) + 1e-15)
            ax.annotate("", xy=sys.x_star[:2] + (a*0.6)*e_true, xytext=sys.x_star[:2], arrowprops=dict(arrowstyle="-|>", color="k", lw=2.0), zorder=8)
            ax.set_title(f"{label}\n{stats}", fontsize=9, pad=8, color=col if "Autonomous" not in label else "#333")
        elif i == 0:
            draw_background(ax, sys, a); draw_invariant_manifolds(ax, sys)
            e_true = sys.w_u_true[:2] / (np.linalg.norm(sys.w_u_true[:2]) + 1e-15)
            ax.annotate("", xy=sys.x_star[:2] + (a*0.6)*e_true, xytext=sys.x_star[:2], arrowprops=dict(arrowstyle="-|>", color="#2b3c96", lw=2.5), zorder=8)
            
        if est_dir is not None:
            e_est = est_dir[:2] / (np.linalg.norm(est_dir[:2]) + 1e-15)
            ax.annotate("", xy=sys.x_star[:2] + (a*0.6)*e_est, xytext=sys.x_star[:2], arrowprops=dict(arrowstyle="-|>", color=col, lw=2.0, ls="dashed"), zorder=9)
            
        cmap = get_gradient_cmap(col); dark_col = cmap(1.0)
        alpha, lw_val, s_val = (1.0, 2.0, 40) if is_single else (max(0.4, min(0.8, 3.0/max(1, len(trajs)))), 1.2, 25)
        
        for X, _, _ in trajs:
            pts = X[:, :2]; segs = np.concatenate([pts[:-1,None,:], pts[1:,None,:]], axis=1)
            lc = LineCollection(segs, cmap=cmap, norm=plt.Normalize(0, len(X)), lw=lw_val, alpha=alpha, zorder=4)
            lc.set_array(np.arange(len(X)-1)); ax.add_collection(lc)
            ax.scatter(*pts[-1], s=s_val, color=dark_col, edgecolors="w", lw=0.6, zorder=6)
            
        if combined: legend_handles.append(mlines.Line2D([], [], color=dark_col, lw=2.0, label=label))
        if not combined or i == n-1:
            ax.set(xlim=sys.plot_xlim, ylim=sys.plot_ylim, aspect="equal"); ax.set_xticks([]); ax.set_yticks([])
            
    if combined: ax.legend(handles=legend_handles, loc='upper center', ncol=3, fontsize=9, framealpha=0.9, bbox_to_anchor=(0.5, 1.0))
    if filepath:
        fig.savefig(filepath, bbox_inches="tight"); fig.savefig(filepath.replace('.png','.pdf'), bbox_inches="tight")
        print(f"Saved {'Combined' if combined else 'Grid'} {'single' if is_single else 'multi'} traj -> {filepath}")
    plt.close(fig)

# ──────────────────────────────────────────────────────────────────────────────
# 6b. Convergence data generation & plotting
# ──────────────────────────────────────────────────────────────────────────────

def generate_convergence_data(sys, a=0.5, D_train=0.01, dt=0.01, T_train=500, r_neigh=np.inf, num_trajs=20, tau=10, use_whitened=True, csv_filepath="figures/convergence_data.csv"):
    print(f" -> Generating convergence data ({csv_filepath})...")
    os.makedirs(os.path.dirname(csv_filepath), exist_ok=True)
    with open(csv_filepath, 'w', newline='') as f:
        writer = csv.writer(f); writer.writerow(["Seed", "Step", "Cos_Sim", "Lambda_u", "D_val"])
        for seed in range(num_trajs):
            tracker = OnlineCCATracker(dim=sys.x_star.size, dt=dt, gamma=0.01, tau=tau, min_steps=20, a=a, rng_seed=seed, use_whitened=use_whitened)
            rng = np.random.default_rng(seed); x = sys.x_star + rng.uniform(-0.1, 0.1, sys.x_star.size)
            limit = r_neigh if r_neigh < np.inf else max(2.0, a * 2.0)
            for t in range(T_train):
                x_next = x + sys.drift(t*dt, x)*dt + np.sqrt(2.0*D_train) * rng.standard_normal(sys.x_star.size)*np.sqrt(dt)
                if np.linalg.norm(x_next - sys.x_star) > limit:
                    x = sys.x_star + rng.normal(0, 0.05, sys.x_star.size); tracker.reset(); continue
                tracker.update(x, x_next, sys.w_u_true); x = x_next
                if tracker.n_updates >= tracker.min_steps:
                    p = tracker.get_params(sys.w_u_true)
                    writer.writerow([seed, t, abs(float(np.dot(p['w_u'], sys.w_u_true))), p['lambda_u'], p['D']])

def plot_convergence_from_data(csv_filepath, sys_lambda_true=2.0, sys_D_true=0.01, filepath="figures/convergence.pdf", layout="stacked", label_fontsize=12, tick_fontsize=11, burn_in=30):
    set_ieee_style()
    if not os.path.exists(csv_filepath): print(f" [!] Missing {csv_filepath}"); return
    
    data = defaultdict(lambda: {'cos': [], 'lam': [], 'D': []})
    with open(csv_filepath, 'r') as f:
        for r in csv.DictReader(f):
            s = int(r['Step'])
            data[s]['cos'].append(float(r['Cos_Sim'])); data[s]['lam'].append(float(r['Lambda_u'])); data[s]['D'].append(float(r['D_val']))
            
    steps = sorted(data.keys())
    if not steps: return
    valid_steps = [s for s in steps if s >= steps[min(burn_in, max(0, len(steps)-10))]]
    metrics = {'cos': [], 'lam': [], 'D': []}
    
    for s in valid_steps:
        for k in metrics:
            arr = np.array(data[s][k])
            metrics[k].append((np.median(arr), np.percentile(arr, 25), np.percentile(arr, 75)))
            
    metrics = {k: list(zip(*v)) for k, v in metrics.items()}
    c_dir, c_eig, c_D = "#2d307b", "#19421e", "#a01b2a"
    
    if layout == "stacked":
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(5.5, 6), sharex=True); fig.subplots_adjust(hspace=0.1)
        axes_data = [(ax1, 'cos', c_dir, r"$|\cos(\hat{w}_u, w_u)|$", 1.0), 
                     (ax2, 'lam', c_eig, r"$\hat{\lambda}_u$", sys_lambda_true), 
                     (ax3, 'D', c_D, r"$\hat{D}$", sys_D_true)]
        for ax, k, c, ylab, true_val in axes_data:
            ax.plot(valid_steps, metrics[k][0], color=c, lw=2.0)
            ax.fill_between(valid_steps, metrics[k][1], metrics[k][2], color=c, alpha=0.2)
            ax.axhline(true_val, color=c, ls='--', lw=1.2, alpha=0.6)
            ax.set_ylabel(ylab, color=c, fontsize=label_fontsize)
            ax.tick_params(axis='y', labelcolor=c, labelsize=tick_fontsize)
            ax.grid(True, ls='-', alpha=0.3, color='gray')
        ax3.set_xlabel("Training step", fontsize=label_fontsize); ax3.set_xlim([valid_steps[0], valid_steps[-1]])
        fig.align_ylabels((ax1, ax2, ax3))
    elif layout == "combined":
        fig, ax1 = plt.subplots(figsize=(6.5, 3.5)); fig.subplots_adjust(right=0.78)
        ax2 = ax1.twinx(); ax3 = ax1.twinx(); ax3.spines["right"].set_position(("axes", 1.15))
        axes_data = [(ax1, 'cos', c_dir, r"$|\cos(\hat{w}_u, w_u)|$", 1.0, r"Direction $|\cos(\hat{w}_u, w_u)|$"),
                     (ax2, 'lam', c_eig, r"$\hat{\lambda}_u$", sys_lambda_true, r"Eigenvalue $\hat{\lambda}_u$"),
                     (ax3, 'D', c_D, r"$\hat{D}$", sys_D_true, r"Noise $\hat{D}$")]
        for ax, k, c, ylab, true_val, lab in axes_data:
            ax.plot(valid_steps, metrics[k][0], color=c, lw=2.0, label=lab)
            ax.fill_between(valid_steps, metrics[k][1], metrics[k][2], color=c, alpha=0.2)
            ax.axhline(true_val, color=c, ls='--', lw=1.2, alpha=0.6)
            ax.set_ylabel(ylab, color=c, fontsize=label_fontsize)
            ax.tick_params(axis='y', labelcolor=c, labelsize=tick_fontsize)
        ax1.grid(True, ls='-', alpha=0.3, color='gray'); ax1.set_xlabel("Training step", fontsize=label_fontsize)
        ax1.set_xlim([valid_steps[0], valid_steps[-1]])
        lines = sum([ax.get_legend_handles_labels()[0] for ax in (ax1, ax2, ax3)], [])
        labels = sum([ax.get_legend_handles_labels()[1] for ax in (ax1, ax2, ax3)], [])
        ax1.legend(lines, labels, loc='lower right', fontsize=10, framealpha=0.95)
        
    fig.savefig(filepath, bbox_inches="tight"); fig.savefig(filepath.replace('.pdf', '.png'), bbox_inches="tight"); plt.close(fig)

# ──────────────────────────────────────────────────────────────────────────────
# 6c/d. Combined Scaling, Noise, & Energy Sweep Plotting
# ──────────────────────────────────────────────────────────────────────────────

def simulate_and_save_noise_energy_sweep(sys, a=0.5, dt=0.01, r_neigh=np.inf, num_trajs=100, T_sim=2000, T_train=2500, tau=10, use_whitened=True, noise_levels=None, csv_filepath="figures/noise_energy_sweep.csv"):
    if noise_levels is None: noise_levels = [0.0, 0.001, 0.005, 0.01, 0.02, 0.05, 0.1]
    w_u_norm = sys.w_u_true / np.linalg.norm(sys.w_u_true)
    w_s = np.random.default_rng(42).standard_normal(sys.x_star.size)
    w_s -= np.dot(w_s, w_u_norm) * w_u_norm; w_s /= np.linalg.norm(w_s) + 1e-15
    os.makedirs(os.path.dirname(csv_filepath), exist_ok=True)
    
    with open(csv_filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["System", "Method", "Noise_D", "Success_Rate", "N_Total", "Mean_Energy_Succ", "Std_Energy_Succ", "N_Success", "Mean_Energy_All", "Std_Energy_All", "N_All"])
        for D_test in noise_levels:
            tracker = CCATracker(dim=sys.x_star.size, dt=dt, tau=tau, rng_seed=0, use_whitened=use_whitened)
            tracker.train(sys.drift, T_train, dt, max(D_test, 0.005), 0, a, sys.w_u_true, init_center=sys.x_star, r_neigh=r_neigh)
            tracker_online = OnlineCCATracker(dim=sys.x_star.size, dt=dt, gamma=0.01, tau=tau, min_steps=20, a=a, rng_seed=0, use_whitened=use_whitened)
            tracker_online.train(sys.drift, T_train, dt, max(D_test, 0.005), 0, sys.w_u_true, sys.x_star, r_neigh=r_neigh)
            
            c_opt = partial(relu_feedback if D_test < 1e-8 else erf_feedback, w_u=sys.w_u_true, lambda_u=sys.lambda_u_true, D=D_test, x_fp=sys.x_star, a=a, r_neigh=r_neigh)
            rng_ic = np.random.default_rng(11)
            ics = [sys.x_star + rng_ic.uniform(0.02, a*0.9)*w_u_norm + rng_ic.uniform(-a*0.4, a*0.4)*w_s for _ in range(num_trajs)]
            
            for method_name, ctrl, use_tr, is_adp, trk in [('Optimal', c_opt, False, False, None), ('CCA', None, True, False, tracker), ('CCA (Adaptive)', None, True, True, tracker_online)]:
                e_succ, e_all = [], []
                for x0 in ics:
                    run_rng = np.random.default_rng(int(rng_ic.integers(0, 10**6)))
                    if use_tr: _, hit, en = simulate_adaptive_trajectory(x0, T_sim, dt, sys.drift, D_test, copy.deepcopy(trk), sys, a, r_neigh, None, run_rng, is_adaptive=is_adp)
                    else: _, hit, en = simulate_trajectory(x0, T_sim, dt, sys.drift, D_test, ctrl, sys, a, run_rng)
                    e_all.append(en)
                    if hit == "reached_A" or (D_test < 1e-8 and hit == "timeout"): e_succ.append(en)
                
                writer.writerow([sys.name, method_name, D_test, len(e_succ)/num_trajs, num_trajs, np.mean(e_succ) if e_succ else 0, np.std(e_succ) if e_succ else 0, len(e_succ), np.mean(e_all), np.std(e_all), len(e_all)])

def plot_scaling_and_energy(csv_filepath, sys_name=None, filepath="figures/scaling_energy.pdf", 
                            mode="stacked", layout=None, include_adaptive=False, 
                            fontsize=16, ylim_sr=(-0.05, 1.05), ylim_en=None,
                            figsize=None, legend_loc=None, legend_bbox=None):
    """
    Unified function to plot Success Rate and Mean Energy vs Noise. 
    Accepts mode="stacked" (2 vertical panels) or mode="combined" (1 panel, dual y-axes).
    """
    # 1. Apply base style FIRST so it doesn't overwrite our custom settings
    set_ieee_style()
    
    # 2. THEN override with CMU Roman and custom dynamic font sizes
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman", "Times New Roman", "serif"],
        "mathtext.fontset": "cm",
        "font.size": fontsize,
        "axes.labelsize": fontsize,
        "xtick.labelsize": fontsize - 2,
        "ytick.labelsize": fontsize - 2,
        "legend.fontsize": fontsize - 2,
    })
    
    C_SUCCESS = "#2c327b"
    C_ENERGY = "#025230"
    
    mode = layout if layout else mode
    data = _read_sweep_csv(csv_filepath, sys_name)
    if not data: print(f" [!] No data found in {csv_filepath}"); return

    noises = sorted(set(d['Noise_D'] for d in data))
    pseudo_zero, map_x, non_zero = _make_log_axis(noises)
    methods = ["Optimal", "CCA"] + (["CCA (Adaptive)"] if include_adaptive else [])
    
    is_stacked = (mode == "stacked")
    if is_stacked:
        fs = figsize if figsize else (6.0, 6.0)
        fig, (ax_sr, ax_en) = plt.subplots(2, 1, figsize=fs, sharex=True)
        fig.subplots_adjust(hspace=0.08)
    else:
        fs = figsize if figsize else (6.5, 4.5)
        fig, ax_sr = plt.subplots(figsize=fs)
        ax_en = ax_sr.twinx()

    handles = []
    for method in methods:
        sty = _get_method_style(method)
        
        # Decouple success arrays from energy arrays
        xs_z_sr, sr_z = [], []
        xs_z_en, en_z, sem_z = [], [], []
        xs_p_sr, sr_p = [], []
        xs_p_en, en_p, sem_p = [], [], []
        
        for n in noises:
            row = next((r for r in data if r['Noise_D'] == n and r['Method'] == method), None)
            if not row: continue
            
            total_success = float(row['Success_Rate']) + float(row.get('Timeout_Rate', 0.0))
            ns = float(row.get('N_Success', 0))
            x_val = map_x(n)
            
            if n == 0.0:
                xs_z_sr.append(x_val)
                sr_z.append(total_success)
                if ns > 0:
                    xs_z_en.append(x_val)
                    en_z.append(float(row['Mean_Energy_Succ']))
                    sem_z.append(float(row.get('Std_Energy_Succ', 0.0)) / np.sqrt(ns))
            else:
                xs_p_sr.append(x_val)
                sr_p.append(total_success)
                if ns > 0:
                    xs_p_en.append(x_val)
                    en_p.append(float(row['Mean_Energy_Succ']))
                    sem_p.append(float(row.get('Std_Energy_Succ', 0.0)) / np.sqrt(ns))

        # Plot zero noise (isolated points)
        if xs_z_sr:
            ax_sr.plot(xs_z_sr, sr_z, marker='o', color=C_SUCCESS, ls='none', lw=sty['lw'], ms=5)
        if xs_z_en:
            ax_en.plot(xs_z_en, en_z, marker='o', color=C_ENERGY, ls='none', lw=sty['lw'], ms=5)

        # Plot positive noise (lines)
        if xs_p_sr:
            ls = sty.get('ls', '-')
            ax_sr.plot(xs_p_sr, sr_p, marker='None', color=C_SUCCESS, ls=ls, lw=sty['lw'])
        if xs_p_en:
            ls = sty.get('ls', '-')
            en_a, sem_a = np.array(en_p), np.array(sem_p)
            ax_en.plot(xs_p_en, en_a, marker='None', color=C_ENERGY, ls=ls, lw=sty['lw'])
            if np.any(sem_a > 0): 
                ax_en.fill_between(xs_p_en, np.maximum(0, en_a - 1.96*sem_a), en_a + 1.96*sem_a, color=C_ENERGY, alpha=0.08)
        
        handles.append(mlines.Line2D([], [], color='#333', marker='None', ls=sty.get('ls','-'), lw=sty['lw'], label=sty['label']))

    # Formatting Success Rate
    ax_sr.set_ylabel("Success rate", color=C_SUCCESS)
    ax_sr.tick_params(axis='y', colors=C_SUCCESS)
    ax_sr.set_ylim(ylim_sr)
    
    # FORCE MIN/MAX TICKS ONLY & FORMAT AS INTEGERS IF POSSIBLE
    sr_ticks = [max(0.0, ylim_sr[0]), min(1.0, ylim_sr[1])]
    ax_sr.set_yticks(sr_ticks)
    ax_sr.set_yticklabels([f"{t:g}" for t in sr_ticks])
    
    # Formatting Energy
    all_en = [d['Mean_Energy_Succ'] for d in data if d['Method'] in methods and d.get('N_Success', 0) > 0]
    if ylim_en is not None:
        y_min, y_max = ylim_en
    elif all_en: 
        y_min, y_max = min(all_en)*0.8, max(all_en)*1.25
    else:
        y_min, y_max = 0, 1  # Fallback
        
    ax_en.set_ylim(y_min, y_max)
    ax_en.set_ylabel("Control energy", color=C_ENERGY)
    ax_en.tick_params(axis='y', colors=C_ENERGY)
    
    # FORCE MIN/MAX TICKS ONLY (Calculated Limits)
    en_ticks = [y_min, y_max]
    ax_en.set_yticks(en_ticks)
    ax_en.set_yticklabels([f"{t:g}" for t in en_ticks])

    # Shared formatting
    base_ax = ax_en if is_stacked else ax_sr
    base_ax.set_xscale('log')
    base_ax.set_xticks([map_x(n) for n in noises])
    base_ax.set_xticklabels(['0' if n == 0.0 else f"{n:.3g}" for n in noises], rotation=45, ha='right')
    base_ax.set_xlabel("Noise strength ($D$)")
    
    if non_zero:
        for ax in ([ax_sr, ax_en] if is_stacked else [ax_sr]): 
            ax.axvline(x=pseudo_zero * 1.5, color='gray', ls=':', alpha=0.4, lw=0.6)

    # Apply dynamic legend placement
    if legend_loc is None:
        legend_loc = 'center left' if is_stacked else 'lower left'
    if legend_bbox is None:
        legend_bbox = (1.15, 0.5) if is_stacked else None
        
    ax_sr.legend(handles=handles, loc=legend_loc, bbox_to_anchor=legend_bbox, frameon=False)
                 
    if is_stacked: fig.align_ylabels([ax_sr, ax_en])
    _save_fig(fig, filepath)

def plot_noise_sweep_panel(csv_filepath, filepath, sys_name=None, ylim_success=(0.9, 1.0), ylim_energy=(0.0, 1.5),
                           figsize=(6.4, 3.0), xtick_values=(0.0, 0.001, 0.01, 0.1), legend_kwargs=None):
    """Success rate (left axis) and mean control energy of successful trajectories (right axis) vs noise D.

    Panel C of the double-well and pendulum figures. D = 0 is drawn as isolated markers left of a dotted line.
    Energy bands are 95% confidence intervals of the mean, using the actual number of successful trajectories.
    legend_kwargs are passed to the legend (default: one row above the axes). Returns the list of overlaps found
    between text labels, the legend and the plotted data (empty if none).
    """
    import matplotlib.font_manager as fm
    for f in fm.findSystemFonts(fontpaths=[os.path.expandvars(r"%LOCALAPPDATA%\Microsoft\Windows\Fonts")]):
        try: fm.fontManager.addfont(f)
        except Exception: pass
    if not fm.findfont(fm.FontProperties(family="CMU Serif")).lower().endswith("cmunrm.ttf"):
        print("  [!] CMU Serif not found; falling back to the default serif font")
    style = {"font.family": "serif", "font.serif": ["CMU Serif", "DejaVu Serif"], "mathtext.fontset": "cm",
             "font.size": 16, "axes.labelsize": 18, "xtick.labelsize": 15, "ytick.labelsize": 15,
             "legend.fontsize": 15, "axes.linewidth": 1.2, "lines.linewidth": 2.2, "legend.frameon": False,
             "pdf.fonttype": 42, "ps.fonttype": 42}
    c_success, c_energy = "#0072B2", "#D55E00"  # Okabe-Ito blue and vermillion
    methods = [("Optimal", "Model-based", "-", "o"), ("CCA", "Data-enabled", "--", "s")]
    data = _read_sweep_csv(csv_filepath, sys_name)
    noises = sorted({d['Noise_D'] for d in data})
    pseudo_zero, map_x, non_zero = _make_log_axis(noises)

    with plt.rc_context(style):
        fig, ax_sr = plt.subplots(figsize=figsize)
        ax_en = ax_sr.twinx()
        for i, (method, _, ls, marker) in enumerate(methods):
            rows = {d['Noise_D']: d for d in data if d['Method'] == method}
            if 0.0 in rows:
                r0, x0 = rows[0.0], map_x(0.0) * (0.8, 1.25)[i]  # side by side, so equal values stay visible
                ax_sr.plot(x0, r0['Success_Rate'], marker=marker, color=c_success, ls='none', ms=8)
                if r0['N_Success'] > 0:
                    ax_en.plot(x0, r0['Mean_Energy_Succ'], marker=marker, color=c_energy, ls='none', ms=8)
            xs = [n for n in non_zero if n in rows]
            ax_sr.plot(xs, [rows[n]['Success_Rate'] for n in xs], color=c_success, ls=ls, marker=marker, ms=6)
            xe = [n for n in xs if rows[n]['N_Success'] > 0]
            en = np.array([rows[n]['Mean_Energy_Succ'] for n in xe])
            half = np.array([1.96 * rows[n]['Std_Energy_Succ'] / np.sqrt(rows[n]['N_Success']) for n in xe])
            ax_en.plot(xe, en, color=c_energy, ls=ls, marker=marker, ms=6)
            ax_en.fill_between(xe, np.maximum(0.0, en - half), en + half, color=c_energy, alpha=0.15, lw=0)

        from matplotlib import ticker as mticker
        ax_sr.set_xscale('log')
        ticks = [n for n in xtick_values if n in noises]
        ax_sr.set_xticks([map_x(n) for n in ticks])
        ax_sr.set_xticklabels(['0' if n == 0.0 else f"{n:g}" for n in ticks])
        ax_sr.xaxis.set_minor_locator(mticker.FixedLocator([map_x(n) for n in noises if n not in ticks]))
        ax_sr.xaxis.set_minor_formatter(mticker.NullFormatter())
        ax_sr.axvline(pseudo_zero * 1.8, color='gray', ls=':', lw=1.0, alpha=0.6)
        ax_sr.set_xlabel(r"Noise strength $D$")
        for ax, lim, color, label in ((ax_sr, ylim_success, c_success, "Success rate"),
                                      (ax_en, ylim_energy, c_energy, "Control energy")):
            ax.set_ylim(lim[0], lim[1] + 0.03 * (lim[1] - lim[0]))  # headroom so markers at the top value are not clipped
            ax.set_yticks(list(lim))
            ax.set_yticklabels([f"{t:g}" for t in lim])
            ax.set_ylabel(label, color=color)
            ax.tick_params(axis='y', colors=color)
            ax.spines['top'].set_visible(False)
        ax_sr.spines['right'].set_visible(False)
        ax_en.spines['left'].set_visible(False)
        handles = [mlines.Line2D([], [], color='#333333', ls=ls, marker=marker, ms=6, label=label)
                   for _, label, ls, marker in methods]
        kw = dict(loc='lower center', ncol=2, bbox_to_anchor=(0.5, 1.0), borderaxespad=0.2)
        kw.update(legend_kwargs or {})
        ax_en.legend(handles=handles, **kw)
        overlaps = _find_panel_overlaps(fig)
        for o in overlaps:
            print(f"  [!] overlap: {o}")
        d = os.path.dirname(filepath)
        if d: os.makedirs(d, exist_ok=True)
        fig.savefig(filepath, bbox_inches="tight")
        fig.savefig(os.path.splitext(filepath)[0] + ".png", dpi=300, bbox_inches="tight")
        plt.close(fig)
    print(f"  Saved -> {filepath}")
    return overlaps


def _find_panel_overlaps(fig, pad_pt=2.0):
    """Overlaps (display coordinates) between text labels and the legend, and between the legend and plotted data."""
    from matplotlib.path import Path
    from matplotlib.transforms import Bbox
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    px = fig.dpi / 72.0
    pad = pad_pt * px
    boxes = []
    for ax in fig.axes:
        labels = (ax.get_xticklabels() if ax.xaxis.get_visible() else []) + ax.get_yticklabels()
        for t in labels:
            if t.get_visible() and t.get_text():
                boxes.append((f"tick label '{t.get_text()}'", t.get_window_extent(renderer)))
        for axis in (ax.xaxis, ax.yaxis):
            if axis.get_visible() and axis.label.get_visible() and axis.label.get_text():
                boxes.append((f"axis label '{axis.label.get_text()}'", axis.label.get_window_extent(renderer)))
    legends = [ax.get_legend() for ax in fig.axes if ax.get_legend() is not None] + list(fig.legends)
    boxes += [("legend", leg.get_window_extent(renderer)) for leg in legends]
    found = []
    for i, (name_a, a) in enumerate(boxes):
        for name_b, b in boxes[i + 1:]:
            if a.x0 < b.x1 + pad and b.x0 < a.x1 + pad and a.y0 < b.y1 + pad and b.y0 < a.y1 + pad:
                found.append(f"{name_a} / {name_b}")
    for leg in legends:
        lb = leg.get_window_extent(renderer)
        for ax in fig.axes:
            for line in ax.get_lines():
                xy = line.get_xydata()
                if len(xy) == 0:
                    continue
                verts = line.get_transform().transform(xy)
                has_marker = line.get_marker() not in (None, 'None', '', ' ')
                has_line = line.get_linestyle() not in (None, 'None', '', ' ')
                grow = max(line.get_markersize() / 2 if has_marker else 0.0, line.get_linewidth() / 2) * px + pad
                bb = Bbox.from_extents(lb.x0 - grow, lb.y0 - grow, lb.x1 + grow, lb.y1 + grow)
                if (has_line and len(verts) > 1 and Path(verts).intersects_bbox(bb, filled=False)) or \
                        (has_marker and any(bb.contains(x, y) for x, y in verts)):
                    found.append("legend / plotted line or marker")
            for coll in ax.collections:
                for p in coll.get_paths():
                    bb = Bbox.from_extents(lb.x0 - pad, lb.y0 - pad, lb.x1 + pad, lb.y1 + pad)
                    if Path(coll.get_transform().transform(p.vertices)).intersects_bbox(bb, filled=True):
                        found.append("legend / shaded band")
    return sorted(set(found))

# ──────────────────────────────────────────────────────────────────────────────
# 6e. Control comparison
# ──────────────────────────────────────────────────────────────────────────────

def plot_control_comparison(sys, a=0.5, D_eval=0.01, dt=0.01, T_sim=1000, r_neigh=np.inf, T_train=2500, tau=10, use_whitened=True, num_trajs=10, filepath="figures/control_comparison.pdf"):
    set_ieee_style()
    tracker = OnlineCCATracker(dim=sys.x_star.size, dt=dt, gamma=0.01, tau=tau, min_steps=20, a=a, rng_seed=0, use_whitened=use_whitened)
    tracker.train(sys.drift, T_train, dt, max(D_eval, 0.005), 0, sys.w_u_true, sys.x_star, r_neigh=r_neigh)
    
    rng, w_u_norm = np.random.default_rng(42), sys.w_u_true / np.linalg.norm(sys.w_u_true)
    c_opt = partial(erf_feedback, w_u=sys.w_u_true, lambda_u=sys.lambda_u_true, D=D_eval, x_fp=sys.x_star, a=a, r_neigh=r_neigh)
    
    fig, axes = plt.subplots(2, 1, figsize=(6, 4.5), sharex=True)
    c_opt_col, c_cca_col = PLOT_CONFIG["COLORS"].get("Optimal", "k"), PLOT_CONFIG["COLORS"].get("CCA", "b")
    alpha_val = 0.8 if num_trajs == 1 else max(0.02, 2.0 / num_trajs)
    
    for i in range(num_trajs):
        x0, noise_seq = sys.x_star + rng.uniform(0.1, 0.3) * w_u_norm, np.sqrt(2.0 * D_eval) * rng.standard_normal((T_sim, sys.x_star.size)) * np.sqrt(dt)
        x_opt, xu_opt, u_opt = x0.copy(), [], []
        x_cca, xu_cca, u_cca, tr_copy = x0.copy(), [], [], copy.deepcopy(tracker); tr_copy.reset()
        
        for t in range(T_sim):
            # Optimal Simulation
            x_u = float(np.dot(sys.w_u_true, x_opt - sys.x_star))
            if abs(x_u) < a:
                u = c_opt(x_opt); xu_opt.append(x_u); u_opt.append(float(np.dot(sys.w_u_true, u)))
                x_opt += (sys.drift(t*dt, x_opt) + u)*dt + noise_seq[t]
            
            # CCA Simulation
            x_u_cca = float(np.dot(sys.w_u_true, x_cca - sys.x_star))
            if abs(x_u_cca) < a:
                p = tr_copy.get_params(sys.w_u_true)
                u = np.clip(erf_feedback(x_cca, p['w_u'], p['lambda_u'], p['D'], p['x_star'], a, r_neigh), -50.0, 50.0)
                xu_cca.append(x_u_cca); u_cca.append(float(np.dot(sys.w_u_true, u)))
                drift_val = sys.drift(t*dt, x_cca)
                x_next = x_cca + (drift_val + u)*dt + noise_seq[t]
                tr_copy.update(x_cca, x_cca + drift_val*dt + noise_seq[t], sys.w_u_true); x_cca = x_next
                
        time_opt, time_cca = np.arange(len(xu_opt))*dt, np.arange(len(xu_cca))*dt
        axes[0].plot(time_opt, xu_opt, color=c_opt_col, lw=1.2, label=PLOT_CONFIG["LABELS"].get("Optimal") if i==0 else "", alpha=alpha_val)
        axes[0].plot(time_cca, xu_cca, color=c_cca_col, lw=1.2, label=PLOT_CONFIG["LABELS"].get("CCA") if i==0 else "", alpha=alpha_val)
        axes[1].plot(time_opt, u_opt, color=c_opt_col, lw=1.2, alpha=alpha_val)
        axes[1].plot(time_cca, u_cca, color=c_cca_col, lw=1.2, alpha=alpha_val)

    axes[0].axhline(0, color='grey', ls=':', lw=0.8); axes[0].axhline(a, color='#cc0000', ls='--', lw=0.8, alpha=0.7); axes[0].axhline(-a, color='#208820', ls='--', lw=0.8, alpha=0.7)
    axes[0].set_ylabel(r"$x_u = w_u^\top(x - x^\ast)$"); axes[0].legend(loc="upper right")
    axes[1].axhline(0, color='grey', ls=':', lw=0.8); axes[1].set(xlabel="Time", ylabel=r"$w_u^\top u(x)$ (control effort)")
    fig.tight_layout(); fig.savefig(filepath, bbox_inches="tight"); fig.savefig(filepath.replace('.pdf', '.png'), bbox_inches="tight"); plt.close(fig)
    
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
    print(f"  True:           lambda_u={sys.lambda_u_true:.3f}, w_u={sys.w_u_true}")
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
        n_success = sum(1 for _, hs, _ in t_list if hs == "reached_A")
        n_time = sum(1 for _, hs, _ in t_list if hs == "timeout")
        n_fail = sum(1 for _, hs, _ in t_list if hs == "hit_B")
        energies_succ = [en for _, hs, en in t_list if hs == "reached_A"]
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

def run_dimensionality_scaling_experiment(dt=0.01, num_trajs=1000, T_sim=5000, use_whitened=True, 
                                         dimensions=[2, 5, 10, 20], lambdas=[2.0, 5.0],
                                         noise_levels=[0.0, 0.001, 0.005, 0.01, 0.02, 0.05, 0.075, 0.1],
                                         energy_csv_reference="figures/noise_sweep_stats_WHITENED_CCA.csv"):
    print("\n" + "="*80)
    mode_str = "WHITENED CCA" if use_whitened else "STANDARD CCA (Algo 1)"
    print(f" Dimensionality Scaling Experiment | Mode: {mode_str}")
    print("="*80)
    
    a, r_neigh = 0.5, np.inf
    clean_mode_str = mode_str.replace(" ", "_").replace("(", "").replace(")", "").replace(".", "")
    dim_filepath = f"figures/dimensionality_scaling_stats_{clean_mode_str}.csv"
    
    # --- Data Collection ---
    with open(dim_filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Dimension", "Lambda_u", "Noise_D", "Method", "Success_Rate", "Timeout_Rate", "Fail_Rate"])
        
        for dim in dimensions:
            for lam_u in lambdas:
                # [cite: 52] System dynamics linearization [cite: 80]
                sys = make_random_linear_system(dim, lam_u)
                w_u_norm = sys.w_u_true / np.linalg.norm(sys.w_u_true)
                
                # Orthogonal stable direction for initial condition sampling [cite: 233]
                w_s = np.random.default_rng(42).standard_normal(dim)
                w_s -= np.dot(w_s, w_u_norm) * w_u_norm
                w_s /= np.linalg.norm(w_s) + 1e-15
                
                for D_test in noise_levels:
                    rng_ic = np.random.default_rng(11)
                    # ICs sampled on undesired side [cite: 233, 646]
                    ics = [sys.x_star + rng_ic.uniform(0.02, a*0.9)*w_u_norm + 
                           rng_ic.uniform(-a*0.4, a*0.4)*w_s for _ in range(num_trajs)]
                    
                    # Oracle Controller (ReLU for D=0, Erf/Committor for D>0) [cite: 125, 156]
                    kw = dict(a=a, r_neigh=r_neigh)
                    if D_test < 1e-8: 
                        c_opt = partial(relu_feedback, w_u=sys.w_u_true, lambda_u=sys.lambda_u_true, D=D_test, x_fp=sys.x_star, **kw)
                    else: 
                        c_opt = partial(erf_feedback, w_u=sys.w_u_true, lambda_u=sys.lambda_u_true, D=D_test, x_fp=sys.x_star, **kw)
                    
                    # Simulation [cite: 166, 642]
                    opt_results = [simulate_trajectory(x0, T_sim, dt, sys.drift, D_test, c_opt, sys, a, np.random.default_rng())[1] for x0 in ics]
                    hits_opt = sum(1 for hs in opt_results if hs == "reached_A")
                    
                    # Data-Enabled Controller Training (Algorithm 1) [cite: 190, 602]
                    D_train = max(D_test, 0.005)
                    alg = CCATracker(dim=dim, dt=dt, tau=10, rng_seed=0, use_whitened=use_whitened)
                    alg.train(sys.drift, 2500, dt, D_train, 0, a, sys.w_u_true, init_center=sys.x_star)
                    
                    # Evaluation [cite: 266, 675]
                    cca_results = [simulate_adaptive_trajectory(x0, T_sim, dt, sys.drift, D_test, copy.deepcopy(alg), sys, a, r_neigh, None, np.random.default_rng(), is_adaptive=False)[1] for x0 in ics]
                    hits_cca = sum(1 for hs in cca_results if hs == "reached_A")
                    time_cca = sum(1 for hs in cca_results if hs == "timeout")
                    fail_cca = sum(1 for hs in cca_results if hs == "hit_B")
                    
                    writer.writerow([dim, lam_u, D_test, "CCA", hits_cca/num_trajs, time_cca/num_trajs, fail_cca/num_trajs])
                    # (Simplified row writing for brevity, include "Optimal" if you wish to track it in scaling plots)
                    
                    print(f" Dim {dim:2}, lam_u {lam_u:3}, D {D_test:5.3f} | Opt [A: {hits_opt/num_trajs:.2f}] | CCA [A: {hits_cca/num_trajs:.2f}]")

    print(f"Data saved -> {dim_filepath}")

    # Top panel: Scaling results (Success) | Bottom panel: Reference energy results
    combined_plot_path = f"figures/combined_scaling_energy_{clean_mode_str}.pdf"
    plot_combined_scaling_and_energy(
        dim_csv_filepath=dim_filepath, 
        energy_csv_filepath=energy_csv_reference, 
        filepath=combined_plot_path
    )
    print(f"Combined plot saved -> {combined_plot_path}")

# ──────────────────────────────────────────────────────────────────────────────
# 8. Baseline Comparisons & Ablations
# ──────────────────────────────────────────────────────────────────────────────

def run_baseline_comparison(sys, a=0.5, r_neigh=np.inf, D_eval=0.01, dt=0.01,
                            num_trajs=500, T_sim=2000, T_train=2500, tau=10,
                            noise_levels=None, csv_filepath="figures/baselines.csv"):
    """Compare committor controller against LQR, sign-gated, and direct VAR baselines."""
    if noise_levels is None: noise_levels = [0.0, 0.001, 0.005, 0.01, 0.02, 0.05, 0.1, .5, 1.]
    w_u_norm = sys.w_u_true / np.linalg.norm(sys.w_u_true)
    w_s = np.random.default_rng(42).standard_normal(sys.x_star.size)
    w_s -= np.dot(w_s, w_u_norm) * w_u_norm; w_s /= np.linalg.norm(w_s) + 1e-15
    os.makedirs(os.path.dirname(csv_filepath), exist_ok=True)
    print(f"\n  Baseline comparison: {sys.name}")
    with open(csv_filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["System", "Method", "Noise_D", "Success_Rate", "N_Total",
                         "Mean_Energy_Succ", "Std_Energy_Succ", "N_Success",
                         "Mean_Energy_All", "Std_Energy_All", "N_All"])
        for D_test in noise_levels:
            D_train = max(D_test, 0.005)
            kw = dict(a=a, r_neigh=r_neigh)
            # --- Oracle controllers ---
            if D_test < 1e-8:
                c_committor = partial(relu_feedback, w_u=sys.w_u_true, lambda_u=sys.lambda_u_true, D=D_test, x_fp=sys.x_star, **kw)
            else:
                c_committor = partial(erf_feedback, w_u=sys.w_u_true, lambda_u=sys.lambda_u_true, D=D_test, x_fp=sys.x_star, **kw)
            c_lqr_oracle = partial(lqr_feedback, w_u=sys.w_u_true, lambda_u=sys.lambda_u_true, D=D_test, x_fp=sys.x_star, **kw)
            c_relu_oracle = partial(sign_gated_feedback, w_u=sys.w_u_true, lambda_u=sys.lambda_u_true, D=D_test, x_fp=sys.x_star, **kw)

            # --- Whitened SVD estimator ---
            alg_wsvd = CCATracker(dim=sys.x_star.size, dt=dt, tau=tau, rng_seed=0, use_whitened=True)
            alg_wsvd.train(sys.drift, T_train, dt, D_train, 0, a, sys.w_u_true, init_center=sys.x_star, r_neigh=r_neigh)
            p_wsvd = alg_wsvd.get_params(sys.w_u_true)

            # --- Direct VAR(1) estimator ---
            alg_var = DirectVARTracker(dim=sys.x_star.size, dt=dt, rng_seed=0)
            alg_var.train(sys.drift, T_train, dt, D_train, 0, a, sys.w_u_true, init_center=sys.x_star, r_neigh=r_neigh)
            p_var = alg_var.get_params(sys.w_u_true)

            # --- Build all controllers ---
            def _make_ctrl(feedback_fn, params, D_ctrl):
                return partial(feedback_fn, w_u=params['w_u'], lambda_u=params['lambda_u'], D=D_ctrl, x_fp=params['x_star'], **kw)

            methods = [
                ("Committor (oracle)",   c_committor),
                ("LQR (oracle)",         c_lqr_oracle),
                ("ReLU (oracle)",        c_relu_oracle),
                ("Committor (wSVD)",     _make_ctrl(erf_feedback if D_test >= 1e-8 else relu_feedback, p_wsvd, p_wsvd['D'])),
                ("LQR (wSVD)",           _make_ctrl(lqr_feedback, p_wsvd, p_wsvd['D'])),
                ("ReLU (wSVD)",          _make_ctrl(sign_gated_feedback, p_wsvd, p_wsvd['D'])),
                ("Committor (VAR)",      _make_ctrl(erf_feedback if D_test >= 1e-8 else relu_feedback, p_var, p_var['D'])),
                ("ReLU (VAR)",           _make_ctrl(sign_gated_feedback, p_var, p_var['D'])),
            ]

            rng_ic = np.random.default_rng(11)
            ics = [sys.x_star + rng_ic.uniform(0, a * 0.75) * w_u_norm + rng_ic.uniform(-a * 0.75, a * 0.75) * w_s for _ in range(num_trajs)]

            for method_name, ctrl in methods:
                successes = 0
                energies_succ, energies_all = [], []
                for x0 in ics:
                    run_rng = np.random.default_rng(int(rng_ic.integers(0, 10**6)))
                    _, hit_side, energy = simulate_trajectory(x0, T_sim, dt, sys.drift, D_test, ctrl, sys, a, run_rng)
                    energies_all.append(energy)
                    if hit_side == "reached_A": successes += 1; energies_succ.append(energy)
                sr = successes / num_trajs
                me_s = np.mean(energies_succ) if energies_succ else 0.0
                se_s = np.std(energies_succ) if energies_succ else 0.0
                me_a = np.mean(energies_all) if energies_all else 0.0
                se_a = np.std(energies_all) if energies_all else 0.0
                writer.writerow([sys.name, method_name, D_test, sr, num_trajs,
                                 me_s, se_s, len(energies_succ), me_a, se_a, len(energies_all)])
                print(f"    D={D_test:.4f} | {method_name:<22} | SR={sr:.3f} | E_s={me_s:.3f} | E_a={me_a:.3f}")
    print(f"  Saved -> {csv_filepath}")


def plot_baseline_comparison(
    csv_filepath="figures/baselines.csv", 
    filepath="figures/baseline_comparison.pdf", 
    ylim_sr=(-0.05, 1.05), 
    ylim_energy=None, 
    yticks_sr="minmax",        
    yticks_energy="minmax",    
    orientation="horizontal",
    figsize_ratio=None,
    base_size=4.0,
    min_D=None,
    legend_loc="lower left",
    legend_ax=1,               
    font_size=10               
):
    """Plots Success Rate and Mean Control Energy for the baseline comparisons."""
    if not os.path.exists(csv_filepath): 
        print(f" [!] Data file {csv_filepath} not found.")
        return
        
    data = []
    with open(csv_filepath, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader: 
            noise_val = float(row['Noise_D'])
            if min_D is not None and noise_val < min_D:
                continue
            data.append({
                'Method': row['Method'], 
                'Noise_D': noise_val, 
                'Success_Rate': float(row['Success_Rate']),
                'Mean_Energy': float(row['Mean_Energy']),
                'Std_Energy': float(row.get('Std_Energy', 0.0))
            })
            
    plotted_methods = set(d['Method'] for d in data)
            
    noises = sorted(list(set(d['Noise_D'] for d in data)))
    has_zero = 0.0 in noises
    non_zero_noises = [n for n in noises if n > 0]
    pseudo_zero = min(non_zero_noises) / 3.0 if non_zero_noises else 0.0001
    def map_x(n): return pseudo_zero if n == 0.0 else n

    if figsize_ratio is None:
        figsize_ratio = (2.2, 1) if orientation == "horizontal" else (1, 1)
    
    width = base_size * figsize_ratio[0]
    height = base_size * figsize_ratio[1]

    if orientation == "horizontal":
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(width, height))
        fig.subplots_adjust(wspace=0.25)
    else:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(width, height), sharex=True)
        fig.subplots_adjust(hspace=0.1)
    
    # --- 1. ENCODING: COLOR = ESTIMATOR ---
    colors = {
        "Oracle Committor": "#025230",          
        "Oracle ReLU": "#025230",               
        "Est CCA Committor": "#2c327b",         
        "Est CCA ReLU": "#2c327b",              
        "Est VAR Committor": "#8a2028",         
        "Adaptive CCA Committor": "#d17c00"     
    }
    markers = {
        "Oracle Committor": "D", "Oracle ReLU": "D", 
        "Est CCA Committor": "o", "Est CCA ReLU": "o", 
        "Est VAR Committor": "s", "Adaptive CCA Committor": "X"        
    }
    linestyles = {
        "Oracle Committor": "-", "Oracle ReLU": "--", 
        "Est CCA Committor": "-", "Est CCA ReLU": "--", 
        "Est VAR Committor": "-", "Adaptive CCA Committor": "-"        
    }
    
    for method in colors.keys():
        if method not in plotted_methods:
            continue
            
        xs_zero, ys_sr_zero, ys_en_zero, sem_en_zero = [], [], [], []
        xs_pos, ys_sr_pos, ys_en_pos, sem_en_pos = [], [], [], []
        
        for n in noises:
            row = next((r for r in data if r['Method'] == method and r['Noise_D'] == n), None)
            if row:
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
        
        is_oracle = "Oracle" in method
        line_w = 3.0 if is_oracle else 1.5
        m_size = 7 if is_oracle else 5
        
        if xs_zero: ax1.plot(xs_zero, ys_sr_zero, marker=mrk, color=col, ls='none', lw=line_w, markersize=m_size)
        if xs_pos: ax1.plot(xs_pos, ys_sr_pos, marker=mrk, color=col, ls=ls, lw=line_w, markersize=m_size)
        
        if xs_zero: 
            ax2.plot(xs_zero, ys_en_zero, marker=mrk, color=col, ls='none', lw=line_w, markersize=m_size)
            if any(e > 0 for e in ys_en_zero):
                lower_err = np.minimum(ys_en_zero, 1.96 * np.array(sem_en_zero)) 
                ax2.errorbar(xs_zero, ys_en_zero, yerr=[lower_err, 1.96*np.array(sem_en_zero)], color=col, fmt='none', alpha=0.5)

        if xs_pos: 
            ax2.plot(xs_pos, ys_en_pos, marker=mrk, color=col, ls=ls, lw=line_w, markersize=m_size)
            ys_en_pos_arr = np.array(ys_en_pos)
            sem_en_pos_arr = np.array(sem_en_pos)
            ci_lower = np.maximum(0, ys_en_pos_arr - 1.96 * sem_en_pos_arr)
            ci_upper = ys_en_pos_arr + 1.96 * sem_en_pos_arr
            ax2.fill_between(xs_pos, ci_lower, ci_upper, color=col, alpha=0.15 if not is_oracle else 0.05)
        
    # --- Formatting ---
    if ylim_sr is not None: ax1.set_ylim(ylim_sr)
    if ylim_energy is not None: ax2.set_ylim(ylim_energy)
    else: ax2.set_ylim(bottom=min(-0.1, ax2.get_ylim()[0]))

    # Helper function to strip .0 from integers
    def fmt_tick(y):
        return f"{int(y)}" if y == int(y) else f"{y}"

    # Apply bulletproof Y-Ticks
    if yticks_sr == "minmax": 
        ticks = list(ax1.get_ylim())
        ax1.set_yticks(ticks)
        ax1.set_yticklabels([fmt_tick(y) for y in ticks])
    elif yticks_sr is not None: 
        ax1.set_yticks(yticks_sr)
        ax1.set_yticklabels([fmt_tick(y) for y in yticks_sr])

    if yticks_energy == "minmax": 
        ticks = list(ax2.get_ylim())
        ax2.set_yticks(ticks)
        ax2.set_yticklabels([fmt_tick(y) for y in ticks])
    elif yticks_energy is not None: 
        ax2.set_yticks(yticks_energy)
        ax2.set_yticklabels([fmt_tick(y) for y in yticks_energy])

    ax1.set_ylabel("Success rate", fontsize=font_size)
    ax2.set_ylabel("Control energy", fontsize=font_size)
    
    for ax in [ax1, ax2]:
        ax.set_xscale('log')
        ax.set_xticks([map_x(n) for n in noises])
        
        if orientation == "horizontal" or ax == ax2:
            ax.set_xticklabels(['0' if n == 0.0 else f"{n:.3g}" for n in noises], rotation=45, ha='right', fontsize=font_size)
            ax.set_xlabel("Noise strength ($D$)", fontsize=font_size)
        elif orientation == "vertical" and ax == ax1:
            plt.setp(ax.get_xticklabels(), visible=False)
            
        ax.tick_params(axis='y', labelsize=font_size)
        ax.grid(True, linestyle='-', alpha=0.3, color='gray')
        if has_zero and non_zero_noises: 
            ax.axvline(x=pseudo_zero * 1.5, color='gray', linestyle=':', alpha=0.5)

    # --- 2. STREAMLINED 2-GROUP LEGEND ---
    custom_handles = []

    custom_handles.append(Line2D([0], [0], color='none', label=r'$\mathbf{Estimator}$'))
    if any("Oracle" in m for m in plotted_methods):
        custom_handles.append(Line2D([0], [0], color='#025230', lw=2, marker='D', markersize=6, label='Oracle'))
    if any("CCA" in m for m in plotted_methods):
        custom_handles.append(Line2D([0], [0], color='#2c327b', lw=2, marker='o', markersize=6, label='CCA Estimator'))
    if any("VAR" in m for m in plotted_methods):
        custom_handles.append(Line2D([0], [0], color='#8a2028', lw=2, marker='s', markersize=6, label='VAR Estimator'))
    if any("Adaptive" in m for m in plotted_methods):
        custom_handles.append(Line2D([0], [0], color='#d17c00', lw=2, marker='X', markersize=6, label='Adaptive CCA'))

    custom_handles.append(Line2D([0], [0], color='none', label=' ')) 
    
    custom_handles.append(Line2D([0], [0], color='none', label=r'$\mathbf{Controller}$'))
    if any("Committor" in m for m in plotted_methods):
        custom_handles.append(Line2D([0], [0], color='gray', lw=1.5, ls='-', label='Committor'))
    if any("ReLU" in m for m in plotted_methods):
        custom_handles.append(Line2D([0], [0], color='gray', lw=1.5, ls='--', label='ReLU'))

    target_ax = ax1 if legend_ax == 1 else ax2
    target_ax.legend(
        handles=custom_handles,
        loc=legend_loc, 
        fontsize=font_size * 0.9, 
        labelspacing=0.2,       
        handlelength=1.5,       
        handletextpad=0.5,      
        borderaxespad=0.5,      
        framealpha=0.9
    )
    
    if orientation == "vertical":
        fig.align_ylabels([ax1, ax2])
    
    fig.tight_layout()
    fig.savefig(filepath, bbox_inches="tight")
    fig.savefig(filepath.replace('.pdf', '.png'), bbox_inches="tight")
    print(f" Saved Grouped Baseline Plot (legend_ax={legend_ax}) -> {filepath}")
    plt.close(fig)


def run_trust_region_ablation(sys, a=0.5, D_eval=0.01, dt=0.01, num_trajs=500, T_sim=2000,
                               T_train=2500, tau=10, r_neigh_values=None,
                               csv_filepath="figures/trust_region_ablation.csv"):
    """Ablation over trust-region radius r_neigh."""
    if r_neigh_values is None: r_neigh_values = [0.25, 0.5, 1.0, 2.0, np.inf]
    w_u_norm = sys.w_u_true / np.linalg.norm(sys.w_u_true)
    w_s = np.random.default_rng(42).standard_normal(sys.x_star.size)
    w_s -= np.dot(w_s, w_u_norm) * w_u_norm; w_s /= np.linalg.norm(w_s) + 1e-15
    D_train = max(D_eval, 0.005)
    os.makedirs(os.path.dirname(csv_filepath), exist_ok=True)
    print(f"\n  Trust-region ablation: {sys.name}")
    with open(csv_filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["System", "R_neigh", "Cos_Sim", "Lambda_u_hat", "D_hat", "Success_Rate", "Mean_Energy", "Std_Energy", "N_Success"])
        rng_ic = np.random.default_rng(11)
        ics = [sys.x_star + rng_ic.uniform(0, a * 0.75) * w_u_norm + rng_ic.uniform(-a * 0.75, a * 0.75) * w_s for _ in range(num_trajs)]
        for rn in r_neigh_values:
            alg = CCATracker(dim=sys.x_star.size, dt=dt, tau=tau, rng_seed=0, use_whitened=True)
            alg.train(sys.drift, T_train, dt, D_train, 0, a, sys.w_u_true, init_center=sys.x_star, r_neigh=rn)
            p = alg.get_params(sys.w_u_true)
            cos_sim = abs(float(np.dot(p['w_u'], sys.w_u_true)))
            kw = dict(a=a, r_neigh=rn)
            if D_eval < 1e-8: ctrl = partial(relu_feedback, w_u=p['w_u'], lambda_u=p['lambda_u'], D=D_eval, x_fp=p['x_star'], **kw)
            else: ctrl = partial(erf_feedback, w_u=p['w_u'], lambda_u=p['lambda_u'], D=p['D'], x_fp=p['x_star'], **kw)
            successes, energies = 0, []
            for x0 in ics:
                run_rng = np.random.default_rng(int(rng_ic.integers(0, 10**6)))
                _, hit_side, energy = simulate_trajectory(x0, T_sim, dt, sys.drift, D_eval, ctrl, sys, a, run_rng)
                if hit_side == "reached_A": successes += 1; energies.append(energy)
            sr = successes / num_trajs; me = np.mean(energies) if energies else 0.0; se = np.std(energies) if energies else 0.0
            rn_str = f"{rn:.2f}" if rn < np.inf else "inf"
            writer.writerow([sys.name, rn_str, cos_sim, p['lambda_u'], p['D'], sr, me, se, len(energies)])
            print(f"    r_neigh={rn_str:>6} | cos={cos_sim:.4f} | lam={p['lambda_u']:.3f} | D={p['D']:.5f} | SR={sr:.3f} | E={me:.3f}")
    print(f"  Saved -> {csv_filepath}")

def run_table1_neighborhood_ablation(sys, rho_values=(0.1, 0.25, 0.5, 1.0, np.inf), a=0.5, dt=0.01,
                                     D_train=0.01, D_eval=0.01, tau=2, T_train=5000, train_seed=42,
                                     estimator="batch", gamma=0.01, use_whitened=True,
                                     evaluate=True, num_trajs=1000, T_sim=2000, ic_seed=11,
                                     ic_u_range=(0.0, 0.75), ic_s_spread=0.75, gate_control=False,
                                     csv_filepath="figures/table_neighborhood_ablation_repro.csv"):
    """Table I: identification radius rho vs. estimation error and control performance.

    For each rho the estimator is trained on passive data kept within ||x - x*|| <= rho (restart at the
    saddle otherwise); the frozen committor controller is then evaluated from x* + U(ic_u_range)*a*w_u
    + U(-ic_s_spread*a, ic_s_spread*a)*w_s. With gate_control the controller is also off outside rho.
    Rows use the columns of figures/table_neighborhood_ablation.csv; returns the rows.

    Defaults reproduce figures/table_neighborhood_ablation.csv (Table I): the estimation columns exactly,
    Success/Energy up to Monte Carlo error (the original evaluation seeds were not saved).
    """
    w_u_norm = sys.w_u_true / np.linalg.norm(sys.w_u_true)
    w_s = np.random.default_rng(42).standard_normal(sys.x_star.size)
    w_s -= np.dot(w_s, w_u_norm) * w_u_norm; w_s /= np.linalg.norm(w_s) + 1e-15
    rows = []
    for rho in rho_values:
        if estimator == "batch":
            trk = CCATracker(dim=sys.x_star.size, dt=dt, tau=tau, rng_seed=0, use_whitened=use_whitened)
            trk.train(sys.drift, T_train, dt, D_train, train_seed, a, sys.w_u_true, init_center=sys.x_star, r_neigh=rho)
        else:
            trk = OnlineCCATracker(dim=sys.x_star.size, dt=dt, gamma=gamma, tau=tau, a=a, rng_seed=0, use_whitened=use_whitened)
            trk.train(sys.drift, T_train, dt, D_train, train_seed, sys.w_u_true, sys.x_star, r_neigh=rho)
        p = trk.get_params(sys.w_u_true)
        cos_sim = min(1.0, abs(float(np.dot(p['w_u'], w_u_norm))))
        row = {"a_train": str(float(rho)),
               "Condition": "Ungated (Infinite)" if np.isinf(rho) else f"Gated (a={float(rho)})",
               "Est_Lambda_u": f"{p['lambda_u']:.2f}", "Lam_Error": f"{abs(p['lambda_u'] - sys.lambda_u_true):.2f}",
               "w_u_Error_Deg": f"{np.degrees(np.arccos(cos_sim)):.1f}",
               "Est_D": f"{p['D']:.4f}", "D_Error": f"{abs(p['D'] - D_train):.4f}",
               "Success_Rate": "", "Mean_Energy": ""}
        if evaluate:
            kw = dict(a=a, r_neigh=rho if gate_control else np.inf)
            feedback = relu_feedback if D_eval < 1e-8 else erf_feedback
            ctrl = partial(feedback, w_u=p['w_u'], lambda_u=p['lambda_u'], D=p['D'], x_fp=p['x_star'], **kw)
            rng_ic = np.random.default_rng(ic_seed)
            ics = [sys.x_star + rng_ic.uniform(a * ic_u_range[0], a * ic_u_range[1]) * w_u_norm
                   + rng_ic.uniform(-a * ic_s_spread, a * ic_s_spread) * w_s for _ in range(num_trajs)]
            energies = []
            for x0 in ics:
                run_rng = np.random.default_rng(int(rng_ic.integers(0, 10**6)))
                _, hit_side, energy = simulate_trajectory(x0, T_sim, dt, sys.drift, D_eval, ctrl, sys, a, run_rng)
                if hit_side == "reached_A": energies.append(energy)
            row["Success_Rate"] = f"{len(energies) / num_trajs:.2f}"
            row["Mean_Energy"] = f"{np.mean(energies) if energies else 0.0:.2f}"
        rows.append(row)
    if csv_filepath:
        os.makedirs(os.path.dirname(csv_filepath) or ".", exist_ok=True)
        with open(csv_filepath, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader(); writer.writerows(rows)
    return rows

# Earlier draft of the Table I ablation; it does not run (sys.dim, wrong simulate_trajectory arguments).
# Use run_table1_neighborhood_ablation instead.
def run_neighborhood_ablation(sys, a_eval, a_train_levels, dt, tau, D_train, D_test, 
                              num_eval_trajs, csv_filepath):
    """
    Evaluates how the training neighborhood size (a) affects estimator 
    accuracy and control performance.
    """
    fieldnames = ["Config", "dLambda", "dW_angle", "dD", "Success", "Energy"]
    
    with open(csv_filepath, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for a_train in a_train_levels:
            # 1. Train the Estimator (Algorithm 1)
            # Use a_train to set the "limit" or "r_neigh" in the training phase
            tracker = CCATracker(dim=sys.dim, dt=dt, tau=tau, use_whitened=True)
            
            # The 'train' method uses the neighborhood size to gate data 
            tracker.train(
                sys.drift, T=5000, dt=dt, D=D_train, seed=42, 
                a=a_eval, sign_ref=sys.w_u_true, 
                init_center=sys.x_star, r_neigh=a_train
            )
            
            params = tracker.get_params(sys.w_u_true)
            
            # 2. Compute Estimation Errors 
            d_lambda = abs(params['lambda_u'] - sys.lambda_u_true)
            
            # Angular error in degrees 
            cos_sim = np.clip(np.dot(params['w_u'], sys.w_u_true), -1.0, 1.0)
            dw_angle = np.degrees(np.arccos(cos_sim))
            
            d_D = abs(params['D'] - D_test)

            # 3. Evaluate Control Performance 
            # Sample initial conditions near the saddle [cite: 243]
            rng = np.random.default_rng(123)
            success_count = 0
            total_energy = 0.0
            
            for _ in range(num_eval_trajs):
                # Standard IC sampling from the manuscript [cite: 243]
                x0 = sys.x_star + rng.uniform(0, 0.75*a_eval) * sys.w_u_true
                
                # simulate_trajectory should return (path, hit_status, energy)
                _, status, energy = simulate_trajectory(
                    x0, T_sim=2000, dt=dt, drift_func=sys.drift, 
                    D=D_test, controller=tracker, sys=sys, a=a_eval, rng=rng
                )
                
                if status == "reached_A": # Success 
                    success_count += 1
                    total_energy += energy

            success_rate = success_count / num_eval_trajs
            avg_energy = total_energy / success_count if success_count > 0 else 0.0

            # 4. Save results 
            config_label = f"a={a_train}" if a_train != np.inf else "Ungated"
            row = {
                "Config": config_label,
                "dLambda": f"{d_lambda:.2f}",
                "dW_angle": f"{dw_angle:.1f}°",
                "dD": f"{d_D:.4f}",
                "Success": f"{success_rate:.2f}",
                "Energy": f"{avg_energy:.2f}"
            }
            writer.writerow(row)
            print(f"Finished {config_label}: Success={success_rate:.2f}, Angle Err={dw_angle:.1f}°")

    print(f"Ablation results saved to {csv_filepath}")

def run_hyperparameter_ablation(sys, a=0.5, r_neigh=np.inf, D_eval=0.01, dt=0.01,
                                 num_trajs=500, T_sim=2000, T_train=2500,
                                 tau_values=None, gamma_values=None,
                                 csv_filepath="figures/hyperparam_ablation.csv"):
    """Ablation over lag tau and EMA discount gamma."""
    if tau_values is None: tau_values = [1, 5, 10, 20, 50]
    if gamma_values is None: gamma_values = [0.001, 0.005, 0.01, 0.05, 0.1]
    w_u_norm = sys.w_u_true / np.linalg.norm(sys.w_u_true)
    w_s = np.random.default_rng(42).standard_normal(sys.x_star.size)
    w_s -= np.dot(w_s, w_u_norm) * w_u_norm; w_s /= np.linalg.norm(w_s) + 1e-15
    D_train = max(D_eval, 0.005)
    os.makedirs(os.path.dirname(csv_filepath), exist_ok=True)
    print(f"\n  Hyperparameter ablation: {sys.name}")
    with open(csv_filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["System", "Tau", "Gamma", "Cos_Sim", "Lambda_u_hat", "D_hat", "Success_Rate", "Mean_Energy"])
        rng_ic = np.random.default_rng(11)
        ics = [sys.x_star + rng_ic.uniform(0, a * 0.75) * w_u_norm + rng_ic.uniform(-a * 0.75, a * 0.75) * w_s for _ in range(num_trajs)]
        default_gamma = 0.01; default_tau = 10
        # Sweep tau with fixed gamma
        for tau_val in tau_values:
            alg = CCATracker(dim=sys.x_star.size, dt=dt, tau=tau_val, rng_seed=0, use_whitened=True)
            alg.train(sys.drift, T_train, dt, D_train, 0, a, sys.w_u_true, init_center=sys.x_star, r_neigh=r_neigh)
            p = alg.get_params(sys.w_u_true)
            cos_sim = abs(float(np.dot(p['w_u'], sys.w_u_true)))
            kw = dict(a=a, r_neigh=r_neigh)
            ctrl = partial(erf_feedback, w_u=p['w_u'], lambda_u=p['lambda_u'], D=p['D'], x_fp=p['x_star'], **kw)
            successes, energies = 0, []
            for x0 in ics:
                run_rng = np.random.default_rng(int(rng_ic.integers(0, 10**6)))
                _, hit_side, energy = simulate_trajectory(x0, T_sim, dt, sys.drift, D_eval, ctrl, sys, a, run_rng)
                if hit_side == "reached_A": successes += 1; energies.append(energy)
            sr = successes / num_trajs; me = np.mean(energies) if energies else 0.0
            writer.writerow([sys.name, tau_val, default_gamma, cos_sim, p['lambda_u'], p['D'], sr, me])
            print(f"    tau={tau_val:3}, gamma={default_gamma:.3f} | cos={cos_sim:.4f} | lam={p['lambda_u']:.3f} | SR={sr:.3f}")
        # Sweep gamma with fixed tau (using OnlineCCATracker)
        for gamma_val in gamma_values:
            alg = OnlineCCATracker(dim=sys.x_star.size, dt=dt, gamma=gamma_val, tau=default_tau, min_steps=20, a=a, rng_seed=0, use_whitened=True)
            alg.train(sys.drift, T_train, dt, D_train, 0, sys.w_u_true, sys.x_star, r_neigh=r_neigh)
            p = alg.get_params(sys.w_u_true)
            cos_sim = abs(float(np.dot(p['w_u'], sys.w_u_true)))
            kw = dict(a=a, r_neigh=r_neigh)
            ctrl = partial(erf_feedback, w_u=p['w_u'], lambda_u=p['lambda_u'], D=p['D'], x_fp=p['x_star'], **kw)
            successes, energies = 0, []
            for x0 in ics:
                run_rng = np.random.default_rng(int(rng_ic.integers(0, 10**6)))
                _, hit_side, energy = simulate_trajectory(x0, T_sim, dt, sys.drift, D_eval, ctrl, sys, a, run_rng)
                if hit_side == "reached_A": successes += 1; energies.append(energy)
            sr = successes / num_trajs; me = np.mean(energies) if energies else 0.0
            writer.writerow([sys.name, default_tau, gamma_val, cos_sim, p['lambda_u'], p['D'], sr, me])
            print(f"    tau={default_tau:3}, gamma={gamma_val:.3f} | cos={cos_sim:.4f} | lam={p['lambda_u']:.3f} | SR={sr:.3f}")
    print(f"  Saved -> {csv_filepath}")

def plot_trust_region_ablation(csv_filepath, sys_name, filepath, label_fontsize=11, legend_fontsize=9):
    """Plot trust-region ablation: parameter quality and success rate vs r_neigh."""
    set_ieee_style()
    if not os.path.exists(csv_filepath): return
    data = []
    with open(csv_filepath, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['System'] == sys_name:
                data.append({k: row[k] for k in row})
    if not data: return
    rns = [d['R_neigh'] for d in data]
    cos_vals = [float(d['Cos_Sim']) for d in data]
    lam_vals = [float(d['Lambda_u_hat']) for d in data]
    sr_vals = [float(d['Success_Rate']) for d in data]
    en_vals = [float(d['Mean_Energy']) for d in data]
    x = np.arange(len(rns))

    fig, axes = plt.subplots(2, 2, figsize=(8, 5)); fig.subplots_adjust(hspace=0.35, wspace=0.35)
    axes[0, 0].bar(x, cos_vals, color='#2d307b', alpha=0.8); axes[0, 0].set_xticks(x); axes[0, 0].set_xticklabels(rns, rotation=45)
    axes[0, 0].set_ylabel(r"$|\cos(\hat{w}_u, w_u)|$"); axes[0, 0].set_ylim(0.5, 1.05); axes[0, 0].set_xlabel(r"$r_{\mathrm{neigh}}$")
    axes[0, 1].bar(x, lam_vals, color='#19421e', alpha=0.8); axes[0, 1].set_xticks(x); axes[0, 1].set_xticklabels(rns, rotation=45)
    axes[0, 1].axhline(y=2.0, color='gray', ls='--', lw=1); axes[0, 1].set_ylabel(r"$\hat{\lambda}_u$"); axes[0, 1].set_xlabel(r"$r_{\mathrm{neigh}}$")
    axes[1, 0].bar(x, sr_vals, color='#1a6faf', alpha=0.8); axes[1, 0].set_xticks(x); axes[1, 0].set_xticklabels(rns, rotation=45)
    axes[1, 0].set_ylabel("Success rate"); axes[1, 0].set_ylim(0, 1.05); axes[1, 0].set_xlabel(r"$r_{\mathrm{neigh}}$")
    axes[1, 1].bar(x, en_vals, color='#a01b2a', alpha=0.8); axes[1, 1].set_xticks(x); axes[1, 1].set_xticklabels(rns, rotation=45)
    axes[1, 1].set_ylabel("Mean energy"); axes[1, 1].set_xlabel(r"$r_{\mathrm{neigh}}$")
    fig.savefig(filepath, bbox_inches="tight"); fig.savefig(filepath.replace('.pdf', '.png'), bbox_inches="tight")
    print(f"     Saved trust-region ablation -> {filepath}"); plt.close(fig)

def plot_hyperparameter_ablation(csv_filepath, sys_name, filepath, true_lambda=2.0, label_fontsize=11):
    """Plot hyperparameter ablation: tau sweep and gamma sweep."""
    set_ieee_style()
    if not os.path.exists(csv_filepath): return
    data = []
    with open(csv_filepath, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['System'] == sys_name: data.append({k: row[k] for k in row})
    if not data: return
    # Split tau sweep vs gamma sweep
    tau_data = [d for d in data if float(d['Gamma']) == 0.01]
    gam_data = [d for d in data if int(d['Tau']) == 10]

    fig, axes = plt.subplots(2, 2, figsize=(8, 5)); fig.subplots_adjust(hspace=0.4, wspace=0.35)
    # Tau sweep
    if tau_data:
        taus = [int(d['Tau']) for d in tau_data]
        ax = axes[0, 0]; ax.plot(taus, [float(d['Cos_Sim']) for d in tau_data], 'o-', color='#2d307b')
        ax.set_xlabel(r"$\tau$"); ax.set_ylabel(r"$|\cos(\hat{w}_u, w_u)|$"); ax.set_ylim(0.5, 1.05)
        ax = axes[0, 1]; ax.plot(taus, [float(d['Success_Rate']) for d in tau_data], 's-', color='#1a6faf')
        ax.set_xlabel(r"$\tau$"); ax.set_ylabel("Success rate"); ax.set_ylim(0, 1.05)
    # Gamma sweep
    if gam_data:
        gammas = [float(d['Gamma']) for d in gam_data]
        ax = axes[1, 0]; ax.plot(gammas, [float(d['Cos_Sim']) for d in gam_data], 'o-', color='#2d307b')
        ax.set_xlabel(r"$\gamma$"); ax.set_ylabel(r"$|\cos(\hat{w}_u, w_u)|$"); ax.set_xscale('log'); ax.set_ylim(0.5, 1.05)
        ax = axes[1, 1]; ax.plot(gammas, [float(d['Success_Rate']) for d in gam_data], 's-', color='#1a6faf')
        ax.set_xlabel(r"$\gamma$"); ax.set_ylabel("Success rate"); ax.set_xscale('log'); ax.set_ylim(0, 1.05)
    fig.savefig(filepath, bbox_inches="tight"); fig.savefig(filepath.replace('.pdf', '.png'), bbox_inches="tight")
    print(f"     Saved hyperparam ablation -> {filepath}"); plt.close(fig)

def plot_combined_scaling_and_energy(dim_csv_filepath, energy_csv_filepath,
                                     filepath="figures/combined_scaling_energy.pdf",
                                     include_adaptive=False,
                                     success_ylim=(0.45, 1.05),
                                     energy_ylim=None,
                                     figsize=(6, 6.5),
                                     label_fontsize=12, legend_fontsize=9):
    """Two-panel plot: top = dimensionality scaling from dim CSV, bottom = energy from sweep CSV."""
    set_ieee_style()
    fig, (ax_scale, ax_energy) = plt.subplots(2, 1, figsize=figsize, sharex=True)
    fig.subplots_adjust(hspace=0.08)
 
    noises, non_zero = [], []
    pseudo_zero = 0.0001
    def map_x(n): return pseudo_zero if n == 0.0 else n
 
    # --- Top panel: dimensionality scaling ---
    if os.path.exists(dim_csv_filepath):
        data_dim = []
        with open(dim_csv_filepath, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row['Method'] != "CCA": continue
                data_dim.append({
                    'Dimension': int(row['Dimension']),
                    'Lambda_u': float(row['Lambda_u']),
                    'Noise_D': float(row['Noise_D']),
                    'Value': float(row['Success_Rate']) + float(row.get('Timeout_Rate', 0.0))
                })
        if data_dim:
            dims = sorted(set(d['Dimension'] for d in data_dim))
            lams = sorted(set(d['Lambda_u'] for d in data_dim))
            noises = sorted(set(d['Noise_D'] for d in data_dim))
            non_zero = [n for n in noises if n > 0]
            pseudo_zero = min(non_zero) / 2.0 if non_zero else 0.0001
 
            palettes = {
                lams[0]: [plt.cm.Purples(x) for x in np.linspace(0.85, 0.4, len(dims))],
                lams[1]: [plt.cm.Oranges(x) for x in np.linspace(0.85, 0.4, len(dims))],
            }
            markers = ['o', 'D', 'X', '^', 's']
            for lam in lams:
                for j, dim in enumerate(dims):
                    xs, ys = [], []
                    for n in noises:
                        row = next((r for r in data_dim if r['Dimension']==dim and r['Lambda_u']==lam and r['Noise_D']==n), None)
                        if row: xs.append(map_x(n)); ys.append(row['Value'])
                    if xs:
                        ax_scale.plot(xs, ys, marker=markers[j%len(markers)], markersize=5,
                                      color=palettes[lam][j], ls='-', lw=1.3, alpha=0.8)
 
            ax_scale.set_ylim(success_ylim)
            ax_scale.set_ylabel("Success rate", fontsize=label_fontsize)
            if non_zero: ax_scale.axvline(x=pseudo_zero*1.5, color='gray', ls=':', alpha=0.4, lw=0.6)
 
            leg = [Line2D([], [], color='none', label=r'$\lambda_u$')]
            for lam in lams:
                leg.append(Line2D([], [], color=palettes[lam][len(dims)//2], lw=2, label=f" {int(lam)}"))
            leg.append(Line2D([], [], color='none', label='Dim.'))
            for j, dim in enumerate(dims):
                leg.append(Line2D([], [], color='gray', marker=markers[j%len(markers)], ls='-', lw=1.2, label=f" {dim}"))
            ax_scale.legend(handles=leg, loc='lower left', fontsize=legend_fontsize, ncol=2, framealpha=0.9)
 
    # --- Bottom panel: energy from sweep CSV ---
    data_en = _read_sweep_csv(energy_csv_filepath)
    if data_en:
        methods_en = ["Optimal", "CCA"]
        if not noises:
            noises = sorted(set(d['Noise_D'] for d in data_en))
            non_zero = [n for n in noises if n > 0]
            pseudo_zero = min(non_zero) / 2.0 if non_zero else 0.0001
        for method in methods_en:
            sty = _get_method_style(method)
            xs, ys, sems = [], [], []
            for n in noises:
                if method == "Optimal" and n == 0.0: continue
                row = next((r for r in data_en if r['Noise_D']==n and r['Method']==method), None)
                if row:
                    ns = row.get('N_Success', 0)
                    sem = row['Std_Energy_Succ'] / np.sqrt(max(ns, 1)) if ns > 0 else 0.0
                    xs.append(map_x(n)); ys.append(row['Mean_Energy_Succ']); sems.append(sem)
            if xs:
                ys_a, sems_a = np.array(ys), np.array(sems)
                ax_energy.plot(xs, ys_a, marker=sty['marker'], color=C_ENERGY, ls=sty['ls'],
                               lw=sty['lw'], markersize=5, label=sty['label'])
                if np.any(sems_a > 0):
                    ax_energy.fill_between(xs, np.maximum(0, ys_a - 1.96*sems_a),
                                           ys_a + 1.96*sems_a, color=C_ENERGY, alpha=0.08)
        if energy_ylim: ax_energy.set_ylim(energy_ylim)
        else:
            all_en = [d['Mean_Energy_Succ'] for d in data_en if d['Method'] in methods_en and d['Mean_Energy_Succ'] > 0]
            if all_en: ax_energy.set_ylim(min(all_en)*0.8, max(all_en)*1.25)
        if non_zero: ax_energy.axvline(x=pseudo_zero*1.5, color='gray', ls=':', alpha=0.4, lw=0.6)
        ax_energy.set_ylabel("Mean control energy", fontsize=label_fontsize, color=C_ENERGY)
        ax_energy.tick_params(axis='y', colors=C_ENERGY)
        ax_energy.legend(loc='upper left', fontsize=legend_fontsize)
 
    ax_energy.set_xscale('log')
    if noises:
        ax_energy.set_xticks([map_x(n) for n in noises])
        ax_energy.set_xticklabels(['0' if n==0 else f"{n:.3g}" for n in noises], rotation=45, ha='right')
    ax_energy.set_xlabel("Noise strength ($D$)", fontsize=label_fontsize)
    fig.align_ylabels([ax_scale, ax_energy])
    _save_fig(fig, filepath)
