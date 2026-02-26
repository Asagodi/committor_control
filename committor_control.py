"""
committor_control.py
====================
Data-driven committor-based optimal control for stochastic saddle-point systems.

Implements Algorithm 1 from the paper:
  "The neuron as a feedback controller of stochastic dynamics"
  Ságodi, Pughe-Sanford, Ding, Chklovskii

Generalised to work for any system (double-well, pendulum, …) by using the
*left* unstable eigenvector w_u to define the committor boundaries and the
control projection.  The "rotation" of the boundaries reduces to projecting
the state onto w_u before evaluating the 1-D committor.

Structure
---------
  committor_*       – closed-form committor math (erf-based)
  erf_feedback      – optimal committor control in state space
  relu_feedback     – deterministic-limit (ReLU) controller
  NbhdVARTracker    – neighbourhood-based system-ID (Algorithm 1)
  make_*            – system factory functions
  simulate_*        – trajectory utilities
  run_grid_eval     – evaluate controllers on a grid of ICs
  learn_and_eval    – full pipeline: ID → control → evaluation
  plot_*            – visualisation helpers
"""

from __future__ import annotations

import itertools
import os
from functools import partial
from typing import Callable, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy.special import erf

# ──────────────────────────────────────────────────────────────────────────────
# 1.  Utility helpers
# ──────────────────────────────────────────────────────────────────────────────

def noise_strength_to_D(noise_strength, dt: float, dim: int = 2,
                        convention: str = "sqrt2D") -> np.ndarray:
    """
    Convert mean-squared residual to diffusion coefficient D.

    For  dx = f(x)dt + sqrt(2D) dW_t  the per-step residual covariance is
    2D·dt·I, so  E[||ε||²] = 2D·dt·dim.

    Parameters
    ----------
    noise_strength : float | ndarray  – mean squared residual norm
    dt             : float            – time step
    dim            : int              – state dimension
    convention     : str              – "sqrt2D" (standard)
    """
    if convention != "sqrt2D":
        raise ValueError(f"Unknown SDE convention: {convention}")
    return np.asarray(noise_strength) / (2.0 * dt * dim)


def _sign_align(v: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Flip v so that it points in the same half-space as reference."""
    return v if np.dot(v, reference) >= 0 else -v


# ──────────────────────────────────────────────────────────────────────────────
# 2.  Committor mathematics (1-D, closed form via error function)
# ──────────────────────────────────────────────────────────────────────────────

def committor_1d(x_u: np.ndarray, a: float, r: float) -> np.ndarray:
    """
    Forward committor  q(x_u) = P(hit A={x_u ≤ -a} before B={x_u ≥ a}).

    Closed form for the 1-D OU boundary-value problem (Eq. 9 in paper):

        q(x_u) = ∫_{x_u}^{a} exp(-r y²) dy  /  ∫_{-a}^{a} exp(-r y²) dy

    Evaluated exactly using  erf(√r · x):

        = [erf(√r · a) - erf(√r · x_u)] / [2 erf(√r · a)]   (r > 0)

    Parameters
    ----------
    x_u : float | ndarray  – unstable coordinate value(s) in (-a, a)
    a   : float            – half-width of the control region
    r   : float            – r = λ_u / (2D)  (sharpness parameter)
    """
    x_u = np.asarray(x_u, dtype=float)
    if r > 1e-10:
        sr = np.sqrt(r)
        num = erf(sr * a) - erf(sr * x_u)
        den = 2.0 * erf(sr * a)
        return np.clip(num / (den + 1e-300), 0.0, 1.0)
    else:
        # flat committor (r → 0)
        return np.clip((a - x_u) / (2.0 * a), 0.0, 1.0)


def committor_score_1d(x_u: np.ndarray, a: float, r: float) -> np.ndarray:
    """
    d/dx_u  log q(x_u)  =  -exp(-r x_u²) / ∫_{x_u}^{a} exp(-r y²) dy

    This is the 1-D factor of the optimal control (Eq. 10, 11).
    Using the erf identity the denominator becomes
        √(π/(4r)) · [erf(√r · a) - erf(√r · x_u)]

    For large r (poorly estimated λ_u/D), both erf arguments saturate to 1
    and the ratio becomes 0/0.  In that regime integration by parts gives the
    asymptotic score  →  -2r·x_u,  which is used as a fallback.
    ``r`` is also clipped to ``_MAX_R`` to prevent exp overflow.
    """
    _MAX_R = 1e4
    x_u = np.asarray(x_u, dtype=float)
    r   = float(np.clip(r, 0.0, _MAX_R))
    if r > 1e-10:
        sr  = np.sqrt(r)
        erf_a  = erf(sr * a)
        erf_xu = erf(sr * x_u)
        erf_diff = erf_a - erf_xu          # ≥ 0 for x_u ≤ a

        # Asymptotic fallback when erf has saturated (diff < numerical noise)
        # limit: score → -2r·x_u
        with np.errstate(invalid='ignore', divide='ignore'):
            numerator = -np.exp(-r * x_u ** 2)
            denom     = np.sqrt(np.pi / (4.0 * r)) * erf_diff
            denom_safe = np.where(np.abs(denom) < 1e-300,
                                  1e-300 * np.sign(denom + 1e-300), denom)
            score_erf  = numerator / denom_safe
            score_asym = -2.0 * r * x_u         # asymptotic limit for large r·x_u²

        # Use asymptotic whenever erf_diff ≈ 0 (saturated)
        saturated = np.abs(erf_diff) < 1e-12
        return np.where(saturated, score_asym, score_erf)
    else:
        return -1.0 / (2.0 * a) * np.ones_like(x_u)


def optimal_control_scalar(x_u: np.ndarray, a: float,
                            lambda_u: float, D: float) -> np.ndarray:
    """
    Optimal control magnitude along the unstable direction (Eq. 11):

        u*(x_u) = 2D · d/dx_u log q(x_u)

    Returns a scalar (or array) value; the full 2-D control vector is
    obtained by multiplying by  w_u  (left unstable eigenvector).
    """
    r = lambda_u / (2.0 * D) if D > 0 else 1e12
    return 2.0 * D * committor_score_1d(x_u, a, r)


# ──────────────────────────────────────────────────────────────────────────────
# 3.  State-space feedback controllers
# ──────────────────────────────────────────────────────────────────────────────

def _unstable_proj(x: np.ndarray, w_u: np.ndarray, x_fp: np.ndarray) -> float:
    """Projection of (x - x*) onto the left unstable eigenvector w_u."""
    return float(np.dot(w_u, x - x_fp))


def _in_neighborhood(x: np.ndarray, x_fp: np.ndarray,
                     r_neigh: float, shape: str = "circular") -> bool:
    """Return True if x lies within the control neighbourhood of x_fp."""
    d = x - x_fp
    if shape == "circular":
        return float(np.dot(d, d)) <= r_neigh ** 2
    elif shape == "rectangular":
        return bool(np.all(np.abs(d) <= r_neigh))
    else:          # "none" – always active
        return True


def erf_feedback(x: np.ndarray,
                 w_u: np.ndarray,
                 lambda_u: float,
                 D: float,
                 x_fp: np.ndarray,
                 a: float,
                 r_neigh: float = 0.5,
                 nbhd_shape: str = "circular") -> np.ndarray:
    """
    Committor-based (erf / score) optimal control.

    Control direction  w_u  is the normalised left unstable eigenvector.
    The scalar control magnitude is u*(x_u) = 2D ∇_{x_u} log q(x_u).

    Parameters
    ----------
    x          : current state
    w_u        : left unstable eigenvector  (||w_u|| = 1)
    lambda_u   : continuous-time unstable eigenvalue
    D          : diffusion coefficient
    x_fp       : saddle fixed point
    a          : committor boundary half-width
    r_neigh    : neighbourhood radius for control activation
    nbhd_shape : "circular" | "rectangular" | "none"
    """
    u = np.zeros_like(x, dtype=float)
    if not _in_neighborhood(x, x_fp, r_neigh, nbhd_shape):
        return u
    x_u = _unstable_proj(x, w_u, x_fp)
    if abs(x_u) >= a:            # outside committor region – no control
        return u
    u_scalar = optimal_control_scalar(x_u, a, lambda_u, D)
    return float(u_scalar) * w_u


# Keep alias for backwards compatibility
high_precision_feedback = erf_feedback



def cca_feedback(x: np.ndarray,
                 v0: np.ndarray,
                 lambda0: float,
                 Dv: float,
                 x_star: np.ndarray,
                 a: float,
                 r_neigh: float = 0.5,
                 nbhd_shape: str = "circular") -> np.ndarray:
    """
    Algorithm 2 control law (paper step 10).

    Uses the CCA outputs directly — no continuous-time conversion:

        u(x) = 2·Dv · exp(−λ_0 q²/(4·Dv)) / ∫_a^{q} exp(−λ_0 y²/(4·Dv)) dy · v_0

    where q(x) = v_0^T x − Δ  (Δ baked into x_star: q = v_0^T(x − x̂*) · sign...).

    Parameters
    ----------
    x        : current state
    v0       : C_pp-normalised canonical direction (from CCATracker)
    lambda0  : λ_0 = ρ_0 v_0^T C_ff u_0
    Dv       : v_0^T D v_0  (projected diffusion, from CCATracker)
    x_star   : estimated fixed point (q(x̂*) = 0)
    a        : committor half-width
    r_neigh  : neighbourhood radius
    nbhd_shape : "circular" | "rectangular" | "none"
    """
    u = np.zeros_like(x, dtype=float)
    if not _in_neighborhood(x, x_star, r_neigh, nbhd_shape):
        return u
    q = float(np.dot(v0, x - x_star))
    if abs(q) >= a:
        return u

    # r = λ_0 / (4 Dv)  — paper's exponent coefficient
    r = lambda0 / max(4.0 * Dv, 1e-15)

    # reuse committor_score_1d: score = -exp(-r q²) / ∫_q^a exp(-r y²) dy
    # paper integrates from a to q (flipped limits → sign flip already in score)
    score = committor_score_1d(np.array([q]), a=a, r=r)[0]
    # paper: u = 2Dv · score · v0  (2Dv = 2 v_0^T D v_0)
    u_scalar = 2.0 * Dv * score
    return float(u_scalar) * v0

def relu_feedback(x: np.ndarray,
                  w_u: np.ndarray,
                  lambda_u: float,
                  D: float,
                  x_fp: np.ndarray,
                  a: float = None,
                  r_neigh: float = 0.5,
                  nbhd_shape: str = "circular") -> np.ndarray:
    """
    ReLU control (deterministic limit D → 0, Eq. 14):

        u*(x_u) = 0                if x_u < 0
                  -2 λ_u · x_u    if x_u ≥ 0
    """
    u = np.zeros_like(x, dtype=float)
    if not _in_neighborhood(x, x_fp, r_neigh, nbhd_shape):
        return u
    x_u = _unstable_proj(x, w_u, x_fp)
    if x_u >= 0.0:
        u = -2.0 * lambda_u * x_u * w_u
    return u


# ──────────────────────────────────────────────────────────────────────────────
# 4.  System factories
# ──────────────────────────────────────────────────────────────────────────────

class SystemSpec:
    """
    Holds everything needed to run the control pipeline for one system.

    Attributes
    ----------
    name         : str
    drift        : callable(t, x) → ndarray
    x_star       : ndarray           – saddle fixed point
    J            : ndarray (2,2)     – Jacobian at saddle
    w_u_true     : ndarray (2,)      – true left unstable eigenvector
    lambda_u_true: float             – true continuous-time unstable eigenvalue
    lambda_s_true: float             – true stable eigenvalue
    potential    : callable | None   – V(X, Y) for plotting
    """
    def __init__(self, name, drift, x_star, J, w_u_true,
                 lambda_u_true, lambda_s_true, potential=None):
        self.name = name
        self.drift = drift
        self.x_star = np.asarray(x_star, dtype=float)
        self.J = np.asarray(J, dtype=float)
        self.w_u_true = np.asarray(w_u_true, dtype=float)
        self.w_u_true /= np.linalg.norm(self.w_u_true)
        self.lambda_u_true = float(lambda_u_true)
        self.lambda_s_true = float(lambda_s_true)
        self.potential = potential


def make_double_well_curved(lambda_u: float = 2.0,
                            lambda_s: float = 10.0,
                            c: float = 0.5,
                            x_shift: float = 0.0,
                            y_shift: float = 0.0) -> SystemSpec:
    """
    Curved double-well potential (Fig. 3 in paper).

        V = (λ_u/4)(X_s² - 1)² + (λ_s/2)(Y_s - c X_s²)²

    Saddle at (x_shift, y_shift); left unstable eigenvector = [1, 0].
    """
    def potential(X, Y):
        Xs = X - x_shift;  Ys = Y - y_shift
        return lambda_u / 4 * (Xs ** 2 - 1) ** 2 + lambda_s / 2 * (Ys - c * Xs ** 2) ** 2

    def drift(t, x):
        Xs = x[0] - x_shift;  Ys = x[1] - y_shift
        dx = -lambda_u * Xs * (Xs ** 2 - 1) + 2 * c * lambda_s * Xs * (Ys - c * Xs ** 2)
        dy = -lambda_s * (Ys - c * Xs ** 2)
        return np.array([dx, dy])

    # Jacobian at saddle (Xs=0, Ys=0):
    #   ∂f_x/∂X_s = -λ_u(3·0-1) = λ_u,   ∂f_x/∂Y_s = 2c·λ_s·0 = 0
    #   ∂f_y/∂X_s = 2c·λ_s·0 = 0,          ∂f_y/∂Y_s = -λ_s
    J = np.array([[lambda_u, 0.0], [0.0, -lambda_s]])
    x_star = np.array([x_shift, y_shift])
    w_u = np.array([1.0, 0.0])   # left eigenvector of J for λ_u

    return SystemSpec("curved_double_well", drift, x_star, J, w_u,
                      lambda_u, -lambda_s, potential)


def make_double_well_linear(lambda_u: float = 2.0,
                            lambda_s: float = 10.0,
                            x_shift: float = 0.0,
                            y_shift: float = 0.0) -> SystemSpec:
    """
    Axis-aligned double-well (linear stable direction).
    """
    def potential(X, Y):
        Xs = X - x_shift;  Ys = Y - y_shift
        return lambda_u * (Xs ** 2 - 1) ** 2 + 2 * Ys ** 2

    def drift(t, x):
        Xs = x[0] - x_shift;  Ys = x[1] - y_shift
        return np.array([-lambda_u * Xs * (Xs ** 2 - 1),
                         -lambda_s * Ys])

    J = np.array([[lambda_u, 0.0], [0.0, -lambda_s]])
    x_star = np.array([x_shift, y_shift])
    return SystemSpec("linear_double_well", drift, x_star, J,
                      np.array([1.0, 0.0]), lambda_u, -lambda_s, potential)


def make_pendulum(g: float = 9.81, length: float = 1.0,
                  damping: float = 1.5) -> SystemSpec:
    """
    Damped pendulum – prevent over-the-top swings.

    State  x = [θ, ω]  (angle, angular velocity).
    Saddle at (π, 0)  (inverted equilibrium).

    The Jacobian at the saddle is *non-diagonal*, so the left unstable
    eigenvector  w_u  is NOT axis-aligned.  This is the key difference
    from the double-well: the committor boundaries are *rotated* hyperplanes
    in (θ, ω) space, automatically handled by using  x_u = w_u^T(x - x*).
    """
    def drift(t, x):
        theta, omega = x[0], x[1]
        dtheta = omega
        domega = -(g / length) * np.sin(theta) - damping * omega
        return np.array([dtheta, domega])

    x_star = np.array([np.pi, 0.0])

    # Jacobian at (π, 0):  cos(π) = -1
    J = np.array([[0.0,            1.0],
                  [g / length,  -damping]])

    # Eigendecomposition of J
    eigvals, V_right = np.linalg.eig(J)
    # Sort: unstable (largest real part) first
    order = np.argsort(eigvals.real)[::-1]
    eigvals = eigvals[order].real
    V_right = V_right[:, order].real

    lambda_u_true = float(eigvals[0])
    lambda_s_true = float(eigvals[1])

    # Left eigenvectors: J^T w = λ w
    eigvals_l, V_left = np.linalg.eig(J.T)
    order_l = np.argsort(eigvals_l.real)[::-1]
    eigvals_l = eigvals_l[order_l].real
    V_left = V_left[:, order_l].real

    w_u = V_left[:, 0].real
    w_u /= np.linalg.norm(w_u)

    # Sign convention: w_u^T v_u > 0
    v_u = V_right[:, 0]
    if np.dot(w_u, v_u) < 0:
        w_u = -w_u

    return SystemSpec("pendulum", drift, x_star, J, w_u,
                      lambda_u_true, lambda_s_true)


# ──────────────────────────────────────────────────────────────────────────────
# 5.  Neighbourhood VAR tracker  (Algorithm 1)
# ──────────────────────────────────────────────────────────────────────────────

class NbhdVARTracker:
    """
    Neighbourhood-based VAR(1) system-identification + parameter tracker.

    Open-loop Phase (Algorithm 1, steps 1–3)
    -----------------------------------------
    1. Simulate n_trajs open-loop trajectories of T steps near x*.
    2. Collect all (x_t, x_{t+1}) pairs.
    3. Iterate n_iterations times:
       a. Select pairs with  ||x_t - x̂*|| ≤ radius.
       b. Fit local VAR(1):  x_{t+1} = A x_t + b + ε   (least-squares).
       c. Update x̂* ← (I - A)^{-1}(mean_f - A·mean_p)      (Eq. 17).
    4. Extract  ŵ_u  by power-iteration on A^T.
    5. Estimate  λ̂_u  from Rayleigh quotient.
    6. Estimate  D̂  from residual covariance.

    The generalisaton to non-axis-aligned systems (pendulum) is *automatic*:
    step 4 recovers the correct  ŵ_u  regardless of alignment.
    """

    def __init__(self, dim: int = 2,
                 radius: Optional[float] = None,
                 x_star_init: Optional[np.ndarray] = None,
                 x_star_init_range: float = 0.3,
                 rng_seed: Optional[int] = None,
                 store_history: bool = True):
        self.dim = int(dim)
        self.store_history = bool(store_history)
        self.radius = None if radius is None else float(radius)
        self.rng = np.random.default_rng(rng_seed)

        # Initial fixed-point guess
        if x_star_init is not None:
            self.x_star = np.asarray(x_star_init, dtype=float).ravel()
        else:
            self.x_star = self.rng.uniform(-x_star_init_range,
                                           x_star_init_range, size=self.dim)

        # Model state
        self.A = np.eye(self.dim, dtype=float)
        self.residual_cov = np.eye(self.dim, dtype=float)
        self.noise_strength = 1e-5

        # Eigenvector state
        v0 = self.rng.standard_normal(self.dim)
        v0 /= np.linalg.norm(v0) + 1e-15
        self.w_u = v0.copy()          # left unstable eigenvector estimate
        self.lambda_disc = 1.0        # discrete-time eigenvalue magnitude
        self.D_est = 0.0

        # Pair storage (filled by train())
        self.pairs: List[Tuple[np.ndarray, np.ndarray]] = []

        # History (initial values as first entry)
        if self.store_history:
            self.A_history          = [self.A.copy()]
            self.residual_cov_history = [self.residual_cov.copy()]
            self.noise_history      = [self.noise_strength]
            self.wu_history         = [self.w_u.copy()]
            self.lambda_disc_history = [self.lambda_disc]
            self.xstar_history      = [self.x_star.copy()]
            self.nlocal_history     = [0]

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _pairs_from_trajectories(self, all_X: List[np.ndarray]) -> None:
        self.pairs = []
        for X in all_X:
            X = np.asarray(X)
            for t in range(X.shape[0] - 1):
                self.pairs.append((X[t].copy(), X[t + 1].copy()))

    def _fit_local_var(self, local_pairs, x_star=None):
        """
        VAR(1) fit matching Algorithm 1 exactly.

        Uses z_i = x_i - x̂* as the centered coordinate (step 4-5 of the
        algorithm), NOT the sample mean.  This is the correct centering when
        x̂* is the known/estimated fixed point.

        D̂ is returned as a full matrix (step 9: D̂ = E·E^T / (2t·dt));
        the scalar used for control is ŵ_u^T D̂ ŵ_u (variance of residuals
        projected onto the unstable direction).
        """
        Xp = np.vstack([p[0] for p in local_pairs])
        Xf = np.vstack([p[1] for p in local_pairs])
        n  = Xp.shape[0]

        # center by x̂* (Algorithm 1, step 4)
        # fall back to sample mean if x_star not provided
        c  = x_star if x_star is not None else Xp.mean(0)
        Zp = Xp - c
        Zf = Xf - c

        # covariances (step 6) — no further mean subtraction
        Cpp = Zp.T @ Zp / n
        Cfp = Zf.T @ Zp / n

        # transition matrix (step 7)
        A_hat = Cfp @ np.linalg.pinv(Cpp)

        # residuals (step 8) and D̂ as matrix (step 9)
        resid  = Zf - Zp @ A_hat.T          # E = Z̃_f - Â Z̃_p
        D_mat  = resid.T @ resid / n        # E·E^T / t  (dt factor applied later)
        mse    = float(np.mean(np.sum(resid ** 2, axis=1)))

        # means needed for fixed-point update (z-space means, step 12)
        mp = Zp.mean(0)
        mf = Zf.mean(0)

        return A_hat, D_mat, mse, mp, mf

    def _update_fixed_point(self, A_hat, mp, mf):
        """Update x* ← (I - A)^{-1}(mean_f - A mean_p)  (Eq. 17)."""
        rhs = mf - A_hat @ mp
        IminA = np.eye(self.dim) - A_hat
        try:
            return np.linalg.solve(IminA, rhs)
        except np.linalg.LinAlgError:
            return np.linalg.pinv(IminA) @ rhs

    def _power_iterate(self, A: np.ndarray,
                       n_iter: int = 40, tol: float = 1e-12) -> None:
        """Power iteration matching Algorithm 1 steps 10-11.

        Step 10: ŵ_u ← ŵ_u A / ||ŵ_u A||   (left eigenvector iteration)
        Step 11: λ̂_u = (1/dt) log ||ŵ_u^(new) A||
                 i.e. the growth rate of the *updated* direction under A.
        """
        v = self.w_u.copy()
        for _ in range(n_iter):
            # treat w_u as row vector: w A  (= A^T w as column)
            v_new = A.T @ v
            nrm = np.linalg.norm(v_new)
            if nrm < tol:
                break
            v_new /= nrm
            if np.linalg.norm(v_new - v) < tol:
                v = v_new
                break
            v = v_new
        self.w_u = v.copy()
        # step 11: lambda_disc = ||ŵ_u^(k) A|| = ||A^T ŵ_u^(k)||
        Av = A.T @ v
        self.lambda_disc = float(np.linalg.norm(Av))

        if self.store_history:
            self.wu_history.append(self.w_u.copy())
            self.lambda_disc_history.append(self.lambda_disc)

    def _record(self, A_hat, D_mat, mse, n_local=None) -> None:
        if not self.store_history:
            return
        self.A_history.append(A_hat.copy())
        self.residual_cov_history.append(D_mat.copy())
        self.noise_history.append(mse)
        self.xstar_history.append(self.x_star.copy())
        if n_local is not None:
            self.nlocal_history.append(n_local)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def train(self, drift_func: Callable,
              n_trajs: int = 50, T: int = 25,
              dt: float = 0.01, D: float = 0.01,
              x_shift: float = 0.0, y_shift: float = 0.0,
              seed: int = 0, init_bounds: float = 0.1,
              n_iterations: int = 15, min_pairs: int = 10,
              return_trajectories: bool = False):
        """
        Simulate open-loop trajectories and identify local VAR parameters.

        Parameters
        ----------
        drift_func       : callable(t, x) → ndarray
        n_trajs          : number of trajectories
        T                : steps per trajectory
        dt               : time step
        D                : diffusion coefficient
        x_shift, y_shift : centre of initial-condition perturbations
        seed             : RNG seed for reproducibility
        init_bounds      : half-width of uniform IC perturbation
        n_iterations     : refinement iterations for x* / local VAR
        min_pairs        : skip fitting if fewer local pairs
        return_trajectories : return list of trajectory arrays

        Returns
        -------
        all_X : list of (T, dim) arrays  [only if return_trajectories=True]
        """
        noise_std = np.sqrt(2.0 * D)
        rng = np.random.default_rng(seed)

        # ---- simulate open-loop trajectories ----
        all_X = []
        center = np.zeros(self.dim)
        center[0] = x_shift
        if self.dim > 1:
            center[1] = y_shift

        for _ in range(int(n_trajs)):
            X = np.zeros((int(T), self.dim), dtype=float)
            X[0] = center + rng.uniform(-init_bounds, init_bounds, size=self.dim)
            for t in range(1, int(T)):
                dx = drift_func(t * dt, X[t - 1])
                X[t] = (X[t - 1]
                        + dx * dt
                        + noise_std * rng.standard_normal(self.dim) * np.sqrt(dt))
            all_X.append(X)

        self._pairs_from_trajectories(all_X)

        # ---- iterative local fitting ----
        A_hat = self.A.copy()
        b_hat = np.zeros(self.dim)
        rescov = self.residual_cov.copy()
        mse = self.noise_strength

        D_mat = np.eye(self.dim)   # running D̂ matrix

        if self.radius is None:
            # Global fit (no neighbourhood restriction)
            A_hat, D_mat, mse, mp, mf = self._fit_local_var(self.pairs, self.x_star)
            self.noise_strength = mse
            self._power_iterate(A_hat)
            self.D_mat = D_mat
            self.D_est = float(self.w_u @ D_mat @ self.w_u) / (2.0 * dt)
            self._record(A_hat, D_mat, mse, len(self.pairs))
        else:
            for _ in range(int(n_iterations)):
                local = [(xp, xf) for xp, xf in self.pairs
                         if np.linalg.norm(xp - self.x_star) <= self.radius]
                n_loc = len(local)

                if self.store_history:
                    self.nlocal_history.append(n_loc)

                if n_loc < min_pairs:
                    self._power_iterate(A_hat)
                    self._record(A_hat, D_mat, mse)
                    continue

                A_hat, D_mat, mse, mp, mf = self._fit_local_var(local, self.x_star)
                self.x_star = self._update_fixed_point(A_hat, mp, mf)
                self.noise_strength = mse
                self._power_iterate(A_hat)
                self.D_mat = D_mat
                self.D_est = float(self.w_u @ D_mat @ self.w_u) / (2.0 * dt)
                self._record(A_hat, D_mat, mse)

        self.A = A_hat

        return all_X if return_trajectories else None

    def get_params(self, dt: float, sign_ref: Optional[np.ndarray] = None) -> dict:
        """
        Return estimated control parameters.

        Parameters
        ----------
        dt       : time step (to convert discrete → continuous eigenvalue)
        sign_ref : optional reference vector to fix sign of  ŵ_u

        Returns
        -------
        dict with keys: w_u, lambda_u, D, x_star
        """
        w_u = self.w_u.copy()
        if sign_ref is not None:
            w_u = _sign_align(w_u, sign_ref)

        lam_disc = abs(self.lambda_disc)
        lambda_u_ct = float(np.log(lam_disc) / dt) if lam_disc > 1e-12 else 0.0
        D_mat = getattr(self, 'D_mat', np.eye(self.dim))
        D_est = float(w_u @ D_mat @ w_u) / (2.0 * dt)

        return dict(w_u=w_u, lambda_u=lambda_u_ct,
                    D=D_est, x_star=self.x_star.copy())


# ──────────────────────────────────────────────────────────────────────────────
# 6.  Trajectory simulation
# ──────────────────────────────────────────────────────────────────────────────

def simulate_trajectory(x0: np.ndarray, T: int, dt: float,
                        drift_func: Callable, D: float,
                        control_func: Optional[Callable] = None,
                        x_fp: Optional[np.ndarray] = None,
                        w_u: Optional[np.ndarray] = None,
                        a: Optional[float] = None,
                        rng: Optional[np.random.Generator] = None
                        ) -> Tuple[np.ndarray, str]:
    """
    Simulate one trajectory with optional control and committor-boundary stopping.

    Stopping condition (if a, w_u, x_fp are given):
        hit  B = {w_u^T (x - x*) ≥  a}  →  returns "positive"
        hit  A = {w_u^T (x - x*) ≤ -a}  →  returns "negative"

    Returns
    -------
    X        : (steps_taken, dim) trajectory array
    hit_side : "positive" | "negative" | "none"
    """
    if rng is None:
        rng = np.random.default_rng()

    dim = len(x0)
    noise_std = np.sqrt(2.0 * D)
    X = np.zeros((T, dim), dtype=float)
    x = x0.copy()
    hit_side = "none"

    for t in range(T):
        X[t] = x

        # ---- check absorbing boundaries ----
        if a is not None and w_u is not None and x_fp is not None:
            x_u = float(np.dot(w_u, x - x_fp))
            if x_u >= a:
                return X[:t + 1], "positive"
            if x_u <= -a:
                return X[:t + 1], "negative"

        # ---- step ----
        f = drift_func(t * dt, x)
        u = control_func(x) if control_func is not None else np.zeros(dim)
        dW = noise_std * rng.standard_normal(dim) * np.sqrt(dt)
        x = x + (f + u) * dt + dW

    return X, hit_side


# ──────────────────────────────────────────────────────────────────────────────
# 7.  Grid evaluation
# ──────────────────────────────────────────────────────────────────────────────

def _make_control_fn(ctrl_type, label,
                     w_u_true, lambda_u_true, D_true, x_fp_true,
                     w_u_est, lambda_u_est, D_est, x_fp_est,
                     a, r_neigh, nbhd_shape):
    """Build a bound control callable based on label / ctrl_type."""
    if ctrl_type is None:
        return None

    if "Optimal" in label or "model" in label.lower():
        return partial(ctrl_type,
                       w_u=w_u_true, lambda_u=lambda_u_true, D=D_true,
                       x_fp=x_fp_true, a=a,
                       r_neigh=r_neigh, nbhd_shape=nbhd_shape)
    else:
        return partial(ctrl_type,
                       w_u=w_u_est, lambda_u=lambda_u_est, D=D_est,
                       x_fp=x_fp_est, a=a,
                       r_neigh=r_neigh, nbhd_shape=nbhd_shape)


def run_grid_eval(T: int, dt: float, D_true: float,
                  drift_func: Callable,
                  x_fp_true: np.ndarray,
                  w_u_true: np.ndarray,
                  lambda_u_true: float,
                  w_u_est: np.ndarray,
                  lambda_u_est: float,
                  D_est: float,
                  x_fp_est: np.ndarray,
                  controllers: Dict[str, Optional[Callable]],
                  a: float,
                  n_grid: int = 5,
                  x_max: float = 0.5,
                  grid_side: str = "positive",
                  grid_rotation_deg: float = 0.0,
                  r_neigh: float = 0.5,
                  nbhd_shape: str = "circular",
                  seed: int = 0) -> Dict:
    """
    Evaluate controllers on a 2-D grid of initial conditions.

    The grid is defined in a rotated frame aligned with  w_u  (so "positive"
    means initial conditions where  x_u > 0,  i.e. on the undesired side).
    This generalises the axis-aligned grids used for the double-well to
    arbitrarily oriented saddles (e.g. pendulum).

    Parameters
    ----------
    controllers : dict  {label: callable | None}
        callable signature: (x, w_u, lambda_u, D, x_fp, a, r_neigh, nbhd_shape)
    grid_rotation_deg : extra rotation (degrees) applied on top of w_u alignment

    Returns
    -------
    dict keyed by label, each with:
        trajectories, hit_proportion, control_costs, mean_cost
    """
    dim = len(x_fp_true)
    rng_master = np.random.default_rng(seed)

    # ---- build grid in the (w_u, w_s) rotated frame ----
    if grid_side == "positive":
        first = np.linspace(0.0, x_max, n_grid, endpoint=False)
    elif grid_side == "negative":
        first = np.linspace(-x_max, 0.0, n_grid + 1, endpoint=True)[1:]
    else:
        first = np.linspace(-x_max, x_max, n_grid + 2)[1:-1]

    other = np.linspace(-x_max, x_max, n_grid + 2)[1:-1]
    raw_grid = np.array(list(itertools.product(first, other)))   # (N, 2)

    # Rotation matrix aligning  first-axis  →  w_u direction
    angle_rad = np.arctan2(w_u_true[1], w_u_true[0]) + np.radians(grid_rotation_deg)
    R = np.array([[np.cos(angle_rad), -np.sin(angle_rad)],
                  [np.sin(angle_rad),  np.cos(angle_rad)]])
    grid_pts = raw_grid @ R.T + x_fp_true   # (N, 2)

    n_traj = len(grid_pts)
    rng_seeds = rng_master.integers(0, 2**32, size=n_traj)

    results = {}
    for label, ctrl_type in controllers.items():
        ctrl_fn = _make_control_fn(
            ctrl_type, label,
            w_u_true, lambda_u_true, D_true, x_fp_true,
            w_u_est, lambda_u_est, D_est, x_fp_est,
            a, r_neigh, nbhd_shape)

        trajs, hit_sides, costs = [], [], []
        for i, x0 in enumerate(grid_pts):
            rng_i = np.random.default_rng(int(rng_seeds[i]))
            X, hs = simulate_trajectory(
                x0, T, dt, drift_func, D_true,
                control_func=ctrl_fn,
                x_fp=x_fp_true, w_u=w_u_true, a=a,
                rng=rng_i)
            trajs.append(X)
            hit_sides.append(hs)
            # control cost = (1/2) ∫ ||u||² dt  (discretised)
            if ctrl_fn is not None:
                u_vals = np.vstack([ctrl_fn(x) for x in X])
                costs.append(0.5 * float(np.sum(np.sum(u_vals ** 2, 1))) * dt)
            else:
                costs.append(0.0)

        hs = np.array(hit_sides)
        results[label] = dict(
            trajectories=trajs,
            hit_proportion=dict(
                positive=float(np.mean(hs == "positive")),
                negative=float(np.mean(hs == "negative")),
                none=float(np.mean(hs == "none"))),
            control_costs=costs,
            mean_cost=float(np.mean(costs)))

    return results


# ──────────────────────────────────────────────────────────────────────────────
# 8.  Estimation-quality evaluation over training iterations
# ──────────────────────────────────────────────────────────────────────────────

def evaluate_over_iterations(
        tracker: NbhdVARTracker,
        sys: SystemSpec,
        dt: float,
        a: float = 0.5,
        T_eval: int = 500,
        n_grid: int = 5,
        x_max: float = 0.5,
        r_neigh: float = 0.5,
        nbhd_shape: str = "circular",
        grid_rotation_deg: float = 0.0,
        feedback_function: Callable = erf_feedback,
        eval_interval: int = 1,
        seed: int = 0) -> Tuple[List[int], Dict]:
    """
    Evaluate controller quality at each tracked iteration.

    For each stored iteration of the tracker, uses the estimated parameters
    to build the data-driven controller and evaluates it against the
    model-based optimal controller on a grid of initial conditions.

    Returns
    -------
    iter_points      : list of iteration indices evaluated
    results_by_iter  : {iter_idx: {label: {hit_proportion, mean_cost}}}
    """
    n_hist = len(tracker.wu_history)
    iter_points = list(range(0, n_hist, eval_interval))
    results_by_iter = {}

    for it in iter_points:
        w_u_est = _sign_align(tracker.wu_history[it], sys.w_u_true)
        lam_disc = abs(tracker.lambda_disc_history[it])
        lambda_u_est = float(np.log(lam_disc) / dt) if lam_disc > 1e-12 else 0.0
        D_est_i = noise_strength_to_D(tracker.noise_history[it], dt, sys.x_star.size)
        x_fp_est = tracker.xstar_history[it]

        r = run_grid_eval(
            T=T_eval, dt=dt,
            D_true=sys.x_star.size,   # placeholder – overridden below
            drift_func=sys.drift,
            x_fp_true=sys.x_star,
            w_u_true=sys.w_u_true,
            lambda_u_true=sys.lambda_u_true,
            w_u_est=w_u_est,
            lambda_u_est=lambda_u_est,
            D_est=float(D_est_i),
            x_fp_est=x_fp_est,
            controllers={
                "Optimal (model)": feedback_function,
                "Data-driven":     feedback_function},
            a=a, n_grid=n_grid, x_max=x_max,
            grid_side="positive",
            grid_rotation_deg=grid_rotation_deg,
            r_neigh=r_neigh, nbhd_shape=nbhd_shape,
            seed=seed)

        results_by_iter[it] = {
            lbl: dict(hit_proportion=v["hit_proportion"],
                      mean_cost=v["mean_cost"])
            for lbl, v in r.items()}

    return iter_points, results_by_iter


# ──────────────────────────────────────────────────────────────────────────────
# 9.  Full pipeline
# ──────────────────────────────────────────────────────────────────────────────

def learn_and_eval(
        sys: SystemSpec,
        D_true: float = 0.01,
        dt: float = 0.01,
        n_trajs: int = 50,
        T_train: int = 25,
        init_bounds: float = 0.1,
        radius: float = 0.3,
        x_star_init_range: float = 0.2,
        n_iterations: int = 15,
        a: float = 0.5,
        T_eval: int = 1000,
        n_grid: int = 5,
        x_max: float = 0.5,
        r_neigh: float = 0.5,
        nbhd_shape: str = "circular",
        grid_rotation_deg: float = 0.0,
        feedback_function: Callable = erf_feedback,
        seed: int = 42,
        verbose: bool = True) -> Tuple[NbhdVARTracker, List, Dict]:
    """
    End-to-end pipeline:
      1. Run NbhdVARTracker to identify saddle parameters.
      2. Evaluate uncontrolled / model-based / data-driven controllers.

    The  grid_rotation_deg  parameter rotates the IC grid so that it is
    expressed in coordinates aligned with the unstable direction of the
    *true* system.  For double-well (w_u = [1,0]) this is 0°; for the
    pendulum it should be set to the angle of  w_u  in state space.

    Returns
    -------
    tracker     : trained NbhdVARTracker
    all_X       : open-loop trajectories used for training
    eval_results: dict {label: {hit_proportion, control_costs, …}}
    """
    dim = sys.x_star.size

    # ---- system identification ----
    tracker = NbhdVARTracker(
        dim=dim,
        radius=radius,
        x_star_init_range=x_star_init_range,
        store_history=True,
        rng_seed=seed)

    all_X = tracker.train(
        drift_func=sys.drift,
        n_trajs=n_trajs, T=T_train, dt=dt, D=D_true,
        x_shift=float(sys.x_star[0]),
        y_shift=float(sys.x_star[1]) if dim > 1 else 0.0,
        seed=seed, init_bounds=init_bounds,
        n_iterations=n_iterations,
        return_trajectories=True)

    params = tracker.get_params(dt, sign_ref=sys.w_u_true)
    w_u_est    = params["w_u"]
    lambda_u_est = params["lambda_u"]
    D_est      = params["D"]
    x_fp_est   = params["x_star"]

    if verbose:
        print(f"\n{'─'*50}")
        print(f"System: {sys.name}")
        print(f"  True  x*    : {sys.x_star}")
        print(f"  Est.  x*    : {x_fp_est.round(4)}")
        print(f"  True  w_u   : {sys.w_u_true.round(4)}")
        print(f"  Est.  w_u   : {w_u_est.round(4)}")
        print(f"  True  λ_u   : {sys.lambda_u_true:.4f}")
        print(f"  Est.  λ_u   : {lambda_u_est:.4f}")
        print(f"  True  D     : {D_true:.4f}")
        print(f"  Est.  D     : {D_est:.4f}")

    # ---- evaluate controllers ----
    controllers = {
        "Uncontrolled":   None,
        "Optimal (model)": feedback_function,
        "Data-driven":     feedback_function}

    eval_results = run_grid_eval(
        T=T_eval, dt=dt, D_true=D_true,
        drift_func=sys.drift,
        x_fp_true=sys.x_star,
        w_u_true=sys.w_u_true,
        lambda_u_true=sys.lambda_u_true,
        w_u_est=w_u_est, lambda_u_est=lambda_u_est,
        D_est=D_est, x_fp_est=x_fp_est,
        controllers=controllers,
        a=a, n_grid=n_grid, x_max=x_max,
        grid_side="positive",
        grid_rotation_deg=grid_rotation_deg,
        r_neigh=r_neigh, nbhd_shape=nbhd_shape,
        seed=seed)

    if verbose:
        print("\n  Controller evaluation:")
        for lbl, res in eval_results.items():
            hp = res["hit_proportion"]
            print(f"    {lbl:22s}  P(A)={hp['negative']:.3f}  "
                  f"P(B)={hp['positive']:.3f}  cost={res['mean_cost']:.4f}")

    return tracker, all_X, eval_results


# ──────────────────────────────────────────────────────────────────────────────
# 10.  Multi-run evaluation (Fig. 3E in paper)
# ──────────────────────────────────────────────────────────────────────────────

def multi_run_eval(sys: SystemSpec, D_true: float = 0.01, dt: float = 0.01,
                   n_runs: int = 20, n_trajs: int = 50, T_train: int = 25,
                   init_bounds: float = 0.1, radius: float = 0.3,
                   x_star_init_range: float = 0.2, n_iterations: int = 15,
                   a: float = 0.5, T_eval: int = 500, n_grid: int = 5,
                   x_max: float = 0.5, r_neigh: float = 0.5,
                   nbhd_shape: str = "circular",
                   grid_rotation_deg: float = 0.0,
                   feedback_function: Callable = erf_feedback,
                   eval_interval: int = 1,
                   verbose: bool = True) -> Tuple[List[int], Dict]:
    """
    Run n_runs independent trackers and aggregate evaluation metrics.

    Returns
    -------
    iter_points      : list of iteration indices
    aggregated       : {label: {P_desired_mean, P_desired_std, cost_mean, cost_std}}
    """
    all_P = {}
    all_C = {}

    for run_idx in range(n_runs):
        seed = run_idx * 13 + 7
        tracker = NbhdVARTracker(dim=sys.x_star.size, radius=radius,
                                 x_star_init_range=x_star_init_range,
                                 store_history=True, rng_seed=seed)
        tracker.train(
            drift_func=sys.drift,
            n_trajs=n_trajs, T=T_train, dt=dt, D=D_true,
            x_shift=float(sys.x_star[0]),
            y_shift=float(sys.x_star[1]) if sys.x_star.size > 1 else 0.0,
            seed=seed, init_bounds=init_bounds,
            n_iterations=n_iterations)

        iter_pts, res_it = evaluate_over_iterations(
            tracker, sys, dt, a=a, T_eval=T_eval, n_grid=n_grid,
            x_max=x_max, r_neigh=r_neigh, nbhd_shape=nbhd_shape,
            grid_rotation_deg=grid_rotation_deg,
            feedback_function=feedback_function,
            eval_interval=eval_interval, seed=seed)

        for it in iter_pts:
            for lbl, vals in res_it[it].items():
                key = (it, lbl)
                p = vals["hit_proportion"]["negative"]  # P(desired)
                c = vals["mean_cost"]
                all_P.setdefault(key, []).append(p)
                all_C.setdefault(key, []).append(c)

        if verbose:
            print(f"  run {run_idx+1}/{n_runs} done")

    # Aggregate
    labels = list({lbl for (_, lbl) in all_P.keys()})
    aggregated = {lbl: dict(
        iter_points=iter_pts,
        P_desired_mean=np.array([np.mean(all_P[(it, lbl)]) for it in iter_pts]),
        P_desired_std =np.array([np.std( all_P[(it, lbl)]) for it in iter_pts]),
        cost_mean     =np.array([np.mean(all_C[(it, lbl)]) for it in iter_pts]),
        cost_std      =np.array([np.std( all_C[(it, lbl)]) for it in iter_pts]),
    ) for lbl in labels}

    return iter_pts, aggregated


# ──────────────────────────────────────────────────────────────────────────────
# 11.  Visualisation
# ──────────────────────────────────────────────────────────────────────────────

def plot_committor_and_control(a: float = 0.5, lambda_u: float = 2.0,
                               D_values=None, save_dir: str = "figures"):
    """Reproduce Fig. 2: committor and optimal control vs x_u."""
    if D_values is None:
        D_values = [0.5, 0.1, 0.01]
    os.makedirs(save_dir, exist_ok=True)

    xu = np.linspace(-a, a, 400)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    colors = plt.cm.viridis(np.linspace(0.1, 0.85, len(D_values)))

    for D, col in zip(D_values, colors):
        r = lambda_u / (2.0 * D)
        q_vals = committor_1d(xu, a, r)
        u_vals = optimal_control_scalar(xu, a, lambda_u, D)
        axes[0].plot(xu, q_vals, color=col, lw=2, label=f"D = {D}")
        axes[1].plot(xu, u_vals, color=col, lw=2, label=f"D = {D}")

    # ReLU limit
    xu_pos = xu[xu >= 0]
    axes[1].plot(xu[xu < 0], np.zeros(np.sum(xu < 0)),
                 "k--", lw=2.5, label="ReLU (D → 0)")
    axes[1].plot(xu_pos, -2 * lambda_u * xu_pos, "k--", lw=2.5)

    for ax, ylabel, title in zip(
            axes, ["q(x_u)", "u*(x_u)"],
            ["Forward Committor", "Optimal Control"]):
        ax.axvline(-a, color="green", ls="--", alpha=.7, label="∂A")
        ax.axvline( a, color="red",   ls="--", alpha=.7, label="∂B")
        ax.axvline( 0, color="gray",  ls=":",  alpha=.5)
        ax.set_xlabel("Unstable coordinate  $x_u$", fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(title, fontsize=13)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(save_dir, "committor_control.pdf")
    fig.savefig(path, bbox_inches="tight", dpi=150)
    print(f"  saved → {path}")
    plt.show()


def plot_estimation_errors(tracker: NbhdVARTracker, sys: SystemSpec,
                           dt: float, D_true: float,
                           save_dir: str = "figures"):
    """Plot relative parameter estimation errors over training iterations."""
    os.makedirs(save_dir, exist_ok=True)
    n = len(tracker.A_history)
    steps = np.arange(n)

    # λ_u error (relative)
    lam_cts = np.array([
        np.log(abs(l)) / dt if abs(l) > 1e-12 else 0.0
        for l in tracker.lambda_disc_history])
    lambda_err = np.abs(lam_cts - sys.lambda_u_true) / max(abs(sys.lambda_u_true), 1e-12)

    # ŵ_u angle error
    wu_err = np.array([
        min(np.linalg.norm(w - sys.w_u_true),
            np.linalg.norm(w + sys.w_u_true))
        for w in tracker.wu_history])

    # D̂ error (relative)
    D_hats = np.array([
        float(np.trace(R) / (2.0 * dt * sys.x_star.size))
        for R in tracker.residual_cov_history])
    D_err = np.abs(D_hats - D_true) / max(D_true, 1e-12)

    # x* error
    xs_err = np.array([np.linalg.norm(xs - sys.x_star)
                       for xs in tracker.xstar_history])

    plt.figure(figsize=(8, 4))
    plt.semilogy(steps, lambda_err, color="green",   lw=2,
                 label=r"$|\hat\lambda_u - \lambda_u|/|\lambda_u|$")
    plt.semilogy(steps, wu_err,     color="darkblue", lw=2,
                 label=r"$\|\hat{w}_u - w_u\|$")
    plt.semilogy(steps, D_err,      color="firebrick", lw=2,
                 label=r"$|\hat D - D|/D$")
    plt.semilogy(steps, xs_err,     color="#6B4226",  lw=2,
                 label=r"$\|\hat{x}^* - x^*\|$")
    plt.xlabel("Iteration", fontsize=13)
    plt.ylabel("Error", fontsize=13)
    plt.title(f"Parameter estimation: {sys.name}", fontsize=13)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(save_dir, f"estimation_{sys.name}.pdf")
    plt.savefig(path, bbox_inches="tight")
    print(f"  saved → {path}")
    plt.show()


def plot_trajectories(eval_results: Dict, sys: SystemSpec, a: float,
                      save_dir: str = "figures", n_show: int = 6):
    """Plot example closed-loop trajectories for each controller."""
    os.makedirs(save_dir, exist_ok=True)
    n_ctrl = len(eval_results)
    fig, axes = plt.subplots(1, n_ctrl, figsize=(5 * n_ctrl, 4.5),
                             sharex=True, sharey=True)
    if n_ctrl == 1:
        axes = [axes]

    color_map = {"Uncontrolled": "gray",
                 "Optimal (model)": "darkblue",
                 "Data-driven": "firebrick"}

    w_u = sys.w_u_true
    w_s = np.array([-w_u[1], w_u[0]])     # perpendicular (stable direction)
    x_max_for_plot = 0.8
    t_line = np.linspace(-x_max_for_plot, x_max_for_plot, 60)

    for ax, (label, res) in zip(axes, eval_results.items()):
        col = color_map.get(label, "steelblue")
        for X in res["trajectories"][:n_show]:
            ax.plot(X[:, 0], X[:, 1], alpha=0.6, color=col, lw=1.5)

        # Committor boundaries (hyperplanes ⊥ to w_u at ±a from x*)
        for sign, bc in [(+1, "red"), (-1, "green")]:
            centre = sys.x_star + sign * a * w_u
            line = centre[:, None] + np.outer(w_s, t_line)
            ax.plot(line[0], line[1], "--", color=bc, alpha=0.8, lw=1.5)

        ax.plot(*sys.x_star, "k+", ms=12, mew=2.5, label="saddle")
        hp = res["hit_proportion"]
        ax.set_title(f"{label}\nP(A)={hp['negative']:.2f}  "
                     f"P(B)={hp['positive']:.2f}", fontsize=11)
        ax.set_xlabel("$x_1$", fontsize=11)
        ax.set_ylabel("$x_2$", fontsize=11)
        ax.grid(True, alpha=0.3)

    plt.suptitle(f"Controlled trajectories – {sys.name}", fontsize=13)
    plt.tight_layout()
    path = os.path.join(save_dir, f"trajectories_{sys.name}.pdf")
    plt.savefig(path, bbox_inches="tight")
    print(f"  saved → {path}")
    plt.show()


def plot_multirun_results(iter_pts: List[int], aggregated: Dict,
                          sys: SystemSpec, save_dir: str = "figures"):
    """Plot P(desired set) and control cost over training iterations."""
    os.makedirs(save_dir, exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    colors = {"Optimal (model)": "darkblue", "Data-driven": "firebrick"}
    for lbl, agg in aggregated.items():
        its = np.array(agg["iter_points"])
        col = colors.get(lbl, "gray")
        pm = agg["P_desired_mean"]
        ps = agg["P_desired_std"]
        cm = agg["cost_mean"]
        cs = agg["cost_std"]

        ax1.plot(its, pm, color=col, lw=2, label=lbl)
        ax1.fill_between(its, pm - ps, pm + ps, color=col, alpha=0.25)
        ax2.semilogy(its, cm, color=col, lw=2, label=lbl)
        ax2.fill_between(its, np.maximum(cm - cs, 1e-10), cm + cs,
                         color=col, alpha=0.25)

    ax1.set_ylim(0, 1.05)
    ax1.set_xlabel("Training iteration", fontsize=13)
    ax1.set_ylabel("P(reaching desired set A)", fontsize=13)
    ax1.set_title(f"Success rate – {sys.name}", fontsize=13)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.set_xlabel("Training iteration", fontsize=13)
    ax2.set_ylabel("Mean control cost", fontsize=13)
    ax2.set_title(f"Control cost – {sys.name}", fontsize=13)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(save_dir, f"multirun_{sys.name}.pdf")
    plt.savefig(path, bbox_inches="tight")
    print(f"  saved → {path}")
    plt.show()


# ──────────────────────────────────────────────────────────────────────────────
# 12.  Demo / entry point
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    os.makedirs("figures", exist_ok=True)

    # ── shared parameters ──────────────────────────────────────────────────
    dt      = 0.01
    D_true  = 0.01
    a       = 0.5
    seed    = 42

    # ── Fig. 2: committor and control curves ───────────────────────────────
    print("Plotting committor / control curves …")
    plot_committor_and_control(a=a, lambda_u=2.0, D_values=[0.5, 0.1, 0.01])

    # ══════════════════════════════════════════════════════════════════════
    # Demo 1 – Curved Double-Well
    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "═"*60)
    print("  Demo 1: Curved Double-Well")
    print("═"*60)

    sys_dw = make_double_well_curved(lambda_u=2.0, lambda_s=10.0, c=0.5)

    tracker_dw, all_X_dw, results_dw = learn_and_eval(
        sys_dw,
        D_true=D_true, dt=dt,
        n_trajs=50, T_train=25, init_bounds=0.1,
        radius=0.3, x_star_init_range=0.2, n_iterations=15,
        a=a, T_eval=1000, n_grid=5, x_max=a,
        r_neigh=0.5, nbhd_shape="circular",
        grid_rotation_deg=0.0,       # w_u = [1,0], no extra rotation needed
        feedback_function=erf_feedback,
        seed=seed)

    plot_estimation_errors(tracker_dw, sys_dw, dt, D_true)
    plot_trajectories(results_dw, sys_dw, a=a, n_show=5)

    # ══════════════════════════════════════════════════════════════════════
    # Demo 2 – Pendulum (rotate committor boundaries with w_u)
    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "═"*60)
    print("  Demo 2: Pendulum – prevent over-the-top swing")
    print("═"*60)

    sys_pend = make_pendulum(g=9.81, length=1.0, damping=1.5)
    print(f"  Saddle  x*        = {sys_pend.x_star}")
    print(f"  λ_u  (cont.)      = {sys_pend.lambda_u_true:.4f}")
    print(f"  Left unstable w_u = {sys_pend.w_u_true.round(4)}")

    # The left unstable eigenvector is NOT axis-aligned, so the evaluation
    # grid must be rotated to place initial conditions on the "undesired"
    # side (x_u > 0).  grid_rotation_deg is already encoded in w_u_true
    # inside run_grid_eval, but we add 0 extra rotation here.
    tracker_pend, all_X_pend, results_pend = learn_and_eval(
        sys_pend,
        D_true=D_true, dt=dt,
        n_trajs=100, T_train=50, init_bounds=0.3,
        radius=0.6, x_star_init_range=0.5, n_iterations=20,
        a=a, T_eval=500, n_grid=5, x_max=a,
        r_neigh=0.8, nbhd_shape="circular",
        grid_rotation_deg=0.0,       # rotation encoded in w_u_true via run_grid_eval
        feedback_function=erf_feedback,
        seed=seed)

    plot_estimation_errors(tracker_pend, sys_pend, dt, D_true)
    plot_trajectories(results_pend, sys_pend, a=a, n_show=5)

    print("\nAll done – figures saved to ./figures/")


# ──────────────────────────────────────────────────────────────────────────────


# ──────────────────────────────────────────────────────────────────────────────



# ──────────────────────────────────────────────────────────────────────────────
# 12b. Algorithm 1 (single-trajectory variant) – direct VAR on one long trajectory
# ──────────────────────────────────────────────────────────────────────────────

class DirectVARTracker:
    """
    Algorithm 1 (single-trajectory variant): local VAR on one long trajectory.

    Key contrast with Algorithm 1 (NbhdVARTracker)
    ───────────────────────────────────────────────
    • Input: ONE long continuous trajectory  {x_0, x_1, …, x_{T+τ}}
             started near the saddle.
    • NbhdVARTracker input: N short trajectories each started fresh near the
             saddle from different initial conditions.

    Both algorithms then:
      1. Collect (x_t, x_{t+1}) pairs that lie within radius r of x̂*.
      2. Fit a local VAR(1) on those pairs.
      3. Update x̂* = (I − Â)⁻¹(mean_f − Â·mean_p).
      4. Extract ŵ_u by power-iteration on Â^T.

    Because the single trajectory escapes to a well quickly (under low
    noise), it accumulates far fewer local pairs than the multi-trajectory
    approach, making estimation noisier — this is the real comparison.
    """

    def __init__(self, dim: int = 2,
                 radius: float = 0.3,
                 x_star_init: Optional[np.ndarray] = None,
                 x_star_init_range: float = 0.3,
                 store_history: bool = True,
                 rng_seed: Optional[int] = None):
        self.dim = int(dim)
        self.radius = float(radius)
        self.store_history = store_history
        self.rng = np.random.default_rng(rng_seed)

        if x_star_init is not None:
            self.x_star = np.asarray(x_star_init, dtype=float).ravel()
        else:
            self.x_star = self.rng.uniform(
                -x_star_init_range, x_star_init_range, size=self.dim)

        self.A             = np.eye(dim, dtype=float)
        self.residual_cov  = np.eye(dim, dtype=float)
        self.noise_strength = 1e-5
        self.D_est         = 0.0

        v0 = self.rng.standard_normal(dim)
        v0 /= np.linalg.norm(v0) + 1e-15
        self.w_u          = v0.copy()
        self.lambda_disc  = 1.0

        # all pairs from the single trajectory
        self.pairs: List[Tuple[np.ndarray, np.ndarray]] = []

        self.D_mat = np.eye(int(dim), dtype=float) * 1e-5

        if store_history:
            self.A_history            = [self.A.copy()]
            self.residual_cov_history = [self.D_mat.copy()]
            self.noise_history        = [self.noise_strength]
            self.wu_history           = [self.w_u.copy()]
            self.lambda_disc_history  = [self.lambda_disc]
            self.xstar_history        = [self.x_star.copy()]
            self.nlocal_history       = [0]

    # ------------------------------------------------------------------ #
    def _fit_local(self, dt: float,
                   sign_ref: Optional[np.ndarray] = None) -> None:
        """Fit local VAR on pairs within radius of current x*."""
        local = [(xp, xf) for xp, xf in self.pairs
                 if np.linalg.norm(xp - self.x_star) <= self.radius]
        n_loc = len(local)

        if self.store_history:
            self.nlocal_history.append(n_loc)

        if n_loc < self.dim + 1:
            if self.store_history:
                self.wu_history.append(self.w_u.copy())
                self.lambda_disc_history.append(self.lambda_disc)
                self.xstar_history.append(self.x_star.copy())
                self.A_history.append(self.A.copy())
                self.residual_cov_history.append(self.residual_cov.copy())
                self.noise_history.append(self.noise_strength)
            return

        Xp = np.vstack([p[0] for p in local])
        Xf = np.vstack([p[1] for p in local])

        # center by x̂* (Algorithm 1, step 4)
        Zp = Xp - self.x_star
        Zf = Xf - self.x_star
        mp = Zp.mean(0);  mf = Zf.mean(0)

        Cpp = Zp.T @ Zp / n_loc
        Cfp = Zf.T @ Zp / n_loc
        A_hat = Cfp @ np.linalg.pinv(Cpp)

        # residuals and D̂ matrix (steps 8-9)
        resid  = Zf - Zp @ A_hat.T
        D_mat  = resid.T @ resid / n_loc
        mse    = float(np.mean(np.sum(resid ** 2, 1)))

        # Update x̂* in original coords: x̂*_new = x̂* + (I-A)^{-1}(z̄_f - A z̄_p)
        IminA = np.eye(self.dim) - A_hat
        try:
            x_star_cand = self.x_star + np.linalg.solve(IminA, mf - A_hat @ mp)
        except np.linalg.LinAlgError:
            x_star_cand = self.x_star + np.linalg.pinv(IminA) @ (mf - A_hat @ mp)
        if np.linalg.norm(x_star_cand - Xp.mean(0)) < 3.0 * self.radius:
            self.x_star = x_star_cand
        else:
            self.x_star = Xp.mean(0)

        self.A             = A_hat
        self.D_mat         = D_mat
        self.noise_strength = mse

        # power iteration steps 10-11
        v = self.w_u.copy()
        for _ in range(40):
            v_new = A_hat.T @ v
            nrm   = np.linalg.norm(v_new)
            if nrm < 1e-12: break
            v_new /= nrm
            if np.linalg.norm(v_new - v) < 1e-12:
                v = v_new; break
            v = v_new
        if sign_ref is not None:
            v = _sign_align(v, sign_ref)
        self.w_u = v
        self.lambda_disc = float(np.linalg.norm(A_hat.T @ v))
        self.D_est = float(v @ D_mat @ v) / (2.0 * dt)

        if self.store_history:
            self.A_history.append(A_hat.copy())
            self.residual_cov_history.append(D_mat.copy())
            self.noise_history.append(mse)
            self.wu_history.append(self.w_u.copy())
            self.lambda_disc_history.append(self.lambda_disc)
            self.xstar_history.append(self.x_star.copy())

    # ------------------------------------------------------------------ #
    def train(self, drift_func: Callable,
              T: int = 1250,
              dt: float = 0.01, D: float = 0.01,
              x_shift: float = 0.0, y_shift: float = 0.0,
              seed: int = 0, init_bounds: float = 0.1,
              n_iterations: int = 15,
              sign_ref: Optional[np.ndarray] = None,
              return_trajectories: bool = False,
              a: Optional[float] = None,
              init_spread: float = 0.05):
        """
        Simulate ONE long trajectory near the saddle, then fit local VAR.

        If ``a`` is provided, the trajectory is restarted near the current
        x̂* estimate whenever it crosses the committor boundary
        |ŵ_u · (x − x̂*)| ≥ a.  This keeps the data stream near-saddle for
        the full budget T, eliminating the windowing problem without changing
        the neighbourhood-filter logic.  Each restart uses a small Gaussian
        perturbation around x̂* (std = ``init_spread``).

        Parameters
        ----------
        T            : total number of steps (budget)
        a            : committor boundary — trigger restart when projection
                       onto ŵ_u exceeds this.  None = no restarts (original).
        init_spread  : std of Gaussian restart perturbation around x̂*
        n_iterations : number of x* refinement passes after data collection
        """
        noise_std = np.sqrt(2.0 * D)
        rng = np.random.default_rng(seed)
        center = np.zeros(self.dim)
        center[0] = x_shift
        if self.dim > 1:
            center[1] = y_shift

        X_all = []
        x = center + rng.uniform(-init_bounds, init_bounds, self.dim)

        for t in range(int(T)):
            X_all.append(x.copy())
            dx  = drift_func(t * dt, x)
            x_next = x + dx * dt + noise_std * rng.standard_normal(self.dim) * np.sqrt(dt)

            self.pairs.append((x.copy(), x_next.copy()))

            # restart if boundary crossed (only once we have a non-trivial estimate)
            if a is not None and len(self.pairs) > self.dim + 1:
                proj = float(np.dot(x_next - self.x_star, self.w_u))
                if abs(proj) >= a:
                    x_next = self.x_star + rng.normal(0, init_spread, self.dim)

            x = x_next

        X_all.append(x.copy())
        X = np.array(X_all)

        # ── iterative local fitting ───────────────────────────────────
        for _ in range(int(n_iterations)):
            self._fit_local(dt, sign_ref=sign_ref)

        return [X] if return_trajectories else None

    # ------------------------------------------------------------------ #
    def get_params(self, dt: float,
                   sign_ref: Optional[np.ndarray] = None) -> dict:
        """Return estimated control parameters (same interface as NbhdVARTracker)."""
        w_u = self.w_u.copy()
        if sign_ref is not None:
            w_u = _sign_align(w_u, sign_ref)
        lam_disc = abs(self.lambda_disc)
        lambda_u_ct = (np.log(lam_disc) / dt) if lam_disc > 1e-12 else 0.0
        return dict(w_u=w_u, lambda_u=lambda_u_ct,
                    D=self.D_est, x_star=self.x_star.copy())



# ──────────────────────────────────────────────────────────────────────────────
# 13.  Algorithm 2 (batch) – past-future CCA on raw states
# ──────────────────────────────────────────────────────────────────────────────

class CCATracker:
    """
    Algorithm 2: past-future CCA on consecutive states.

    Uses p_t = x_t,  f_t = x_{t+1}  (τ dropped — raw states, not increments).

    Steps follow the paper exactly:
      3. C_pp, C_ff, C_pf, C_fp  from {x_t}, {x_{t+1}}.
      4. CCA eigenproblems for (v_0, u_0, ρ_0).
      5. Normalise: v_0^T C_pp v_0 = u_0^T C_ff u_0 = 1.
      6. λ_0 = ρ_0 · v_0^T C_ff u_0.
      7. Δ = (1/T) Σ_t (v_0^T x_{t+1} − λ_0 v_0^T x_t) / (1 − λ_0).
      8. q(x) = v_0^T x − Δ.
      9. 2 v_0^T D v_0 = (1/T) Σ_t [q(x_{t+1}) − λ_0 q(x_t)]^2.
     10. u(x) from paper Eq (control step).

    The control law is applied via ``cca_feedback``, which uses λ_0 and
    v_0^T D v_0 directly (not their continuous-time conversions).

    Both batch and restarted modes supported via ``a`` in ``train()``.

    Parameters
    ----------
    dim          : state dimension
    store_history: record per-segment estimates
    rng_seed     : random seed
    """

    def __init__(self, dim: int = 2,
                 store_history: bool = True,
                 rng_seed: Optional[int] = None):
        self.dim           = int(dim)
        self.store_history = store_history
        self.rng           = np.random.default_rng(rng_seed)

        v0 = self.rng.standard_normal(dim)
        v0 /= np.linalg.norm(v0) + 1e-15
        self.v0          = v0.copy()
        self.u0          = v0.copy()
        self.lambda_disc = 0.0    # λ_0 = ρ_0 v_0^T C_ff u_0
        self.Delta       = 0.0    # q(x̂*) = 0
        self.Dv          = 0.0    # v_0^T D v_0  (scalar, from paper step 9)
        self.x_star      = np.zeros(dim, dtype=float)

        if store_history:
            self.v0_history          = [self.v0.copy()]
            self.lambda_disc_history = [self.lambda_disc]
            self.xstar_history       = [self.x_star.copy()]
            self.npairs_history      = [0]

    # ------------------------------------------------------------------ #
    def _fit(self, X: np.ndarray,
             sign_ref: Optional[np.ndarray] = None) -> bool:
        """Fit CCA on trajectory segment X (shape N × dim)."""
        X = X[np.all(np.isfinite(X), axis=1)]
        if len(X) < self.dim + 3:
            return False

        Xp = X[:-1]; Xf = X[1:]   # p_t = x_t,  f_t = x_{t+1}
        n  = len(Xp)

        # step 3: covariance matrices
        C_pp = Xp.T @ Xp / n
        C_ff = Xf.T @ Xf / n
        C_pf = Xp.T @ Xf / n
        C_fp = Xf.T @ Xp / n

        # step 4: CCA eigenproblems
        Cpp_inv = np.linalg.pinv(C_pp)
        Cff_inv = np.linalg.pinv(C_ff)
        M_v = Cpp_inv @ C_pf @ Cff_inv @ C_fp
        M_u = Cff_inv @ C_fp @ Cpp_inv @ C_pf

        try:
            evals_v, evecs_v = np.linalg.eigh(M_v)
            evals_u, evecs_u = np.linalg.eigh(M_u)
        except np.linalg.LinAlgError:
            return False

        v0 = evecs_v[:, np.argmax(evals_v)]
        u0 = evecs_u[:, np.argmax(evals_u)]
        rho0 = np.sqrt(max(float(np.max(evals_v)), 0.0))

        # step 5: normalise
        vCv = float(v0 @ C_pp @ v0)
        uCu = float(u0 @ C_ff @ u0)
        if vCv > 1e-15: v0 /= np.sqrt(vCv)
        if uCu > 1e-15: u0 /= np.sqrt(uCu)

        if sign_ref is not None:
            v0 = _sign_align(v0, sign_ref)
        if np.dot(u0, v0) < 0:
            u0 = -u0

        # step 6: λ_0
        lam0 = rho0 * float(v0 @ C_ff @ u0)

        # step 7: Δ
        pn = Xf @ v0; pc = Xp @ v0
        if abs(1.0 - lam0) > 1e-10:
            Delta = float(np.mean((pn - lam0 * pc) / (1.0 - lam0)))
        else:
            Delta = float(np.mean(Xp @ v0))

        # step 9: 2 v_0^T D v_0
        qn = pn - Delta; qc = pc - Delta
        two_Dv = float(np.mean((qn - lam0 * qc) ** 2))   # = 2 v_0^T D v_0

        # x̂* in state space
        v0_norm2  = float(v0 @ v0)
        x_mean    = Xp.mean(0)
        self.x_star = x_mean + (Delta - float(v0 @ x_mean)) / max(v0_norm2, 1e-15) * v0

        self.v0          = v0
        self.u0          = u0
        self.lambda_disc = lam0
        self.Delta       = Delta
        self.Dv          = two_Dv / 2.0   # v_0^T D v_0
        return True

    # ------------------------------------------------------------------ #
    def train(self, drift_func: Callable,
              T: int = 2500,
              dt: float = 0.01, D: float = 0.01,
              seed: int = 0, init_bounds: float = 0.1,
              sign_ref: Optional[np.ndarray] = None,
              return_trajectories: bool = False,
              a: Optional[float] = None,
              init_spread: float = 0.05):
        """
        Simulate trajectory and fit CCA.

        If ``a`` is set, restart near x̂* when |v_0^T(x−x̂*)/||v_0||| ≥ a.
        """
        noise_std = np.sqrt(2.0 * D)
        rng = np.random.default_rng(seed)
        x = rng.uniform(-init_bounds, init_bounds, self.dim)
        X_seg = [x.copy()]

        for t in range(int(T)):
            dx     = drift_func(t * dt, x)
            x_next = (x + dx * dt
                      + noise_std * rng.standard_normal(self.dim) * np.sqrt(dt))

            if a is not None and len(X_seg) > self.dim + 3:
                v0_nrm = np.linalg.norm(self.v0)
                proj   = float(np.dot(x_next - self.x_star, self.v0))
                if v0_nrm > 1e-15 and abs(proj / v0_nrm) >= a:
                    self._fit(np.array(X_seg), sign_ref=sign_ref)
                    if self.store_history:
                        self.v0_history.append(self.v0.copy())
                        self.lambda_disc_history.append(self.lambda_disc)
                        self.xstar_history.append(self.x_star.copy())
                        self.npairs_history.append(len(X_seg))
                    x_next = self.x_star + rng.normal(0, init_spread, self.dim)
                    X_seg  = [x_next.copy()]
                    x      = x_next
                    continue

            X_seg.append(x_next.copy())
            x = x_next

        self._fit(np.array(X_seg), sign_ref=sign_ref)
        if self.store_history:
            self.v0_history.append(self.v0.copy())
            self.lambda_disc_history.append(self.lambda_disc)
            self.xstar_history.append(self.x_star.copy())
            self.npairs_history.append(len(X_seg))

        return [np.array(X_seg)] if return_trajectories else None

    # ------------------------------------------------------------------ #
    def get_params(self, dt: float = None,
                   sign_ref: Optional[np.ndarray] = None) -> dict:
        """
        Return control parameters for use with ``cca_feedback``.

        Returns λ_0, v_0^T D v_0, v_0 (C_pp-normalised), Δ, x̂* directly —
        not converted to continuous-time.  Pass to ``cca_feedback``, not
        ``erf_feedback``.
        """
        v0 = self.v0.copy()
        if sign_ref is not None:
            v0 = _sign_align(v0, sign_ref)
        return dict(v0=v0, lambda0=self.lambda_disc,
                    Dv=self.Dv, Delta=self.Delta,
                    x_star=self.x_star.copy())


# ──────────────────────────────────────────────────────────────────────────────
# 14.  Algorithm 1 (online) – streaming VAR via EMA covariances
# ──────────────────────────────────────────────────────────────────────────────

class OnlineVARTracker:
    """
    Algorithm 1 (online variant): updates VAR parameters incrementally as new
    observations arrive, using exponential moving averages of the covariance
    matrices.

    At each step t, given a new pair (x_t, x_{t+1}) near the saddle:

        C_pp ← (1-γ) C_pp + γ z_t   z_t^T          (z_t = x_t   - x̂*)
        C_fp ← (1-γ) C_fp + γ z_{t+1} z_t^T         (z_{t+1} = x_{t+1} - x̂*)
        Â    = C_fp C_pp^{-1}
        ŵ_u  ← Â^T ŵ_u / ||Â^T ŵ_u||              (one power step)
        λ̂_u  = (1/dt) log ||Â^T ŵ_u||
        D̂    = (1-γ) D̂ + γ (e_t e_t^T) / (2 dt)   (e_t = z_{t+1} - Â z_t)
        x̂*   ← (I - Â)^{-1} (z̄_f - Â z̄_p)

    This is the same VAR logic as NbhdVARTracker / DirectVARTracker, but run
    step-by-step rather than in batch.  The neighbourhood filter is implicit:
    only pairs with ||x_t - x̂*|| ≤ radius are used to update covariances.

    Parameters
    ----------
    dim        : state dimension
    radius     : neighbourhood radius (None = accept all)
    gamma      : EMA decay per step  (smaller = longer memory)
    min_steps  : minimum pairs before fitting is attempted
    rng_seed   : random seed for initial w_u
    """

    def __init__(self, dim: int = 2,
                 radius: Optional[float] = 0.3,
                 gamma: float = 0.02,
                 min_steps: int = 50,
                 store_history: bool = True,
                 rng_seed: Optional[int] = None):
        self.dim        = int(dim)
        self.radius     = radius
        self.gamma      = float(gamma)
        self.min_steps  = int(min_steps)
        self.store_history = store_history
        self.rng        = np.random.default_rng(rng_seed)

        v0 = self.rng.standard_normal(dim)
        v0 /= np.linalg.norm(v0) + 1e-15
        self.w_u         = v0.copy()
        self.lambda_disc = 1.0
        self.D_mat       = np.eye(dim, dtype=float) * 1e-5
        self.D_est       = 0.0
        self.x_star      = np.zeros(dim, dtype=float)

        # EMA covariance accumulators (centred by x̂*)
        self.C_pp = np.eye(dim, dtype=float) * 1e-5
        self.C_fp = np.zeros((dim, dim), dtype=float)
        self.A    = np.eye(dim, dtype=float)

        self.n_updates = 0   # number of local pairs processed

        if store_history:
            self.wu_history          = [self.w_u.copy()]
            self.lambda_disc_history = [self.lambda_disc]
            self.xstar_history       = [self.x_star.copy()]
            self.D_est_history       = [self.D_est]

    # ------------------------------------------------------------------ #
    def update(self, x_t: np.ndarray, x_next: np.ndarray,
               dt: float, sign_ref: Optional[np.ndarray] = None) -> None:
        """
        Incorporate one new transition (x_t → x_next) and update all estimates.

        Skips the update if x_t is outside the neighbourhood of x̂*.
        """
        if (self.radius is not None and
                np.linalg.norm(x_t - self.x_star) > self.radius):
            return

        g = self.gamma
        z_t    = x_t   - self.x_star
        z_next = x_next - self.x_star

        # EMA covariance update (Algorithm 1, steps 6-7, streaming version)
        self.C_pp = (1 - g) * self.C_pp + g * np.outer(z_t,    z_t)
        self.C_fp = (1 - g) * self.C_fp + g * np.outer(z_next, z_t)
        self.n_updates += 1

        if self.n_updates < self.min_steps:
            return

        # Transition matrix (step 7)
        A = self.C_fp @ np.linalg.pinv(self.C_pp)

        # Residual and D̂ update (steps 8-9)
        e = z_next - A @ z_t
        self.D_mat = (1 - g) * self.D_mat + g * np.outer(e, e) / (2.0 * dt)

        # Power step for left unstable eigenvector (steps 10-11)
        v = A.T @ self.w_u
        nrm = np.linalg.norm(v)
        if nrm > 1e-12:
            self.w_u = v / nrm
            if sign_ref is not None:
                self.w_u = _sign_align(self.w_u, sign_ref)
            self.lambda_disc = nrm   # = ||Â^T ŵ_u||

        # D_est = ŵ_u^T D̂ ŵ_u / (2 dt) — already divided by 2dt above in D_mat
        self.D_est = float(self.w_u @ self.D_mat @ self.w_u)

        # Fixed-point update (step 12): x̂* = (I-A)^{-1}(z̄_f - A z̄_p)
        # In z-space, z̄_f - A z̄_p ≈ 0 when x̂* is correct.
        # EMA z̄_p ≈ 0, z̄_f ≈ 0 by definition since z = x - x̂*.
        # Refine via: x̂*_new = x̂* + (I-A)^{-1} * residual
        # residual = z_next - A @ z_t  minus noise ≈ (A-I)(x̂*_true - x̂*)
        # → x̂*_new = x̂* - (I-A)^{-1} @ e  where e is the drift residual
        # Simple stable version: nudge x̂* toward (I-A)^{-1} @ mean_z_f
        # Use instantaneous z values as a noisy estimate of means
        IminA = np.eye(self.dim) - A
        try:
            # (I-A)^{-1}(z_next - A z_t) gives correction in x-space
            correction = np.linalg.solve(IminA, z_next - A @ z_t)
            x_star_cand = self.x_star + self.gamma * correction
        except np.linalg.LinAlgError:
            x_star_cand = self.x_star
        # guard: only accept if within 3×radius of current data
        if (self.radius is None or
                np.linalg.norm(x_star_cand - x_t) < 3.0 * self.radius):
            self.x_star = x_star_cand

        self.A = A

        if self.store_history:
            self.wu_history.append(self.w_u.copy())
            self.lambda_disc_history.append(self.lambda_disc)
            self.xstar_history.append(self.x_star.copy())
            self.D_est_history.append(self.D_est)

    # ------------------------------------------------------------------ #
    def train(self, drift_func: Callable,
              T: int = 2500,
              dt: float = 0.01, D: float = 0.01,
              seed: int = 0, init_bounds: float = 0.1,
              sign_ref: Optional[np.ndarray] = None,
              return_trajectories: bool = False):
        """Simulate one trajectory, calling update() at every step."""
        noise_std = np.sqrt(2.0 * D)
        rng = np.random.default_rng(seed)
        x = rng.uniform(-init_bounds, init_bounds, self.dim)
        X = [x.copy()]
        for t in range(int(T)):
            dx     = drift_func(t * dt, x)
            x_next = (x + dx * dt
                      + noise_std * rng.standard_normal(self.dim) * np.sqrt(dt))
            self.update(x, x_next, dt, sign_ref=sign_ref)
            X.append(x_next.copy())
            x = x_next
        return [np.array(X)] if return_trajectories else None

    # ------------------------------------------------------------------ #
    def get_params(self, dt: float,
                   sign_ref: Optional[np.ndarray] = None) -> dict:
        w_u = self.w_u.copy()
        if sign_ref is not None:
            w_u = _sign_align(w_u, sign_ref)
        lam = abs(self.lambda_disc)
        lambda_u_ct = np.log(lam) / dt if lam > 1e-12 else 0.0
        D_est = float(w_u @ self.D_mat @ w_u)
        return dict(w_u=w_u, lambda_u=lambda_u_ct,
                    D=D_est, x_star=self.x_star.copy())


# ──────────────────────────────────────────────────────────────────────────────
# 15.  Algorithm 2 (online) – streaming CCA via EMA covariances
# ──────────────────────────────────────────────────────────────────────────────

class OnlineCCATracker:
    """
    Algorithm 2 (online variant): updates CCA parameters incrementally as new
    observations arrive, using exponential moving averages of the four
    covariance matrices.

    At each step t, given a new pair (x_t, x_{t+1}):

        C_pp ← (1-γ) C_pp + γ x_t   x_t^T
        C_ff ← (1-γ) C_ff + γ x_{t+1} x_{t+1}^T
        C_pf ← (1-γ) C_pf + γ x_t   x_{t+1}^T
        C_fp ← (1-γ) C_fp + γ x_{t+1} x_t^T

    Then a full CCA solve is run on the current accumulators (cheap for dim=2)
    to extract (v_0, u_0, λ_0, Δ, Dv) exactly as in CCATracker._fit().

    The neighbourhood filter (radius) is applied before updating: pairs
    outside ||x_t - x̂*|| > radius are skipped.

    Parameters
    ----------
    dim        : state dimension
    radius     : neighbourhood radius (None = accept all)
    gamma      : EMA decay per step
    min_steps  : minimum pairs before CCA solve is attempted
    rng_seed   : random seed for initial v_0
    """

    def __init__(self, dim: int = 2,
                 radius: Optional[float] = None,
                 gamma: float = 0.02,
                 min_steps: int = 50,
                 a: Optional[float] = None,
                 init_spread: float = 0.05,
                 store_history: bool = True,
                 rng_seed: Optional[int] = None):
        self.dim         = int(dim)
        self.radius      = radius
        self.gamma       = float(gamma)
        self.min_steps   = int(min_steps)
        self.a           = a            # boundary for restarts (None = no restarts)
        self.init_spread = float(init_spread)
        self.store_history = store_history
        self.rng         = np.random.default_rng(rng_seed)

        v0 = self.rng.standard_normal(dim)
        v0 /= np.linalg.norm(v0) + 1e-15
        self.v0          = v0.copy()
        self.u0          = v0.copy()
        self.lambda_disc = 0.0
        self.Delta       = 0.0
        self.Dv          = 0.0
        self.x_star      = np.zeros(dim, dtype=float)

        # EMA covariance accumulators
        eps = 1e-5
        self.C_pp = np.eye(dim, dtype=float) * eps
        self.C_ff = np.eye(dim, dtype=float) * eps
        self.C_pf = np.zeros((dim, dim), dtype=float)
        self.C_fp = np.zeros((dim, dim), dtype=float)

        # running sum for Δ: numerator and denominator (EMA scalars)
        self.Delta_num = 0.0
        self.Delta_den = 1.0

        self.n_updates   = 0
        self._x_mean     = np.zeros(dim, dtype=float)  # EMA of restart origins
        self._n_restarts = 0

        if store_history:
            self.v0_history          = [self.v0.copy()]
            self.lambda_disc_history = [self.lambda_disc]
            self.xstar_history       = [self.x_star.copy()]

    # ------------------------------------------------------------------ #
    def _solve_cca(self, sign_ref: Optional[np.ndarray] = None) -> bool:
        """Run CCA on current EMA accumulators. Returns True on success."""
        try:
            Cpp_inv = np.linalg.pinv(self.C_pp)
            Cff_inv = np.linalg.pinv(self.C_ff)
            M_v = Cpp_inv @ self.C_pf @ Cff_inv @ self.C_fp
            M_u = Cff_inv @ self.C_fp @ Cpp_inv @ self.C_pf
            evals_v, evecs_v = np.linalg.eigh(M_v)
            evals_u, evecs_u = np.linalg.eigh(M_u)
        except np.linalg.LinAlgError:
            return False

        v0 = evecs_v[:, np.argmax(evals_v)]
        u0 = evecs_u[:, np.argmax(evals_u)]
        rho0 = np.sqrt(max(float(np.max(evals_v)), 0.0))

        vCv = float(v0 @ self.C_pp @ v0)
        uCu = float(u0 @ self.C_ff @ u0)
        if vCv > 1e-15: v0 /= np.sqrt(vCv)
        if uCu > 1e-15: u0 /= np.sqrt(uCu)

        if sign_ref is not None:
            v0 = _sign_align(v0, sign_ref)
        if np.dot(u0, v0) < 0:
            u0 = -u0

        lam0 = rho0 * float(v0 @ self.C_ff @ u0)

        # Δ from running EMA accumulator
        Delta = self.Delta_num / max(self.Delta_den, 1e-15)

        # Dv from residual variance
        # 2Dv ≈ v0^T C_ff v0 - 2λ0 v0^T C_pf^T v0 + λ0^2 v0^T C_pp v0
        # (expands mean[(q_{t+1} - λ0 q_t)^2] using EMA covariances)
        two_Dv = (float(v0 @ self.C_ff @ v0)
                  - 2.0 * lam0 * float(v0 @ self.C_fp @ v0)
                  + lam0**2 * float(v0 @ self.C_pp @ v0))

        # x̂* is NOT updated here — it is updated from restart positions in train().
        # _x_mean is an EMA of restart origins which is stable and near-saddle.
        pass

        self.v0          = v0
        self.u0          = u0
        self.lambda_disc = lam0
        self.Delta       = Delta
        self.Dv          = max(two_Dv / 2.0, 1e-15)
        return True

    # ------------------------------------------------------------------ #
    def update(self, x_t: np.ndarray, x_next: np.ndarray,
               sign_ref: Optional[np.ndarray] = None) -> None:
        """Incorporate one new transition (x_t → x_next)."""
        if (self.radius is not None and
                np.linalg.norm(x_t - self.x_star) > self.radius):
            return

        g = self.gamma
        self.C_pp = (1 - g) * self.C_pp + g * np.outer(x_t,   x_t)
        self.C_ff = (1 - g) * self.C_ff + g * np.outer(x_next, x_next)
        self.C_pf = (1 - g) * self.C_pf + g * np.outer(x_t,   x_next)
        self.C_fp = (1 - g) * self.C_fp + g * np.outer(x_next, x_t)
        self.n_updates += 1

        if self.n_updates < self.min_steps:
            return

        # Update Δ accumulator: EMA of (v0^T x_{t+1} - λ0 v0^T x_t) / (1-λ0)
        lam0 = self.lambda_disc
        if abs(1.0 - lam0) > 1e-10:
            contrib = (float(self.v0 @ x_next) - lam0 * float(self.v0 @ x_t)) / (1.0 - lam0)
            self.Delta_num = (1 - g) * self.Delta_num + g * contrib
            self.Delta_den = (1 - g) * self.Delta_den + g

        self._solve_cca(sign_ref=sign_ref)

        if self.store_history:
            self.v0_history.append(self.v0.copy())
            self.lambda_disc_history.append(self.lambda_disc)
            self.xstar_history.append(self.x_star.copy())

    # ------------------------------------------------------------------ #
    def train(self, drift_func: Callable,
              T: int = 2500,
              dt: float = 0.01, D: float = 0.01,
              seed: int = 0, init_bounds: float = 0.1,
              sign_ref: Optional[np.ndarray] = None,
              return_trajectories: bool = False):
        """
        Simulate one trajectory, calling update() at every step.

        If ``self.a`` is set (passed in __init__), restarts near x̂* whenever
        the trajectory crosses the boundary |v_0^T(x - x̂*)/||v_0||| >= a.
        This keeps all EMA covariance updates near-saddle, ensuring λ_0 > 1.
        """
        noise_std = np.sqrt(2.0 * D)
        rng = np.random.default_rng(seed)
        x = rng.uniform(-init_bounds, init_bounds, self.dim)
        X = [x.copy()]
        for t in range(int(T)):
            dx     = drift_func(t * dt, x)
            x_next = (x + dx * dt
                      + noise_std * rng.standard_normal(self.dim) * np.sqrt(dt))

            # boundary-triggered restart: keep trajectory near x̂*
            # Before min_steps we use Euclidean distance (v0 not yet estimated).
            # After min_steps we use projected distance along v0.
            if self.a is not None:
                v0_nrm = np.linalg.norm(self.v0)
                if self.n_updates < self.min_steps or v0_nrm < 1e-15:
                    # Euclidean guard: prevents trajectory poisoning _x_mean
                    dist = float(np.linalg.norm(x_next - self.x_star))
                else:
                    proj = float(np.dot(self.v0, x_next - self.x_star)) / v0_nrm
                    dist = abs(proj)
                if dist >= self.a:
                    x_next = self.x_star + rng.normal(0, self.init_spread, self.dim)
                    # update x_star EMA from the restart target (stable near-saddle pos)
                    self._x_mean = ((1 - self.gamma) * self._x_mean
                                    + self.gamma * x_next)
                    self._n_restarts += 1
                    if self._n_restarts >= 3:
                        self.x_star = self._x_mean.copy()

            self.update(x, x_next, sign_ref=sign_ref)
            X.append(x_next.copy())
            x = x_next
        return [np.array(X)] if return_trajectories else None

    # ------------------------------------------------------------------ #
    def get_params(self, dt: float = None,
                   sign_ref: Optional[np.ndarray] = None) -> dict:
        """Returns same dict as CCATracker.get_params() — use with cca_feedback."""
        v0 = self.v0.copy()
        if sign_ref is not None:
            v0 = _sign_align(v0, sign_ref)
        return dict(v0=v0, lambda0=self.lambda_disc,
                    Dv=self.Dv, Delta=self.Delta,
                    x_star=self.x_star.copy())