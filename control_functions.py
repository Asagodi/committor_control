import numpy as np
from typing import Callable, Tuple, Optional, Literal

def relu_control_1d(u, lambda_u):
    """1D ReLU control along unstable coordinate."""
    signal = 2 * lambda_u * u
    return -np.maximum(signal, 0.0)


def relu_scalar(u_scalar, l_unstable, D):
    # ReLU control in scalar form
    return 2 * D * np.maximum(0, u_scalar) * l_unstable


def relu_feedback(x, l_unstable, lambda_u, D, x_fp=None, r_neigh=1.0, nbhd_shape="rectangular"):
    if x_fp is None:
        x_fp = np.zeros_like(x)

    if not in_neighborhood(x, x_fp, l_unstable, r_neigh, nbhd_shape):
        return np.zeros_like(x)

    proj = np.dot(l_unstable, x - x_fp)
    signal = 2 * lambda_u * proj
    #print(f"x: {x}, lambda_u: {lambda_u}, Projection: {proj}, Control: {np.maximum(signal, 0.0) * l_unstable}")
    return - np.maximum(signal, 0.0) * l_unstable


def committor_1d(u_vals, a, r):
    """
    Compute committor q(u) for 1D OU unstable coordinate for an array of u values.
    """
    def q_single(u):
        num, _ = quad(lambda y: np.exp(-0.5 * r * y**2), -a, u)
        den, _ = quad(lambda y: np.exp(-0.5 * r * y**2), -a, a)
        return num / den
    
    return np.array([q_single(u) for u in u_vals])

def committor_control_quad(tracker, x, r=1.0, x_fp=None, a=1.0, D=1.0):
    """
    Optimal committor control for stochastic dynamics near saddle.

    Parameters
    ----------
    tracker : ExponentialCovarianceTracker
        Holds l_unstable (unit vector) and lambda_t (unstable eigenvalue).
    x : np.ndarray
        Current state.
    r : float
        Radius of neighborhood around fixed point where control is active.
    x_fp : np.ndarray or None
        Fixed point estimate. If None, assume origin.
    a : float
        Absorbing boundary threshold in unstable coordinate.
    D : float
        Diffusion coefficient (inverse temperature).
    """
    if x_fp is None:
        x_fp = np.zeros_like(x)

    # Distance to fixed point
    dist = np.linalg.norm(x - x_fp)
    if dist > r:
        return np.zeros_like(x)

    l_u = tracker.l_unstable
    lambda_u = tracker.lambda_t

    # Projection onto unstable coordinate
    u = np.dot(l_u, x - x_fp)

    # Parameter r = lambda_u / (2D)
    rho = lambda_u / (2*D)

    # Committor function and derivative ratio
    q_u = committor_1d(u, a, rho)
    # q'(u) = exp(-0.5*r*u^2) / ∫_{-a}^a exp(-0.5*r*y^2) dy
    den, _ = quad(lambda y: np.exp(-0.5 * rho * y**2), -a, a)
    q_prime = np.exp(-0.5 * rho * u**2) / den

    # Optimal committor control
    control_scalar = 2 * D * (q_prime / max(q_u, 1e-12))  # avoid div by zero
    control = control_scalar * l_u

    return control

def committor_control_quad_1d(u, a: float, lambda_u: float, D: float):
    """
    Vectorized 1D quadrature-based committor control along unstable mode.

    Parameters
    ----------
    u : float or np.ndarray
        Projection onto unstable coordinate.
    a : float
        Absorbing boundary threshold.
    lambda_u : float
        Unstable eigenvalue.
    D : float
        Diffusion coefficient.

    Returns
    -------
    control : float or np.ndarray
        Optimal committor control 2D q'/q along the unstable coordinate.
    """
    # Parameter rho = lambda_u / (2 D)
    rho = lambda_u / (2 * D)

    # Compute normalization denominator once (scalar)
    den, _ = quad(lambda y: np.exp(-0.5 * rho * y**2), -a, a)

    # Ensure u is array-like
    u_arr = np.atleast_1d(u)

    # Committor q(u) via integral
    q_u = np.array([quad(lambda y: np.exp(-0.5 * rho * y**2), -a, ui)[0] / den for ui in u_arr])

    # Derivative q'(u)
    q_prime = np.exp(-0.5 * rho * u_arr**2) / den

    # Optimal control: 2 D q'/q, safe division
    control = np.where(q_u > 1e-12, 4 * D * q_prime / q_u, 0.0)

    # Return scalar if input was scalar
    if np.isscalar(u):
        return control[0]
    return control


from scipy import special
def make_committor_and_control_from_projection(lambda_u: float, D: float, a: float):
    r = lambda_u / (2 * D)
    sqrt_r = np.sqrt(r)
    Z = special.erf(a * sqrt_r) - special.erf(-a * sqrt_r)

    def committor_function(p: float) -> float:
        return (special.erf(p * sqrt_r) - special.erf(-a * sqrt_r)) / Z

    def committor_control(p: float) -> float:
        q_val = committor_function(p)
        if q_val < 1e-8:
            return 0.0  # Avoid division by zero
        dq_dp = (2 / np.sqrt(np.pi)) * sqrt_r * np.exp(-r * p**2) / Z
        return 2 * D * dq_dp / q_val
    
    def committor_control(p: float) -> float:
        return 2 * D * np.exp(-r * p**2) / (special.erf(p * sqrt_r) - special.erf(-a * sqrt_r))

    return committor_function, committor_control

def committor_control_erf(p, lambda_u: float, D: float, a: float):
    """
    Vectorized committor control using erf formulation (simplified).

    Parameters
    ----------
    p : float or np.ndarray
        Projection onto unstable coordinate.
    lambda_u : float
        Unstable eigenvalue.
    D : float
        Diffusion constant.
    a : float
        Absorbing boundary threshold.

    Returns
    -------
    control : float or np.ndarray
        Optimal committor control.
    """
    r = lambda_u / (2 * D)
    sqrt_r = np.sqrt(r)

    denom = special.erf(p * sqrt_r) - special.erf(-a * sqrt_r)

    # Safe division (avoid division by zero or tiny denom)
    control = np.where(
        np.abs(denom) > 1e-12,
        (4 * D * sqrt_r / np.sqrt(np.pi)) * np.exp(-r * p**2) / denom,
        0.0,
    )
    return control


def committor_control_erf_scalar(p: float, lambda_u: float, D: float, a: float) -> float:
    r = lambda_u / (2 * D)
    sqrt_r = np.sqrt(r)
    denom = special.erf(p * sqrt_r) - special.erf(-a * sqrt_r)
    if denom < 1e-12:  # avoid div by zero
        return 0.0
    return (4 * D * sqrt_r / np.sqrt(np.pi)) * np.exp(-r * p**2) / denom



import mpmath   #for high precision calculations
mpmath.mp.dps = 200  # decimal places
def high_precision_control(xs: np.ndarray, a: float, r: float, D: float, precision: int = 50) -> np.ndarray:
    """
    Compute high-precision control values using mpmath for integration and exponentiation.
    
    Parameters:
        xs : np.ndarray
            Points at which to evaluate the control.
        a : float
            Integration limit (-a to x).
        r : float
            Rate parameter in the exponential.
        D : float
            Scaling constant.
        precision : int
            Decimal precision for mpmath.

    Returns:
        np.ndarray of control values, dtype float64, filled with np.nan where integral is too small.
    """
    mpmath.mp.dps = precision
    a_mpf = mpmath.mpf(a)
    r_mpf = mpmath.mpf(r)
    D_mpf = mpmath.mpf(D)

    control_exact = np.zeros_like(xs, dtype=np.float64)

    for i, x in tqdm(enumerate(xs)):
        x_mpf = mpmath.mpf(x)

        # Define the integrand
        integrand = lambda y: mpmath.exp(-r_mpf * y**2)

        # High-precision definite integral
        try:
            integral_val = mpmath.quad(integrand, [-a_mpf, x_mpf])
        except Exception:
            integral_val = mpmath.mpf(0)

        numerator = 2 * D_mpf * mpmath.exp(-r_mpf * x_mpf**2)
        control_exact[i] = float(numerator / (integral_val + mpmath.mpf('1e-500')))
        # if mpmath.fabs(integral_val) > mpmath.mpf('1e-1000'):
        #     numerator = 2 * D_mpf * mpmath.exp(-r_mpf * x_mpf**2)
        #     control_exact[i] = float(numerator / integral_val)
        # else:
        #     control_exact[i] = np.nan

    return control_exact


def high_precision_control_scalar(x: float, a: float, r: float, D: float, precision: int = 50) -> float:
    """
    Compute high-precision control value for a single scalar using mpmath.

    Parameters
    ----------
    x : float
        Point at which to evaluate the control.
    a : float
        Integration limit (-a to x).
    r : float
        Rate parameter in the exponential.
    D : float
        Scaling constant.
    precision : int
        Decimal precision for mpmath.

    Returns
    -------
    float
        Control value. Returns np.nan if integral is too small.
    """
    mpmath.mp.dps = precision
    a_mpf = mpmath.mpf(a)
    r_mpf = mpmath.mpf(r)
    D_mpf = mpmath.mpf(D)
    x_mpf = mpmath.mpf(x)

    # Define the integrand
    integrand = lambda y: mpmath.exp(-r_mpf * y**2)

    # High-precision definite integral
    try:
        integral_val = mpmath.quad(integrand, [-a_mpf, x_mpf])
    except Exception:
        integral_val = mpmath.mpf(0)

    # Compute control with tiny offset to avoid division by zero
    control_val = float(2 * D_mpf * mpmath.exp(-r_mpf * x_mpf**2) / (integral_val + mpmath.mpf('1e-1500')))

    return control_val


def in_neighborhood(x, x_fp, l_unstable, r_neigh, nbhd_shape="rectangular"):
    """
    Check whether x is inside the neighborhood around x_fp.
    
    Args:
        x : np.ndarray
        x_fp : fixed point (np.ndarray)
        l_unstable : unstable direction (np.ndarray)
        r_neigh : float, radius or bound
        nbhd_shape : "circular" or "rectangular"
    """
    if nbhd_shape == "circular":
        dist = np.linalg.norm(x - x_fp)
        return dist <= r_neigh

    elif nbhd_shape == "rectangular":  # committor-based
        u_scalar = np.dot(l_unstable, x - x_fp)
        return abs(u_scalar) <= r_neigh

    else:
        raise ValueError(f"Unknown nbhd_shape: {nbhd_shape}")
    
def erf_feedback(x, l_unstable, lambda_u, D, a, 
                x_fp=None, r_neigh=1.0, nbhd_shape="rectangular"):
    if np.allclose(D, 0):
        D = 1e-8  # avoid div by zero
    if x_fp is None:
        x_fp = np.zeros_like(x)

    if not in_neighborhood(x, x_fp, l_unstable, r_neigh, nbhd_shape):
        return np.zeros_like(x)

    u_scalar = -np.dot(l_unstable, x - x_fp)
    r = lambda_u / (2 * D)

    return -committor_control_erf_scalar(u_scalar, lambda_u=lambda_u, D=D, a=a) * l_unstable

def high_precision_feedback(x, l_unstable, lambda_u, D, a, 
                            x_fp=None, r_neigh=1.0, nbhd_shape="rectangular"):
    if x_fp is None:
        x_fp = np.zeros_like(x)

    if not in_neighborhood(x, x_fp, l_unstable, r_neigh, nbhd_shape):
        return np.zeros_like(x)

    u_scalar = -np.dot(l_unstable, x - x_fp)
    r = lambda_u / (2 * D)

    return -high_precision_control_scalar(u_scalar, a=a, r=r, D=D, precision=25) * l_unstable



class LS_Controller:
    def __init__(
        self,
        control_vector,
        tracker=None,           # <- tracker instance
        C=None,
        target_proj=0.0,
        control_noise_std=0.0,
        t_start=0,
        window=200,
        u_max=None,
        ls_ridge=1e-6,
        gamma_v=0.0,
        radius=1.0,            # for adaptive control vector update
        apply_radius: float = 1.0,  # radius around x* to apply control
        fp_init_range: float = 2.0,
        rs: int | np.random.Generator = 42,
    ):
        v = np.asarray(control_vector, dtype=float)
        v_norm = np.linalg.norm(v)
        if v_norm == 0:
            raise ValueError("control_vector must be nonzero.")
        self.v = v / v_norm
        self.s_hist: list[float] = []
        self.u_hist: list[float] = []
        self.theta: Optional[np.ndarray] = None
        self.C = np.eye(len(control_vector)) if C is None else C
        self.ls_ridge = ls_ridge
        self.u_max = u_max
        self.window = window
        self.t_start = t_start
        self.target_proj = target_proj
        self.control_noise_std = control_noise_std
        self.fp_init_range = fp_init_range  

        self.gamma_v = gamma_v
        self.radius = radius
        self.apply_radius = apply_radius
        self.tracker = tracker  # reference to tracker for current x*
        self.rng = np.random.default_rng(rs)

    def _fit_ab(self) -> Optional[np.ndarray]:
        if len(self.u_hist) < 2:
            return None
        s = np.asarray(self.s_hist, dtype=float)
        u = np.asarray(self.u_hist, dtype=float)
        K = u.size
        X = np.stack([s[:K], u], axis=1)
        y = s[1:K+1]
        lam = self.ls_ridge
        XtX = X.T @ X + lam * np.eye(2)
        Xty = X.T @ y
        try:
            theta = np.linalg.solve(XtX, Xty)
        except np.linalg.LinAlgError:
            theta = np.linalg.pinv(XtX) @ Xty
        return theta

    def _update_control_vector(self, x: np.ndarray):
        if self.gamma_v > 0:
            v = self.v.reshape(-1, 1)
            x_in = x.reshape(-1, 1)
            v = v - self.gamma_v * (v.T @ x_in / self.radius**2) * x_in
            v /= np.linalg.norm(v)
            self.v = v.flatten()

    def __call__(self, t: float, x: np.ndarray, dt) -> np.ndarray:
        # Determine current fixed point
        if self.tracker is not None and len(self.tracker.xstar_history) > 0:
            x_star = self.tracker.xstar_history[-1]
        else:
            x_star = np.zeros_like(x)
            self.x_star_guess = self.rng.uniform(-self.fp_init_range, self.fp_init_range, size=len(self.control_vector))

        # Apply control only within radius around x*
        if np.linalg.norm(x - x_star) > self.apply_radius:
            return np.zeros_like(x)

        s = float(self.v @ x)

        if t < self.t_start or len(self.s_hist) == 0:
            u = np.random.normal(0.0, self.control_noise_std)
            self.s_hist.append(s)
            self.u_hist.append(u)
            self.s_hist.append(s)
            self._update_control_vector(x)
            return u * self.v

        theta = self._fit_ab()
        if theta is not None:
            self.theta = theta
        a, b = (self.theta if self.theta is not None else np.array([1.0, 1e-3]))
        if abs(b) < 1e-6:
            b = np.sign(b) * 1e-6 if b != 0.0 else 1e-6

        u = (self.target_proj - a * s) / b
        if self.u_max is not None:
            u = float(np.clip(u, -self.u_max, self.u_max))
        if self.control_noise_std > 0:
            u += np.random.normal(0.0, self.control_noise_std)

        # Update histories
        if len(self.s_hist) > 0 and len(self.u_hist) > 0:
            self.s_hist[-1] = s
        else:
            self.s_hist.append(s)
        self.u_hist.append(u)
        self.s_hist.append(s)

        # Update control vector
        self._update_control_vector(x)

        return u * self.v