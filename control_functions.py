import numpy as np
from typing import Callable, Tuple, Optional, Literal

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