import numpy as np

def estimate_fixed_point(A, X_p, X_f):
    """
    Estimate the fixed point b in a linear system x_{k+1} = A x_k + b
    using least squares:
        b = (I - A)^+ (x_t - A x_{t-1})
    
    Parameters
    ----------
    A : ndarray (d x d)
        Linear operator matrix
    X_p : ndarray (d x T-1)
        Past states: [x_0, ..., x_{T-1}]
    X_f : ndarray (d x T-1)
        Future states: [x_1, ..., x_T]

    Returns
    -------
    b : ndarray (d,)
        Estimated fixed point
    """
    # Compute the difference for each timestep
    dX = X_f - A @ X_p  # shape: (d, T-1)

    # Average over all timesteps (least squares)
    mean_dX = np.mean(dX, axis=1)  # shape: (d,)

    # Compute pseudoinverse of (I - A)
    I = np.eye(A.shape[0])
    b = np.linalg.pinv(I - A) @ mean_dX

    return b


def sliding_window_fp(X, window_size=50):
    """
    Estimate A_t and b_t in a sliding window over trajectory X.
    
    X: (T, d) trajectory
    window_size: number of steps in the sliding window
    
    Returns:
        A_list: list of estimated A matrices
        b_list: list of estimated b vectors
    """
    T, d = X.shape
    A_list = []
    b_list = []

    for t0 in range(T - window_size):
        X_p = X[t0 : t0 + window_size - 1].T  # past states (d, W-1)
        X_f = X[t0 + 1 : t0 + window_size].T  # future states (d, W-1)

        # Estimate A
        A = X_f @ X_p.T @ np.linalg.pinv(X_p @ X_p.T)

        # Estimate b (fixed point)
        dX = X_f - A @ X_p
        mean_dX = np.mean(dX, axis=1)
        I = np.eye(d)
        b = np.linalg.pinv(I - A) @ mean_dX

        A_list.append(A)
        b_list.append(b)

    return np.array(A_list), np.array(b_list)


def ema_update_and_residuals(mean_prev, x_prev, x_new, eta):
    """
    Exponential Moving Average (EMA) update with residuals.

    Args:
        mean_prev : mean before assimilating x_{t-1} (i.e. bar{x}_{t-2})
        x_prev    : sample x_{t-1}
        x_new     : sample x_t
        eta       : smoothing factor (0 < eta <= 1)

    Returns:
        mean_after_prev : mean after assimilating x_{t-1}
        resid_prev_after: residual of x_{t-1} w.r.t. mean_after_prev
        mean_after_new  : mean after assimilating x_t
        resid_new_after : residual of x_t w.r.t. mean_after_new
    """
    # assimilate x_{t-1}
    mean_after_prev = (1 - eta) * mean_prev + eta * x_prev
    resid_prev_after = x_prev - mean_after_prev  # = (1-eta)*(x_prev - mean_prev)

    # assimilate x_t
    mean_after_new = (1 - eta) * mean_after_prev + eta * x_new
    resid_new_after = x_new - mean_after_new     # = (1-eta)*(x_new - mean_after_prev)

    return mean_after_prev, resid_prev_after, mean_after_new, resid_new_after


def hebbian_update_Mfp_Mpf(
    Mfp_prev, Mpf_prev, 
    p_mean_prev, # f_mean_prev
    f_t, p_t, 
    eta_mean, eta_M
):
    """
    Update EMA means and Hebbian cross-covariances Mfp, Mpf.

    Args:
        Mfp_prev (np.ndarray): previous Mfp (F, P)
        Mpf_prev (np.ndarray): previous Mpf (P, F)
        f_mean_prev (np.ndarray): EMA mean of f before update (F,)
        p_mean_prev (np.ndarray): EMA mean of p before update (P,)
        f_t (np.ndarray): current f activity (F,)
        p_t (np.ndarray): current p activity (P,)
        eta_mean (float): EMA step size for means
        eta_M (float): EMA step size for covariances

    Returns:
        Mfp (np.ndarray): updated Mfp (F, P)
        Mpf (np.ndarray): updated Mpf (P, F)
        f_mean (np.ndarray): updated EMA mean of f
        p_mean (np.ndarray): updated EMA mean of p
        df (np.ndarray): residual of f_t wrt updated mean
        dp (np.ndarray): residual of p_t wrt updated mean
    """
    # update EMA means
    #f_mean = (1 - eta_mean) * f_mean_prev + eta_mean * f_t
    p_mean = (1 - eta_mean) * p_mean_prev + eta_mean * p_t

    # residuals wrt updated means (as in Yang)
    df = f_t - p_mean # f_mean?
    dp = p_t - p_mean

    # Hebbian updates for cross-covariances
    Mfp = (1 - eta_M) * Mfp_prev + eta_M * np.outer(df, dp)
    Mpf = (1 - eta_M) * Mpf_prev + eta_M * np.outer(dp, df)

    return Mfp, Mpf, p_mean, df, dp # f_mean



def update_Wpp(Wpp_prev, p_mean_prev, p_t, eta_mean, eta_w):
    """
    Update EMA mean of p and approximate inverse covariance W_pp
    using Sherman–Morrison formula in the y_t form.

    Args:
        Wpp_prev     (np.ndarray): previous W_pp, shape (P, P)
        p_mean_prev  (np.ndarray): previous EMA mean of p, shape (P,)
        p_t          (np.ndarray): current p activity, shape (P,)
        eta_mean     (float): EMA step size for the mean
        eta_w        (float): EMA step size for W_pp

    Returns:
        Wpp         (np.ndarray): updated W_pp
        p_mean      (np.ndarray): updated EMA mean of p
        dp          (np.ndarray): residual p_t - p_mean
        y_t         (np.ndarray): W_pp @ dp
    """
    # 1) Update EMA mean of p
    p_mean = (1 - eta_mean) * p_mean_prev + eta_mean * p_t

    # 2) Compute residual
    dp = p_t - p_mean

    # 3) Compute y_t
    y_t = Wpp_prev @ dp

    # 4) Update W_pp using Sherman-Morrison in y_t form
    dpy = dp.T @ y_t  # scalar
    Wpp = (1 / (1 - eta_w)) * Wpp_prev - (eta_w / (1 - eta_w)) / (1 - eta_w + eta_w * dpy) * np.outer(y_t, y_t)

    return Wpp, p_mean, dp, y_t



def online_fixed_point_step(b, x_t, x_prev, Mfp, Mpf, Wpp):
    """
    One step of online fixed-point inference using the learned recurrent circuit.

    Args:
        b      (np.ndarray): current prediction of p (shape P,)
        x_t    (np.ndarray): current observation (shape P,)
        x_prev (np.ndarray): previous observation (shape P,)
        Mfp    (np.ndarray): learned cross-covariance M_fp (F, P)
        Mpf    (np.ndarray): learned cross-covariance M_pf (P, F)
        Wpp    (np.ndarray): approximate inverse covariance (P, P)

    Returns:
        b_new (np.ndarray): updated prediction b
        et    (np.ndarray): error vector
    """
    # 1) residual from previous b
    ct = x_prev - b

    # 2) decorrelated residual
    dt = Wpp @ ct

    # 3) prediction error
    et = (x_t - b) - Mfp @ dt

    # 4) gradient w.r.t. b
    gt = - Mpf @ et

    # 5) delta update
    delta_b = et + Wpp @ gt

    # 6) update b
    b_new = b + delta_b

    return b_new, et


def linear_dynamics_tracker(
    x_prev, x_t, b, 
    Mfp, Mpf, Wpp, f_mean, p_mean,
    eta_mean=0.05, eta_M=0.01, eta_w=0.01
):
    """
    Online tracking of linearized dynamics x = Ax + b
    with learned recurrent circuit (Mfp, Mpf, Wpp).
    """
    # 1) Update Hebbian cross-covariances
    Mfp_new, Mpf_new, f_mean_new, p_mean_new, df, dp = hebbian_update_Mfp_Mpf(
        Mfp, Mpf, p_mean, x_prev, x_t, eta_mean, eta_M
    )

    # 2) Update Wpp
    Wpp_new, p_mean_new, dp, y_t = update_Wpp(Wpp, p_mean_new, x_t, eta_mean, eta_w)

    # 3) Update b using the fixed-point step function
    b_new, et = online_fixed_point_step(b, x_t, x_prev, Mfp_new, Mpf_new, Wpp_new)

    return b_new, Mfp_new, Mpf_new, Wpp_new, f_mean_new, p_mean_new, dp, df, et


class OnlineCovarianceTracker:
    def __init__(self, dim, eta_mean=0.05, eta_M=0.01, eta_w=0.01, eta_e=0.01, n_iter_lu=10, store_history=True):
        self.dim = dim
        self.eta_mean = eta_mean
        self.eta_M = eta_M
        self.eta_w = eta_w
        self.eta_e = eta_e
        self.n_iter_lu = n_iter_lu

        self.b = np.zeros(dim)
        self.Mfp = np.zeros((dim, dim))
        self.Mpf = np.zeros((dim, dim))
        self.Wpp = np.eye(dim)
        self.p_mean = np.zeros(dim)
        self.f_mean = np.zeros(dim)

        # Eigen-tracking state
        self.l_unstable = np.random.randn(dim)
        self.l_unstable /= np.linalg.norm(self.l_unstable)
        self.l_stable = self.l_unstable.copy()
        self.lambda_t = np.random.randn(1)

        # Noise strength
        self.noise_strength = 0.0

        self.store_history = store_history
        if store_history:
            self.b_history = []
            self.dp_history = []
            self.df_history = []
            self.et_history = []
            self.y_history = []
            self.xstar_history = []
            self.A_history = []
            self.lu_history = []
            self.ls_history = []
            self.lambda_history = []
            self.noise_history = []

            # Add plotting histories like ExponentialCovarianceTracker
            self.pred_history = []
            self.pred_error_history = []

    def _update_left_eigenvectors(self, n_iter=10, tol=1e-12):
        A_t = self.Mfp @ self.Wpp

        v = self.l_unstable.copy()

        # Power iteration with multiple steps for stability
        for _ in range(n_iter):
            v_new = A_t.T @ v
            norm_v = np.linalg.norm(v_new)
            if norm_v < tol:
                break  # Avoid collapse
            v_new /= norm_v

            if np.linalg.norm(v_new - v) < tol:
                v = v_new
                break

            v = v_new

        self.l_unstable = v

        # Rayleigh quotient for eigenvalue
        Av = A_t @ self.l_unstable
        num = self.l_unstable @ Av
        den = self.l_unstable @ self.l_unstable
        self.lambda_t = float(num / den)

        # # Stable eigenvector (scaled)
        # if abs(self.lambda_t) > tol:
        #     self.l_stable = v / self.lambda_t

        # Store histories
        if self.store_history:
            self.lu_history.append(self.l_unstable.copy())
            #self.ls_history.append(self.l_stable.copy())
            self.lambda_history.append(self.lambda_t)

    def _update_noise_strength(self, et):
        new_val = float(np.dot(et, et))
        self.noise_strength = (1 - self.eta_e) * self.noise_strength + self.eta_e * new_val
        if self.store_history:
            self.noise_history.append(self.noise_strength)

    def predict(self, x_prev):
        """Predict next state using current Mfp and Wpp."""
        return (self.Mfp @ self.Wpp @ x_prev).ravel()

    def step(self, x_prev, x_t):
        self.Mfp, self.Mpf, self.p_mean, df, dp = hebbian_update_Mfp_Mpf(
            self.Mfp, self.Mpf, self.p_mean, x_prev, x_t,
            self.eta_mean, self.eta_M
        )

        self.Wpp, self.p_mean, dp, y_t = update_Wpp(
            self.Wpp, self.p_mean, x_t,
            self.eta_mean, self.eta_w
        )

        self.b, et = online_fixed_point_step(
            self.b, x_t, x_prev, self.Mfp, self.Mpf, self.Wpp
        )

        # Update eigen-tracking
        self._update_left_eigenvectors(n_iter=self.n_iter_lu)

        # Update noise strength
        self._update_noise_strength(et)

        # --- Store histories for plotting ---
        if self.store_history:
            self.b_history.append(self.b.copy())
            self.dp_history.append(dp.copy())
            self.df_history.append(df.copy())
            self.et_history.append(et.copy())
            self.y_history.append(y_t.copy())

            # Effective system matrix
            try:
                A_tilde = self.Mfp @ self.Wpp
                self.A_history.append(A_tilde)
                x_star = -np.linalg.inv(A_tilde) @ self.b
            except np.linalg.LinAlgError:
                self.A_history.append(np.full((self.dim, self.dim), np.nan))
                x_star = np.full_like(self.b, np.nan)
            self.xstar_history.append(x_star)

            # Predictions & errors like ExponentialCovarianceTracker
            x_pred = self.predict(x_prev)
            e_pred = x_t - x_pred
            self.pred_history.append(x_pred.copy())
            self.pred_error_history.append(np.linalg.norm(e_pred))

        return self.b




class ExponentialCovarianceTracker:
    def __init__(self, dim, gamma=0.95, eta_e=0.01, store_history=True):
        self.dim = dim
        self.gamma = gamma
        self.cov = np.eye(2*dim)        # joint covariance of [X_t; X_{t-1}]
        self.mean = np.zeros(2*dim)     # running mean of joint vector
        self.store_history = store_history

        # Eigen-tracking
        self.l_unstable = np.random.randn(dim)
        self.l_unstable /= np.linalg.norm(self.l_unstable)
        self.l_stable = self.l_unstable.copy()
        self.lambda_t = np.random.randn(1)

        # Noise strength
        self.eta_e = eta_e
        self.noise_strength = 0.0

        # Initialize histories
        if store_history:
            self.pred_error_history = []
            self.pred_history = []
            self.A_history = []
            self.b_history = []
            self.lu_history = []
            self.ls_history = []
            self.lambda_history = []
            self.noise_history = []

    def step(self, x_t, x_prev):
        x_t = np.asarray(x_t).reshape(-1)
        x_prev = np.asarray(x_prev).reshape(-1)
        z = np.concatenate([x_t, x_prev])
        g = self.gamma

        # Update mean
        self.mean = g*self.mean + (1-g)*z
        z0 = z - self.mean

        # Update joint covariance
        self.cov = g*self.cov + (1-g)*np.outer(z0, z0)

        # Predict X_t
        x_pred = self.predict(x_prev)

        # Prediction error
        e_t = (x_t - x_pred).ravel()

        # Update noise strength (EMA of squared error norm)
        new_val = float(np.dot(e_t, e_t))
        self.noise_strength = (1 - self.eta_e) * self.noise_strength + self.eta_e * new_val

        # Store histories
        if self.store_history:
            self.pred_history.append(x_pred.copy())
            self.pred_error_history.append(np.linalg.norm(e_t))
            self.A_history.append(self.estimate_A())
            self.b_history.append(self.mean[:self.dim].copy())
            self.noise_history.append(self.noise_strength)

        # Update left unstable eigenvector and eigenvalue
        self._update_left_eigenvectors()

    def predict(self, x_prev):
        """Predict X_t from X_{t-1} using block covariance"""
        d = self.dim
        Σ = self.cov
        Σ_xy = Σ[:d, d:]
        Σ_yy = Σ[d:, d:]
        x_prev = np.asarray(x_prev).reshape(-1)
        return (Σ_xy @ np.linalg.pinv(Σ_yy) @ x_prev).ravel()

    def estimate_A(self):
        """Return current linear map estimate"""
        d = self.dim
        Σ = self.cov
        Σ_xy = Σ[:d, d:]
        Σ_yy = Σ[d:, d:]
        return Σ_xy @ np.linalg.pinv(Σ_yy)

    def _update_left_eigenvectors(self, n_iter=10, tol=1e-12):
        """Update unstable eigenvector and its eigenvalue using Rayleigh quotient with multiple iterations"""
        A_t = self.estimate_A()
        v = self.l_unstable.copy()

        # Power iteration with multiple steps for stability
        for _ in range(n_iter):
            v_new = A_t.T @ v
            norm_v = np.linalg.norm(v_new)
            if norm_v < tol:
                break  # Avoid collapse
            v_new /= norm_v

            if np.linalg.norm(v_new - v) < tol:
                v = v_new
                break

            v = v_new

        self.l_unstable = v

        # Rayleigh quotient for eigenvalue
        Av = A_t @ self.l_unstable
        num = self.l_unstable @ Av
        den = self.l_unstable @ self.l_unstable
        self.lambda_t = float(num / den)

        # Stable eigenvector (scaled)
        if abs(self.lambda_t) > tol:
            self.l_stable = v / self.lambda_t

        # Store histories
        if self.store_history:
            self.lu_history.append(self.l_unstable.copy())
            self.ls_history.append(self.l_stable.copy())
            self.lambda_history.append(self.lambda_t)





#closed loop

class ClosedLoopExponentialCovarianceTracker:
    def __init__(self, dim, udim, gamma=0.95, eta_e=0.01, store_history=True):
        self.dim = dim       # state dimension
        self.udim = udim     # control dimension
        self.gamma = gamma
        self.cov = np.eye(dim + dim + udim)   # joint covariance [x_t ; x_{t-1} ; u_{t-1}]
        self.mean = np.zeros(dim + dim + udim)
        self.eta_e = eta_e
        self.noise_strength = 0.0

        # eigen-tracking
        self.l_unstable = np.random.randn(dim)
        self.l_unstable /= np.linalg.norm(self.l_unstable)
        self.lambda_t = 0.0

        self.store_history = store_history
        if store_history:
            self.A_history, self.B_history = [], []
            self.pred_history, self.pred_error_history = [], []
            self.noise_history, self.lu_history, self.lambda_history = [], [], []

    def step(self, x_t, x_prev, u_prev):
        x_t = np.asarray(x_t).reshape(-1)
        x_prev = np.asarray(x_prev).reshape(-1)
        u_prev = np.asarray(u_prev).reshape(-1)
        z = np.concatenate([x_t, x_prev, u_prev])
        g = self.gamma

        # mean + covariance update
        self.mean = g*self.mean + (1-g)*z
        z0 = z - self.mean
        self.cov = g*self.cov + (1-g)*np.outer(z0, z0)

        # predict
        x_pred = self.predict(x_prev, u_prev)
        e_t = (x_t - x_pred).ravel()

        # noise update
        new_val = float(np.dot(e_t, e_t))
        self.noise_strength = (1 - self.eta_e) * self.noise_strength + self.eta_e * new_val

        # eigen-tracking
        self._update_eigenvectors()

        if self.store_history:
            self.pred_history.append(x_pred.copy())
            self.pred_error_history.append(np.linalg.norm(e_t))
            self.A_history.append(self.estimate_A())
            self.B_history.append(self.estimate_B())
            self.noise_history.append(self.noise_strength)
            self.lu_history.append(self.l_unstable.copy())
            self.lambda_history.append(self.lambda_t)

    def estimate_A(self):
        d, u = self.dim, self.udim
        Σ = self.cov
        Σ_xx = Σ[:d, d:d+ d]
        Σ_uu = Σ[:d, d+d:d+d+u]
        Σ_prev = np.block([[Σ[d:d+d, d:d+d], Σ[d:d+d, d+d:d+d+u]],
                           [Σ[d+d:d+d+u, d:d+d], Σ[d+d:d+d+u, d+d:d+d+u]]])
        Σ_cross = np.hstack([Σ_xx, Σ_uu])
        return Σ_cross @ np.linalg.pinv(Σ_prev)[:, :d]  # extract A

    def estimate_B(self):
        d, u = self.dim, self.udim
        Σ = self.cov
        Σ_xx = Σ[:d, d:d+ d]
        Σ_uu = Σ[:d, d+d:d+d+u]
        Σ_prev = np.block([[Σ[d:d+d, d:d+d], Σ[d:d+d, d+d:d+d+u]],
                           [Σ[d+d:d+d+u, d:d+d], Σ[d+d:d+d+u, d+d:d+d+u]]])
        Σ_cross = np.hstack([Σ_xx, Σ_uu])
        return Σ_cross @ np.linalg.pinv(Σ_prev)[:, d:]  # extract B

    def predict(self, x_prev, u_prev):
        return self.estimate_A() @ x_prev + self.estimate_B() @ u_prev

    def _update_eigenvectors(self):
        A_hat = self.estimate_A()
        v = A_hat.T @ self.l_unstable
        norm_v = np.linalg.norm(v)
        if norm_v > 1e-12:
            self.l_unstable = v / norm_v
        num = self.l_unstable @ (A_hat @ self.l_unstable)
        den = self.l_unstable @ self.l_unstable
        self.lambda_t = float(num / den) if abs(den) > 1e-12 else 0.0



class ClosedLoopOnlineCovarianceTracker(OnlineCovarianceTracker):
    def step(self, x_prev, u_prev, x_t):
        # Update Hebbian estimates with control included
        xu_prev = np.concatenate([x_prev, u_prev])
        if not hasattr(self, "Mxu"):
            self.Mxu = np.zeros((self.dim, len(xu_prev)))  # joint regression weights

        # Hebbian update for Mxu
        self.Mxu = (1 - self.eta_M)*self.Mxu + self.eta_M*np.outer(x_t, xu_prev)

        # Predict
        x_pred = self.Mxu @ xu_prev
        e_t = x_t - x_pred

        # Noise update
        self._update_noise_strength(e_t)

        # Eigen update (on A block of Mxu)
        A_hat = self.Mxu[:, :self.dim]
        v = A_hat.T @ self.l_unstable
        norm_v = np.linalg.norm(v)
        if norm_v > 1e-12:
            self.l_unstable = v / norm_v
        num = self.l_unstable @ (A_hat @ self.l_unstable)
        den = self.l_unstable @ self.l_unstable
        self.lambda_t = float(num / den) if abs(den) > 1e-12 else 0.0

        if self.store_history:
            self.pred_history.append(x_pred.copy())
            self.pred_error_history.append(np.linalg.norm(e_t))
            self.A_history.append(A_hat.copy())
            self.b_history.append(self.Mxu[:, self.dim:].copy())
            self.lu_history.append(self.l_unstable.copy())
            self.lambda_history.append(self.lambda_t)

        return self.Mxu
