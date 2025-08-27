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



class LinearDynamicsTracker:
    def __init__(self, dim, eta_mean=0.05, eta_M=0.01, eta_w=0.01, store_history=True):
        self.dim = dim
        self.eta_mean = eta_mean
        self.eta_M = eta_M
        self.eta_w = eta_w

        self.b = np.zeros(dim)
        self.Mfp = np.zeros((dim, dim))
        self.Mpf = np.zeros((dim, dim))
        self.Wpp = np.eye(dim)
        self.p_mean = np.zeros(dim)
        self.f_mean = np.zeros(dim)

        # Eigen-tracking state
        self.l_unstable = np.random.randn(dim)  # init random
        self.l_unstable /= np.linalg.norm(self.l_unstable)
        self.l_stable = self.l_unstable.copy()
        self.lambda_t = 1.0

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
            self.At_history = []
            self.lu_history = []
            self.ls_history = []
            self.lambda_history = []
            self.noise_history = []

    def _update_left_eigenvectors(self):
        """Update unstable eigenvector, its eigenvalue, and stable eigenvector."""
        # Apply transformation
        v = self.l_unstable @ self.Mfp @ self.Wpp

        # Normalize to get next unstable eigenvector
        norm_v = np.linalg.norm(v)
        if norm_v > 1e-12:
            self.l_unstable = v / norm_v

        # Eigenvalue estimate
        self.lambda_t = 1.0 - float(self.l_unstable @ self.l_unstable)

        # Update stable eigenvector (scaled by lambda_t)
        if abs(self.lambda_t) > 1e-12:
            self.l_stable = v / self.lambda_t

        if self.store_history:
            self.lu_history.append(self.l_unstable.copy())
            self.ls_history.append(self.l_stable.copy())
            self.lambda_history.append(self.lambda_t)

    def _update_noise_strength(self, et):
        """Update noise strength as squared norm of prediction error."""
        self.noise_strength = float(np.dot(et, et))
        if self.store_history:
            self.noise_history.append(self.noise_strength)

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
        self._update_left_eigenvectors()

        # Update noise strength
        self._update_noise_strength(et)

        if self.store_history:
            self.b_history.append(self.b.copy())
            self.dp_history.append(dp.copy())
            self.df_history.append(df.copy())
            self.et_history.append(et.copy())
            self.y_history.append(y_t.copy())

            # Effective system matrix
            try:
                A_tilde = self.Mfp @ self.Wpp
                self.At_history.append(A_tilde)
                # tracker’s current fixed point
                x_star = -np.linalg.inv(A_tilde) @ self.b
            except np.linalg.LinAlgError:
                self.At_history.append(np.full((self.dim, self.dim), np.nan))
                x_star = np.full_like(self.b, np.nan)
            self.xstar_history.append(x_star)

        return self.b