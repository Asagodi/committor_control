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



# Trackers  
class NaiveVARTracker:
    def __init__(self, dim, eta_e=0.01, store_history=True):
        self.dim = dim
        self.eta_e = eta_e
        self.store_history = store_history

        # Data storage (optional — still keep points if you like)
        self.data = []

        # Explicit transitions (pairs inside same trajectory)
        self.pairs = []  # list of (x_prev, x_curr)

        # Helper to know last point of the *current* trajectory
        self._last_point = None

        # Model parameters
        self.A = np.eye(dim)
        self.b = np.zeros(dim)
        self.residual_cov = np.eye(dim)

        # Eigen-tracking
        v = np.random.randn(dim)
        v /= np.linalg.norm(v)
        self.l_unstable = v.copy()
        self.l_stable = v.copy()
        self.lambda_t = 0.0

        # Noise strength (EMA of squared error norm)
        self.noise_strength = 0.0

        # Histories
        if store_history:
            self.pred_history = []
            self.pred_error_history = []
            self.A_history = []
            self.b_history = []
            self.noise_history = []
            self.residual_cov_history = []
            self.lu_history = []
            self.ls_history = []
            self.lambda_history = []

    def update(self, x_t, new_trajectory=False):
        """
        Add a new observation.
        If new_trajectory is True, this point is considered the start of a new trajectory
        (no pair is formed with the previous global point).
        """
        x_t = np.asarray(x_t).ravel()
        self.data.append(x_t.copy())

        if new_trajectory:
            # mark start of a new trajectory: don't create a pair
            self._last_point = x_t.copy()
            return

        if self._last_point is None:
            # no previous point available (first ever point) — treat as start
            self._last_point = x_t.copy()
            return

        # otherwise form a pair inside the same trajectory
        self.pairs.append((self._last_point.copy(), x_t.copy()))
        self._last_point = x_t.copy()

        # need at least 1 pair to estimate
        if len(self.pairs) < 1:
            return

        X_p = np.vstack([p[0] for p in self.pairs])  # shape (n_pairs, dim)
        X_f = np.vstack([p[1] for p in self.pairs])  # shape (n_pairs, dim)
        n = X_p.shape[0]

        # Means
        mean_p = X_p.mean(axis=0)
        mean_f = X_f.mean(axis=0)

        # Covariances 
        X_p0 = X_p - mean_p
        X_f0 = X_f - mean_f
        C_pp = (X_p0.T @ X_p0) #/ n
        C_fp = (X_f0.T @ X_p0) #/ n

        # VAR parameters
        A_hat = C_fp @ np.linalg.pinv(C_pp)
        b_hat = mean_f - A_hat @ mean_p

        # Residuals and residual covariance
        pred = X_p @ A_hat.T + b_hat  # (n, dim)
        residuals = X_f - pred
        residual_cov = (residuals.T @ residuals) / n
        mse = float(np.mean(np.sum(residuals**2, axis=1)))

        # Update state
        self.A = A_hat
        self.b = b_hat
        self.residual_cov = residual_cov
        self.noise_strength = mse

        # Store histories
        if self.store_history:
            self.A_history.append(A_hat.copy())
            self.b_history.append(b_hat.copy())
            self.pred_history.append(pred[-1].copy())
            self.pred_error_history.append(np.sqrt(mse))
            self.noise_history.append(self.noise_strength)
            self.residual_cov_history.append(residual_cov.copy())

        # Update eigen-tracking
        self._update_left_eigenvectors()

    def _update_left_eigenvectors(self, n_iter=1, tol=1e-12):
        if self.A is None:
            return
        A_t = self.A
        v = self.l_unstable.copy()
        for _ in range(n_iter):
            v_new = A_t.T @ v
            norm_v = np.linalg.norm(v_new)
            if norm_v < tol:
                break
            v_new /= norm_v
            if np.linalg.norm(v_new - v) < tol:
                v = v_new
                break
            v = v_new
        self.l_unstable = v
        Av = A_t @ v
        den = v @ v
        self.lambda_t = (v @ Av) / den if den > tol else 0.0
        # alternative:
        self.lambda_t = np.linalg.norm(Av)
        if abs(self.lambda_t) > tol:
            self.l_stable = v / self.lambda_t
        if self.store_history:
            self.lu_history.append(self.l_unstable.copy())
            self.ls_history.append(self.l_stable.copy())
            self.lambda_history.append(self.lambda_t)

    def train(self, drift_func, n_trajs=200, T=100, dt=0.01, D=0.1,
              x_shift=0.0, y_shift=0.0, seed=20, init_bounds=0.1):
        """Simulate trajectories and update parameters, signalling new trajectories correctly."""
        def D_to_noise_std(D):
            return np.sqrt(2 * D)

        noise_std = D_to_noise_std(D)
        rng = np.random.default_rng(seed)

        # Initialize trajectories
        initial_positions = [
            np.array([x_shift, y_shift]) + rng.uniform(-init_bounds, init_bounds, size=(self.dim,))
            for _ in range(n_trajs)
        ]

        all_X_uncontrolled = []
        for x0 in initial_positions:
            X = np.zeros((T, self.dim))
            X[0] = x0
            # Start new trajectory so tracker won't pair x0 with previous trajectory's last point
            self.update(X[0], new_trajectory=True)
            for t in range(1, T):
                dx = drift_func(t, X[t-1])
                X[t] = X[t-1] + dx * dt + noise_std * rng.standard_normal(self.dim) * np.sqrt(dt)
                self.update(X[t])  # pairs formed only within this trajectory
            all_X_uncontrolled.append(X)

        return all_X_uncontrolled

    # Utilities
    def predict(self, x_prev):
        return self.A @ x_prev + self.b

    def get_latest_params(self):
        return self.A, self.b, self.residual_cov


## Neighborhood based VAR tracker with fixed point tracking
class NbhdVARTracker:
    """
    Neighborhood VAR tracker (batch / offline version).

    Offline training:
      1) generate n_trajs trajectories
      2) build all past-future pairs
      3) starting from x_star (x_star_init or random within [-x_star_init_range, x_star_init_range]^dim), iterate n_iterations times:
           - select pairs within radius of x_star
           - compute local A_hat, b_hat, residuals
           - update x_star via x_star = (I - A)^-1 (mean_f - A mean_p)
      Histories are recorded for all iterations, with **initial values as the first entry**.
    """

    def __init__(self, dim,
                 store_history=True,
                 x_star_init=None,
                 x_star_init_range=None,
                 radius=None,
                 rng_seed=None):
        self.dim = int(dim)
        self.store_history = bool(store_history)

        # Model state
        self.A = np.eye(self.dim, dtype=float)
        self.b = np.zeros(self.dim, dtype=float)
        self.residual_cov = np.eye(self.dim, dtype=float)
        self.noise_strength = np.random.rand()*0.00001

        # Random generator
        self.rng = np.random.default_rng(rng_seed)

        # Fixed point / neighborhood
        self.radius = None if radius is None else float(radius)
        if x_star_init is None:
            if self.radius is None:
                self.x_star = np.zeros(self.dim, dtype=float)
            else:
                if x_star_init_range is None:
                    x_star_init_range = self.radius
                self.x_star = self.rng.uniform(-x_star_init_range, x_star_init_range, size=self.dim)
        else:
            self.x_star = np.asarray(x_star_init, dtype=float).ravel()

        # Pair storage
        self.pairs = []

        # Eigen-tracking
        v = self.rng.standard_normal(self.dim)
        v /= (np.linalg.norm(v) + 1e-12)
        self.l_unstable = v.copy()
        self.l_stable = v.copy()
        self.lambda_t = 0.

        # Histories (initial values included)
        if self.store_history:
            self.A_history = [self.A.copy()]
            self.b_history = [self.b.copy()]
            self.residual_cov_history = [self.residual_cov.copy()]
            self.noise_history = [self.noise_strength]

            self.lu_history = [self.l_unstable.copy()]
            self.ls_history = [self.l_stable.copy()]
            self.lambda_history = [self.lambda_t]
            self.xstar_history = [self.x_star.copy()]
            self.nlocal_history = [0]

    # -----------------------------
    # Build pairs from trajectories
    # -----------------------------
    def _compute_all_pairs_from_trajectories(self, all_X):
        self.pairs = []
        for X in all_X:
            X = np.asarray(X)
            for t in range(X.shape[0] - 1):
                self.pairs.append((X[t].copy(), X[t+1].copy()))

    # -----------------------------
    # Estimate VAR from local pairs
    # -----------------------------
    def _estimate_VAR_from_pairs(self, local_pairs):
        X_p = np.vstack([p[0] for p in local_pairs])
        X_f = np.vstack([p[1] for p in local_pairs])
        n = X_p.shape[0]

        mean_p = X_p.mean(axis=0)
        mean_f = X_f.mean(axis=0)

        X_p0 = X_p - mean_p
        X_f0 = X_f - mean_f

        C_pp = X_p0.T @ X_p0
        C_fp = X_f0.T @ X_p0

        A_hat = C_fp @ np.linalg.pinv(C_pp)
        b_hat = mean_f - A_hat @ mean_p

        pred = X_p @ A_hat.T + b_hat
        residuals = X_f - pred
        residual_cov = (residuals.T @ residuals) / max(n, 1)
        mse = float(np.mean(np.sum(residuals**2, axis=1)))

        return A_hat, b_hat, residual_cov, mse, mean_p, mean_f

    # -----------------------------
    # Left-eigenvector update
    # -----------------------------
    def _update_left_eigenvectors(self, A, n_iter=20, tol=1e-12):
        v = self.l_unstable.copy()
        for _ in range(n_iter):
            v_new = A.T @ v
            nrm = np.linalg.norm(v_new)
            if nrm <= tol:
                break
            v_new /= nrm
            if np.linalg.norm(v_new - v) < tol:
                v = v_new
                break
            v = v_new

        self.l_unstable = v.copy()
        Av = A @ v
        den = max(v @ v, 1e-12)
        lambda_est = float((v @ Av) / den)
        self.lambda_t = lambda_est
        self.l_stable = v / (lambda_est if abs(lambda_est) > 1e-12 else 1.0)

        if self.store_history:
            self.lu_history.append(self.l_unstable.copy())
            self.ls_history.append(self.l_stable.copy())
            self.lambda_history.append(self.lambda_t)

    # -----------------------------
    # Offline training
    # -----------------------------
    def train(self, drift_func,
              n_trajs=200, T=100, dt=0.01, D=0.1,
              x_shift=0.0, y_shift=0.0, seed=0, init_bounds=0.1,
              n_iterations=15, min_pairs=None,
              return_trajectories=False):
        def noise_std(D):
            return np.sqrt(2.0 * D)

        rng = np.random.default_rng(seed)
        all_X = []

        # generate trajectories
        for _ in range(int(n_trajs)):
            X = np.zeros((int(T), self.dim), dtype=float)
            X[0] = np.array([x_shift, y_shift]) + rng.uniform(-init_bounds, init_bounds, size=self.dim)
            for t in range(1, int(T)):
                dx = drift_func(t, X[t-1])
                X[t] = X[t-1] + dx * dt + noise_std(D) * rng.standard_normal(self.dim) * np.sqrt(dt)
            all_X.append(X)

        # build all pairs
        self._compute_all_pairs_from_trajectories(all_X)

        if min_pairs is None:
            min_pairs = max(self.dim + 1, 5)

        # global fit if radius is None
        if self.radius is None:
            A_hat, b_hat, resid_cov, mse, _, _ = self._estimate_VAR_from_pairs(self.pairs)
            self.A, self.b, self.residual_cov = A_hat, b_hat, resid_cov
            self.noise_strength = mse
            self._update_left_eigenvectors(self.A)
            if self.store_history:
                self.A_history.append(self.A.copy())
                self.b_history.append(self.b.copy())
                self.residual_cov_history.append(self.residual_cov.copy())
                self.noise_history.append(self.noise_strength)
                self.xstar_history.append(self.x_star.copy())
                self.nlocal_history.append(len(self.pairs))
            return all_X if return_trajectories else None

        # iterative local-fitting loop (ignore x_star convergence)
        A_hat = self.A.copy()
        b_hat = self.b.copy()
        resid_cov = self.residual_cov.copy()
        mse = self.noise_strength

        for it in range(int(n_iterations)):
            # select local pairs
            local_pairs = [(xp.copy(), xf.copy()) for (xp, xf) in self.pairs
                           if np.linalg.norm(xp - self.x_star) <= self.radius]
            n_local = len(local_pairs)
            if self.store_history:
                self.nlocal_history.append(n_local)

            if n_local < min_pairs:
                # not enough pairs, record history and skip estimation
                if self.store_history:
                    self.xstar_history.append(self.x_star.copy())
                    self.A_history.append(A_hat.copy())
                    self.b_history.append(b_hat.copy())
                    self.residual_cov_history.append(resid_cov.copy())
                    self.noise_history.append(mse)
                    self._update_left_eigenvectors(A_hat)
                continue

            # estimate local VAR
            A_hat, b_hat, resid_cov, mse, mean_p, mean_f = self._estimate_VAR_from_pairs(local_pairs)

            # update x_star
            I_minus_A = np.eye(self.dim) - A_hat
            try:
                x_star_new = np.linalg.solve(I_minus_A, mean_f - A_hat @ mean_p)
            except np.linalg.LinAlgError:
                x_star_new = np.linalg.pinv(I_minus_A) @ (mean_f - A_hat @ mean_p)
            
            self.x_star = x_star_new.copy()

            # update eigenvectors and record history
            self._update_left_eigenvectors(A_hat)
            if self.store_history:
                self.A_history.append(A_hat.copy())
                self.b_history.append(b_hat.copy())
                self.residual_cov_history.append(resid_cov.copy())
                self.noise_history.append(mse)
                self.xstar_history.append(self.x_star.copy())

        # assign final parameters
        self.A, self.b, self.residual_cov, self.noise_strength = A_hat, b_hat, resid_cov, mse

        return all_X if return_trajectories else None

    # -----------------------------
    # Predict / getters
    # -----------------------------
    def predict(self, x_prev):
        x_prev = np.asarray(x_prev).ravel()
        return self.A @ x_prev + self.b

    def get_latest_params(self):
        return self.A.copy(), self.b.copy(), self.residual_cov.copy()



    
    # def enable_fixed_point_tracking(self, x_star_init, radius):
    #     """
    #     Enable fixed point tracking with given initial estimate and radius.
    #     After calling this, fixed point will be updated with each new observation.
        
    #     Parameters:
    #     -----------
    #     x_star_init : array-like
    #         Initial guess for the fixed point
    #     radius : float
    #         Fixed radius for the neighborhood around the fixed point
    #     """
    #     self.x_star = np.asarray(x_star_init).ravel().copy()
    #     self.radius = radius
    #     if self.store_history:
    #         self.xstar_history = [self.x_star.copy()]

    # def find_fixed_point_iterative(self, x_star_init, radius, max_iters=10, 
    #                                min_pairs=20, tol=1e-6, verbose=True):
    #     """
    #     Iteratively find the fixed point (saddle node) using a fixed neighborhood radius.
        
    #     Parameters:
    #     -----------
    #     x_star_init : array-like
    #         Initial guess for the fixed point
    #     radius : float
    #         Fixed radius for the neighborhood around the fixed point
    #     max_iters : int
    #         Maximum number of iterations
    #     min_pairs : int
    #         Minimum number of pairs required for estimation
    #     tol : float
    #         Convergence tolerance (change in fixed point estimate)
    #     verbose : bool
    #         Print iteration details
            
    #     Returns:
    #     --------
    #     x_star : array
    #         Estimated fixed point
    #     A_local : array
    #         Local linear approximation matrix A
    #     b_local : array
    #         Local linear approximation offset b
    #     history : dict
    #         Dictionary containing iteration history
    #     """
    #     x_star = np.asarray(x_star_init).ravel().copy()
    #     history = {
    #         'x_star': [x_star.copy()],
    #         'A': [],
    #         'b': [],
    #         'n_pairs': [],
    #         'residual_norm': []
    #     }
        
    #     for iteration in range(max_iters):
    #         # Crop pairs to those within the fixed radius of current x_star
    #         local_pairs = []
    #         for x_prev, x_curr in self.pairs:
    #             # Include pair if x_prev is within radius of x_star
    #             if np.linalg.norm(x_prev - x_star) <= radius:
    #                 local_pairs.append((x_prev, x_curr))
            
    #         n_local = len(local_pairs)
            
    #         if n_local < min_pairs:
    #             if verbose:
    #                 print(f"Iteration {iteration}: Only {n_local} pairs found (< {min_pairs}). Stopping.")
    #             break
            
    #         # Compute local VAR parameters
    #         X_p = np.vstack([p[0] for p in local_pairs])
    #         X_f = np.vstack([p[1] for p in local_pairs])
            
    #         mean_p = X_p.mean(axis=0)
    #         mean_f = X_f.mean(axis=0)
            
    #         X_p0 = X_p - mean_p
    #         X_f0 = X_f - mean_f
    #         C_pp = X_p0.T @ X_p0
    #         C_fp = X_f0.T @ X_p0
            
    #         A_local = C_fp @ np.linalg.pinv(C_pp)
    #         b_local = mean_f - A_local @ mean_p
            
    #         # Update fixed point estimate: solve (I - A) x_star = b
    #         # x_star = (I - A)^{-1} b
    #         I_minus_A = np.eye(self.dim) - A_local
    #         try:
    #             x_star_new = np.linalg.solve(I_minus_A, b_local)
    #         except np.linalg.LinAlgError:
    #             if verbose:
    #                 print(f"Iteration {iteration}: Singular matrix (I - A). Using pseudoinverse.")
    #             x_star_new = np.linalg.pinv(I_minus_A) @ b_local
            
    #         # Compute residual
    #         pred = X_p @ A_local.T + b_local
    #         residuals = X_f - pred
    #         residual_norm = np.sqrt(np.mean(np.sum(residuals**2, axis=1)))
            
    #         # Store history
    #         history['x_star'].append(x_star_new.copy())
    #         history['A'].append(A_local.copy())
    #         history['b'].append(b_local.copy())
    #         history['n_pairs'].append(n_local)
    #         history['residual_norm'].append(residual_norm)
            
    #         # Check convergence
    #         delta = np.linalg.norm(x_star_new - x_star)
            
    #         if verbose:
    #             print(f"Iteration {iteration}: n_pairs={n_local}, "
    #                   f"delta={delta:.6e}, residual={residual_norm:.6e}, "
    #                   f"x_star={x_star_new}")
            
    #         if delta < tol:
    #             if verbose:
    #                 print(f"Converged at iteration {iteration}!")
    #             x_star = x_star_new
    #             break
            
    #         x_star = x_star_new
        
    #     return x_star, A_local, b_local, history



class OfflineVARTrackerBatched:
    """
    Offline least-squares VAR(1) estimator.
    Estimates A, b, eigenvectors, eigenvalues, and residual covariance
    using closed-form sample covariances on groups of trajectories.
    """

    def __init__(self, dim, eta_e=0.01, store_history=True):
        self.dim = int(dim)
        self.eta_e = float(eta_e)
        self.store_history = bool(store_history)

        # Parameters / state
        self.mean = np.zeros(2 * self.dim)     # joint mean [x_t; x_{t-1}]
        self.cov = np.eye(2 * self.dim)        # joint covariance

        # Eigen-tracking state
        v = np.random.randn(self.dim)
        v /= np.linalg.norm(v)
        self.l_unstable = v.copy()
        self.l_stable = v.copy()
        self.lambda_t = 0.0

        # Noise strength (EMA of squared error norm)
        self.noise_strength = 0.0

        # Histories
        if self.store_history:
            self.pred_error_history = []
            self.pred_history = []
            self.A_history = []
            self.b_history = []
            self.lu_history = []
            self.ls_history = []
            self.lambda_history = []
            self.noise_history = []
            self.residual_cov_history = []

    def offline_train(self, all_X, n_traj_cov=50):
        n_trajs = len(all_X)
        n_groups = n_trajs // n_traj_cov
        if n_groups == 0:
            raise ValueError("Not enough trajectories for one group")

        A_list, b_list = [], []

        for g in range(n_groups):
            group = all_X[g * n_traj_cov:(g + 1) * n_traj_cov]

            X_p, X_f = [], []
            for X in group:
                X_p.append(X[:-1])
                X_f.append(X[1:])
            X_p = np.vstack(X_p)
            X_f = np.vstack(X_f)
            N = X_p.shape[0]

            # Means
            mean_p = X_p.mean(axis=0)
            mean_f = X_f.mean(axis=0)

            # Covariances
            X_p0 = X_p - mean_p
            X_f0 = X_f - mean_f
            C_pp = (X_p0.T @ X_p0) / N
            C_fp = (X_f0.T @ X_p0) / N

            # Closed-form estimates
            A_hat = C_fp @ np.linalg.pinv(C_pp)
            b_hat = mean_f - A_hat @ mean_p

            # Residuals
            pred = X_p @ A_hat.T + b_hat
            residuals = X_f - pred
            residual_cov = (residuals.T @ residuals) / N
            mse = float(np.mean(np.sum(residuals ** 2, axis=1)))

            # Joint mean/covariance for [x_t; x_{t-1}]
            mean_joint = np.concatenate([mean_f, mean_p])
            Z0 = np.hstack([X_f0, X_p0])
            cov_joint = (Z0.T @ Z0) / N
            self.mean = mean_joint
            self.cov = cov_joint

            # Update noise estimate
            self.noise_strength = (1 - self.eta_e) * self.noise_strength + self.eta_e * mse

            # Store
            A_list.append(A_hat)
            b_list.append(b_hat)
            if self.store_history:
                self.A_history.append(A_hat.copy())
                self.b_history.append(b_hat.copy())
                self.pred_history.append(pred.mean(axis=0).copy())
                self.pred_error_history.append(np.sqrt(mse))
                self.noise_history.append(self.noise_strength)
                self.residual_cov_history.append(residual_cov.copy())

            # Update eigen-tracking
            self._update_left_eigenvectors()

        return A_list, b_list

    def _update_left_eigenvectors(self, n_iter=20, tol=1e-12):
        if not self.A_history:
            return
        A_t = self.A_history[-1]
        v = self.l_unstable.copy()
        for _ in range(n_iter):
            v_new = A_t.T @ v
            norm_v = np.linalg.norm(v_new)
            if norm_v < tol:
                break
            v_new /= norm_v
            if np.linalg.norm(v_new - v) < tol:
                v = v_new
                break
            v = v_new
        self.l_unstable = v
        Av = A_t @ v
        den = v @ v
        self.lambda_t = (v @ Av) / den if den > tol else 0.0
        if abs(self.lambda_t) > tol:
            self.l_stable = v / self.lambda_t
        if self.store_history:
            self.lu_history.append(self.l_unstable.copy())
            self.ls_history.append(self.l_stable.copy())
            self.lambda_history.append(self.lambda_t)

    def train(self, drift_func, n_trajs=200, T=100, dt=0.01, D=0.1,
              x_shift=0.0, y_shift=0.0, seed=20, n_traj_cov=50, init_bounds=0.1):
        def D_to_noise_std(D):
            return np.sqrt(2 * D)

        noise_std = D_to_noise_std(D)
        rng = np.random.default_rng(seed)

        # initialize trajectories
        initial_positions = [
            np.array([x_shift, y_shift]) + rng.uniform(-init_bounds, init_bounds, size=(self.dim,))
            for _ in range(n_trajs)
        ]

        all_X_uncontrolled = []
        for x0 in initial_positions:
            X = np.zeros((T, self.dim))
            X[0] = x0
            for t in range(1, T):
                dx = drift_func(t, X[t-1])
                X[t] = X[t-1] + dx*dt + noise_std*rng.standard_normal(self.dim)*np.sqrt(dt)
            all_X_uncontrolled.append(X)

        # Offline VAR estimation in groups
        self.offline_train(all_X_uncontrolled, n_traj_cov=n_traj_cov)
        return all_X_uncontrolled

    # ---------------- utilities ----------------
    def predict_from_params(self, A, b, x_prev):
        return (A @ x_prev + b).ravel()

    def get_latest_A_b(self):
        if not self.A_history:
            return None, None
        return self.A_history[-1], self.b_history[-1]


class OnlineCovarianceTracker:
    def __init__(self, dim, eta_mean=0.05, eta_M=0.01, eta_w=0.01, eta_e=0.01, eta_v=1., n_iter_lu=10, store_history=True):
        self.dim = dim
        self.eta_mean = eta_mean
        self.eta_M = eta_M
        self.eta_w = eta_w
        self.eta_e = eta_e
        self.eta_v = eta_v
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

        #self.l_unstable = v
        self.l_unstable = (1 - self.eta_v) * self.l_unstable + self.eta_v * v

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

        # Update joint mean
        self.mean = g*self.mean + (1-g)*z
        z0 = z - self.mean

        # Update joint covariance
        self.cov = g*self.cov + (1-g)*np.outer(z0, z0)

        # Predict X_t using current estimate of A and b
        A_hat = self.estimate_A()
        mean_x_t = self.mean[:self.dim]
        mean_x_prev = self.mean[self.dim:]
        b_hat = mean_x_t - A_hat @ mean_x_prev
        x_pred = A_hat @ x_prev + b_hat

        # Prediction error
        e_t = (x_t - x_pred).ravel()

        # Update noise strength (EMA of squared error norm)
        new_val = float(np.dot(e_t, e_t))
        self.noise_strength = (1 - self.eta_e) * self.noise_strength + self.eta_e * new_val

        # Store histories
        if self.store_history:
            self.pred_history.append(x_pred.copy())
            self.pred_error_history.append(np.linalg.norm(e_t))
            self.A_history.append(A_hat)
            self.b_history.append(b_hat.copy())
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

        # Power iteration
        for _ in range(n_iter):
            v_new = A_t.T @ v
            norm_v = np.linalg.norm(v_new)
            if norm_v < tol:
                break
            v_new /= norm_v

            if np.linalg.norm(v_new - v) < tol:
                v = v_new
                break

            v = v_new

        self.l_unstable = v

        # Rayleigh quotient
        Av = A_t @ self.l_unstable
        num = self.l_unstable @ Av
        den = self.l_unstable @ self.l_unstable
        self.lambda_t = float(num / den)

        # Stable eigenvector
        if abs(self.lambda_t) > tol:
            self.l_stable = v / self.lambda_t

        # Store histories
        if self.store_history:
            self.lu_history.append(self.l_unstable.copy())
            self.ls_history.append(self.l_stable.copy())
            self.lambda_history.append(self.lambda_t)

    def train(self, drift_func, n_trajs=200, T=100, dt=0.01, D=0.1, x_shift=0.0, y_shift=0.0, seed=20, init_bounds=0.1):
        """
        Simulate trajectories under drift dynamics and update tracker online.

        Returns
        -------
        all_X_uncontrolled : list[np.ndarray]
            List of simulated trajectories (each shape (T, dim)).
        """
        def D_to_noise_std(D):
            return np.sqrt(2 * D)

        noise_std = D_to_noise_std(D)

        rng = np.random.default_rng(seed)
        initial_positions = [
            np.array([x_shift, y_shift]) + rng.uniform(-init_bounds, init_bounds, size=(self.dim,))
            for _ in range(n_trajs)
        ]

        all_X_uncontrolled = []
        for x0 in initial_positions:
            X = np.zeros((T, self.dim))
            X[0] = x0

            for t in range(1, T):
                dx = drift_func(t, X[t - 1])
                X[t] = (
                    X[t - 1] + dx * dt + noise_std * rng.standard_normal(self.dim) * np.sqrt(dt)
                )
                self.step(X[t], X[t - 1])

            all_X_uncontrolled.append(X)

        return all_X_uncontrolled





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
