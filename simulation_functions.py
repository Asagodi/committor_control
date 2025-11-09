import numpy as np
from typing import Callable, Tuple, Optional, Literal



def D_to_noise_std(D):
    """
    Convert diffusion coefficient D to per-step noise_std used in Euler-Maruyama.

    Parameters:
        D (float): Diffusion coefficient in the SDE (dX = f dt + sqrt(2D) dW)

    Returns:
        noise_std (float): Standard deviation to use in X[t] = X[t-1] + f*dt + noise_std*sqrt(dt)*randn()
    """
    return np.sqrt(2 * D)


def noise_std_to_D(noise_std, dim=2):
    """
    Convert per-step noise_std back to diffusion coefficient D.

    Parameters:
        noise_std (float): Standard deviation used in Euler-Maruyama step

    Returns:
        D (float): Diffusion coefficient in the SDE
    """
    return noise_std**2 / dim


def noise_strength_to_D(noise_strength, dt, dim=1, sde_convention="sqrt2D"):
    """
    Convert tracker noise_strength to diffusion coefficient D.

    Parameters
    ----------
    noise_strength : float
        Exponential moving average of squared noise increments (⟨et^2⟩).
    dt : float
        Simulation time step.
    dim : int, default=1
        Dimension of the system (length of et).
    sde_convention : str, default="sqrt2D"
        - "sqrt2D": et ~ sqrt(2 D dt) * xi  (standard SDE definition)
        - "sqrtD":  et ~ sqrt(D dt) * xi    (common simulation convention without factor 2)

    Returns
    -------
    D : float
        Estimated per-component diffusion coefficient.
    """
    if sde_convention == "sqrt2D":
        return noise_strength / (2 * dt * dim)
    elif sde_convention == "sqrtD":
        return noise_strength / (dt * dim)
    else:
        raise ValueError("sde_convention must be 'sqrt2D' or 'sqrtD'")


# def generate_initial_positions(n_traj, dim, seed=0):
#     np.random.seed(seed)
#     return [np.random.uniform(-0.5, 0.5, size=(dim,)) for _ in range(n_traj)]

def generate_noise(initial_positions, T, dim, noise_std, dt, seed=0):
    np.random.seed(seed)
    all_noise = []
    for _ in initial_positions:
        noise = np.random.randn(T, dim) * noise_std * np.sqrt(dt)
        all_noise.append(noise)
    return all_noise

def simulate_trajectory(x0, T, dim, dt, drift_func, noise, control_func=None):
    X = np.zeros((T, dim))
    X[0] = x0
    for t in range(1, T):
        dx = drift_func(t, X[t-1])
        u_t = control_func(X[t-1]) if control_func is not None else np.zeros(dim)
        X[t] = X[t-1] + dx*dt + u_t*dt + noise[t]
    return X

def generate_initial_conditions(
    n: int,
    bound: float,
    circle: bool = False,
    along_stable: Optional[np.ndarray] = None,
    angle_range: float = np.pi / 8,
    angle_distribution: str = "uniform",
) -> np.ndarray:
    """
    If circle=True: points on a circle of radius=bound.
    If along_stable is provided, put half near that direction and half opposite.
    Else: uniform in a square [-bound, bound]^2.
    """
    if circle:
        if along_stable is not None:
            along_stable = np.asarray(along_stable, dtype=float)
            along_stable_angle = np.arctan2(along_stable[1], along_stable[0])
            m = n // 2
            if angle_distribution == "normal":
                angles_1 = np.random.normal(along_stable_angle, angle_range, m)
                angles_2 = np.random.normal(along_stable_angle + np.pi, angle_range, m)
            else:
                angles_1 = np.random.uniform(along_stable_angle - angle_range,
                                             along_stable_angle + angle_range, m)
                angles_2 = np.random.uniform(along_stable_angle + np.pi - angle_range,
                                             along_stable_angle + np.pi + angle_range, m)
            angles = np.concatenate([angles_1, angles_2])
            if angles.size < n:  # odd n
                angles = np.concatenate([angles, np.array([along_stable_angle])])
            return np.stack([bound * np.cos(angles), bound * np.sin(angles)], axis=1)
        else:
            angles = np.random.uniform(0, 2 * np.pi, n)
            return np.stack([bound * np.cos(angles), bound * np.sin(angles)], axis=1)
    else:
        return np.random.uniform(-bound, bound, size=(n, 2))



def _sample_from_basin(
    basin: str, rng: np.random.Generator, spread: float = 0.2
) -> np.ndarray:
    if basin == "left":
        center = np.array([-1.0, 0.0])
    elif basin == "right":
        center = np.array([1.0, 0.0])
    else:  # random: choose left or right randomly
        center = np.array([-1.0, 0.0]) if rng.random() < 0.5 else np.array([1.0, 0.0])
    return center + rng.normal(scale=spread, size=2)



def stopping_condition(x: np.ndarray, bound: float, circle: bool = False) -> bool:
    """
    Stop when we exit the feasible region.
    - circle=True: outside a circle of radius 'bound'
    - circle=False: outside the square [-bound, bound]^d (ANY coord beyond bound)
    """
    if circle:
        return np.linalg.norm(x) >= bound
    else:
        return np.any(np.abs(x) >= bound)


# ------------------------------
# Euler–Maruyama simulator
# ------------------------------
def simulate_euler_maruyama(
    drift_func: Callable[[np.ndarray, float], np.ndarray],
    x0: np.ndarray,
    t_span: Tuple[float, float],
    dt: float,
    noise_std: float | np.ndarray = 0.0,
    *,
    bound: float = np.inf,
    circle: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Additive-noise EM:  x_{k+1} = x_k + f(x_k,t_k) dt + sigma * sqrt(dt) * N(0,I)
    """
    t0, t1 = t_span
    t_vals = np.arange(t0, t1 + 1e-12, dt)
    x = np.asarray(x0, dtype=float)
    d = x.size
    traj = np.zeros((t_vals.size, d), dtype=float)
    traj[0] = x

    noise_std = np.broadcast_to(np.asarray(noise_std, dtype=float), (d,))
    sqrt_dt = np.sqrt(dt)

    for i in range(1, t_vals.size):
        t = t_vals[i-1]
        drift = np.asarray(drift_func(t, x), dtype=float)
        noise = np.random.normal(size=d) * noise_std * sqrt_dt
        x = x + drift * dt + noise
        traj[i] = x
        if stopping_condition(x, bound=bound, circle=circle):
            traj = traj[: i]
            t_vals = t_vals[: i]
            break
    return t_vals, traj


# ------------------------------
# EM simulator with control
# ------------------------------
def simulate_euler_maruyama_control(
    drift_func: Callable[[np.ndarray, float], np.ndarray],
    control_func: Callable[[float, np.ndarray, float], np.ndarray],
    x0: np.ndarray,
    t_span: Tuple[float, float],
    dt: float,
    noise_std: float | np.ndarray = 0.0,
    *,
    bound: float = np.inf,
    circle: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    x_{k+1} = x_k + [f(x_k,t_k) + u_k] dt + sigma * sqrt(dt) * N(0,I)
    control_func(t_k, x_k, dt) -> vector control u_k (same shape as x)
    """
    t0, t1 = t_span
    t_vals = np.arange(t0, t1 + 1e-12, dt)
    x = np.asarray(x0, dtype=float)
    d = x.size
    traj = np.zeros((t_vals.size, d), dtype=float)
    ctrls = np.zeros((t_vals.size, d), dtype=float)
    traj[0] = x

    noise_std = np.broadcast_to(np.asarray(noise_std, dtype=float), (d,))
    sqrt_dt = np.sqrt(dt)

    for i in range(1, t_vals.size):
        t = t_vals[i-1]
        drift = np.asarray(drift_func(t, x), dtype=float)
        u = np.asarray(control_func(t, x, dt), dtype=float)
        noise = np.random.normal(size=d) * noise_std * sqrt_dt
        x = x + (drift + u) * dt + noise

        traj[i] = x
        ctrls[i] = u

        if stopping_condition(x, bound=bound, circle=circle):
            traj = traj[: i + 1]
            ctrls = ctrls[: i + 1]
            t_vals = t_vals[: i + 1]
            break

    return t_vals, traj, ctrls


def simulate_euler_maruyama_control_restart(
    drift_func: Callable[[float, np.ndarray], np.ndarray],
    control_func: Callable[[float, np.ndarray, float], np.ndarray],
    x0: np.ndarray,
    t_span: Tuple[float, float],
    dt: float,
    noise_std: float | np.ndarray = 0.0,
    *,
    bound: float = 1.0,
    circle: bool = True,
    max_restarts: int = 10,
    rs: int | np.random.Generator = 42,
    along_stable: Optional[np.ndarray] = None,
    angle_range: float = np.pi / 8,
    angle_distribution: str = "uniform",
    on_restart: Optional[Callable[[int, np.ndarray], None]] = None,
    restart_mode: str = "random",  # "reset" | "random" 
) -> Tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    """
    Euler–Maruyama simulation with control, restarting system when bounds are exceeded.
    Controller keeps its internal state/history unless `on_restart` resets it.
    """
    t0, t1 = t_span
    t_vals_full: list[np.ndarray] = []
    traj_full: list[np.ndarray] = []
    ctrls_full: list[np.ndarray] = []

    x = np.asarray(x0, dtype=float)
    rng = np.random.default_rng(rs)

    noise_std = np.broadcast_to(np.asarray(noise_std, dtype=float), (x.size,))
    sqrt_dt = np.sqrt(dt)

    restarts = 0
    t_current = t0

    while t_current < t1 and restarts <= max_restarts:
        t_vals = np.arange(t_current, t1 + 1e-12, dt)
        traj = np.zeros((t_vals.size, x.size))
        ctrls = np.zeros((t_vals.size, x.size))
        traj[0] = x

        for i in range(1, t_vals.size):
            t = t_vals[i - 1]
            drift = np.asarray(drift_func(t, x), dtype=float)
            u = np.asarray(control_func(t, x, dt), dtype=float)
            noise = rng.normal(size=x.size) * noise_std * sqrt_dt
            x = x + (drift + u) * dt + noise

            traj[i] = x
            ctrls[i] = u

            if stopping_condition(x, bound=bound, circle=circle):
                # choose new state
                if restart_mode == "reset":
                    x = np.array(x0, dtype=float)
                # elif restart_mode == "basin":  # "random", but restricted to a basin
                #     x = _sample_from_basin(basin_init, rng)
                else:  # "random"
                    x = generate_initial_conditions(
                        1,
                        bound,
                        circle=circle,
                        along_stable=along_stable,
                        angle_range=angle_range,
                        angle_distribution=angle_distribution,
                    )[0]
                restarts += 1

                # let caller react (e.g., tracker.reset(...))
                if on_restart is not None:
                    on_restart(restarts, x)

                # cut current segment
                traj = traj[: i + 1]
                ctrls = ctrls[: i + 1]
                t_vals = t_vals[: i + 1]
                break

        t_current = t_vals[-1] + dt
        t_vals_full.append(t_vals)
        traj_full.append(traj)
        ctrls_full.append(ctrls)

    return t_vals_full, traj_full, ctrls_full