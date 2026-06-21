import numpy as np


def apply_rampup(
    d: np.ndarray, md_since_ps: np.ndarray, alpha: float = 1.0, tau: float = 85.0,
) -> np.ndarray:
    """Dampen TVT increments near PS: d *= alpha*(1 - exp(-t/tau))."""
    return d * alpha * (1.0 - np.exp(-md_since_ps / tau))


def blend_pf(d_model: np.ndarray, d_pf: np.ndarray, w_pf: float = 0.09) -> np.ndarray:
    return (1.0 - w_pf) * d_model + w_pf * d_pf


def savgol_smooth(y: np.ndarray, window: int = 17, poly: int = 3) -> np.ndarray:
    """SG filter. Uses scipy if available, otherwise numpy fallback."""
    try:
        from scipy.signal import savgol_filter
        return savgol_filter(y, window, poly)
    except Exception:
        half = window // 2
        x = np.arange(-half, half + 1, dtype=float)
        P = np.column_stack([x ** i for i in range(poly + 1)])
        coeff = np.linalg.pinv(P)[0]
        out = y.copy()
        for i in range(half, len(y) - half):
            out[i] = coeff @ y[i - half:i + half + 1]
        return out


def robust_polyfit(
    x: np.ndarray, y: np.ndarray, degree: int = 4, n_iters: int = 4, c: float = 2.0,
) -> np.ndarray:
    """IRLS polynomial fit with Tukey bisquare weights. Returns fitted values."""
    xn = x / (x[-1] + 1e-10)
    A  = np.column_stack([xn ** i for i in range(degree + 1)])
    w  = np.ones(len(y))
    for _ in range(n_iters):
        Aw = A * w[:, None]
        coeffs, _, _, _ = np.linalg.lstsq(Aw, y * w, rcond=None)
        resid = y - A @ coeffs
        s = np.median(np.abs(resid)) / 0.6745
        u = resid / (c * s + 1e-10)
        w = np.where(np.abs(u) < 1.0, (1.0 - u ** 2) ** 2, 0.0)
    return A @ coeffs


def apply_uspace(
    tvt_pred: np.ndarray, z: np.ndarray, anchor_tvt: float,
    degree: int = 4, robust_iters: int = 4, c: float = 2.0,
) -> np.ndarray:
    """Project TVT through U=TVT+Z space. Enforces geological planarity."""
    U      = tvt_pred + z - anchor_tvt
    s      = np.arange(len(U), dtype=float)
    U_proj = robust_polyfit(s, U, degree, robust_iters, c)
    return anchor_tvt + U_proj - z


def postprocess_well(
    tvt_increments: np.ndarray,
    pf_increments: np.ndarray,
    z: np.ndarray,
    md_since_ps: np.ndarray,
    last_known_tvt: float,
    params: dict,
    uspace_cfg: dict,
) -> np.ndarray:
    """Full post-processing pipeline for one well. Returns TVT trajectory."""
    d = blend_pf(tvt_increments, pf_increments, params['w_pf'])
    d = apply_rampup(d, md_since_ps, params['alpha'], params['tau'])
    tvt = last_known_tvt + np.cumsum(d)
    tvt = savgol_smooth(tvt)
    tvt = apply_uspace(tvt, z, last_known_tvt, **uspace_cfg)
    return tvt


import optuna


def tune_postprocess(well_data: list[dict], n_trials: int = 500) -> dict:
    """Optuna TPE over (alpha, tau, w_pf). well_data: per-well dicts with model outputs."""
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial: optuna.Trial) -> float:
        alpha = trial.suggest_float('alpha', 0.5, 2.0)
        tau   = trial.suggest_float('tau',   20.0, 200.0)
        w_pf  = trial.suggest_float('w_pf',  0.0, 0.3)
        total_sq, total_n = 0.0, 0
        for wd in well_data:
            d = blend_pf(wd['d_model'], wd['d_pf'], w_pf)
            d = apply_rampup(d, wd['md_since_ps'], alpha, tau)
            tvt_pred = wd['last_known_tvt'] + np.cumsum(d)
            tvt_true = wd['last_known_tvt'] + np.cumsum(wd['target_increments'])
            total_sq += float(np.sum((tvt_pred - tvt_true) ** 2))
            total_n  += len(d)
        return float(np.sqrt(total_sq / total_n))

    study = optuna.create_study(
        direction='minimize',
        sampler=optuna.samplers.TPESampler(seed=42),
    )
    study.optimize(objective, n_trials=n_trials)
    return study.best_params
