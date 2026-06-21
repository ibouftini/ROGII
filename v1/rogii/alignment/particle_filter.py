import numpy as np


def _resample(weights: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    n = len(weights)
    pos = (rng.uniform() + np.arange(n)) / n
    return np.searchsorted(np.cumsum(weights), pos)


def _run_batch(
    hw_gr: np.ndarray, tw_tvt: np.ndarray, tw_gr: np.ndarray,
    tvt_start: float, n_particles: int, sigma_0: float,
    gr_scale: int, obs_sigma: float, n_seeds: int, use_velocity: bool,
) -> np.ndarray:
    """Run all n_seeds instances in one vectorised pass (S, P) → mean trajectory.

    Replaces n_seeds sequential Python function calls with a single T-step loop
    operating on (n_seeds, n_particles) arrays. Python loop overhead: O(T) instead
    of O(n_seeds * T).
    """
    rng = np.random.default_rng(0)
    k = np.ones(gr_scale) / gr_scale
    hw_s = np.convolve(hw_gr, k, 'same')
    tw_s = np.convolve(tw_gr, k, 'same')
    T, S, P = len(hw_gr), n_seeds, n_particles

    tvt_p = rng.normal(tvt_start, sigma_0, (S, P))
    log_w = np.zeros((S, P))
    traj  = np.empty((S, T))
    vel_p = rng.normal(0.0, 0.02, (S, P)) if use_velocity else None

    for t in range(T):
        if use_velocity:
            vel_p = vel_p + rng.normal(0, 0.005, (S, P))
            tvt_p = tvt_p + vel_p
        else:
            tvt_p += rng.normal(0.0, 0.3, (S, P))

        if not np.isnan(hw_s[t]):
            pred = np.interp(tvt_p.ravel(), tw_tvt, tw_s).reshape(S, P)
            log_w += -0.5 * ((hw_s[t] - pred) / obs_sigma) ** 2

        log_w -= log_w.max(axis=1, keepdims=True)
        w = np.exp(log_w)
        w /= w.sum(axis=1, keepdims=True)
        traj[:, t] = (tvt_p * w).sum(axis=1)

        ess = 1.0 / (w * w).sum(axis=1)
        for s in np.where(ess < P / 2)[0]:
            pos = (rng.random() + np.arange(P)) / P
            idx = np.searchsorted(w[s].cumsum(), pos)
            tvt_p[s] = tvt_p[s, idx]
            if use_velocity:
                vel_p[s] = vel_p[s, idx]
            log_w[s] = 0.0

    return traj.mean(axis=0)


def run_pf_variants(
    hw_gr: np.ndarray, tw_tvt: np.ndarray, tw_gr: np.ndarray,
    tvt_start: float, cfg: dict,
) -> dict[str, np.ndarray]:
    """Run pf_ancc (position) and pf_z (velocity) variants over all GR scales."""
    n_p, s0, scales, n_s = cfg['n_particles'], cfg['sigma_0'], cfg['gr_scales'], cfg['n_seeds']
    pf_ancc = np.mean([
        _run_batch(hw_gr, tw_tvt, tw_gr, tvt_start, n_p, s0, sc, 15.0, n_s, False)
        for sc in scales
    ], axis=0)
    pf_z = np.mean([
        _run_batch(hw_gr, tw_tvt, tw_gr, tvt_start, n_p, s0, sc, 15.0, n_s, True)
        for sc in scales
    ], axis=0)
    return {'pf_ancc': pf_ancc, 'pf_z': pf_z}
