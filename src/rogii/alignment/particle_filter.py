import numpy as np


def _resample(weights: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    n = len(weights)
    pos = (rng.uniform() + np.arange(n)) / n
    return np.searchsorted(np.cumsum(weights), pos)


def _run_single(
    hw_gr: np.ndarray, tw_tvt: np.ndarray, tw_gr: np.ndarray,
    tvt_start: float, n_particles: int, sigma_0: float,
    gr_scale: int, obs_sigma: float, seed: int, use_velocity: bool,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    k = np.ones(gr_scale) / gr_scale
    hw_s = np.convolve(hw_gr, k, 'same')
    tw_s = np.convolve(tw_gr, k, 'same')

    tvt_p = rng.normal(tvt_start, sigma_0, n_particles)
    vel_p = rng.normal(0.04, 0.02, n_particles).clip(0.0) if use_velocity else None
    log_w = np.zeros(n_particles)
    traj = np.empty(len(hw_gr))

    for t in range(len(hw_gr)):
        if use_velocity:
            vel_p = (vel_p + rng.normal(0, 0.005, n_particles)).clip(0.0)
            tvt_p = tvt_p + vel_p
        else:
            tvt_p += np.clip(rng.normal(0.04, 0.3, n_particles), 0.0, None)

        if not np.isnan(hw_s[t]):
            pred = np.interp(tvt_p, tw_tvt, tw_s)
            log_w += -0.5 * ((hw_s[t] - pred) / obs_sigma) ** 2

        log_w -= np.max(log_w)
        w = np.exp(log_w)
        w /= w.sum()
        traj[t] = tvt_p @ w

        if 1.0 / (w @ w) < n_particles / 2:
            idx = _resample(w, rng)
            tvt_p = tvt_p[idx]
            if use_velocity:
                vel_p = vel_p[idx]
            log_w = np.zeros(n_particles)

    return traj


def run_pf(
    hw_gr: np.ndarray, tw_tvt: np.ndarray, tw_gr: np.ndarray,
    tvt_start: float, n_particles: int = 500, sigma_0: float = 4.5,
    gr_scale: int = 5, obs_sigma: float = 15.0, n_seeds: int = 64,
    use_velocity: bool = False,
) -> np.ndarray:
    """Ensemble PF. Averages n_seeds runs."""
    trajs = [
        _run_single(hw_gr, tw_tvt, tw_gr, tvt_start, n_particles,
                    sigma_0, gr_scale, obs_sigma, s, use_velocity)
        for s in range(n_seeds)
    ]
    return np.mean(trajs, axis=0)


def run_pf_variants(
    hw_gr: np.ndarray, tw_tvt: np.ndarray, tw_gr: np.ndarray,
    tvt_start: float, cfg: dict,
) -> dict[str, np.ndarray]:
    """Run pf_ancc (position) and pf_z (velocity) variants over all GR scales."""
    n_p, s0, scales, n_s = cfg['n_particles'], cfg['sigma_0'], cfg['gr_scales'], cfg['n_seeds']
    pf_ancc = np.mean([run_pf(hw_gr, tw_tvt, tw_gr, tvt_start, n_p, s0, sc, n_seeds=n_s)
                       for sc in scales], axis=0)
    pf_z    = np.mean([run_pf(hw_gr, tw_tvt, tw_gr, tvt_start, n_p, s0, sc, n_seeds=n_s,
                               use_velocity=True)
                       for sc in scales], axis=0)
    return {'pf_ancc': pf_ancc, 'pf_z': pf_z}
