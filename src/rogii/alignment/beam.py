import numpy as np


def _run_single(
    hw_gr: np.ndarray, tw_tvt: np.ndarray, tw_gr: np.ndarray,
    tvt_start: float, step_max: float, penalty: float, beam_width: int = 20,
) -> np.ndarray:
    steps = np.linspace(0.0, step_max, 15)
    beam = [(0.0, tvt_start)]   # (score, tvt)
    traj = np.empty(len(hw_gr))

    for t in range(len(hw_gr)):
        obs = hw_gr[t]
        cands = []
        for score, last_tvt in beam:
            for s in steps:
                new_tvt = last_tvt + s
                pred = float(np.interp(new_tvt, tw_tvt, tw_gr))
                gr_cost = (obs - pred) ** 2 if not np.isnan(obs) else 0.0
                cands.append((score - gr_cost - penalty * s ** 2, new_tvt))
        cands.sort(reverse=True)
        beam = cands[:beam_width]
        traj[t] = beam[0][1]

    return traj


def run_beam_configs(
    hw_gr: np.ndarray, tw_tvt: np.ndarray, tw_gr: np.ndarray,
    tvt_start: float, configs: list[dict],
) -> dict[str, np.ndarray]:
    """Run all beam configs. Adds beam_ref = (cons + sm5) / 2."""
    results = {}
    for cfg in configs:
        results[cfg['name']] = _run_single(
            hw_gr, tw_tvt, tw_gr, tvt_start,
            cfg['step_max'], cfg['penalty'],
        )
    results['beam_ref'] = (results['cons'] + results['sm5']) / 2.0
    return results
