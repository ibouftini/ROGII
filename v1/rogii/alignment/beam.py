import numpy as np


def _run_single(
    hw_gr: np.ndarray, tw_tvt: np.ndarray, tw_gr: np.ndarray,
    tvt_start: float, step_max: float, penalty: float, beam_width: int = 20,
) -> np.ndarray:
    steps    = np.linspace(-step_max, step_max, 29)
    pen_cost = penalty * steps ** 2          # (29,) precomputed

    beam_tvts   = np.array([tvt_start])
    beam_scores = np.array([0.0])
    traj = np.empty(len(hw_gr))

    for t in range(len(hw_gr)):
        obs = hw_gr[t]
        B   = len(beam_tvts)

        # Expand all beam x step combinations in one shot — (B, n_steps)
        n_steps     = len(steps)
        cand_tvts   = beam_tvts[:, None] + steps[None, :]
        preds       = np.interp(cand_tvts.ravel(), tw_tvt, tw_gr).reshape(B, n_steps)
        gr_cost     = 0.0 if np.isnan(obs) else (obs - preds) ** 2
        cand_scores = beam_scores[:, None] - gr_cost - pen_cost[None, :]

        flat_scores = cand_scores.ravel()
        flat_tvts   = cand_tvts.ravel()

        # O(N) partial-sort instead of O(N log N) full sort
        if len(flat_scores) > beam_width:
            top = np.argpartition(flat_scores, -beam_width)[-beam_width:]
        else:
            top = np.arange(len(flat_scores))

        beam_scores = flat_scores[top]
        beam_tvts   = flat_tvts[top]
        traj[t]     = beam_tvts[np.argmax(beam_scores)]

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
