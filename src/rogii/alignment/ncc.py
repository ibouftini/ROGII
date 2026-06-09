import numpy as np


def compute_sc_trust(known_rows: int) -> float:
    return float(np.clip(known_rows / 200.0, 0.0, 0.6))


def _run_single_scale(
    hw_gr: np.ndarray, tw_tvt: np.ndarray, tw_gr: np.ndarray,
    baseline_tvt: np.ndarray, hw_size: int, stride: int,
    known_rows: int = 0, search_range: float = 50.0,
) -> tuple[np.ndarray, float]:
    offsets = np.arange(-search_range, search_range + 0.5, 0.5)   # (201,)
    n   = len(baseline_tvt)
    traj = baseline_tvt.copy()
    confs = []

    for i in range(0, n, stride):
        center = known_rows + i
        lo = max(0, center - hw_size)
        hi = min(len(hw_gr), center + hw_size + 1)
        hw_win = hw_gr[lo:hi]
        W = len(hw_win)

        if W == 0 or np.isnan(hw_win).mean() > 0.5:
            confs.append(0.0)
            continue

        # Vectorise all 201 offset candidates at once — (201, W)
        rel_pts = np.linspace(-hw_size * 0.5, hw_size * 0.5, W)
        cands   = baseline_tvt[i] + offsets                         # (201,)
        all_pts = cands[:, None] + rel_pts[None, :]                  # (201, W)
        tw_wins = np.interp(all_pts.ravel(), tw_tvt, tw_gr).reshape(len(offsets), W)

        hw_c  = hw_win - hw_win.mean()                               # (W,)
        tw_c  = tw_wins - tw_wins.mean(axis=1, keepdims=True)        # (201, W)
        denom = np.sqrt((hw_c ** 2).sum() * (tw_c ** 2).sum(axis=1))
        scores = (tw_c @ hw_c) / np.maximum(denom, 1e-8)   # (201,)

        best_idx = int(np.argmax(scores))
        end = min(n, i + stride)
        traj[i:end] = cands[best_idx]
        confs.append(float(scores[best_idx]))

    return traj, float(np.mean(confs)) if confs else 0.0


def run_ncc_multiscale(
    hw_gr: np.ndarray, tw_tvt: np.ndarray, tw_gr: np.ndarray,
    baseline_tvt: np.ndarray, known_rows: int,
    hw_sizes: tuple = (8, 15, 25), stride: int = 3,
) -> dict[str, np.ndarray | float]:
    results: dict = {}
    for hw_size in hw_sizes:
        traj, conf = _run_single_scale(hw_gr, tw_tvt, tw_gr, baseline_tvt, hw_size, stride, known_rows)
        results[f'sc{hw_size}_tvt'] = traj
        results[f'sc{hw_size}_conf'] = conf
    sc_trust = compute_sc_trust(known_rows)
    results['sc_trust'] = sc_trust
    results['hyb_ref'] = (1 - sc_trust) * baseline_tvt + sc_trust * results['sc15_tvt']
    return results
