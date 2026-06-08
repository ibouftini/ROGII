import numpy as np


def _ncc(a: np.ndarray, b: np.ndarray) -> float:
    a = a - a.mean()
    b = b - b.mean()
    denom = np.sqrt((a ** 2).sum() * (b ** 2).sum())
    return float(np.dot(a, b) / denom) if denom > 1e-8 else 0.0


def compute_sc_trust(known_rows: int) -> float:
    return float(np.clip(known_rows / 200.0, 0.0, 0.6))


def _run_single_scale(
    hw_gr: np.ndarray, tw_tvt: np.ndarray, tw_gr: np.ndarray,
    baseline_tvt: np.ndarray, hw_size: int, stride: int,
    known_rows: int = 0, search_range: float = 50.0,
) -> tuple[np.ndarray, float]:
    n = len(baseline_tvt)
    offsets = np.arange(-search_range, search_range + 0.5, 0.5)
    traj = baseline_tvt.copy()
    confs = []

    for i in range(0, n, stride):
        center = known_rows + i
        lo, hi = max(0, center - hw_size), min(len(hw_gr), center + hw_size + 1)
        hw_win = hw_gr[lo:hi]
        if np.isnan(hw_win).mean() > 0.5:
            confs.append(0.0)
            continue
        best, best_tvt = -np.inf, baseline_tvt[i]
        for off in offsets:
            cand = baseline_tvt[i] + off
            pts = np.linspace(cand - hw_size * 0.5, cand + hw_size * 0.5, len(hw_win))
            tw_win = np.interp(pts, tw_tvt, tw_gr)
            score = _ncc(hw_win, tw_win)
            if score > best:
                best, best_tvt = score, cand
        end = min(n, i + stride)
        traj[i:end] = best_tvt
        confs.append(best)

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
