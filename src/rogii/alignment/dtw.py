import numpy as np


def _fill_cost_matrix(x: np.ndarray, y: np.ndarray, radius: int) -> np.ndarray:
    n, m = len(x), len(y)
    D = np.full((n + 1, m + 1), np.inf)
    D[0, 0] = 0.0
    for i in range(1, n + 1):
        j_lo, j_hi = max(1, i - radius), min(m, i + radius)
        for j in range(j_lo, j_hi + 1):
            cost = (x[i - 1] - y[j - 1]) ** 2
            D[i, j] = cost + min(D[i - 1, j], D[i, j - 1], D[i - 1, j - 1])
    return D


def _traceback(D: np.ndarray, n: int, m: int) -> np.ndarray:
    i, j = n, m
    path = []
    while i > 0 and j > 0:
        path.append(j - 1)
        prev = np.argmin([D[i - 1, j], D[i, j - 1], D[i - 1, j - 1]])
        if prev == 0:
            i -= 1
        elif prev == 1:
            j -= 1
        else:
            i -= 1; j -= 1
    path.reverse()
    if len(path) < n:
        path = [path[0]] * (n - len(path)) + path
    return np.array(path[:n], dtype=int)


def _dtw_stochastic_single(
    hw_gr: np.ndarray, tw_tvt: np.ndarray, tw_gr: np.ndarray,
    radius: int, k: int, seed: int,
) -> np.ndarray:
    """K stochastic DTW paths. Returns tvt_paths shape (k, n)."""
    rng = np.random.default_rng(seed)
    n, m = len(hw_gr), len(tw_gr)
    D = _fill_cost_matrix(hw_gr, tw_gr, radius)
    paths = []
    for _ in range(k):
        noise = np.zeros_like(D)
        noise[1:, 1:] = rng.gumbel(0, 1, (n, m))
        path_j = _traceback(D + noise, n, m)
        paths.append(tw_tvt[path_j])
    return np.array(paths)


def run_dtw_all_radii(
    hw_gr: np.ndarray, tw_tvt: np.ndarray, tw_gr: np.ndarray,
    radii: tuple | list = (20, 50, 100, 200), k: int = 12,
) -> dict[str, np.ndarray]:
    results = {}
    for r in radii:
        paths = _dtw_stochastic_single(hw_gr, tw_tvt, tw_gr, r, k, seed=r)
        mean = paths.mean(axis=0)
        std  = paths.std(axis=0)
        cv   = std / (np.abs(mean) + 1e-6)
        results[f'dtw_r{r}_mean'] = mean
        results[f'dtw_r{r}_std']  = std
        results[f'dtw_r{r}_cv']   = cv
    return results
