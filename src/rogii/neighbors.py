from collections import defaultdict
import numpy as np
from scipy.spatial import cKDTree

FORMATIONS = ['ANCC', 'ASTNU', 'ASTNL', 'EGFDU', 'EGFDL', 'BUDA']


def assign_cluster(tw_gr_mean: float, y_coord: float) -> int:
    """C0=standard basin, C1=north high-GR, C2=SW high-GR."""
    if tw_gr_mean < 100.0:
        return 0
    return 1 if y_coord > 1_093_000.0 else 2


class FormationPlaneKNN:
    """Spatial KNN for imputing 6 formation depths. Same-cluster neighbors only."""

    def __init__(self, k: int = 10):
        self.k = k
        self._trees: dict[int, cKDTree] = {}
        self._xy: dict[int, np.ndarray] = {}
        self._depths: dict[int, np.ndarray] = {}

    def fit(self, wells: list[tuple[int, float, float, dict]]) -> 'FormationPlaneKNN':
        """wells: list of (cluster_id, mean_x, mean_y, {form: depth})"""
        buckets: dict[int, dict] = defaultdict(lambda: {'xy': [], 'depths': []})
        for cid, mx, my, depths in wells:
            buckets[cid]['xy'].append([mx, my])
            buckets[cid]['depths'].append([depths.get(f, 0.0) for f in FORMATIONS])
        for cid, data in buckets.items():
            self._xy[cid] = np.array(data['xy'])
            self._depths[cid] = np.array(data['depths'])
            self._trees[cid] = cKDTree(self._xy[cid])
        return self

    def predict(self, cluster_id: int, x: float, y: float) -> dict[str, float]:
        """IDW-averaged formation depths from K nearest same-cluster wells."""
        if cluster_id not in self._trees:
            cluster_id = 0
        k = min(self.k, len(self._xy[cluster_id]))
        dists, idx = self._trees[cluster_id].query([[x, y]], k=k)
        dists, idx = dists[0], idx[0]
        w = 1.0 / (dists + 1e-6)
        w /= w.sum()
        avg = (self._depths[cluster_id][idx] * w[:, None]).sum(axis=0)
        return dict(zip(FORMATIONS, avg))
