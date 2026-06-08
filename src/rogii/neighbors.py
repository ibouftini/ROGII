from collections import defaultdict
import numpy as np
import pandas as pd
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


def _tw_signature(tw: pd.DataFrame, n: int = 50) -> str:
    """Hash of first n non-null GR values rounded to 1 decimal."""
    vals = tw['GR'].dropna().values[:n]
    return '|'.join(f'{v:.1f}' for v in vals)

def build_typewell_index(well_pairs: list[tuple[str, str]]) -> dict[str, str]:
    """Build {signature: wellname} from training well pairs (hw_path, tw_path)."""
    from rogii.utils import extract_wellname, load_tw
    index = {}
    for hw_path, tw_path in well_pairs:
        tw = load_tw(tw_path)
        sig = _tw_signature(tw)
        index[sig] = extract_wellname(hw_path)
    return index

def find_tw_match(tw: pd.DataFrame, index: dict[str, str]) -> str | None:
    """Return training wellname whose typewell matches tw, or None."""
    return index.get(_tw_signature(tw))
