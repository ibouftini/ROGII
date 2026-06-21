"""Spatial imputation: FormationPlaneKNN, DenseANCCImputer, typewell index."""
import os
import hashlib
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

FORMATIONS = ['ANCC', 'ASTNU', 'ASTNL', 'EGFDU', 'EGFDL', 'BUDA']


class FormationPlaneKNN:
    """KNN imputation of formation contact depths using (X, Y) coordinates."""

    def __init__(self, train_wids, train_dir, k=10):
        self.k = k
        coords = []
        depths = []
        self.wids = []
        for wid in train_wids:
            hw_path = os.path.join(str(train_dir), f'{wid}__horizontal_well.csv')
            if not os.path.exists(hw_path):
                continue
            hw = pd.read_csv(hw_path)
            x_mean = float(hw['X'].mean())
            y_mean = float(hw['Y'].mean())
            d = []
            for f in FORMATIONS:
                if f in hw.columns:
                    vals = hw[f].dropna()
                    d.append(float(vals.median()) if len(vals) else 0.0)
                else:
                    d.append(0.0)
            coords.append([x_mean, y_mean])
            depths.append(d)
            self.wids.append(wid)
        self.coords = np.array(coords)
        self.depths = np.array(depths)
        self.tree = cKDTree(self.coords)

    def impute(self, xy, self_wid=None):
        """Impute formation depths for points xy (N, 2).
        Returns (depths (N, 6), distances (N,))."""
        dists, idxs = self.tree.query(xy, k=self.k)
        if dists.ndim == 1:
            dists = dists.reshape(-1, 1)
            idxs = idxs.reshape(-1, 1)
        result = np.zeros((len(xy), len(FORMATIONS)))
        mean_dists = np.zeros(len(xy))
        for i in range(len(xy)):
            neighbors = idxs[i]
            d = dists[i]
            # Exclude self
            if self_wid is not None:
                mask = np.array([self.wids[j] != self_wid for j in neighbors])
                neighbors = neighbors[mask]
                d = d[mask]
            if len(neighbors) == 0:
                result[i] = self.depths.mean(axis=0)
                continue
            w = 1.0 / (d + 1e-6)
            w /= w.sum()
            result[i] = (self.depths[neighbors] * w[:, None]).sum(axis=0)
            mean_dists[i] = d.mean()
        return result.astype(np.float32), mean_dists.astype(np.float32)


class DenseANCCImputer:
    """Dense ANCC imputation using IDW from K=20 neighbors."""

    def __init__(self, train_wids, train_dir, k=20, n_samples=60):
        self.k = k
        coords = []
        ancc_vals = []
        self.wids_per_sample = []
        for wid in train_wids:
            hw_path = os.path.join(str(train_dir), f'{wid}__horizontal_well.csv')
            if not os.path.exists(hw_path):
                continue
            hw = pd.read_csv(hw_path)
            if 'ANCC' not in hw.columns:
                continue
            valid = hw.dropna(subset=['ANCC'])
            if len(valid) == 0:
                continue
            step = max(1, len(valid) // n_samples)
            sampled = valid.iloc[::step][:n_samples]
            for _, row in sampled.iterrows():
                coords.append([float(row['X']), float(row['Y'])])
                ancc_vals.append(float(row['ANCC']))
                self.wids_per_sample.append(wid)
        if len(coords) == 0:
            self.tree = None
            return
        self.coords = np.array(coords)
        self.ancc = np.array(ancc_vals)
        self.tree = cKDTree(self.coords)

    def impute(self, xy, self_wid=None):
        """Returns (ancc_values, std_values, distances) for points xy."""
        if self.tree is None:
            n = len(xy)
            return np.zeros(n, np.float32), np.zeros(n, np.float32), np.full(n, 1e6, np.float32)
        dists, idxs = self.tree.query(xy, k=self.k)
        if dists.ndim == 1:
            dists = dists.reshape(-1, 1)
            idxs = idxs.reshape(-1, 1)
        result = np.zeros(len(xy), np.float32)
        std_out = np.zeros(len(xy), np.float32)
        dist_out = np.zeros(len(xy), np.float32)
        for i in range(len(xy)):
            neighbors = idxs[i]
            d = dists[i]
            if self_wid is not None:
                mask = np.array([self.wids_per_sample[j] != self_wid for j in neighbors])
                neighbors = neighbors[mask]
                d = d[mask]
            if len(neighbors) == 0:
                result[i] = self.ancc.mean()
                continue
            w = 1.0 / (d + 1e-6)
            w /= w.sum()
            vals = self.ancc[neighbors]
            result[i] = (vals * w).sum()
            std_out[i] = np.sqrt(((vals - result[i]) ** 2 * w).sum())
            dist_out[i] = d.mean()
        return result, std_out, dist_out


def seg_b_well(ktvt, kz, form_col):
    """Segmented b_well: full, early, mid, late, WLS."""
    resid = ktvt + kz - form_col
    n = len(resid)
    b_full = float(np.median(resid))
    t3 = n // 3
    b_early = float(np.median(resid[:t3])) if t3 > 0 else b_full
    b_mid = float(np.median(resid[t3:2*t3])) if t3 > 0 else b_full
    b_late = float(np.median(resid[2*t3:])) if t3 > 0 else b_full
    # WLS with exponential decay
    if n > 0:
        md_range = np.arange(n, dtype=float)
        w = np.exp(-0.02 * (n - 1 - md_range))
        w /= w.sum()
        b_wls = float((resid * w).sum())
    else:
        b_wls = b_full
    return b_full, b_early, b_mid, b_late, b_wls


def build_typewell_index(well_pairs):
    """Hash typewell GR for matching test wells to training wells."""
    index = {}
    for hw_path, tw_path in well_pairs:
        wid = os.path.basename(hw_path).split('__')[0]
        tw = pd.read_csv(tw_path).sort_values('TVT')
        gr = tw['GR'].fillna(0).values[:50]
        h = hashlib.md5(gr.tobytes()).hexdigest()
        index[h] = wid
    return index


def find_tw_match(tw, tw_index):
    """Check if a test well's typewell matches any training well."""
    tw = tw.sort_values('TVT')
    gr = tw['GR'].fillna(0).values[:50]
    h = hashlib.md5(gr.tobytes()).hexdigest()
    return tw_index.get(h)
