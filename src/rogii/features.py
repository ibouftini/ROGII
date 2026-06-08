import numpy as np
import pandas as pd

FORMATIONS = ['ANCC', 'ASTNU', 'ASTNL', 'EGFDU', 'EGFDL', 'BUDA']
ANCHOR_OFFSETS = [-80, -40, -20, -10, -5, 0, 5, 10, 20, 40, 80]
TVT_SIGNAL_COLS = [
    'pf_ancc', 'pf_z', 'beam_ref',
    'sc8_tvt', 'sc15_tvt', 'sc25_tvt', 'hyb_ref',
    'dtw_r20_mean', 'dtw_r50_mean', 'dtw_r100_mean', 'dtw_r200_mean',
]


def build_alignment_df(hw: pd.DataFrame, ps_idx: int, alignment: dict) -> pd.DataFrame:
    """Pack alignment trajectories into a DataFrame aligned to the eval zone."""
    idx = hw.index[ps_idx:]
    data = {}
    for k, v in alignment.items():
        if isinstance(v, np.ndarray) and len(v) == len(idx):
            data[k] = v
        elif not isinstance(v, np.ndarray):
            data[k] = float(v)  # scalar confidence scores
    return pd.DataFrame(data, index=idx)


def compute_anchor_offsets(
    baseline_tvt: np.ndarray, tw_tvt: np.ndarray, tw_gr: np.ndarray,
    hw_gr: np.ndarray, window: int = 10,
) -> pd.DataFrame:
    """NCC at each of 11 TVT offsets around the baseline prediction."""
    n = len(baseline_tvt)
    result = np.zeros((n, len(ANCHOR_OFFSETS)))

    def _ncc(a: np.ndarray, b: np.ndarray) -> float:
        a, b = a - a.mean(), b - b.mean()
        d = np.sqrt((a ** 2).sum() * (b ** 2).sum())
        return float(np.dot(a, b) / d) if d > 1e-8 else 0.0

    for i in range(n):
        lo, hi = max(0, i - window), min(n, i + window + 1)
        hw_win = hw_gr[lo:hi]
        for j, off in enumerate(ANCHOR_OFFSETS):
            cand = baseline_tvt[i] + off
            pts = np.linspace(cand - window * 0.5, cand + window * 0.5, len(hw_win))
            tw_win = np.interp(pts, tw_tvt, tw_gr)
            result[i, j] = _ncc(hw_win, tw_win)

    cols = [f'anchor_off_{o:+d}' for o in ANCHOR_OFFSETS]
    return pd.DataFrame(result, columns=cols)


def compute_b_well(
    hw: pd.DataFrame, ps_idx: int, formation_depths: dict, decay: float = 0.02,
) -> dict[str, float]:
    """WLS offset b such that TVT ~ -Z + depth + b in the known zone."""
    known = hw.iloc[:ps_idx].dropna(subset=['TVT_input'])
    tvt = known['TVT_input'].values
    z   = known['Z'].values
    md  = known['MD'].values
    w = np.exp(-decay * (md[-1] - md)) if len(md) else np.array([1.0])
    b = {}
    for f, depth in formation_depths.items():
        approx = -z + depth
        resid = tvt - approx
        b[f] = float(np.average(resid, weights=w))
    return b


def compute_formation_features(
    hw: pd.DataFrame, ps_idx: int, formation_depths: dict, b_well: dict,
) -> pd.DataFrame:
    """tvt_fw_f, b_well_f, form_rmse_f for 6 formations."""
    eval_z = hw.iloc[ps_idx:]['Z'].values
    data = {}
    for f in FORMATIONS:
        depth = formation_depths.get(f, 0.0)
        b     = b_well.get(f, 0.0)
        data[f'tvt_fw_{f}']   = -eval_z + depth + b
        data[f'b_well_{f}']   = b
    # form_rmse: residual in known zone
    known = hw.iloc[:ps_idx].dropna(subset=['TVT_input'])
    tvt_k = known['TVT_input'].values
    z_k   = known['Z'].values
    for f in FORMATIONS:
        approx = -z_k + formation_depths.get(f, 0.0) + b_well.get(f, 0.0)
        data[f'form_rmse_{f}'] = float(np.sqrt(np.mean((tvt_k - approx) ** 2)))
    return pd.DataFrame(data, index=hw.index[ps_idx:])
