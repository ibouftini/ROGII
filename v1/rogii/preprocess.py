import numpy as np
import pandas as pd
from scipy.ndimage import label as _label


def detect_ps(hw: pd.DataFrame) -> int:
    """Index of first NaN in TVT_input. Returns len(hw) if no NaN."""
    mask = hw['TVT_input'].isna()
    return int(mask.to_numpy().argmax()) if mask.any() else len(hw)


def interpolate_gr(hw: pd.DataFrame) -> pd.DataFrame:
    """Fill GR gaps: linear for runs <=20, rolling median for longer."""
    hw = hw.copy()
    gr = hw['GR'].copy()
    hw['gr_imputed'] = gr.isna().astype(np.int8)

    if not gr.isna().any():
        return hw

    nan_arr = gr.isna().to_numpy()
    labeled, _ = _label(nan_arr)
    run_sizes = np.zeros(len(gr), dtype=int)
    for i in range(1, labeled.max() + 1):
        mask = labeled == i
        run_sizes[mask] = int(mask.sum())

    short = nan_arr & (run_sizes <= 20)
    if short.any():
        gr_lin = gr.interpolate(method='linear')
        gr[short] = gr_lin[short]

    long = gr.isna()
    if long.any():
        filled = gr.ffill().bfill()
        roll = filled.rolling(51, min_periods=1, center=True).median()
        gr[long] = roll[long]

    hw['GR'] = gr
    return hw


def calibrate_gr(hw: pd.DataFrame, tw: pd.DataFrame, ps_idx: int) -> tuple[float, float, pd.DataFrame]:
    """Affine fit: a*GR_hw + b ~ GR_tw in known zone. Returns (a, b, calibrated_hw)."""
    known = hw.iloc[:ps_idx].dropna(subset=['TVT_input', 'GR'])
    if len(known) < 10:
        return 1.0, 0.0, hw.copy()
    tw_gr = np.interp(known['TVT_input'].values, tw['TVT'].values, tw['GR'].values)
    hw_gr = known['GR'].values
    A = np.column_stack([hw_gr, np.ones(len(hw_gr))])
    coeffs, _, _, _ = np.linalg.lstsq(A, tw_gr, rcond=None)
    a, b = float(coeffs[0]), float(coeffs[1])
    hw = hw.copy()
    hw['GR'] = hw['GR'] * a + b
    return a, b, hw


def extract_scalars(hw: pd.DataFrame, ps_idx: int) -> dict:
    """Per-well scalar features computed from the known zone."""
    known = hw.iloc[:ps_idx]
    tvt_k = known['TVT_input'].dropna()
    md_k = known.loc[tvt_k.index, 'MD'].values
    tvt_v = tvt_k.values
    last_tvt = float(tvt_v[-1]) if len(tvt_v) else 0.0
    md_ps = float(hw['MD'].iloc[min(ps_idx, len(hw) - 1)])

    if len(tvt_v) > 10:
        slope_all = float(np.polyfit(md_k - md_k[0], tvt_v, 1)[0])
        n = min(200, len(tvt_v))
        slope_rec = float(np.polyfit(md_k[-n:] - md_k[-n], tvt_v[-n:], 1)[0])
    else:
        slope_all = slope_rec = 0.04

    return dict(
        last_known_tvt=last_tvt,
        md_at_ps=md_ps,
        slope_tvt_md_all=slope_all,
        slope_tvt_md_recent=slope_rec,
        z_span=float(hw['Z'].max() - hw['Z'].min()),
        eval_zone_length=len(hw) - ps_idx,
        known_zone_length=ps_idx,
    )
