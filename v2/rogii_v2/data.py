"""Well loading, GR interpolation, GR calibration, PS detection, scalar extraction."""
import os
import glob
import numpy as np
import pandas as pd
from scipy.ndimage import label as _label


def extract_wellname(path):
    return os.path.basename(path).split('__')[0]


def list_wells(data_dir):
    hw_paths = sorted(glob.glob(os.path.join(str(data_dir), '*__horizontal_well.csv')))
    tw_map = {extract_wellname(p): p
              for p in glob.glob(os.path.join(str(data_dir), '*__typewell.csv'))}
    return [(p, tw_map[extract_wellname(p)])
            for p in hw_paths if extract_wellname(p) in tw_map]


def load_well(hw_path, tw_path):
    hw = pd.read_csv(hw_path)
    tw = pd.read_csv(tw_path).sort_values('TVT')
    return hw, tw


def detect_ps(hw):
    mask = hw['TVT_input'].isna().to_numpy()
    return int(mask.argmax()) if mask.any() else len(hw)


def interpolate_gr(hw):
    hw = hw.copy()
    gr = hw['GR'].copy()
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


def calibrate_gr(hw, tw, ps_idx):
    kn = hw.iloc[:ps_idx].dropna(subset=['TVT_input', 'GR'])
    if len(kn) < 10:
        return 1.0, 0.0
    tw_gr = np.interp(kn['TVT_input'].values, tw['TVT'].values, tw['GR'].values)
    hw_gr = kn['GR'].values
    A = np.column_stack([hw_gr, np.ones(len(hw_gr))])
    coeffs, _, _, _ = np.linalg.lstsq(A, tw_gr, rcond=None)
    return float(coeffs[0]), float(coeffs[1])


def robust_slope(x, y):
    if len(x) < 3:
        return 0.0
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    dx = x - x[0]
    if dx[-1] == 0:
        return 0.0
    return float(np.polyfit(dx, y, 1)[0])


def extract_scalars(hw, tw, ps_idx):
    kn = hw.iloc[:ps_idx]
    tvt_k = kn['TVT_input'].dropna()
    if len(tvt_k) == 0:
        return None
    md_k = kn.loc[tvt_k.index, 'MD'].values
    tvt_v = tvt_k.values
    last_tvt = float(tvt_v[-1])
    last_z = float(kn.loc[tvt_k.index[-1], 'Z'])
    last_md = float(kn.loc[tvt_k.index[-1], 'MD'])
    tw_tvt = tw['TVT'].values
    tw_gr = tw['GR'].fillna(tw['GR'].mean()).values
    tw_at_k = np.interp(tvt_v, tw_tvt, tw_gr)
    kgr = kn.loc[tvt_k.index, 'GR'].fillna(0).values
    gr_sigma = float(np.clip(np.nanstd(kgr - tw_at_k), 10.0, 60.0))
    # Initial rate from tail
    tail = kn.tail(30)
    dt = np.diff(tail['TVT_input'].dropna().values)
    dz = np.diff(tail.loc[tail['TVT_input'].notna(), 'Z'].values)
    dm = np.diff(tail.loc[tail['TVT_input'].notna(), 'MD'].values)
    m = dm > 0
    init_rate = float(np.median((dt + dz)[m] / dm[m])) if m.sum() >= 3 else 0.0
    ev = hw.iloc[ps_idx:]
    z_eval = ev['Z'].values.astype(float)
    z_span = float(np.ptp(z_eval)) if len(z_eval) else 0.0
    return dict(
        last_known_tvt=last_tvt,
        last_z=last_z,
        last_md=last_md,
        init_pos=last_tvt + last_z,   # U-space init for PF
        init_rate=init_rate,
        gr_sigma=gr_sigma,
        slope_all=robust_slope(md_k, tvt_v),
        slope_50=robust_slope(md_k[-50:], tvt_v[-50:]),
        slope_z=robust_slope(kn.loc[tvt_k.index, 'Z'].values, tvt_v),
        known_len=ps_idx,
        eval_len=len(hw) - ps_idx,
        z_span=z_span,
        ktvt_range=float(np.ptp(tvt_v)),
        ktvt_std=float(tvt_v.std()),
        tw_range=float(np.ptp(tw_tvt)),
        tw_gr_mean=float(tw_gr.mean()),
    )
