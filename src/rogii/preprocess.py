import numpy as np
import pandas as pd
from scipy.ndimage import label as _label


def detect_ps(hw: pd.DataFrame) -> int:
    """Index of first NaN in TVT_input. Returns len(hw) if no NaN."""
    mask = hw['TVT_input'].isna()
    return int(mask.idxmax()) if mask.any() else len(hw)


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
