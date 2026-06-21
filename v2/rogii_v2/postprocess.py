"""Post-processing: selector, warm-up, SG smooth, U-space, contact override."""
import numpy as np
import pandas as pd


def selector_well_code(hw, n_eval_thresh, z_span_thresholds):
    """Classify well into selector bin. Returns (code, n_eval, z_span)."""
    ev = hw[hw['TVT_input'].isna()]
    n_eval = float(len(ev))
    z_eval = ev['Z'].values.astype(float)
    z_span = float(np.ptp(z_eval)) if len(z_eval) else 0.0
    n_bin = int(n_eval > n_eval_thresh)
    z_bin = int(np.searchsorted(z_span_thresholds, z_span, side='right'))
    code = n_bin + 2 * z_bin
    return code, n_eval, z_span


def parse_selector_variant(name):
    """Parse 'pf_scale_X_beam_Y_hold_Z' into (scale, beam_w, hold_w)."""
    parts = name.split('_')
    scale = float(parts[2])
    beam_w = 0.0
    hold_w = 0.0
    if 'beam' in parts:
        beam_w = float(parts[parts.index('beam') + 1])
    if 'hold' in parts:
        hold_w = float(parts[parts.index('hold') + 1])
    return scale, beam_w, hold_w


def apply_selector(variant_name, pf_by_scale, tvt_beam, last_known_tvt):
    """Apply selector variant: blend PF scale + beam + hold-to-last."""
    scale, beam_w, hold_w = parse_selector_variant(variant_name)
    base = pf_by_scale.get(f'scale_{scale:g}',
           pf_by_scale.get('mean', np.full(1, last_known_tvt)))
    pred = (1.0 - beam_w) * base + beam_w * tvt_beam
    pred = (1.0 - hold_w) * pred + hold_w * last_known_tvt
    return pred


def warmup(md_since, tau=85.0):
    """Exponential warm-up ramp."""
    return 1.0 - np.exp(-np.maximum(md_since, 0.0) / tau) if tau > 1e-6 else 1.0


def savgol_smooth(pred, well_groups, window=61, poly=3):
    """Per-well Savitzky-Golay smoothing."""
    from scipy.signal import savgol_filter
    out = pred.copy()
    for _, idx in well_groups.items():
        v = pred[idx]
        n = len(v)
        wl = min(window, n)
        if wl % 2 == 0:
            wl -= 1
        if wl >= poly + 2:
            out[idx] = savgol_filter(v, wl, poly)
    return out


def robust_polyfit(x, y, degree=4, n_iters=4, c=2.0):
    """IRLS polynomial fit with Tukey bisquare. Returns fitted values."""
    xn = x / (x[-1] + 1e-10)
    A = np.column_stack([xn ** i for i in range(degree + 1)])
    w = np.ones(len(y))
    for _ in range(n_iters):
        Aw = A * w[:, None]
        coeffs, _, _, _ = np.linalg.lstsq(Aw, y * w, rcond=None)
        resid = y - A @ coeffs
        s = np.median(np.abs(resid)) / 0.6745
        u = resid / (c * s + 1e-10)
        w = np.where(np.abs(u) < 1.0, (1.0 - u ** 2) ** 2, 0.0)
    return A @ coeffs


def uspace_projection(tvt_pred, z, anchor_tvt, degree=4, n_iters=4, c=2.0, blend=0.75):
    """U-space polynomial projection. Returns blended TVT."""
    U = tvt_pred + z - anchor_tvt
    s = np.arange(len(U), dtype=float)
    U_proj = robust_polyfit(s, U, degree, n_iters, c)
    tvt_proj = anchor_tvt + U_proj - z
    return (1.0 - blend) * tvt_pred + blend * tvt_proj


def tvt_from_contacts(hw_tr, tw_tr, ref_col='EGFDU'):
    """Reconstruct TVT from formation contacts for visible wells."""
    tw_g = tw_tr.dropna(subset=['Geology'])
    ref_tvt = tw_g[tw_g['Geology'] == ref_col]['TVT'].min()
    if np.isnan(ref_tvt):
        ref_col = tw_g['Geology'].iloc[0]
        ref_tvt = tw_g[tw_g['Geology'] == ref_col]['TVT'].min()
    offset = (hw_tr['TVT'] - (ref_tvt - (hw_tr['Z'] - hw_tr[ref_col]))).mean()
    return ref_tvt - (hw_tr['Z'] - hw_tr[ref_col]) + offset


def guarded_contact_override(tvt_pred, hw_test, hw_train, tw_train):
    """Override prediction with contacts if prefix RMSE < 1 ft."""
    try:
        tvt_phys = tvt_from_contacts(hw_train, tw_train)
    except Exception:
        return tvt_pred
    # Validate against known prefix
    kn = hw_test[hw_test['TVT_input'].notna()]
    if len(kn) < 10:
        return tvt_pred
    prefix_pred = tvt_phys.iloc[kn.index].values
    prefix_true = kn['TVT_input'].values
    rmse = float(np.sqrt(np.mean((prefix_pred - prefix_true) ** 2)))
    if rmse < 1.0:
        return tvt_phys.values.astype(float)
    return tvt_pred
