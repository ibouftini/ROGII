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
    """NCC at each of 11 TVT offsets — all offsets computed in one numpy batch per row."""
    n           = len(baseline_tvt)
    result      = np.zeros((n, len(ANCHOR_OFFSETS)))
    offsets_arr = np.array(ANCHOR_OFFSETS, dtype=float)   # (11,)

    for i in range(n):
        lo, hi = max(0, i - window), min(n, i + window + 1)
        hw_win = hw_gr[lo:hi]
        W = len(hw_win)
        if W == 0:
            continue

        # All 11 candidate windows at once — (11, W)
        rel_pts = np.linspace(-window * 0.5, window * 0.5, W)
        cands   = baseline_tvt[i] + offsets_arr              # (11,)
        all_pts = cands[:, None] + rel_pts[None, :]           # (11, W)
        tw_wins = np.interp(all_pts.ravel(), tw_tvt, tw_gr).reshape(11, W)

        hw_c  = hw_win - hw_win.mean()
        tw_c  = tw_wins - tw_wins.mean(axis=1, keepdims=True)
        denom = np.sqrt((hw_c ** 2).sum() * (tw_c ** 2).sum(axis=1))
        result[i] = (tw_c @ hw_c) / np.maximum(denom, 1e-8)

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


def compute_gr_features(
    hw: pd.DataFrame, ps_idx: int, a_cal: float, b_cal: float,
    tw_tvt: np.ndarray, tw_gr: np.ndarray, baseline_tvt: np.ndarray,
) -> pd.DataFrame:
    eval_hw = hw.iloc[ps_idx:]
    gr = eval_hw['GR'].values
    data = {}
    for w in [11, 51, 151]:
        s = pd.Series(gr)
        data[f'gr_roll_mean_{w}'] = s.rolling(w, min_periods=1, center=True).mean().values
        data[f'gr_roll_std_{w}']  = s.rolling(w, min_periods=1, center=True).std(ddof=0).fillna(0).values
    data['hgr_env']  = pd.Series(gr).rolling(21, min_periods=1, center=True).max().values
    data['hgr_nrg']  = np.sqrt(pd.Series(gr ** 2).rolling(21, min_periods=1, center=True).mean().values)
    data['a_cal']    = a_cal
    data['b_cal']    = b_cal
    data['gr_imputed_flag'] = eval_hw['gr_imputed'].values if 'gr_imputed' in eval_hw.columns else 0
    data['tw_gr_at_baseline_tvt'] = np.interp(baseline_tvt, tw_tvt, tw_gr)
    return pd.DataFrame(data, index=eval_hw.index)


def compute_tabular_features(
    hw: pd.DataFrame, ps_idx: int, scalars: dict, cluster_id: int,
    signal_df: pd.DataFrame,
) -> pd.DataFrame:
    eval_hw = hw.iloc[ps_idx:]
    n = len(eval_hw)
    md_from_ps = eval_hw['MD'].values - scalars['md_at_ps']
    sig_cols = [c for c in TVT_SIGNAL_COLS if c in signal_df.columns]
    inter_std = signal_df[sig_cols].std(axis=1).values if sig_cols else np.zeros(n)
    data = dict(
        md_from_ps=md_from_ps,
        row_from_ps=np.arange(n, dtype=float),
        row_frac=np.arange(n) / max(1, n - 1),
        last_known_tvt=scalars['last_known_tvt'],
        slope_tvt_md_all=scalars['slope_tvt_md_all'],
        slope_tvt_md_recent=scalars['slope_tvt_md_recent'],
        z_span=scalars['z_span'],
        eval_zone_length=float(scalars['eval_zone_length']),
        cluster_id=float(cluster_id),
        inter_signal_std=inter_std,
    )
    return pd.DataFrame(data, index=eval_hw.index)


def build_feature_matrix(
    hw: pd.DataFrame, tw: pd.DataFrame, ps_idx: int,
    alignment: dict, formations: dict, b_well: dict,
    scalars: dict, cluster_id: int, a_cal: float, b_cal: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Assemble ~80-column feature matrix and TVT increment target."""
    baseline = alignment.get('beam_ref', alignment.get('pf_ancc'))
    gr_full = hw['GR'].values

    eval_idx  = hw.index[ps_idx:]
    df_align  = build_alignment_df(hw, ps_idx, alignment)
    df_anchor = compute_anchor_offsets(baseline, tw['TVT'].values, tw['GR'].values, gr_full[ps_idx:])
    df_anchor.index = eval_idx
    df_form   = compute_formation_features(hw, ps_idx, formations, b_well)
    df_gr     = compute_gr_features(hw, ps_idx, a_cal, b_cal, tw['TVT'].values, tw['GR'].values, baseline)
    df_tab    = compute_tabular_features(hw, ps_idx, scalars, cluster_id, df_align)

    df = pd.concat([df_align, df_anchor, df_form, df_gr, df_tab], axis=1)

    # target: TVT increment
    tvt = hw['TVT'].values
    tvt_eval = tvt[ps_idx:]
    first_prev = tvt[ps_idx - 1] if ps_idx > 0 else tvt_eval[0]
    tvt_prev = np.concatenate([[first_prev], tvt_eval[:-1]])
    y = tvt_eval - tvt_prev

    return df.values.astype(np.float32), y.astype(np.float32)
