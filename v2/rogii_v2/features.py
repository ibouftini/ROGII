"""Feature engineering: ~150 features + delta target."""
import numpy as np
import pandas as pd
from rogii_v2.spatial import FORMATIONS, seg_b_well

ANCHOR_OFFSETS = [-80, -40, -20, -10, -5, 0, 5, 10, 20, 40, 80]


def build_well_features(hw, tw, ps_idx, scalars,
                        pf_ancc, pf_ancc_std, pf_z, pf_z_std,
                        beam_paths, ncc_results, ncc_ensemble,
                        likpf_dict,
                        form_ev, form_kn, dense_ancc, dense_std, dense_dist,
                        dense_kn, dense_std_kn, knn_dist,
                        is_train, wid):
    """Build ~150-feature DataFrame for one well's eval zone.

    All tracker signals are absolute TVT values; features are stored
    as deltas from last_known_tvt.

    Returns DataFrame with features + 'target' column (if is_train) + 'id' + 'well'.
    """
    kn = hw.iloc[:ps_idx]
    ev = hw.iloc[ps_idx:]
    nh = len(ev)
    if nh == 0:
        return None

    last_tvt = scalars['last_known_tvt']
    tw_tvt = tw['TVT'].values.astype(np.float32)
    tw_gr = tw['GR'].fillna(tw['GR'].mean()).values.astype(np.float32)

    # GR
    gr_full = hw['GR'].interpolate(limit_direction='both').fillna(float(tw_gr.mean()))
    hgr = gr_full.iloc[ev.index].values.astype(np.float32)
    kgr = gr_full.iloc[:ps_idx].values.astype(np.float32)
    ktvt = kn['TVT_input'].dropna().values.astype(np.float32)
    kz = kn.loc[kn['TVT_input'].notna(), 'Z'].values.astype(np.float32)

    # Calibration
    tw_at_k = np.interp(ktvt, tw_tvt, tw_gr)
    a_cal, b_cal = _affine_cal(kgr[:len(ktvt)], tw_at_k)

    # Beam summary
    beam_vals = list(beam_paths.values())
    beam_stack = np.stack([(v - last_tvt) for v in beam_vals], axis=1)

    # NCC
    sc_results = ncc_results  # list of (tvt, scores)
    sc8, sc8s = sc_results[0] if len(sc_results) > 0 else (np.full(nh, last_tvt, np.float32), np.zeros(nh, np.float32))
    sc15, sc15s = sc_results[1] if len(sc_results) > 1 else (sc8.copy(), np.zeros(nh, np.float32))
    sc25, sc25s = sc_results[2] if len(sc_results) > 2 else (sc8.copy(), np.zeros(nh, np.float32))
    sc_cons = (sc8 + sc15 + sc25) / 3.0
    sc_trust = float(np.clip(len(kn) / 200.0, 0.0, 0.6))
    beam_ref = beam_paths.get('beam_ref', beam_paths.get('cons', np.full(nh, last_tvt, np.float32)))
    hyb_ref = (1 - sc_trust) * beam_ref + sc_trust * ncc_ensemble

    # Formation features
    z_ev = ev['Z'].values.astype(np.float32)
    tvt_fs = {}
    form_rmses = {}
    form_list = []
    for fi, fn in enumerate(FORMATIONS):
        b_full, b_early, b_mid, b_late, b_wls = seg_b_well(ktvt, kz, form_kn[:, fi])
        tvt_f = (-z_ev + form_ev[:, fi] + b_full).astype(np.float32)
        tvt_fs[f'tvtF_{fn}'] = tvt_f
        tvt_fs[f'tvtFw_{fn}'] = (-z_ev + form_ev[:, fi] + b_wls).astype(np.float32)
        tvt_fs[f'tvtF50_{fn}'] = (-z_ev + form_ev[:, fi] + b_late).astype(np.float32)
        tvt_fs[f'bw_{fn}'] = np.float32(b_full)
        tvt_fs[f'bww_{fn}'] = np.float32(b_wls)
        tvt_fs[f'bw50_{fn}'] = np.float32(b_late)
        tvt_fs[f'bw_early_{fn}'] = np.float32(b_early)
        tvt_fs[f'bw_mid_{fn}'] = np.float32(b_mid)
        kn_approx = ktvt - (-kz + form_kn[:, fi] + b_full)
        form_rmses[fn] = float(np.sqrt(np.mean(kn_approx ** 2)))
        form_list.append(tvt_f)
    fs = np.stack(form_list, axis=1)

    # Dense ANCC features
    _, b_de, b_dm, b_dl, b_dw = seg_b_well(ktvt, kz, dense_kn)
    b_d = float(np.median(ktvt + kz - dense_kn))
    tvt_dense = (-z_ev + dense_ancc + b_d).astype(np.float32)
    tvt_densew = (-z_ev + dense_ancc + b_dw).astype(np.float32)
    tvt_dense50 = (-z_ev + dense_ancc + b_dl).astype(np.float32)
    res_kn = ktvt + kz - dense_kn
    d_rmse = float(np.sqrt(np.mean(res_kn ** 2)))
    d_bias = float(np.mean(res_kn))
    d_nb_std = float(np.mean(dense_std_kn))

    # Signal ensemble stats
    all_sigs = [pf_ancc] + list(beam_paths.values()) + [sc8, sc15, sc25, ncc_ensemble,
                tvt_fs['tvtF_ANCC'], tvt_dense]
    sig_mat = np.stack(all_sigs, axis=1)
    sig_std = sig_mat.std(axis=1).astype(np.float32)
    sig_mean = (sig_mat.mean(axis=1) - last_tvt).astype(np.float32)

    # GR rolling stats
    gr_s = pd.Series(gr_full.values)
    rolls = {}
    for w in [5, 21, 51, 101]:
        r = gr_s.rolling(w, center=True, min_periods=1)
        rolls[f'grm{w}'] = r.mean().iloc[ev.index].values.astype(np.float32)
        rolls[f'grs{w}'] = r.std().fillna(0).iloc[ev.index].values.astype(np.float32)
    for lag in [1, 5, 15, 30]:
        rolls[f'glag{lag}'] = gr_s.shift(lag).bfill().iloc[ev.index].values.astype(np.float32)
        rolls[f'glead{lag}'] = gr_s.shift(-lag).ffill().iloc[ev.index].values.astype(np.float32)
    gr_d1 = gr_s.diff().fillna(0.0).iloc[ev.index].values.astype(np.float32)
    gr_d2 = gr_s.diff().diff().fillna(0.0).iloc[ev.index].values.astype(np.float32)
    gr_env = gr_s.rolling(21, center=True, min_periods=1).max().iloc[ev.index].values.astype(np.float32)
    gr_nrg = np.sqrt(np.maximum((gr_s ** 2).rolling(21, center=True, min_periods=1).mean(), 0.0)).iloc[ev.index].values.astype(np.float32)

    # Position features
    hmd = ev['MD'].values.astype(np.float32)
    md_since = hmd - scalars['last_md']
    frac = (np.arange(nh, dtype=np.float32) / max(nh - 1, 1))
    slp_b_all = (last_tvt + scalars['slope_all'] * md_since).astype(np.float32)
    slp_b_50 = (last_tvt + scalars['slope_50'] * md_since).astype(np.float32)

    # Trajectory derivatives
    mdd = hw['MD'].diff().replace(0, np.nan)
    dzdmd = (hw['Z'].diff() / mdd).iloc[ev.index].values.astype(np.float32)
    dxdmd = (hw['X'].diff() / mdd).iloc[ev.index].values.astype(np.float32)
    dydmd = (hw['Y'].diff() / mdd).iloc[ev.index].values.astype(np.float32)

    lk = kn.iloc[-1]

    def sc(v):
        return np.full(nh, np.float32(v), np.float32)

    # --- Assemble features dict ---
    feats = {
        'well': wid,
        'id': [f'{wid}_{i}' for i in ev.index],
        'last_known_tvt': sc(last_tvt),
        # PF signals
        'pf_ancc': pf_ancc,
        'pf_ancc_std': pf_ancc_std,
        'pf_ancc_delta': (pf_ancc - last_tvt).astype(np.float32),
        'pf_z': pf_z if len(pf_z) == nh else sc(last_tvt),
        'pf_z_delta': ((pf_z - last_tvt).astype(np.float32) if len(pf_z) == nh else sc(0.0)),
        'pf_vs_z': ((pf_ancc - pf_z).astype(np.float32) if len(pf_z) == nh else sc(0.0)),
        # Beam signals
        **{f'beam_{t}_d': (p - np.float32(last_tvt)).astype(np.float32) for t, p in beam_paths.items()},
        'beam_mean_d': beam_stack.mean(axis=1).astype(np.float32),
        'beam_std_d': beam_stack.std(axis=1).astype(np.float32),
        'beam_med_d': np.median(beam_stack, axis=1).astype(np.float32),
        # NCC signals
        'sc8_d': (sc8 - np.float32(last_tvt)).astype(np.float32),
        'sc8_sc': sc8s,
        'sc15_d': (sc15 - np.float32(last_tvt)).astype(np.float32),
        'sc15_sc': sc15s,
        'sc25_d': (sc25 - np.float32(last_tvt)).astype(np.float32),
        'sc25_sc': sc25s,
        'sc_cons_d': (sc_cons - np.float32(last_tvt)).astype(np.float32),
        'sc_ens_d': (ncc_ensemble - np.float32(last_tvt)).astype(np.float32),
        'sc_trust': sc(sc_trust),
        'hyb_d': (hyb_ref - np.float32(last_tvt)).astype(np.float32),
        # Signal ensemble
        'sig_std': sig_std,
        'sig_mean_d': sig_mean,
        # Formation features
        **tvt_fs,
        **{f'frm_rmse_{fn}': sc(form_rmses[fn]) for fn in FORMATIONS},
        'form_mean_d': (fs.mean(axis=1) - last_tvt).astype(np.float32),
        'form_std_d': fs.std(axis=1).astype(np.float32),
        'form_rng_d': (fs.max(axis=1) - fs.min(axis=1)).astype(np.float32),
        # Dense ANCC
        'dense_ancc': dense_ancc,
        'dense_std': dense_std,
        'dense_dist': dense_dist,
        'tvt_dense_d': (tvt_dense - last_tvt).astype(np.float32),
        'tvt_densew_d': (tvt_densew - last_tvt).astype(np.float32),
        'tvt_dense50_d': (tvt_dense50 - last_tvt).astype(np.float32),
        'dense_rmse': sc(d_rmse),
        'dense_bias': sc(d_bias),
        'dense_nb_std': sc(d_nb_std),
        # Cross-signal
        'pf_vs_spatial': (pf_ancc - tvt_fs['tvtF_ANCC']).astype(np.float32),
        'pf_vs_dense': (pf_ancc - tvt_dense).astype(np.float32),
        'spatial_vs_dense': (tvt_fs['tvtF_ANCC'] - tvt_dense).astype(np.float32),
        'beam_vs_spatial': (beam_paths.get('cons', np.full(nh, last_tvt, np.float32)) - tvt_fs['tvtF_ANCC']).astype(np.float32),
        'sc_vs_beam': (ncc_ensemble - beam_paths.get('cons', np.full(nh, last_tvt, np.float32))).astype(np.float32),
        # Well metadata
        'cal_a': sc(a_cal), 'cal_b': sc(b_cal),
        'pfx_rmse': sc(scalars['gr_sigma']),
        'known_len': sc(scalars['known_len']),
        'eval_len': sc(nh),
        'slp_all': sc(scalars['slope_all']),
        'slp_50': sc(scalars['slope_50']),
        'slp_z': sc(scalars['slope_z']),
        'slp_b_d_all': (slp_b_all - last_tvt).astype(np.float32),
        'slp_b_d_50': (slp_b_50 - last_tvt).astype(np.float32),
        'ktvt_range': sc(scalars['ktvt_range']),
        'ktvt_std': sc(scalars['ktvt_std']),
        'tw_range': sc(scalars['tw_range']),
        'tw_gr_mean': sc(scalars['tw_gr_mean']),
        # Position
        'md_since': md_since,
        'frac': frac,
        'frac2': frac ** 2,
        'sqrt_frac': np.sqrt(frac),
        # Spatial
        'z': z_ev,
        'dx': (ev['X'].values - float(lk['X'])).astype(np.float32),
        'dy': (ev['Y'].values - float(lk['Y'])).astype(np.float32),
        'dz': (z_ev - float(lk['Z'])).astype(np.float32),
        'dxy': np.sqrt((ev['X'].values - float(lk['X'])) ** 2 + (ev['Y'].values - float(lk['Y'])) ** 2).astype(np.float32),
        'dzdmd': dzdmd, 'dxdmd': dxdmd, 'dydmd': dydmd,
        # GR
        'gr': hgr, 'gr_d1': gr_d1, 'gr_d2': gr_d2,
        'gr_env': gr_env, 'gr_nrg': gr_nrg,
        'gr_vs_tw_anc': hgr - np.float32(np.interp(last_tvt, tw_tvt, tw_gr)),
        'gr_vs_slp_all': hgr - np.interp(slp_b_all, tw_tvt, tw_gr).astype(np.float32),
        # GR offset probes (4 anchors x 11 offsets = 44 features)
        **{f'tda{int(o)}': hgr - np.float32(np.interp(last_tvt + o, tw_tvt, tw_gr)) for o in ANCHOR_OFFSETS},
        **{f'tdbc{int(o)}': hgr - np.interp(beam_ref + o, tw_tvt, tw_gr).astype(np.float32) for o in ANCHOR_OFFSETS},
        **{f'tdsc{int(o)}': hgr - np.interp(ncc_ensemble + o, tw_tvt, tw_gr).astype(np.float32) for o in ANCHOR_OFFSETS},
        **{f'tdpf{int(o)}': hgr - np.interp(pf_ancc + o, tw_tvt, tw_gr).astype(np.float32) for o in ANCHOR_OFFSETS},
        **rolls,
    }

    # LikPF features
    for k, v in likpf_dict.items():
        col = 'likpf_' + k.replace('pf_scale_', 'scale_').replace('pf_mean', 'mean')
        feats[col] = v.astype(np.float32) if len(v) == nh else sc(last_tvt)
        feats[col + '_d'] = (feats[col] - np.float32(last_tvt)).astype(np.float32)

    df = pd.DataFrame(feats)

    if is_train and 'TVT' in hw.columns:
        df['target'] = (ev['TVT'].values.astype(np.float32) - np.float32(last_tvt))

    return df


def _affine_cal(kgr, tw_at_k):
    """Affine GR calibration coefficients."""
    if len(kgr) < 10:
        return 1.0, 0.0
    A = np.column_stack([kgr, np.ones(len(kgr))])
    coeffs, _, _, _ = np.linalg.lstsq(A, tw_at_k, rcond=None)
    return float(coeffs[0]), float(coeffs[1])


# Feature columns used for model training (excludes metadata columns)
META_COLS = {'well', 'id', 'last_known_tvt', 'target', 'pf_ancc', 'pf_z'}

def get_feature_cols(df):
    """Return list of feature column names (excludes meta + target)."""
    return [c for c in df.columns if c not in META_COLS and df[c].dtype in (np.float32, np.float64)]
