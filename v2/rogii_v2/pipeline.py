"""End-to-end train + predict orchestration."""
import os
import time
import pickle
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm

from rogii_v2.data import (list_wells, load_well, interpolate_gr, detect_ps,
                           calibrate_gr, extract_scalars, extract_wellname)
from rogii_v2.trackers import (run_pf_ancc, run_pf_z, run_lik_pf,
                               run_beam_configs, multi_scale_ncc)
from rogii_v2.spatial import (FormationPlaneKNN, DenseANCCImputer,
                              build_typewell_index, find_tw_match, FORMATIONS)
from rogii_v2.features import build_well_features, get_feature_cols
from rogii_v2.models import (group_kfold, train_lgb, train_catboost,
                             ridge_stack, save_models, load_models)
from rogii_v2.postprocess import (selector_well_code, parse_selector_variant,
                                  apply_selector, warmup, savgol_smooth,
                                  uspace_projection, guarded_contact_override)


def _process_one_well(hw_path, tw_path, knn, dense_imp, cfg, is_train):
    """Process one well: preprocess, run trackers, build features. Returns DataFrame."""
    wid = extract_wellname(hw_path)
    hw, tw = load_well(hw_path, tw_path)
    hw = interpolate_gr(hw)
    ps_idx = detect_ps(hw)
    if ps_idx == len(hw) or ps_idx < 10:
        return None

    tw_tvt = tw['TVT'].values.astype(float)
    tw_gr = tw['GR'].fillna(tw['GR'].mean()).values.astype(float)
    scalars = extract_scalars(hw, tw, ps_idx)
    if scalars is None:
        return None

    kn = hw.iloc[:ps_idx]
    ev = hw.iloc[ps_idx:]
    last_tvt = scalars['last_known_tvt']

    # Trackers
    pf_a, pf_a_std = run_pf_ancc(hw, tw_tvt, tw_gr, cfg)
    if len(pf_a) == 0:
        return None
    pf_z, pf_z_std = run_pf_z(hw, tw_tvt, tw_gr, cfg)

    gr_interp = hw['GR'].interpolate(limit_direction='both').fillna(float(tw_gr.mean()))
    hgr = gr_interp.iloc[ev.index[0]:].values.astype(np.float32)
    beams = run_beam_configs(hgr, tw_tvt, tw_gr, last_tvt, cfg.BEAM_CONFIGS)

    kgr = gr_interp.iloc[:ps_idx].values.astype(np.float32)
    ktvt = kn['TVT_input'].dropna().values.astype(np.float32)
    ncc_results, ncc_ens = multi_scale_ncc(kgr, ktvt, hgr)

    # LikPF
    likpf_dict, _ = run_lik_pf(hw, tw_tvt, tw_gr, cfg)

    # Spatial imputation
    xy_ev = ev[['X', 'Y']].values.astype(np.float64)
    xy_kn = kn[['X', 'Y']].values.astype(np.float64) if len(kn) > 0 else np.zeros((0, 2))
    self_wid = wid if is_train else None
    form_ev, knn_dist = knn.impute(xy_ev, self_wid=self_wid)
    form_kn, _ = knn.impute(xy_kn, self_wid=self_wid) if len(xy_kn) > 0 else (np.zeros((0, 6), np.float32), None)
    dense_ev, dense_std, dense_dist = dense_imp.impute(xy_ev, self_wid=self_wid)
    dense_kn, dense_std_kn, _ = dense_imp.impute(xy_kn, self_wid=self_wid) if len(xy_kn) > 0 else (np.zeros(0, np.float32), np.zeros(0, np.float32), None)

    df = build_well_features(
        hw, tw, ps_idx, scalars,
        pf_a, pf_a_std, pf_z, pf_z_std,
        beams, ncc_results, ncc_ens,
        likpf_dict,
        form_ev, form_kn, dense_ev, dense_std, dense_dist,
        dense_kn, dense_std_kn, knn_dist,
        is_train, wid,
    )
    return df


def run_pipeline(cfg, mode='train', train_dir=None, test_dir=None,
                 models_dir=None, output_path=None):
    t_start = time.time()
    train_dir = train_dir or str(cfg.TRAIN_DIR)
    test_dir = test_dir or str(cfg.TEST_DIR)
    models_dir = models_dir or str(cfg.MODELS_DIR)
    os.makedirs(models_dir, exist_ok=True)

    train_pairs = list_wells(train_dir)
    train_wids = [extract_wellname(p[0]) for p in train_pairs]
    print(f'=== ROGII v2 Pipeline | mode={mode.upper()} ===')
    print(f'Training wells: {len(train_pairs)}')

    # Build spatial imputers
    knn_path = os.path.join(models_dir, 'knn.pkl')
    dense_path = os.path.join(models_dir, 'dense.pkl')
    twi_path = os.path.join(models_dir, 'tw_index.pkl')

    if mode == 'predict' and os.path.exists(knn_path):
        with open(knn_path, 'rb') as f:
            knn = pickle.load(f)
        with open(dense_path, 'rb') as f:
            dense_imp = pickle.load(f)
        with open(twi_path, 'rb') as f:
            tw_index = pickle.load(f)
        print('[setup] Loaded spatial imputers from disk')
    else:
        print('[setup] Building spatial imputers...')
        t0 = time.time()
        knn = FormationPlaneKNN(train_wids, train_dir, k=cfg.KNN_K)
        dense_imp = DenseANCCImputer(train_wids, train_dir, k=cfg.DENSE_K, n_samples=cfg.DENSE_SAMPLES)
        tw_index = build_typewell_index(train_pairs)
        print(f'[setup] Done ({time.time()-t0:.1f}s)')

    # -----------------------------------------------------------------------
    if mode == 'train':
        print(f'\n[1/3] Processing {len(train_pairs)} wells...')
        t0 = time.time()
        results = Parallel(n_jobs=cfg.N_JOBS, prefer='threads')(
            delayed(_process_one_well)(hp, tp, knn, dense_imp, cfg, True)
            for hp, tp in tqdm(train_pairs, desc='wells')
        )
        parts = [r for r in results if r is not None]
        df = pd.concat(parts, ignore_index=True)
        print(f'[1/3] Done ({time.time()-t0:.1f}s) -- {len(df):,} rows from {len(parts)} wells')

        feat_cols = get_feature_cols(df)
        X = df[feat_cols].values.astype(np.float32)
        y = df['target'].values.astype(np.float32)
        wells = df['well'].values
        unique_wells = np.unique(wells)
        groups = np.searchsorted(unique_wells, wells)

        print(f'\n[2/3] Training {len(feat_cols)} features, {len(unique_wells)} wells')
        print(f'  target: mean={y.mean():.3f}, std={y.std():.3f}')

        folds = group_kfold(groups, cfg.N_SPLITS)
        oof_dict = {}
        models = {}

        # LightGBM variants
        for i, lparams in enumerate(cfg.LGB_VARIANTS):
            oof = np.zeros(len(y))
            for fi, (tr_idx, val_idx) in enumerate(folds):
                m, pred = train_lgb(X[tr_idx], y[tr_idx], X[val_idx], y[val_idx],
                                    lparams.copy(), cfg.EARLY_STOPPING)
                oof[val_idx] = pred
                models[f'lgb{i}_f{fi}'] = m
            rmse = float(np.sqrt(np.mean((oof - y) ** 2)))
            print(f'  LGB-{i} OOF RMSE: {rmse:.4f}')
            oof_dict[f'lgb{i}'] = oof

        # CatBoost variants
        for i, cparams in enumerate(cfg.CB_VARIANTS):
            oof = np.zeros(len(y))
            for fi, (tr_idx, val_idx) in enumerate(folds):
                m, pred = train_catboost(X[tr_idx], y[tr_idx], X[val_idx], y[val_idx],
                                         cparams.copy(), cfg.EARLY_STOPPING)
                oof[val_idx] = pred
                models[f'cb{i}_f{fi}'] = m
            rmse = float(np.sqrt(np.mean((oof - y) ** 2)))
            print(f'  CB-{i} OOF RMSE: {rmse:.4f}')
            oof_dict[f'cb{i}'] = oof

        # Ridge stack
        stack_w, model_names = ridge_stack(oof_dict, y, cfg.RIDGE_ALPHA)
        oof_stacked = np.column_stack([oof_dict[n] for n in model_names]) @ stack_w
        stack_rmse = float(np.sqrt(np.mean((oof_stacked - y) ** 2)))
        print(f'  Ridge stack OOF RMSE: {stack_rmse:.4f}')
        print(f'  Weights: {dict(zip(model_names, stack_w.round(3)))}')

        # Save
        meta = {'stack_w': stack_w, 'model_names': model_names,
                'feat_cols': feat_cols, 'oof': oof_dict}
        save_models(models, meta, models_dir)
        with open(knn_path, 'wb') as f:
            pickle.dump(knn, f, protocol=4)
        with open(dense_path, 'wb') as f:
            pickle.dump(dense_imp, f, protocol=4)
        with open(twi_path, 'wb') as f:
            pickle.dump(tw_index, f, protocol=4)

        print(f'\n[3/3] Models saved to {models_dir}/')
        print(f'Total time: {time.time()-t_start:.1f}s')

    # -----------------------------------------------------------------------
    elif mode == 'predict':
        test_pairs = list_wells(test_dir)
        print(f'Test wells: {len(test_pairs)}')

        loaded_models, meta = load_models(models_dir)
        stack_w = meta['stack_w']
        model_names = meta['model_names']
        feat_cols = meta['feat_cols']
        print(f'Loaded {len(loaded_models)} model files')

        # Process test wells
        print(f'\n[1/2] Processing {len(test_pairs)} test wells...')
        t0 = time.time()
        results = Parallel(n_jobs=cfg.N_JOBS, prefer='threads')(
            delayed(_process_one_well)(hp, tp, knn, dense_imp, cfg, False)
            for hp, tp in tqdm(test_pairs, desc='wells')
        )
        parts = [r for r in results if r is not None]
        df = pd.concat(parts, ignore_index=True)
        print(f'[1/2] Done ({time.time()-t0:.1f}s) -- {len(df):,} rows')

        # Model predictions
        X = df[feat_cols].values.astype(np.float32)

        # Average across folds per base model
        fold_groups = {}
        for mname in sorted(loaded_models):
            base = mname.rsplit('_f', 1)[0]
            fold_groups.setdefault(base, []).append(mname)

        preds = {}
        for base in model_names:
            fold_preds = []
            for mname in fold_groups[base]:
                m = loaded_models[mname]
                if hasattr(m, 'predict'):
                    fold_preds.append(m.predict(X))
                else:
                    import xgboost as xgb
                    fold_preds.append(m.predict(xgb.DMatrix(X)))
            preds[base] = np.mean(fold_preds, axis=0)

        model_delta = np.column_stack([preds[n] for n in model_names]) @ stack_w

        # LikPF delta for blending
        likpf_col = f'likpf_{cfg.PP_LIKPF_SCALE}'
        if likpf_col in df.columns:
            likpf_delta = df[likpf_col].values - df['last_known_tvt'].values
        else:
            likpf_delta = model_delta  # fallback

        # Warm-up + blend model with likpf
        md_since = df['md_since'].values
        sub1 = cfg.PP_ALPHA * warmup(md_since, cfg.PP_TAU) * model_delta
        delta_model = cfg.PP_W_SUB1 * sub1 + (1.0 - cfg.PP_W_SUB1) * likpf_delta

        # Selector per-well
        print(f'\n[2/2] Post-processing...')
        all_rows = []
        for wid, gdf in df.groupby('well', sort=False):
            idx = gdf.index.values
            last_tvt = float(gdf['last_known_tvt'].iloc[0])

            # Get PF scales for this well
            pf_scales = {}
            for sc in cfg.PF_SCALES:
                col = f'likpf_scale_{sc:g}'
                if col in gdf.columns:
                    pf_scales[f'scale_{sc:g}'] = gdf[col].values
            if 'likpf_mean' in gdf.columns:
                pf_scales['mean'] = gdf['likpf_mean'].values

            # Beam reference
            beam_ref_col = 'beam_beam_ref_d' if 'beam_beam_ref_d' in gdf.columns else 'beam_cons_d'
            tvt_beam = gdf[beam_ref_col].values + last_tvt if beam_ref_col in gdf.columns else np.full(len(gdf), last_tvt)

            # Selector
            hw_path = os.path.join(test_dir, f'{wid}__horizontal_well.csv')
            hw_test = pd.read_csv(hw_path) if os.path.exists(hw_path) else None
            if hw_test is not None:
                code, n_eval, z_span = selector_well_code(
                    hw_test, cfg.SELECTOR_N_EVAL_THRESHOLD,
                    cfg.SELECTOR_Z_SPAN_THRESHOLDS)
                variant = cfg.SELECTOR_BIN_VARIANTS.get(code, cfg.SELECTOR_GLOBAL_VARIANT)
                selector_tvt = apply_selector(variant, pf_scales, tvt_beam, last_tvt)
            else:
                selector_tvt = pf_scales.get('scale_5', np.full(len(gdf), last_tvt))

            selector_delta = selector_tvt - last_tvt

            # Blend model + selector
            final_delta = cfg.PP_MODEL_W * delta_model[idx] + cfg.PP_SELECTOR_W * selector_delta

            # Reconstruct TVT
            tvt_pred = last_tvt + final_delta

            # U-space projection
            z_ev = gdf['z'].values if 'z' in gdf.columns else np.zeros(len(gdf))
            tvt_pred = uspace_projection(tvt_pred, z_ev, last_tvt,
                                         cfg.USPACE_DEGREE, cfg.USPACE_ITERS,
                                         cfg.USPACE_C, cfg.USPACE_BLEND)

            # Contact override for visible wells
            if wid in train_wids:
                try:
                    hw_tr, tw_tr = load_well(
                        os.path.join(train_dir, f'{wid}__horizontal_well.csv'),
                        os.path.join(train_dir, f'{wid}__typewell.csv'))
                    tvt_pred = guarded_contact_override(tvt_pred, hw_test, hw_tr, tw_tr)
                except Exception:
                    pass

            for row_id, tvt_val in zip(gdf['id'].values, tvt_pred):
                all_rows.append({'id': row_id, 'tvt': float(tvt_val)})

        sub = pd.DataFrame(all_rows)

        # SG smoothing (needs well grouping)
        well_groups = {}
        for i, (_, row) in enumerate(sub.iterrows()):
            w = row['id'][:8]
            well_groups.setdefault(w, []).append(i)
        sub['tvt'] = savgol_smooth(sub['tvt'].values, well_groups,
                                   cfg.PP_SG_WIN, cfg.PP_SG_POLY)

        out_path = output_path or str(cfg.SUBMISSIONS_DIR / 'submission_v2.csv')
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        sub.to_csv(out_path, index=False)
        print(f'\nSubmission: {len(sub):,} rows -> {out_path}')
        print(f'TVT range: [{sub["tvt"].min():.2f}, {sub["tvt"].max():.2f}]')
        print(f'Total time: {time.time()-t_start:.1f}s')
        return sub
