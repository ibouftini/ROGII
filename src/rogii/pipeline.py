import os
import time
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm

from rogii.utils import WellData, load_hw, load_tw, list_wells, extract_wellname
from rogii.preprocess import detect_ps, interpolate_gr, calibrate_gr, extract_scalars
from rogii.neighbors import (assign_cluster, FormationPlaneKNN,
                              build_typewell_index, find_tw_match, FORMATIONS)
from rogii.alignment.particle_filter import run_pf_variants
from rogii.alignment.beam import run_beam_configs
from rogii.alignment.ncc import run_ncc_multiscale
from rogii.alignment.dtw import run_dtw_all_radii
from rogii.features import (build_alignment_df, compute_anchor_offsets,
                             compute_formation_features, compute_b_well,
                             compute_gr_features, compute_tabular_features,
                             build_feature_matrix)
from rogii.meta import group_kfold, train_lgb, train_xgb, train_catboost
from rogii.meta import ridge_stack, blend_predictions, save_models, load_models
from rogii.postprocess import postprocess_well, tune_postprocess

_BAR_FMT = '{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'


def process_well(
    name: str, hw: pd.DataFrame, tw: pd.DataFrame,
    knn: FormationPlaneKNN, tw_index: dict, cfg,
) -> WellData:
    """Preprocess a single well: interpolate, calibrate, detect PS, extract scalars."""
    hw = interpolate_gr(hw)
    ps_idx = detect_ps(hw)
    a, b, hw = calibrate_gr(hw, tw, ps_idx)
    scalars = extract_scalars(hw, ps_idx)

    tw_gr_mean = float(tw['GR'].mean())
    y_coord    = float(hw['Y'].mean())
    cluster_id = assign_cluster(tw_gr_mean, y_coord)
    formations = knn.predict(cluster_id, float(hw['X'].mean()), y_coord)
    tw_match   = find_tw_match(tw, tw_index)

    return WellData(name=name, hw=hw, tw=tw, ps_idx=ps_idx,
                    scalars=scalars, formations=formations,
                    cluster_id=cluster_id, tw_match=tw_match,
                    a_cal=a, b_cal=b)


def _compute_alignment(wd: WellData, cfg) -> dict:
    """Run all 4 alignment families for one well."""
    hw, tw = wd.hw, wd.tw
    ps = wd.ps_idx
    gr_eval = hw['GR'].values[ps:]
    tw_tvt, tw_gr = tw['TVT'].values, tw['GR'].values
    tvt_start = wd.scalars['last_known_tvt']

    pf_res   = run_pf_variants(gr_eval, tw_tvt, tw_gr, tvt_start, cfg.PF)
    beam_res = run_beam_configs(gr_eval, tw_tvt, tw_gr, tvt_start, cfg.BEAM_CONFIGS)
    ncc_res  = run_ncc_multiscale(hw['GR'].values, tw_tvt, tw_gr,
                                  beam_res['beam_ref'], ps, **cfg.NCC)
    dtw_res  = run_dtw_all_radii(gr_eval, tw_tvt, tw_gr, **cfg.DTW)

    return {**pf_res, **beam_res, **ncc_res, **dtw_res}


def _well_to_rows(wd: WellData, alignment: dict, cfg) -> tuple[np.ndarray, np.ndarray]:
    b_well = compute_b_well(wd.hw, wd.ps_idx, wd.formations,
                            cfg.FEATURES['b_well_decay'])
    return build_feature_matrix(wd.hw, wd.tw, wd.ps_idx, alignment,
                                wd.formations, b_well, wd.scalars,
                                wd.cluster_id, wd.a_cal, wd.b_cal)


def _build_knn(train_pairs: list, cfg) -> FormationPlaneKNN:
    wells = []
    for hw_path, tw_path in train_pairs:
        hw = load_hw(hw_path)
        tw = load_tw(tw_path)
        tw_gr_mean = float(tw['GR'].mean())
        y = float(hw['Y'].mean())
        cid = assign_cluster(tw_gr_mean, y)
        depths = {f: float(hw[f].mean()) if f in hw.columns else 0.0 for f in FORMATIONS}
        wells.append((cid, float(hw['X'].mean()), y, depths))
    return FormationPlaneKNN(k=10).fit(wells)


def _sep(char='=', width=65):
    tqdm.write(char * width)


def run_pipeline(cfg, mode: str = 'train',
                 train_dir: str = None, test_dir: str = None,
                 models_dir: str = None) -> pd.DataFrame | None:
    t_start = time.time()
    train_dir  = train_dir  or cfg.DATA['train_dir']
    test_dir   = test_dir   or cfg.DATA['test_dir']
    models_dir = models_dir or cfg.DATA['models_dir']
    os.makedirs(models_dir, exist_ok=True)

    _sep()
    tqdm.write(f'  ROGII Wellbore Geology Pipeline  |  mode: {mode.upper()}')
    _sep()

    train_pairs = list_wells(train_dir)
    tqdm.write(f'  train dir  : {train_dir}  ({len(train_pairs)} wells)')

    tqdm.write(f'\n[setup] Building FormationPlaneKNN + typewell index ...')
    t0 = time.time()
    knn      = _build_knn(train_pairs, cfg)
    tw_index = build_typewell_index(train_pairs)
    tqdm.write(f'[setup] Done  ({time.time()-t0:.1f}s)  k=10, {len(train_pairs)} anchor wells')

    def _load_and_process(hw_path, tw_path):
        name = extract_wellname(hw_path)
        hw   = load_hw(hw_path)
        tw   = load_tw(tw_path)
        wd   = process_well(name, hw, tw, knn, tw_index, cfg)
        aln  = _compute_alignment(wd, cfg)
        X, y = _well_to_rows(wd, aln, cfg)
        pf   = aln['pf_ancc']
        return wd.name, X, y, pf, wd.scalars

    if mode == 'train':
        tqdm.write(f'\n[1/4] Well processing  (PF × 256 runs + Beam × 7 + NCC × 3 + DTW × 4 per well) ...')
        t0 = time.time()
        results = list(tqdm(
            Parallel(n_jobs=-1, return_as='generator')(
                delayed(_load_and_process)(hp, tp) for hp, tp in train_pairs
            ),
            total=len(train_pairs),
            desc='  wells',
            unit='well',
            bar_format=_BAR_FMT,
            ncols=80,
        ))
        tqdm.write(f'[1/4] Done  ({time.time()-t0:.1f}s)')

        names  = [r[0] for r in results]
        X_all  = np.vstack([r[1] for r in results])
        y_all  = np.concatenate([r[2] for r in results])
        groups = np.concatenate([np.full(len(r[1]), i) for i, r in enumerate(results)])
        pf_all = np.concatenate([r[3] for r in results])

        n_wells = len(np.unique(groups))
        tqdm.write(
            f'\n[2/4] Feature matrix  :  {X_all.shape[0]:,} rows × {X_all.shape[1]} cols'
            f'  ({n_wells} wells,  y μ={y_all.mean():.4f}  σ={y_all.std():.4f})'
        )

        folds    = group_kfold(groups, cfg.CV['n_splits'])
        oof_dict = {}
        models   = {}

        tqdm.write(f'\n[3/4] Ensemble training  ({cfg.CV["n_splits"]}-fold GroupKFold) ...')
        _sep('-')
        t_train = time.time()

        # --- LightGBM variants ---
        for i, lparams in enumerate(cfg.LGB_VARIANTS):
            oof = np.zeros(len(y_all))
            fold_models = []
            desc = f'  LGB-{i}  leaves={lparams["num_leaves"]} lr={lparams["learning_rate"]}'
            bar  = tqdm(enumerate(folds), total=len(folds), desc=desc,
                        unit='fold', bar_format=_BAR_FMT, ncols=80)
            for fold_j, (tr_idx, val_idx) in bar:
                m, pred = train_lgb(X_all[tr_idx], y_all[tr_idx],
                                    X_all[val_idx], y_all[val_idx], lparams.copy())
                oof[val_idx] = pred
                fold_models.append(m)
                rmse = float(np.sqrt(np.mean((pred - y_all[val_idx]) ** 2)))
                bar.set_postfix(fold=f'{fold_j+1}/{len(folds)}', val_rmse=f'{rmse:.5f}')
            oof_rmse_i = float(np.sqrt(np.mean((oof - y_all) ** 2)))
            tqdm.write(f'    LGB-{i}  OOF RMSE: {oof_rmse_i:.5f}')
            oof_dict[f'lgb{i}'] = oof
            for fi, fm in enumerate(fold_models):
                models[f'lgb{i}_f{fi}'] = fm

        # --- XGBoost ---
        xgb_oof = np.zeros(len(y_all))
        xgb_fold_models = []
        desc = f'  XGB    depth={cfg.XGB["max_depth"]} lr={cfg.XGB["learning_rate"]}'
        bar  = tqdm(enumerate(folds), total=len(folds), desc=desc,
                    unit='fold', bar_format=_BAR_FMT, ncols=80)
        for fold_j, (tr_idx, val_idx) in bar:
            m, pred = train_xgb(X_all[tr_idx], y_all[tr_idx],
                                 X_all[val_idx], y_all[val_idx], cfg.XGB.copy())
            xgb_oof[val_idx] = pred
            xgb_fold_models.append(m)
            rmse = float(np.sqrt(np.mean((pred - y_all[val_idx]) ** 2)))
            bar.set_postfix(fold=f'{fold_j+1}/{len(folds)}', val_rmse=f'{rmse:.5f}')
        oof_rmse_xgb = float(np.sqrt(np.mean((xgb_oof - y_all) ** 2)))
        tqdm.write(f'    XGB    OOF RMSE: {oof_rmse_xgb:.5f}')
        oof_dict['xgb'] = xgb_oof
        for fi, fm in enumerate(xgb_fold_models):
            models[f'xgb_f{fi}'] = fm

        # --- CatBoost ---
        cat_feat_idx = []
        cb_oof = np.zeros(len(y_all))
        cb_fold_models = []
        desc = f'  CatBoost depth={cfg.CATBOOST["depth"]} lr={cfg.CATBOOST["learning_rate"]}'
        bar  = tqdm(enumerate(folds), total=len(folds), desc=desc,
                    unit='fold', bar_format=_BAR_FMT, ncols=80)
        for fold_j, (tr_idx, val_idx) in bar:
            m, pred = train_catboost(X_all[tr_idx], y_all[tr_idx],
                                     X_all[val_idx], y_all[val_idx],
                                     cfg.CATBOOST.copy(), cat_feat_idx)
            cb_oof[val_idx] = pred
            cb_fold_models.append(m)
            rmse = float(np.sqrt(np.mean((pred - y_all[val_idx]) ** 2)))
            bar.set_postfix(fold=f'{fold_j+1}/{len(folds)}', val_rmse=f'{rmse:.5f}')
        oof_rmse_cb = float(np.sqrt(np.mean((cb_oof - y_all) ** 2)))
        tqdm.write(f'    CatBoost OOF RMSE: {oof_rmse_cb:.5f}')
        oof_dict['catboost'] = cb_oof
        for fi, fm in enumerate(cb_fold_models):
            models[f'catboost_f{fi}'] = fm

        _sep('-')
        tqdm.write(f'[3/4] Training done  ({time.time()-t_train:.1f}s)')

        # --- Ridge stacking ---
        tqdm.write(f'\n[4/4] Ridge stacking  (NNLS, alpha={cfg.RIDGE["alpha"]}) ...')
        stack_w = ridge_stack(oof_dict, y_all, cfg.RIDGE['alpha'])
        model_names = list(oof_dict.keys())
        weight_str  = '  '.join(f'{n}={w:.3f}' for n, w in zip(model_names, stack_w))
        tqdm.write(f'       weights →  {weight_str}')

        save_models(models, {'oof': oof_dict, 'stack_w': stack_w}, models_dir)

        oof_rmse = float(np.sqrt(np.mean(
            (y_all - np.column_stack(list(oof_dict.values())) @ stack_w) ** 2
        )))

        _sep()
        tqdm.write(f'  OOF increment RMSE (stacked) : {oof_rmse:.5f}')
        tqdm.write(f'  Per-model OOF RMSE           : '
                   + '  '.join(f'{n}={v:.4f}' for n, v in [
                       (f'lgb{i}', float(np.sqrt(np.mean((oof_dict[f"lgb{i}"] - y_all)**2))))
                       for i in range(len(cfg.LGB_VARIANTS))
                   ] + [
                       ('xgb',      float(np.sqrt(np.mean((xgb_oof  - y_all)**2)))),
                       ('catboost', float(np.sqrt(np.mean((cb_oof   - y_all)**2)))),
                   ]))
        tqdm.write(f'  Models saved to              : {models_dir}/')
        tqdm.write(f'  Total elapsed                : {time.time()-t_start:.1f}s')
        _sep()
        return None

    elif mode == 'predict':
        test_pairs = list_wells(test_dir)
        tqdm.write(f'  test  dir  : {test_dir}  ({len(test_pairs)} wells)')

        tqdm.write(f'\n[1/2] Loading models from {models_dir}/ ...')
        loaded_models, meta = load_models(models_dir)
        stack_w = meta.get('stack_w', np.array([1.0]))
        tqdm.write(f'[1/2] Loaded {len(loaded_models)} model files')

        tqdm.write(f'\n[2/2] Running inference on {len(test_pairs)} test wells ...')
        t0 = time.time()
        rows = []
        for hp, tp in tqdm(test_pairs, desc='  test wells', unit='well',
                           bar_format=_BAR_FMT, ncols=80):
            name = extract_wellname(hp)
            hw   = load_hw(hp)
            tw   = load_tw(tp)
            wd   = process_well(name, hw, tw, knn, tw_index, cfg)
            aln  = _compute_alignment(wd, cfg)
            X, _ = _well_to_rows(wd, aln, cfg)

            from collections import defaultdict as _dd
            import xgboost as xgb_lib
            fold_groups: dict = _dd(list)
            for mname in loaded_models:
                base_name = mname.rsplit('_f', 1)[0] if '_f' in mname else mname
                fold_groups[base_name].append(mname)
            preds = []
            for base_name in sorted(fold_groups):
                fold_preds = []
                for mname in fold_groups[base_name]:
                    m = loaded_models[mname]
                    if isinstance(m, xgb_lib.Booster):
                        fold_preds.append(m.predict(xgb_lib.DMatrix(X)))
                    else:
                        fold_preds.append(m.predict(X))
                preds.append(np.mean(fold_preds, axis=0))
            if not preds:
                tqdm.write(f'  [warn] no predictions for {name}, skipping')
                continue
            base    = np.column_stack(preds)
            pf_d    = np.diff(np.concatenate([[wd.scalars['last_known_tvt']], aln['pf_ancc']]))
            d_blend = blend_predictions(base, stack_w[:base.shape[1]], pf_d, cfg.BLEND['w_pf'])
            z_eval  = wd.hw.iloc[wd.ps_idx:]['Z'].values
            md_eval = wd.hw.iloc[wd.ps_idx:]['MD'].values
            tvt_pred = postprocess_well(
                d_blend, pf_d, z_eval,
                md_eval - md_eval[0],
                wd.scalars['last_known_tvt'],
                cfg.PP, cfg.USPACE,
            )
            eval_hw = wd.hw.iloc[wd.ps_idx:]
            for idx, tvt_val in zip(eval_hw.index, tvt_pred):
                row_md = int(wd.hw.loc[idx, 'MD'])
                rows.append({'id': f'{name}_{row_md}', 'tvt': tvt_val})

        df_out = pd.DataFrame(rows)
        _sep()
        tqdm.write(f'  Predictions  : {len(df_out):,} rows  ({len(test_pairs)} wells)')
        tqdm.write(f'  TVT range    : [{df_out["tvt"].min():.2f}, {df_out["tvt"].max():.2f}]  '
                   f'mean={df_out["tvt"].mean():.2f}')
        tqdm.write(f'  Inference time : {time.time()-t0:.1f}s')
        tqdm.write(f'  Total elapsed  : {time.time()-t_start:.1f}s')
        _sep()
        return df_out

    return pd.DataFrame()
