import os
import time
import hashlib
import pickle
from collections import defaultdict
import numpy as np
import pandas as pd
import xgboost as xgb_lib
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


# ---------------------------------------------------------------------------
# Disk cache helpers
# ---------------------------------------------------------------------------

def _cfg_hash(cfg) -> str:
    """8-char hash of alignment+feature config. Cache invalidates on any change."""
    key = str((cfg.PF, cfg.BEAM_CONFIGS, cfg.NCC, cfg.DTW, cfg.FEATURES))
    return hashlib.md5(key.encode()).hexdigest()[:8]


def _cache_path(cache_dir: str, name: str, h: str) -> str:
    return os.path.join(cache_dir, f'{name}_{h}.pkl')


def _save_cache(path: str, data) -> None:
    with open(path, 'wb') as f:
        pickle.dump(data, f, protocol=4)


def _load_cache(path: str):
    if not os.path.exists(path):
        return None
    try:
        with open(path, 'rb') as f:
            return pickle.load(f)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Well processing helpers
# ---------------------------------------------------------------------------

def process_well(
    name: str, hw: pd.DataFrame, tw: pd.DataFrame,
    knn: FormationPlaneKNN, tw_index: dict, cfg,
) -> WellData:
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


# ---------------------------------------------------------------------------
# Module-level predict worker — must be at module scope for joblib pickling
# ---------------------------------------------------------------------------

def _predict_one_well(hw_path, tw_path, knn, tw_index, loaded_models, stack_w,
                      cfg, cache_dir, cfg_h):
    """Process one test well; returns list of {'id': ..., 'tvt': ...} dicts."""
    name = extract_wellname(hw_path)
    hw   = load_hw(hw_path)
    tw   = load_tw(tw_path)
    wd   = process_well(name, hw, tw, knn, tw_index, cfg)

    cpath = _cache_path(cache_dir, wd.name, cfg_h)
    aln   = _load_cache(cpath)
    if aln is None:
        aln = _compute_alignment(wd, cfg)
        _save_cache(cpath, aln)

    X, _ = _well_to_rows(wd, aln, cfg)

    fold_groups: dict = defaultdict(list)
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
        return []

    base     = np.column_stack(preds)
    pf_d     = np.diff(np.concatenate([[wd.scalars['last_known_tvt']], aln['pf_ancc']]))
    d_blend  = blend_predictions(base, stack_w[:base.shape[1]], pf_d, cfg.BLEND['w_pf'])
    z_eval   = wd.hw.iloc[wd.ps_idx:]['Z'].values
    md_eval  = wd.hw.iloc[wd.ps_idx:]['MD'].values
    tvt_pred = postprocess_well(
        d_blend, pf_d, z_eval,
        md_eval - md_eval[0],
        wd.scalars['last_known_tvt'],
        cfg.PP, cfg.USPACE,
    )
    eval_hw = wd.hw.iloc[wd.ps_idx:]
    rows = []
    for idx, tvt_val in zip(eval_hw.index, tvt_pred):
        row_md = int(wd.hw.loc[idx, 'MD'])
        rows.append({'id': f'{name}_{row_md}', 'tvt': tvt_val})
    return rows


# ---------------------------------------------------------------------------
# Train worker — also at module scope for joblib
# ---------------------------------------------------------------------------

def _load_and_process(hw_path, tw_path, knn, tw_index, cfg, cache_dir, cfg_h):
    name = extract_wellname(hw_path)
    hw   = load_hw(hw_path)
    tw   = load_tw(tw_path)
    wd   = process_well(name, hw, tw, knn, tw_index, cfg)

    cpath = _cache_path(cache_dir, wd.name, cfg_h)
    aln   = _load_cache(cpath)
    if aln is None:
        aln = _compute_alignment(wd, cfg)
        _save_cache(cpath, aln)

    X, y = _well_to_rows(wd, aln, cfg)
    return wd.name, X, y, aln['pf_ancc'], wd.scalars


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

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

    cache_dir = cfg.DATA.get('cache_dir', 'data/cache')
    os.makedirs(cache_dir, exist_ok=True)
    cfg_h = _cfg_hash(cfg)

    # KNN / typewell index: load saved artefacts if present, else build from scratch
    knn_path = os.path.join(models_dir, 'knn.pkl')
    twi_path = os.path.join(models_dir, 'tw_index.pkl')

    if mode == 'predict' and os.path.exists(knn_path) and os.path.exists(twi_path):
        with open(knn_path, 'rb') as f:
            knn = pickle.load(f)
        with open(twi_path, 'rb') as f:
            tw_index = pickle.load(f)
        tqdm.write(f'[setup] KNN + typewell index loaded from {models_dir}/')
    else:
        tqdm.write(f'\n[setup] Building FormationPlaneKNN + typewell index ...')
        t0 = time.time()
        knn      = _build_knn(train_pairs, cfg)
        tw_index = build_typewell_index(train_pairs)
        tqdm.write(f'[setup] Done  ({time.time()-t0:.1f}s)  k=10, {len(train_pairs)} anchor wells')

    n_cached = sum(
        1 for hp, _ in train_pairs
        if os.path.exists(_cache_path(cache_dir, extract_wellname(hp), cfg_h))
    )
    tqdm.write(f'[cache] {n_cached}/{len(train_pairs)} train wells cached  '
               f'(dir={cache_dir}/  hash={cfg_h})')

    # -----------------------------------------------------------------------
    # TRAIN
    # -----------------------------------------------------------------------
    if mode == 'train':
        tqdm.write(f'\n[1/4] Well processing  (PF × 256 runs + Beam × 7 + NCC × 3 + DTW × 4 per well) ...')
        t0 = time.time()
        results = list(tqdm(
            Parallel(n_jobs=-1, return_as='generator')(
                delayed(_load_and_process)(hp, tp, knn, tw_index, cfg, cache_dir, cfg_h)
                for hp, tp in train_pairs
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

        # Save KNN + typewell index so predict mode skips reading all training CSVs
        with open(knn_path, 'wb') as f:
            pickle.dump(knn, f, protocol=4)
        with open(twi_path, 'wb') as f:
            pickle.dump(tw_index, f, protocol=4)
        tqdm.write(f'       KNN + typewell index saved to {models_dir}/')

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

    # -----------------------------------------------------------------------
    # PREDICT
    # -----------------------------------------------------------------------
    elif mode == 'predict':
        test_pairs = list_wells(test_dir)
        tqdm.write(f'  test  dir  : {test_dir}  ({len(test_pairs)} wells)')

        tqdm.write(f'\n[1/2] Loading models from {models_dir}/ ...')
        loaded_models, meta = load_models(models_dir)
        stack_w = meta.get('stack_w', np.array([1.0]))
        tqdm.write(f'[1/2] Loaded {len(loaded_models)} model files')

        n_test_cached = sum(
            1 for hp, _ in test_pairs
            if os.path.exists(_cache_path(cache_dir, extract_wellname(hp), cfg_h))
        )
        tqdm.write(f'[cache] {n_test_cached}/{len(test_pairs)} test wells cached')

        tqdm.write(f'\n[2/2] Running inference on {len(test_pairs)} test wells ...')
        t0 = time.time()

        all_rows = list(tqdm(
            Parallel(n_jobs=-1, return_as='generator')(
                delayed(_predict_one_well)(
                    hp, tp, knn, tw_index, loaded_models, stack_w,
                    cfg, cache_dir, cfg_h,
                )
                for hp, tp in test_pairs
            ),
            total=len(test_pairs),
            desc='  test wells',
            unit='well',
            bar_format=_BAR_FMT,
            ncols=80,
        ))

        rows = [row for well_rows in all_rows if well_rows for row in well_rows]
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
