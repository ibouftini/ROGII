import os
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

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


def run_pipeline(cfg, mode: str = 'train',
                 train_dir: str = None, test_dir: str = None,
                 models_dir: str = None) -> pd.DataFrame | None:
    train_dir  = train_dir  or cfg.DATA['train_dir']
    test_dir   = test_dir   or cfg.DATA['test_dir']
    models_dir = models_dir or cfg.DATA['models_dir']
    os.makedirs(models_dir, exist_ok=True)

    train_pairs = list_wells(train_dir)
    knn         = _build_knn(train_pairs, cfg)
    tw_index    = build_typewell_index(train_pairs)

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
        results = Parallel(n_jobs=-1)(
            delayed(_load_and_process)(hp, tp) for hp, tp in train_pairs
        )
        names   = [r[0] for r in results]
        X_all   = np.vstack([r[1] for r in results])
        y_all   = np.concatenate([r[2] for r in results])
        groups  = np.concatenate([np.full(len(r[1]), i) for i, r in enumerate(results)])
        pf_all  = np.concatenate([r[3] for r in results])

        folds    = group_kfold(groups, cfg.CV['n_splits'])
        oof_dict = {}
        models   = {}

        for i, lparams in enumerate(cfg.LGB_VARIANTS):
            oof = np.zeros(len(y_all))
            fold_models = []
            for tr_idx, val_idx in folds:
                m, pred = train_lgb(X_all[tr_idx], y_all[tr_idx],
                                    X_all[val_idx], y_all[val_idx], lparams.copy())
                oof[val_idx] = pred
                fold_models.append(m)
            oof_dict[f'lgb{i}'] = oof
            models[f'lgb{i}'] = fold_models[-1]

        xgb_oof = np.zeros(len(y_all))
        for tr_idx, val_idx in folds:
            m, pred = train_xgb(X_all[tr_idx], y_all[tr_idx],
                                 X_all[val_idx], y_all[val_idx], cfg.XGB.copy())
            xgb_oof[val_idx] = pred
        oof_dict['xgb'] = xgb_oof
        models['xgb'] = m

        cat_feat_idx = []
        cb_oof = np.zeros(len(y_all))
        for tr_idx, val_idx in folds:
            m, pred = train_catboost(X_all[tr_idx], y_all[tr_idx],
                                     X_all[val_idx], y_all[val_idx],
                                     cfg.CATBOOST.copy(), cat_feat_idx)
            cb_oof[val_idx] = pred
        oof_dict['catboost'] = cb_oof
        models['catboost'] = m

        stack_w = ridge_stack(oof_dict, y_all, cfg.RIDGE['alpha'])
        save_models(models, {'oof': oof_dict, 'stack_w': stack_w}, models_dir)
        oof_rmse = float(np.sqrt(np.mean(
            (y_all - np.column_stack(list(oof_dict.values())) @ stack_w) ** 2
        )))
        print(f'OOF increment RMSE: {oof_rmse:.4f}')
        return None

    elif mode == 'predict':
        test_pairs = list_wells(test_dir)
        loaded_models, meta = load_models(models_dir)
        stack_w = meta.get('stack_w', np.array([1.0]))
        rows = []
        for hp, tp in test_pairs:
            name = extract_wellname(hp)
            hw   = load_hw(hp)
            tw   = load_tw(tp)
            wd   = process_well(name, hw, tw, knn, tw_index, cfg)
            aln  = _compute_alignment(wd, cfg)
            X, _ = _well_to_rows(wd, aln, cfg)
            preds = []
            for mname in sorted(loaded_models):
                m = loaded_models[mname]
                if hasattr(m, 'predict'):
                    import xgboost as xgb_lib
                    if isinstance(m, xgb_lib.Booster):
                        preds.append(m.predict(xgb_lib.DMatrix(X)))
                    else:
                        preds.append(m.predict(X))
            if not preds:
                continue
            base = np.column_stack(preds)
            pf_d = np.diff(np.concatenate([[wd.scalars['last_known_tvt']], aln['pf_ancc']]))
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
        return pd.DataFrame(rows)

    return pd.DataFrame()
