# auto-generated bundle — do not edit


# --- src/rogii/features.py ---
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
    """NCC at each of 11 TVT offsets around the baseline prediction."""
    n = len(baseline_tvt)
    result = np.zeros((n, len(ANCHOR_OFFSETS)))

    def _ncc(a: np.ndarray, b: np.ndarray) -> float:
        a, b = a - a.mean(), b - b.mean()
        d = np.sqrt((a ** 2).sum() * (b ** 2).sum())
        return float(np.dot(a, b) / d) if d > 1e-8 else 0.0

    for i in range(n):
        lo, hi = max(0, i - window), min(n, i + window + 1)
        hw_win = hw_gr[lo:hi]
        for j, off in enumerate(ANCHOR_OFFSETS):
            cand = baseline_tvt[i] + off
            pts = np.linspace(cand - window * 0.5, cand + window * 0.5, len(hw_win))
            tw_win = np.interp(pts, tw_tvt, tw_gr)
            result[i, j] = _ncc(hw_win, tw_win)

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


# --- src/rogii/meta.py ---
import numpy as np
import lightgbm as lgb
import xgboost as xgb
import os, pickle


def group_kfold(groups: np.ndarray, n_splits: int = 5) -> list[tuple[np.ndarray, np.ndarray]]:
    """Return (train_idx, val_idx) pairs with no well split across folds."""
    unique = np.unique(groups)
    rng = np.random.default_rng(42)
    rng.shuffle(unique)
    fold_groups = np.array_split(unique, n_splits)
    splits = []
    for val_g in fold_groups:
        val_mask = np.isin(groups, val_g)
        splits.append((np.where(~val_mask)[0], np.where(val_mask)[0]))
    return splits


def train_lgb(
    X_tr: np.ndarray, y_tr: np.ndarray,
    X_val: np.ndarray, y_val: np.ndarray,
    params: dict,
) -> tuple:
    p = params.copy()
    n_est = p.pop('n_estimators', 7000)
    dtrain = lgb.Dataset(X_tr, label=y_tr)
    dval   = lgb.Dataset(X_val, label=y_val)
    model  = lgb.train(
        p, dtrain, num_boost_round=n_est, valid_sets=[dval],
        callbacks=[lgb.early_stopping(200, verbose=False), lgb.log_evaluation(0)],
    )
    return model, model.predict(X_val)


def train_xgb(
    X_tr: np.ndarray, y_tr: np.ndarray,
    X_val: np.ndarray, y_val: np.ndarray,
    params: dict,
) -> tuple:
    p = params.copy()
    n_est = p.pop('n_estimators', 6000)
    dtrain = xgb.DMatrix(X_tr, label=y_tr)
    dval   = xgb.DMatrix(X_val, label=y_val)
    model  = xgb.train(p, dtrain, num_boost_round=n_est,
                       evals=[(dval, 'val')], early_stopping_rounds=200,
                       verbose_eval=False)
    return model, model.predict(xgb.DMatrix(X_val))


def train_catboost(
    X_tr: np.ndarray, y_tr: np.ndarray,
    X_val: np.ndarray, y_val: np.ndarray,
    params: dict, cat_idx: list[int],
) -> tuple:
    from catboost import CatBoostRegressor, Pool
    tr_pool  = Pool(X_tr, y_tr, cat_features=cat_idx)
    val_pool = Pool(X_val, y_val, cat_features=cat_idx)
    model    = CatBoostRegressor(**params)
    model.fit(tr_pool, eval_set=val_pool, early_stopping_rounds=200)
    return model, model.predict(X_val)


def ridge_stack(
    oof_preds: dict[str, np.ndarray], y: np.ndarray, alpha: float = 1.0,
) -> np.ndarray:
    """Positive Ridge weights via projected gradient descent."""
    names = list(oof_preds.keys())
    X = np.column_stack([oof_preds[k] for k in names])   # (n, p)
    p = X.shape[1]
    A = np.vstack([X, np.sqrt(alpha) * np.eye(p)])
    b = np.concatenate([y, np.zeros(p)])
    AtA = A.T @ A
    Atb = A.T @ b
    lr = 1.0 / float(np.linalg.norm(AtA, ord=2))
    w = np.ones(p) / p
    for _ in range(2000):
        w = np.maximum(0.0, w - lr * (AtA @ w - Atb))
    return w


def blend_predictions(
    base_preds: np.ndarray,   # shape (n, n_models)
    weights: np.ndarray,       # shape (n_models,)
    pf_pred: np.ndarray,       # shape (n,)
    w_pf: float = 0.70,
) -> np.ndarray:
    """0.30*Ridge_blend + 0.70*PF_heuristic."""
    ridge_pred = base_preds @ weights
    return (1.0 - w_pf) * ridge_pred + w_pf * pf_pred


def save_models(models: dict, oof_preds: dict, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    for name, model in models.items():
        path = os.path.join(out_dir, f'{name}.pkl')
        with open(path, 'wb') as f:
            pickle.dump(model, f)
    np.save(os.path.join(out_dir, 'oof_preds.npy'), oof_preds)


def load_models(out_dir: str) -> tuple[dict, dict]:
    models, oof_preds = {}, {}
    for fn in os.listdir(out_dir):
        if fn.endswith('.pkl'):
            with open(os.path.join(out_dir, fn), 'rb') as f:
                models[fn[:-4]] = pickle.load(f)
    oof_path = os.path.join(out_dir, 'oof_preds.npy')
    if os.path.exists(oof_path):
        oof_preds = np.load(oof_path, allow_pickle=True).item()
    return models, oof_preds


# --- src/rogii/neighbors.py ---
from collections import defaultdict
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

FORMATIONS = ['ANCC', 'ASTNU', 'ASTNL', 'EGFDU', 'EGFDL', 'BUDA']


def assign_cluster(tw_gr_mean: float, y_coord: float) -> int:
    """C0=standard basin, C1=north high-GR, C2=SW high-GR."""
    if tw_gr_mean < 100.0:
        return 0
    return 1 if y_coord > 1_093_000.0 else 2


class FormationPlaneKNN:
    """Spatial KNN for imputing 6 formation depths. Same-cluster neighbors only."""

    def __init__(self, k: int = 10):
        self.k = k
        self._trees: dict[int, cKDTree] = {}
        self._xy: dict[int, np.ndarray] = {}
        self._depths: dict[int, np.ndarray] = {}

    def fit(self, wells: list[tuple[int, float, float, dict]]) -> 'FormationPlaneKNN':
        """wells: list of (cluster_id, mean_x, mean_y, {form: depth})"""
        buckets: dict[int, dict] = defaultdict(lambda: {'xy': [], 'depths': []})
        for cid, mx, my, depths in wells:
            buckets[cid]['xy'].append([mx, my])
            buckets[cid]['depths'].append([depths.get(f, 0.0) for f in FORMATIONS])
        for cid, data in buckets.items():
            self._xy[cid] = np.array(data['xy'])
            self._depths[cid] = np.array(data['depths'])
            self._trees[cid] = cKDTree(self._xy[cid])
        return self

    def predict(self, cluster_id: int, x: float, y: float) -> dict[str, float]:
        """IDW-averaged formation depths from K nearest same-cluster wells."""
        if cluster_id not in self._trees:
            cluster_id = 0
        k = min(self.k, len(self._xy[cluster_id]))
        dists, idx = self._trees[cluster_id].query([[x, y]], k=k)
        dists, idx = np.atleast_1d(dists[0]), np.atleast_1d(idx[0])
        w = 1.0 / (dists + 1e-6)
        w /= w.sum()
        avg = (self._depths[cluster_id][idx] * w[:, None]).sum(axis=0)
        return dict(zip(FORMATIONS, avg))


def _tw_signature(tw: pd.DataFrame, n: int = 50) -> str:
    """Hash of first n non-null GR values rounded to 1 decimal."""
    vals = tw['GR'].dropna().values[:n]
    return '|'.join(f'{v:.1f}' for v in vals)

def build_typewell_index(well_pairs: list[tuple[str, str]]) -> dict[str, str]:
    """Build {signature: wellname} from training well pairs (hw_path, tw_path)."""
    from rogii.utils import extract_wellname, load_tw
    index = {}
    for hw_path, tw_path in well_pairs:
        tw = load_tw(tw_path)
        sig = _tw_signature(tw)
        index[sig] = extract_wellname(hw_path)
    return index

def find_tw_match(tw: pd.DataFrame, index: dict[str, str]) -> str | None:
    """Return training wellname whose typewell matches tw, or None."""
    return index.get(_tw_signature(tw))


# --- src/rogii/postprocess.py ---
import numpy as np


def apply_rampup(
    d: np.ndarray, md_since_ps: np.ndarray, alpha: float = 1.0, tau: float = 85.0,
) -> np.ndarray:
    """Dampen TVT increments near PS: d *= alpha*(1 - exp(-t/tau))."""
    return d * alpha * (1.0 - np.exp(-md_since_ps / tau))


def blend_pf(d_model: np.ndarray, d_pf: np.ndarray, w_pf: float = 0.09) -> np.ndarray:
    return (1.0 - w_pf) * d_model + w_pf * d_pf


def savgol_smooth(y: np.ndarray, window: int = 17, poly: int = 3) -> np.ndarray:
    """SG filter. Uses scipy if available, otherwise numpy fallback."""
    try:
        from scipy.signal import savgol_filter
        return savgol_filter(y, window, poly)
    except Exception:
        half = window // 2
        x = np.arange(-half, half + 1, dtype=float)
        P = np.column_stack([x ** i for i in range(poly + 1)])
        coeff = np.linalg.pinv(P)[0]
        out = y.copy()
        for i in range(half, len(y) - half):
            out[i] = coeff @ y[i - half:i + half + 1]
        return out


def robust_polyfit(
    x: np.ndarray, y: np.ndarray, degree: int = 4, n_iters: int = 4, c: float = 2.0,
) -> np.ndarray:
    """IRLS polynomial fit with Tukey bisquare weights. Returns fitted values."""
    xn = x / (x[-1] + 1e-10)
    A  = np.column_stack([xn ** i for i in range(degree + 1)])
    w  = np.ones(len(y))
    for _ in range(n_iters):
        Aw = A * w[:, None]
        coeffs, _, _, _ = np.linalg.lstsq(Aw, y * w, rcond=None)
        resid = y - A @ coeffs
        s = np.median(np.abs(resid)) / 0.6745
        u = resid / (c * s + 1e-10)
        w = np.where(np.abs(u) < 1.0, (1.0 - u ** 2) ** 2, 0.0)
    return A @ coeffs


def apply_uspace(
    tvt_pred: np.ndarray, z: np.ndarray, anchor_tvt: float,
    degree: int = 4, robust_iters: int = 4, c: float = 2.0,
) -> np.ndarray:
    """Project TVT through U=TVT+Z space. Enforces geological planarity."""
    U      = tvt_pred + z - anchor_tvt
    s      = np.arange(len(U), dtype=float)
    U_proj = robust_polyfit(s, U, degree, robust_iters, c)
    return anchor_tvt + U_proj - z


def postprocess_well(
    tvt_increments: np.ndarray,
    pf_increments: np.ndarray,
    z: np.ndarray,
    md_since_ps: np.ndarray,
    last_known_tvt: float,
    params: dict,
    uspace_cfg: dict,
) -> np.ndarray:
    """Full post-processing pipeline for one well. Returns TVT trajectory."""
    d = blend_pf(tvt_increments, pf_increments, params['w_pf'])
    d = apply_rampup(d, md_since_ps, params['alpha'], params['tau'])
    tvt = last_known_tvt + np.cumsum(d)
    tvt = savgol_smooth(tvt)
    tvt = apply_uspace(tvt, z, last_known_tvt, **uspace_cfg)
    return tvt


import optuna


def tune_postprocess(well_data: list[dict], n_trials: int = 500) -> dict:
    """Optuna TPE over (alpha, tau, w_pf). well_data: per-well dicts with model outputs."""
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial: optuna.Trial) -> float:
        alpha = trial.suggest_float('alpha', 0.5, 2.0)
        tau   = trial.suggest_float('tau',   20.0, 200.0)
        w_pf  = trial.suggest_float('w_pf',  0.0, 0.3)
        total_sq, total_n = 0.0, 0
        for wd in well_data:
            d = blend_pf(wd['d_model'], wd['d_pf'], w_pf)
            d = apply_rampup(d, wd['md_since_ps'], alpha, tau)
            tvt_pred = wd['last_known_tvt'] + np.cumsum(d)
            tvt_true = wd['last_known_tvt'] + np.cumsum(wd['target_increments'])
            total_sq += float(np.sum((tvt_pred - tvt_true) ** 2))
            total_n  += len(d)
        return float(np.sqrt(total_sq / total_n))

    study = optuna.create_study(
        direction='minimize',
        sampler=optuna.samplers.TPESampler(seed=42),
    )
    study.optimize(objective, n_trials=n_trials)
    return study.best_params


# --- src/rogii/preprocess.py ---
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


# --- src/rogii/utils.py ---
import os
import glob
from dataclasses import dataclass
import pandas as pd


@dataclass
class WellData:
    name: str
    hw: pd.DataFrame
    tw: pd.DataFrame
    ps_idx: int
    scalars: dict
    formations: dict        # {form_name: float} imputed depths
    cluster_id: int
    tw_match: str | None = None
    a_cal: float = 1.0
    b_cal: float = 0.0


def extract_wellname(path: str) -> str:
    """8-char hash from path like .../abc12345__horizontal_well.csv"""
    return os.path.basename(path).split('__')[0]


def load_hw(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def load_tw(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def list_wells(data_dir: str) -> list[tuple[str, str]]:
    """Return sorted list of (hw_path, tw_path) pairs."""
    hw_paths = sorted(glob.glob(os.path.join(data_dir, '*__horizontal_well.csv')))
    tw_map = {extract_wellname(p): p
              for p in glob.glob(os.path.join(data_dir, '*__typewell.csv'))}
    return [(p, tw_map[extract_wellname(p)])
            for p in hw_paths if extract_wellname(p) in tw_map]


# --- src/rogii/alignment/beam.py ---
import numpy as np


def _run_single(
    hw_gr: np.ndarray, tw_tvt: np.ndarray, tw_gr: np.ndarray,
    tvt_start: float, step_max: float, penalty: float, beam_width: int = 20,
) -> np.ndarray:
    steps = np.linspace(0.0, step_max, 15)
    beam = [(0.0, tvt_start)]   # (score, tvt)
    traj = np.empty(len(hw_gr))

    for t in range(len(hw_gr)):
        obs = hw_gr[t]
        cands = []
        for score, last_tvt in beam:
            for s in steps:
                new_tvt = last_tvt + s
                pred = float(np.interp(new_tvt, tw_tvt, tw_gr))
                gr_cost = (obs - pred) ** 2 if not np.isnan(obs) else 0.0
                cands.append((score - gr_cost - penalty * s ** 2, new_tvt))
        cands.sort(reverse=True)
        beam = cands[:beam_width]
        traj[t] = beam[0][1]

    return traj


def run_beam_configs(
    hw_gr: np.ndarray, tw_tvt: np.ndarray, tw_gr: np.ndarray,
    tvt_start: float, configs: list[dict],
) -> dict[str, np.ndarray]:
    """Run all beam configs. Adds beam_ref = (cons + sm5) / 2."""
    results = {}
    for cfg in configs:
        results[cfg['name']] = _run_single(
            hw_gr, tw_tvt, tw_gr, tvt_start,
            cfg['step_max'], cfg['penalty'],
        )
    results['beam_ref'] = (results['cons'] + results['sm5']) / 2.0
    return results


# --- src/rogii/alignment/dtw.py ---
import numpy as np


def _fill_cost_matrix(x: np.ndarray, y: np.ndarray, radius: int) -> np.ndarray:
    n, m = len(x), len(y)
    D = np.full((n + 1, m + 1), np.inf)
    D[0, 0] = 0.0
    for i in range(1, n + 1):
        j_lo, j_hi = max(1, i - radius), min(m, i + radius)
        for j in range(j_lo, j_hi + 1):
            cost = (x[i - 1] - y[j - 1]) ** 2
            D[i, j] = cost + min(D[i - 1, j], D[i, j - 1], D[i - 1, j - 1])
    return D


def _traceback(D: np.ndarray, n: int, m: int) -> np.ndarray:
    i, j = n, m
    path = []
    while i > 0 and j > 0:
        path.append(j - 1)
        prev = np.argmin([D[i - 1, j], D[i, j - 1], D[i - 1, j - 1]])
        if prev == 0:
            i -= 1
        elif prev == 1:
            j -= 1
        else:
            i -= 1; j -= 1
    path.reverse()
    if len(path) < n:
        path = [path[0]] * (n - len(path)) + path
    return np.array(path[:n], dtype=int)


def _dtw_stochastic_single(
    hw_gr: np.ndarray, tw_tvt: np.ndarray, tw_gr: np.ndarray,
    radius: int, k: int, seed: int,
) -> np.ndarray:
    """K stochastic DTW paths. Returns tvt_paths shape (k, n)."""
    rng = np.random.default_rng(seed)
    n, m = len(hw_gr), len(tw_gr)
    D = _fill_cost_matrix(hw_gr, tw_gr, radius)
    paths = []
    for _ in range(k):
        noise = np.zeros_like(D)
        noise[1:, 1:] = rng.gumbel(0, 1, (n, m))
        path_j = _traceback(D + noise, n, m)
        paths.append(tw_tvt[path_j])
    return np.array(paths)


def run_dtw_all_radii(
    hw_gr: np.ndarray, tw_tvt: np.ndarray, tw_gr: np.ndarray,
    radii: tuple | list = (20, 50, 100, 200), k_stochastic: int = 12,
) -> dict[str, np.ndarray]:
    results = {}
    for r in radii:
        paths = _dtw_stochastic_single(hw_gr, tw_tvt, tw_gr, r, k_stochastic, seed=r)
        mean = paths.mean(axis=0)
        std  = paths.std(axis=0)
        cv   = std / (np.abs(mean) + 1e-6)
        results[f'dtw_r{r}_mean'] = mean
        results[f'dtw_r{r}_std']  = std
        results[f'dtw_r{r}_cv']   = cv
    return results


# --- src/rogii/alignment/ncc.py ---
import numpy as np


def _ncc(a: np.ndarray, b: np.ndarray) -> float:
    a = a - a.mean()
    b = b - b.mean()
    denom = np.sqrt((a ** 2).sum() * (b ** 2).sum())
    return float(np.dot(a, b) / denom) if denom > 1e-8 else 0.0


def compute_sc_trust(known_rows: int) -> float:
    return float(np.clip(known_rows / 200.0, 0.0, 0.6))


def _run_single_scale(
    hw_gr: np.ndarray, tw_tvt: np.ndarray, tw_gr: np.ndarray,
    baseline_tvt: np.ndarray, hw_size: int, stride: int,
    known_rows: int = 0, search_range: float = 50.0,
) -> tuple[np.ndarray, float]:
    n = len(baseline_tvt)
    offsets = np.arange(-search_range, search_range + 0.5, 0.5)
    traj = baseline_tvt.copy()
    confs = []

    for i in range(0, n, stride):
        center = known_rows + i
        lo, hi = max(0, center - hw_size), min(len(hw_gr), center + hw_size + 1)
        hw_win = hw_gr[lo:hi]
        if np.isnan(hw_win).mean() > 0.5:
            confs.append(0.0)
            continue
        best, best_tvt = -np.inf, baseline_tvt[i]
        for off in offsets:
            cand = baseline_tvt[i] + off
            pts = np.linspace(cand - hw_size * 0.5, cand + hw_size * 0.5, len(hw_win))
            tw_win = np.interp(pts, tw_tvt, tw_gr)
            score = _ncc(hw_win, tw_win)
            if score > best:
                best, best_tvt = score, cand
        end = min(n, i + stride)
        traj[i:end] = best_tvt
        confs.append(best)

    return traj, float(np.mean(confs)) if confs else 0.0


def run_ncc_multiscale(
    hw_gr: np.ndarray, tw_tvt: np.ndarray, tw_gr: np.ndarray,
    baseline_tvt: np.ndarray, known_rows: int,
    hw_sizes: tuple = (8, 15, 25), stride: int = 3,
) -> dict[str, np.ndarray | float]:
    results: dict = {}
    for hw_size in hw_sizes:
        traj, conf = _run_single_scale(hw_gr, tw_tvt, tw_gr, baseline_tvt, hw_size, stride, known_rows)
        results[f'sc{hw_size}_tvt'] = traj
        results[f'sc{hw_size}_conf'] = conf
    sc_trust = compute_sc_trust(known_rows)
    results['sc_trust'] = sc_trust
    results['hyb_ref'] = (1 - sc_trust) * baseline_tvt + sc_trust * results['sc15_tvt']
    return results


# --- src/rogii/alignment/particle_filter.py ---
import numpy as np


def _resample(weights: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    n = len(weights)
    pos = (rng.uniform() + np.arange(n)) / n
    return np.searchsorted(np.cumsum(weights), pos)


def _run_single(
    hw_gr: np.ndarray, tw_tvt: np.ndarray, tw_gr: np.ndarray,
    tvt_start: float, n_particles: int, sigma_0: float,
    gr_scale: int, obs_sigma: float, seed: int, use_velocity: bool,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    k = np.ones(gr_scale) / gr_scale
    hw_s = np.convolve(hw_gr, k, 'same')
    tw_s = np.convolve(tw_gr, k, 'same')

    tvt_p = rng.normal(tvt_start, sigma_0, n_particles)
    vel_p = rng.normal(0.04, 0.02, n_particles).clip(0.0) if use_velocity else None
    log_w = np.zeros(n_particles)
    traj = np.empty(len(hw_gr))

    for t in range(len(hw_gr)):
        if use_velocity:
            vel_p = (vel_p + rng.normal(0, 0.005, n_particles)).clip(0.0)
            tvt_p = tvt_p + vel_p
        else:
            tvt_p += np.clip(rng.normal(0.04, 0.3, n_particles), 0.0, None)

        if not np.isnan(hw_s[t]):
            pred = np.interp(tvt_p, tw_tvt, tw_s)
            log_w += -0.5 * ((hw_s[t] - pred) / obs_sigma) ** 2

        log_w -= np.max(log_w)
        w = np.exp(log_w)
        w /= w.sum()
        traj[t] = tvt_p @ w

        if 1.0 / (w @ w) < n_particles / 2:
            idx = _resample(w, rng)
            tvt_p = tvt_p[idx]
            if use_velocity:
                vel_p = vel_p[idx]
            log_w = np.zeros(n_particles)

    return traj


def run_pf(
    hw_gr: np.ndarray, tw_tvt: np.ndarray, tw_gr: np.ndarray,
    tvt_start: float, n_particles: int = 500, sigma_0: float = 4.5,
    gr_scale: int = 5, obs_sigma: float = 15.0, n_seeds: int = 64,
    use_velocity: bool = False,
) -> np.ndarray:
    """Ensemble PF. Averages n_seeds runs."""
    trajs = [
        _run_single(hw_gr, tw_tvt, tw_gr, tvt_start, n_particles,
                    sigma_0, gr_scale, obs_sigma, s, use_velocity)
        for s in range(n_seeds)
    ]
    return np.mean(trajs, axis=0)


def run_pf_variants(
    hw_gr: np.ndarray, tw_tvt: np.ndarray, tw_gr: np.ndarray,
    tvt_start: float, cfg: dict,
) -> dict[str, np.ndarray]:
    """Run pf_ancc (position) and pf_z (velocity) variants over all GR scales."""
    n_p, s0, scales, n_s = cfg['n_particles'], cfg['sigma_0'], cfg['gr_scales'], cfg['n_seeds']
    pf_ancc = np.mean([run_pf(hw_gr, tw_tvt, tw_gr, tvt_start, n_p, s0, sc, n_seeds=n_s)
                       for sc in scales], axis=0)
    pf_z    = np.mean([run_pf(hw_gr, tw_tvt, tw_gr, tvt_start, n_p, s0, sc, n_seeds=n_s,
                               use_velocity=True)
                       for sc in scales], axis=0)
    return {'pf_ancc': pf_ancc, 'pf_z': pf_z}


# --- src/rogii/pipeline.py ---
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

# from rogii.utils import WellData, load_hw, load_tw, list_wells, extract_wellname  # bundled
# from rogii.preprocess import detect_ps, interpolate_gr, calibrate_gr, extract_scalars  # bundled
# from rogii.neighbors import (assign_cluster, FormationPlaneKNN,  # bundled
#                               build_typewell_index, find_tw_match, FORMATIONS)  # bundled
# from rogii.alignment.particle_filter import run_pf_variants  # bundled
# from rogii.alignment.beam import run_beam_configs  # bundled
# from rogii.alignment.ncc import run_ncc_multiscale  # bundled
# from rogii.alignment.dtw import run_dtw_all_radii  # bundled
# from rogii.features import (build_alignment_df, compute_anchor_offsets,  # bundled
#                              compute_formation_features, compute_b_well,  # bundled
#                              compute_gr_features, compute_tabular_features,  # bundled
#                              build_feature_matrix)  # bundled
# from rogii.meta import group_kfold, train_lgb, train_xgb, train_catboost  # bundled
# from rogii.meta import ridge_stack, blend_predictions, save_models, load_models  # bundled
# from rogii.postprocess import postprocess_well, tune_postprocess  # bundled

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
