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
