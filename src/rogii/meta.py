import numpy as np
import lightgbm as lgb
import xgboost as xgb


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
        callbacks=[lgb.early_stopping(200, verbose=False), lgb.log_evaluation(500)],
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
                       verbose_eval=500)
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
