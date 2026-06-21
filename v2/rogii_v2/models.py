"""Model training, stacking, save/load."""
import os
import pickle
import numpy as np
import lightgbm as lgb


def group_kfold(groups, n_splits=5, seed=42):
    """GroupKFold: no well split across folds."""
    unique = np.unique(groups)
    rng = np.random.default_rng(seed)
    rng.shuffle(unique)
    fold_groups = np.array_split(unique, n_splits)
    splits = []
    for val_g in fold_groups:
        val_mask = np.isin(groups, val_g)
        splits.append((np.where(~val_mask)[0], np.where(val_mask)[0]))
    return splits


def train_lgb(X_tr, y_tr, X_val, y_val, params, early_stopping=250):
    p = params.copy()
    n_est = p.pop('n_estimators', 5000)
    dtrain = lgb.Dataset(X_tr, label=y_tr)
    dval = lgb.Dataset(X_val, label=y_val)
    model = lgb.train(
        p, dtrain, num_boost_round=n_est, valid_sets=[dval],
        callbacks=[lgb.early_stopping(early_stopping, verbose=False),
                   lgb.log_evaluation(0)],
    )
    return model, model.predict(X_val)


def train_catboost(X_tr, y_tr, X_val, y_val, params, early_stopping=250):
    from catboost import CatBoostRegressor, Pool
    tr_pool = Pool(X_tr, y_tr)
    val_pool = Pool(X_val, y_val)
    model = CatBoostRegressor(**params)
    model.fit(tr_pool, eval_set=val_pool, early_stopping_rounds=early_stopping)
    return model, model.predict(X_val)


def ridge_stack(oof_preds, y, alpha=1.66):
    """Non-negative Ridge via projected gradient descent.
    oof_preds: dict of name -> array. Returns (weights, names)."""
    names = list(oof_preds.keys())
    X = np.column_stack([oof_preds[k] for k in names])
    p = X.shape[1]
    A = np.vstack([X, np.sqrt(alpha) * np.eye(p)])
    b = np.concatenate([y, np.zeros(p)])
    AtA = A.T @ A
    Atb = A.T @ b
    lr = 1.0 / float(np.linalg.norm(AtA, ord=2))
    w = np.ones(p) / p
    for _ in range(2000):
        w = np.maximum(0.0, w - lr * (AtA @ w - Atb))
    return w, names


def save_models(models, meta, out_dir):
    """Save all models + meta (oof, stack_w, model_names)."""
    os.makedirs(out_dir, exist_ok=True)
    for name, model in models.items():
        path = os.path.join(out_dir, f'{name}.pkl')
        with open(path, 'wb') as f:
            pickle.dump(model, f, protocol=4)
    np.save(os.path.join(out_dir, 'meta.npy'), meta)


def load_models(out_dir):
    """Load all models + meta. Returns (models_dict, meta_dict)."""
    models = {}
    for fn in sorted(os.listdir(out_dir)):
        if fn.endswith('.pkl') and fn not in ('knn.pkl', 'tw_index.pkl', 'dense.pkl'):
            with open(os.path.join(out_dir, fn), 'rb') as f:
                models[fn[:-4]] = pickle.load(f)
    meta_path = os.path.join(out_dir, 'meta.npy')
    meta = np.load(meta_path, allow_pickle=True).item() if os.path.exists(meta_path) else {}
    return models, meta
