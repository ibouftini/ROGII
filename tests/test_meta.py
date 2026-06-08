import numpy as np
import pytest
from rogii.meta import group_kfold, train_lgb, train_xgb


@pytest.fixture
def small_dataset():
    rng = np.random.default_rng(0)
    n = 2000
    groups = np.repeat(np.arange(20), 100)  # 20 wells, 100 rows each
    X = rng.normal(0, 1, (n, 10)).astype(np.float32)
    y = rng.normal(0, 0.3, n).astype(np.float32)
    return X, y, groups


def test_group_kfold_no_leakage(small_dataset):
    X, y, groups = small_dataset
    folds = group_kfold(groups, n_splits=5)
    assert len(folds) == 5
    for train_idx, val_idx in folds:
        train_groups = set(groups[train_idx])
        val_groups   = set(groups[val_idx])
        assert train_groups.isdisjoint(val_groups)


def test_train_lgb_oof_shape(small_dataset):
    X, y, groups = small_dataset
    folds = group_kfold(groups, n_splits=5)
    train_idx, val_idx = folds[0]
    params = dict(num_leaves=31, learning_rate=0.1, n_estimators=50,
                  verbose=-1, device='cpu')
    model, oof = train_lgb(X[train_idx], y[train_idx], X[val_idx], y[val_idx], params)
    assert oof.shape == (len(val_idx),)


def test_train_xgb_oof_shape(small_dataset):
    X, y, groups = small_dataset
    folds = group_kfold(groups, n_splits=5)
    train_idx, val_idx = folds[0]
    params = dict(max_depth=3, learning_rate=0.1, n_estimators=50,
                  device='cpu', seed=0, verbosity=0)
    model, oof = train_xgb(X[train_idx], y[train_idx], X[val_idx], y[val_idx], params)
    assert oof.shape == (len(val_idx),)
