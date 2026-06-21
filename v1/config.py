# config.py
PF = dict(n_particles=500, sigma_0=4.5, gr_scales=[5, 8], n_seeds=16)

BEAM_CONFIGS = [
    dict(name='vcons',  step_max=0.3, penalty=8.0),
    dict(name='cons',   step_max=0.8, penalty=4.0),
    dict(name='sm5',    step_max=1.5, penalty=2.0),
    dict(name='mod',    step_max=2.5, penalty=1.2),
    dict(name='loose',  step_max=4.0, penalty=0.7),
    dict(name='vloose', step_max=6.0, penalty=0.3),
    dict(name='ultra',  step_max=10.0, penalty=0.1),
]

NCC = dict(hw_sizes=[8, 15, 25], stride=3)
DTW = dict(radii=[20, 50, 100, 200], k_stochastic=4)

FEATURES = dict(
    anchor_offsets=[-80, -40, -20, -10, -5, 0, 5, 10, 20, 40, 80],
    gr_roll_windows=[11, 51, 151],
    gr_env_window=21,
    slope_recent_rows=200,
    b_well_decay=0.02,
)

LGB_VARIANTS = [
    dict(num_leaves=255, learning_rate=0.03,  seed=42,  n_estimators=7000),
    dict(num_leaves=200, learning_rate=0.025, seed=123, n_estimators=7000),
    dict(num_leaves=300, learning_rate=0.035, seed=777, n_estimators=7000),
]

CATBOOST = dict(depth=8, learning_rate=0.025, iterations=6000, l2_leaf_reg=2,
                random_seed=42, task_type='GPU', verbose=0)

XGB = dict(max_depth=8, learning_rate=0.04, reg_lambda=10, n_estimators=6000,
           device='cuda', seed=42)

RIDGE = dict(alpha=1.0)
BLEND = dict(w_ridge=0.30, w_pf=0.70)
PP = dict(alpha=1.0, tau=85.0, w_pf=0.09)
USPACE = dict(degree=4, robust_iters=4, c=2.0)
CV = dict(n_splits=5)

DATA = dict(train_dir='data/train', test_dir='data/test',
            models_dir='models', submissions_dir='submissions',
            cache_dir='data/cache')
