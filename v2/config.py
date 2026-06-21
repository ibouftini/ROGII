# v2/config.py
from pathlib import Path
import os

# --- Paths ---
DATA_DIR = Path(os.environ.get('ROGII_DATA', 'data'))
TRAIN_DIR = DATA_DIR / 'train'
TEST_DIR = DATA_DIR / 'test'
MODELS_DIR = Path(os.environ.get('ROGII_MODELS', 'models_v2'))
CACHE_DIR = DATA_DIR / 'cache_v2'
SUBMISSIONS_DIR = Path('submissions')

# --- Particle Filter ---
PF_N_PARTICLES = 600
PF_N_SEEDS = 128
PF_SCALES = (3.0, 5.0, 8.0, 12.0)
PF_INIT_SPREAD = 4.5
# ANCC variant
PF_ANCC_ALPHA = 0.998   # momentum
PF_ANCC_RN = 0.002      # rate noise
PF_ANCC_PN = 0.005      # position noise
PF_ANCC_IS = 0.3        # initial spread for rate
PF_ANCC_RP = 0.1        # resampling position jitter
PF_ANCC_RR = 0.001      # resampling rate jitter
PF_ANCC_RESAMP = 0.5    # ESS threshold fraction
# Z-velocity variant
PF_Z_MOM = 0.993
PF_Z_VN = 0.005
PF_Z_PN = 0.01
PF_Z_GR_WT = 0.3        # weight on smoothed GR channel
PF_Z_RP = 0.2
PF_Z_RV = 0.003
PF_Z_RESAMP = 0.5

# --- Beam Search ---
BEAM_CONFIGS = [
    # (beam_size, move_cost, energy_scale, smooth_radius, name)
    (10, 20.0, 144.0, 2, 'cons'),
    (10, 8.0, 64.0, 2, 'loose'),
    (8, 35.0, 220.0, 1, 'vcons'),
    (10, 14.0, 90.0, 5, 'sm5'),
    (20, 4.0, 36.0, 3, 'vloose'),
    (12, 12.0, 100.0, 3, 'mid'),
    (15, 25.0, 180.0, 2, 'stiff'),
]

# --- NCC ---
NCC_HWS = (8, 15, 25)
NCC_STRIDE = 3

# --- Spatial ---
FORMATIONS = ['ANCC', 'ASTNU', 'ASTNL', 'EGFDU', 'EGFDL', 'BUDA']
KNN_K = 10
DENSE_K = 20
DENSE_SAMPLES = 60
B_WELL_DECAY = 0.02

# --- Features ---
GR_ROLL_WINDOWS = [5, 21, 51, 101]
GR_LAGS = [1, 5, 15, 30]
ANCHOR_OFFSETS = [-80, -40, -20, -10, -5, 0, 5, 10, 20, 40, 80]

# --- Models ---
N_SPLITS = 5
EARLY_STOPPING = 250

LGB_VARIANTS = [
    dict(num_leaves=255, learning_rate=0.03, n_estimators=5000,
         min_child_samples=20, subsample=0.8, colsample_bytree=0.8, seed=42),
    dict(num_leaves=64, learning_rate=0.0093, n_estimators=10000,
         min_child_samples=20, subsample=0.8, colsample_bytree=0.8, seed=0),
    dict(num_leaves=64, learning_rate=0.0093, n_estimators=10000,
         min_child_samples=20, subsample=0.8, colsample_bytree=0.8, seed=29),
]

CB_VARIANTS = [
    dict(depth=7, learning_rate=0.02, iterations=8000, l2_leaf_reg=2,
         random_seed=42, verbose=0),
    dict(depth=7, learning_rate=0.03, iterations=8000, l2_leaf_reg=2,
         random_seed=123, verbose=0),
]

RIDGE_ALPHA = 1.66

# --- Post-Processing ---
PP_ALPHA = 1.0
PP_TAU = 85.0
PP_W_SUB1 = 0.60       # weight on learned model vs likpf
PP_LIKPF_SCALE = 'scale_5'
PP_SG_WIN = 61
PP_SG_POLY = 3

# Selector blend: 0.30 * model + 0.70 * selector
PP_MODEL_W = 0.30
PP_SELECTOR_W = 0.70

# Selector bins
SELECTOR_N_EVAL_THRESHOLD = 4840.0
SELECTOR_Z_SPAN_THRESHOLDS = (136.73, 185.51)
SELECTOR_BIN_VARIANTS = {
    0: 'pf_scale_5_hold_0.2',
    1: 'pf_scale_3_hold_0.15',
    2: 'pf_scale_12_beam_0.2_hold_0.15',
    3: 'pf_scale_5_hold_0.15',
    4: 'pf_scale_5_beam_0.05_hold_0.05',
    5: 'pf_scale_12_beam_0.2_hold_0.05',
}
SELECTOR_GLOBAL_VARIANT = 'pf_scale_8_hold_0.2'

# U-space
USPACE_DEGREE = 4
USPACE_ITERS = 4
USPACE_C = 2.0
USPACE_BLEND = 0.75   # 0.75 * projected + 0.25 * raw

# --- Parallel ---
N_JOBS = -1  # all cores
