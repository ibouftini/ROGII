# ROGII — Wellbore Geology Prediction

Kaggle competition: predicting TVT (True Vertical Thickness) along horizontal wellbore eval zones using GR signal alignment feeding a GBDT meta-learner.

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
pip install -e .
```

### 2. Verify the setup

```bash
conda run -n base python3 -m pytest tests/ -q
```

Expected: 51 passed, 1 skipped.

### 3. Train on labeled wells

```bash
conda run -n base python3 run.py --mode train --data_dir data/train --out_dir models/
```

### 4. Predict on eval wells

```bash
conda run -n base python3 run.py --mode predict --data_dir data/test --model_dir models/ --out submission.csv
```

### 5. Bundle source into a single Kaggle notebook file

```bash
conda run -n base python3 scripts/bundle.py
# outputs: notebooks/rogii.py
```

## Data layout

```
data/
  <hash>__horizontal_well.csv
  <hash>__typewell.csv
```

## Project layout

```
src/rogii/
  utils.py          WellData, load helpers
  preprocess.py     GR interpolation, affine calibration, scalar features
  neighbors.py      cluster assignment, FormationPlaneKNN, typewell index
  alignment/
    particle_filter.py
    beam.py
    ncc.py
    dtw.py
  features.py       tabular feature matrix builder
  meta.py           LGB/XGB/CatBoost training, Ridge stacking, CV
  postprocess.py    ramp-up, U-space projection, Optuna tuning
  pipeline.py       train / predict orchestration
config.py           all hyperparameters
run.py              CLI entry point
scripts/bundle.py   Kaggle flat-file bundler
```
