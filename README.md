# ROGII — Wellbore Geology Prediction

Kaggle competition: predicting TVT (True Vertical Thickness) along horizontal wellbore eval zones using GR signal alignment feeding a GBDT meta-learner.

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Make `rogii` importable — pick one

**Option A — editable install (installs the package once):**
```bash
pip install -e .
python run.py --mode train
```

**Option B — no install needed (set path at runtime):**
```bash
PYTHONPATH=src python run.py --mode train
```

Both are equivalent. Option B requires no extra step if you're just copying files to a server.

### 3. Verify the setup

```bash
PYTHONPATH=src python -m pytest tests/ -q
# expected: 51 passed, 1 skipped
```

### 4. Train on labeled wells

```bash
PYTHONPATH=src python run.py --mode train
```

### 5. Predict on eval wells

```bash
PYTHONPATH=src python run.py --mode predict
```

### 6. Bundle source into a single Kaggle notebook file

```bash
PYTHONPATH=src python scripts/bundle.py
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
