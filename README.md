# ROGII — Wellbore Geology Prediction

## Setup

```bash
pip install -r requirements.txt
```

> `rogii` is not a published package — make it importable by prefixing commands with `PYTHONPATH=src`, **or** install it once with `pip install -e .`

---

## Run

### 1. Train

```bash
PYTHONPATH=src python run.py --mode train
```

Trains LightGBM / XGBoost / CatBoost on `data/train/`, saves models to `models/`.

### 2. Predict

```bash
PYTHONPATH=src python run.py --mode predict
```

Runs inference on `data/test/`, writes **`submissions/submission.csv`**.

---

## Data layout

```
data/
  train/
    <hash>__horizontal_well.csv
    <hash>__typewell.csv
  test/
    <hash>__horizontal_well.csv
    <hash>__typewell.csv
```

---

## GPU note

XGBoost and CatBoost use GPU by default (`config.py`).
LightGBM requires a GPU build — install via conda for automatic CUDA detection:

```bash
conda install -c conda-forge lightgbm
```

---

## Project layout

```
src/rogii/
  utils.py            WellData, load helpers
  preprocess.py       GR interpolation, affine calibration, scalar features
  neighbors.py        cluster assignment, FormationPlaneKNN, typewell index
  alignment/
    particle_filter.py
    beam.py
    ncc.py
    dtw.py
  features.py         tabular feature matrix builder
  meta.py             LGB/XGB/CatBoost training, Ridge stacking, CV
  postprocess.py      ramp-up, U-space projection, Optuna tuning
  pipeline.py         train / predict orchestration
config.py             all hyperparameters
run.py                CLI entry point
```
