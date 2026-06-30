# ARMOTE-CV-HEA

**ARMOTE-CV** (Automated Regression with Multi-Objective Tuning and Evaluation via Cross-Validation) is a nested cross-validation engine for regression with automated multi-objective hyperparameter optimization.

This repository applies ARMOTE-CV to predict yield strength (YS) and Vickers hardness (HV) of high-entropy alloys (HEAs) from grain size and composition features.

It is designed so that hyperparameter tuning and model evaluation are fully decoupled — the test fold is never seen during optimization. For each outer fold it:

1. Runs multi-objective Bayesian hyperparameter optimization (Optuna, TPE) on the outer **training** split only.
2. Optimizes two objectives simultaneously: minimize MSE and maximize R². The best trial is selected from the Pareto front by highest R².
3. Retrains the winning configuration on the full outer training split.
4. Evaluates on the held-out outer test fold and aggregates metrics across all folds.

**Models supported out of the box:** Linear Regression, Bayesian Ridge, SVR, Decision Tree, Random Forest, XGBoost, Gaussian Process Regressor (GPR), Neural Network Regressor (NNR).

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run 5-fold CV on all feature sets and targets (default)
python run_5fold.py

# 3. Run with specific options
python run_5fold.py -f S1 S2 -t YS -m RandomForest XGBoost --n-trials 30
```

---

## Dataset

The dataset (`data/inputs.csv`) contains HEA compositions and processing conditions across 6 Bayesian optimization batches (BBA, BBB, BBC, CBA, CBB, CBC), referred to as `Iteration`. Each row is one alloy sample with measured YS (MPa) and HV.

Feature columns are organized into blocks defined in `data/inputs_feature_manifest.csv`. The four feature sets are **cumulative**:

| Set | Features |
|-----|----------|
| S1 | Grain size descriptors only |
| S2 | S1 + Wen model features |
| S3 | S2 + Processing parameters |
| S4 | S3 + Composition + solid-solution strengthening |

Running S4 uses the full feature set; running S1 tests how well grain size alone predicts properties.

---

## CV Protocols

Three scripts wrap the same `armote_cv.py` engine. The only difference is the outer splitter.

| Protocol | Script | Outer splitter | # Folds | What it tests |
|----------|--------|----------------|---------|---------------|
| 5-Fold | `run_5fold.py` | `KFold(n=5, shuffle=True)` | 5 | General interpolation |
| LOO | `run_loo.py` | `LeaveOneOut()` | n (~93–94) | Per-sample prediction |
| LOBO | `run_lobo.py` | `LeaveOneGroupOut()` | 6 | Cross-batch extrapolation |

**When to use which:**
- **5-Fold** — fast, standard benchmark. Good first run.
- **LOO** — most pessimistic per-sample estimate. Expensive: ~23 500 model fits per model at defaults.
- **LOBO** — tests whether the model generalizes to a completely unseen BO batch. Relevant when deployment means predicting on new experimental campaigns.

### R² reporting by protocol

**5-Fold:** Per-fold R² is computed normally. `Avg Test R2` and `Std Test R2` are the mean and std across 5 folds.

**LOBO:** Each test fold is a full batch (multiple samples), so per-fold R² is computable and stored in the `All Fold Test R2` column. However, `Avg Test R2` reports the **pooled (PRESS) R²** — computed from all out-of-fold predictions concatenated — which is more robust than the mean of 6 fold R²s given the small fold count. `Std Test R2` is `NaN` because it is not meaningful alongside a pooled metric.

**LOO:** Each test fold is a single sample, so per-fold R² is undefined. `All Fold Test R2` entries are `null`. `Avg Test R2` again reports the pooled PRESS R²:

$$R^2 = 1 - \frac{\sum_i (y_i - \hat{y}_i)^2}{\sum_i (y_i - \bar{y})^2}$$

The inner CV for Optuna is always a fast KFold (default: 5 folds) regardless of the outer splitter.

---

## Running the Scripts

### 5-Fold CV

```bash
python run_5fold.py                              # all defaults
python run_5fold.py -f S1 S2 -t YS              # S1 and S2, YS only
python run_5fold.py -f S4 -t HV -m GPR NNR      # S4, HV, two models
python run_5fold.py --cv 10 --n-trials 100       # 10 folds, 100 Optuna trials
python run_5fold.py --list                       # print valid choices and exit
```

### LOO CV

```bash
python run_loo.py                                # all defaults
python run_loo.py -f S3 -t YS -m SVR XGBoost    # narrow scope to cut runtime
python run_loo.py --n-trials 20 --inner-cv 3     # reduce Optuna cost
```

> **Runtime warning:** LOO with defaults runs ~23 500 fits per model (94 folds × 50 trials × 5 inner folds). Use `--models` to run one or two models at a time.

### LOBO CV

```bash
python run_lobo.py                               # all defaults
python run_lobo.py -f S4 -t YS HV
python run_lobo.py -f S1 -t YS -m RandomForest XGBoost --n-trials 50
```

---

## Output

Each run writes to an auto-created directory:

```
{S1..S4}_{YS|HV}_Results_{protocol}/
├── models/     # .pkl files (sklearn/XGBoost) and .keras files (NNR) — fitted model + scaler per fold
├── plots/      # Y-Y scatter plots, Optuna history plots per fold
├── studies/    # Optuna study objects (reusable for analysis)
└── {target}_results_{n_folds}_fold_CV.csv   # summary metrics table
```

The CSV has one row per model with columns for average and std of R², MSE, MAPE across folds, best hyperparameters per fold, and per-fold raw values as JSON arrays.

### Output Safety

Re-running a script on an existing output directory raises `FileExistsError` by default to prevent overwriting results.

| Flag | Behavior |
|------|----------|
| *(default)* | Error if CSV exists |
| `--overwrite` | Replace the existing CSV entirely |
| `--append` | Skip already-complete models, append new results to existing CSV |

`--append` is useful when a run was interrupted or only some models were run:

```bash
# Add the 7 remaining models without touching the NNR row already in the CSV
python run_loo.py --models LinearRegression BayesianRidge SVR DecisionTree RandomForest XGBoost GPR --append
```

---

## CLI Reference

### All three scripts share these flags

| Argument | Default | Description |
|----------|---------|-------------|
| `-f`, `--feature-sets` | all (S1–S4) | Feature sets to run |
| `-t`, `--targets` | `YS HV` | Target properties |
| `-m`, `--models` | all 8 | Models to run |
| `--n-trials` | `50` | Optuna trials per inner fold |
| `--overwrite` | off | Replace existing output CSV |
| `--append` | off | Append missing models to existing CSV |
| `--list` | — | Print valid choices and exit |

### `run_5fold.py` only

| Argument | Default | Description |
|----------|---------|-------------|
| `--cv` | `5` | Outer folds (also sets inner CV for Optuna) |

### `run_loo.py` and `run_lobo.py` only

| Argument | Default | Description |
|----------|---------|-------------|
| `--inner-cv` | `5` | KFold folds for Optuna inner CV |

---

## Using `armote_cv.py` in Your Own Project

`run_workflow()` is the public API. It requires your data, models, and hyperparameter search spaces; everything else has sensible defaults.

```python
from armote_cv import run_workflow
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

X = pd.read_csv("my_features.csv")
y = pd.read_csv("my_targets.csv")["target"]

models = {
    "RandomForest": RandomForestRegressor(random_state=42),
    "XGBoost":      XGBRegressor(random_state=42),
}

param_spaces = {
    "RandomForest": {
        "n_estimators": ("int",   50, 300),
        "max_depth":    ("int",   3,  20),
    },
    "XGBoost": {
        "n_estimators":    ("int",   50, 300),
        "learning_rate":   ("float", 1e-3, 0.3, "log"),
        "max_depth":       ("int",   3, 8),
    },
}

results_df = run_workflow(
    X, y,
    models=models,
    param_spaces=param_spaces,
    cv=5,
    output_name="target",
    output_folder_name="my_output",
    n_trials=50,
)
```

### Key `run_workflow()` parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `cv` | `5` | Outer CV folds |
| `output_name` | `"Objective 1"` | Target name used in file/plot labels |
| `output_folder_name` | `"workflow_output"` | Root output directory |
| `n_trials` | `50` | Optuna trials per inner fold |
| `inner_cv` | `5` | Inner KFold folds for Optuna |
| `splitter` | `None` (KFold) | Custom outer splitter (e.g. `LeaveOneOut()`) |
| `groups` | `None` | Group labels for `LeaveOneGroupOut` |
| `pool_oof_metrics` | `False` | Compute pooled R² instead of per-fold mean (set `True` for LOO/LOBO) |
| `overwrite` | `False` | Overwrite existing output CSV |
| `append` | `False` | Append new models to existing output CSV |

### Hyperparameter space format

```python
param_spaces = {
    "ModelName": {
        "int_param":         ("int",         low, high),
        "float_param":       ("float",       low, high),
        "log_float_param":   ("float",       low, high, "log"),
        "categorical_param": ("categorical", [val1, val2, val3]),
    }
}
```

---

## Citation

A paper describing this work is currently under preparation. If you use this code or dataset in the meantime, please cite this repository:

```
Padhy, S. P. (2026). ARMOTE-CV-HEA: Automated Regression with Multi-Objective Tuning
and Evaluation via Cross-Validation applied to HEA property prediction. GitHub.
https://github.com/Shakti-95/ARMOTE-CV-HEA
```

This section will be updated with the full journal citation and DOI upon publication.

---

## Repository Structure

```
├── armote_cv.py                       # ARMOTE-CV engine — import run_workflow from here
├── run_5fold.py                       # Entry point: 5-fold KFold CV
├── run_loo.py                         # Entry point: Leave-One-Out CV
├── run_lobo.py                        # Entry point: Leave-One-Batch-Out CV
├── data/
│   ├── inputs.csv                     # HEA dataset
│   └── inputs_feature_manifest.csv    # Maps columns to feature blocks S1–S4
├── requirements.txt
├── LICENSE
└── {S1..S4}_{YS|HV}_Results_{protocol}/   # Auto-generated per run
```

---

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.
