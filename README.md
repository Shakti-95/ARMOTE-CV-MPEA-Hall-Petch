# Grain-size-YS-HV-ARMOTE-CV

Machine learning regression workflow for predicting yield strength (YS) and hardness (HV) of high-entropy alloys (HEAs) from grain size and composition features.

## ARMOTE-CV

**ARMOTE-CV** (Automated Regression with Multi-Objective Tuning and Evaluation via Cross-Validation) is a nested cross-validation workflow that:

1. For each outer fold, runs multi-objective Bayesian hyperparameter optimization (Optuna, TPE sampler) on the outer training data only — the outer test fold is never seen during tuning.
2. Retrains the final model with the fold-specific best hyperparameters on the full outer training set.
3. Evaluates on the held-out outer test fold and aggregates metrics across all folds.

Objectives optimized simultaneously: minimize MSE and maximize R². Best trial selected from the Pareto front by highest R².

**Models included:** Linear Regression, Bayesian Ridge, SVR, Decision Tree, Random Forest, XGBoost, Gaussian Process Regressor (GPR), Neural Network Regressor (NNR).

## CV Protocols

Three protocols share the same engine (`armote_cv.py`); only the outer splitter changes.

| Protocol | Script | Splitter | # Folds | Test fold | Measures |
|----------|--------|----------|---------|-----------|----------|
| 5-Fold | `run_5fold.py` | `KFold(n_splits=5, shuffle=True, random_state=42)` | 5 | ~18–19 random points | Within-cluster interpolation |
| LOO | `run_loo.py` | `LeaveOneOut()` | n (~93/94) | 1 alloy | Within-cluster, leave-one-sample |
| LOBO | `run_lobo.py` | `LeaveOneGroupOut(groups=Iteration)` | 6 | One BO batch | Cross-batch extrapolation |

`Iteration` (the 6 batches BBA/BBB/BBC/CBA/CBB/CBC) is the grouping column — it is already in `inputs.csv`.

**LOO/LOBO R²** is the pooled (PRESS) R² computed over all out-of-fold predictions:

$$R^2_\text{LOO} = 1 - \frac{\sum_i (y_i - \hat{y}_i)^2}{\sum_i (y_i - \bar{y})^2}$$

Per-fold R² is undefined for single-point folds (LOO) and stored as `NaN`; only the pooled value is reported. The inner CV for Optuna is always a fast KFold (default 5) regardless of outer splitter.

## Repository Structure

```
├── armote_cv.py                  # Core workflow: ARMOTE-CV engine (v1.1.0)
├── run_5fold.py                  # Entry point: 5-fold KFold CV
├── run_loo.py                    # Entry point: Leave-One-Out CV
├── run_lobo.py                   # Entry point: Leave-One-Batch-Out CV
├── data/
│   ├── inputs.csv                # Dataset
│   └── inputs_feature_manifest.csv  # Maps feature columns to blocks (S1–S4)
├── requirements.txt
└── {S1..S4}_{YS|HV}_Results_{protocol}/   # Auto-generated output directories
    ├── models/    # Saved model + scaler .pkl files per fold
    ├── plots/     # Y-Y plots, Optuna optimization plots per fold
    ├── studies/   # Optuna study objects per fold
    └── {target}_results_{n_folds}_fold_CV.csv  # Summary metrics table
```

### Feature Sets (cumulative)

| Set | Features included |
|-----|-------------------|
| S1  | Grain size descriptors |
| S2  | S1 + Wen model features |
| S3  | S2 + Processing parameters |
| S4  | S3 + Composition + solid-solution strengthening |

## Installation

```bash
pip install -r requirements.txt
```

## Running

### 5-Fold CV

```bash
# All feature sets, both targets, all models (defaults)
python run_5fold.py

# Specific feature sets and target
python run_5fold.py -f S1 S2 -t YS

# Specific models only
python run_5fold.py -f S3 S4 -t HV -m SVR XGBoost GPR

# Override CV folds and Optuna trials
python run_5fold.py --cv 10 --n-trials 100

# Print valid choices and exit
python run_5fold.py --list
```

### LOO CV

```bash
python run_loo.py
python run_loo.py -f S1 S2 -t YS
python run_loo.py -f S3 -t HV -m SVR XGBoost --n-trials 20 --inner-cv 5
```

### LOBO CV

```bash
python run_lobo.py
python run_lobo.py -f S4 -t YS HV
python run_lobo.py -f S1 -t YS -m RandomForest XGBoost --n-trials 50
```

## CLI Arguments

### `run_5fold.py`

| Argument | Default | Description |
|----------|---------|-------------|
| `-f`, `--feature-sets` | all (S1–S4) | Feature sets to run |
| `-t`, `--targets` | `YS HV` | Target properties |
| `-m`, `--models` | all 8 models | Models to run |
| `--cv` | `5` | Number of outer CV folds (also sets inner CV for Optuna) |
| `--n-trials` | `50` | Optuna trials per inner fold |
| `--list` | — | Print choices and exit |

> **Total Optuna trials per model:** `--cv × --n-trials`

### `run_loo.py` and `run_lobo.py`

| Argument | Default | Description |
|----------|---------|-------------|
| `-f`, `--feature-sets` | all (S1–S4) | Feature sets to run |
| `-t`, `--targets` | `YS HV` | Target properties |
| `-m`, `--models` | all 8 models | Models to run |
| `--inner-cv` | `5` | KFold folds for Optuna inner CV |
| `--n-trials` | `50` | Optuna trials per inner fold |
| `--list` | — | Print choices and exit |

> **Note (LOO cost):** Total fits per model ≈ `n_samples × --n-trials × --inner-cv`. With n≈94 and defaults: 94 × 50 × 5 = 23 500 fits. Use `--models` to narrow scope.
