# Grain-size-YS-HV-ARMOTE-CV

Machine learning regression workflow for predicting yield strength (YS) and hardness (HV) of high-entropy alloys (HEAs) from grain size and composition features.

## ARMOTE-CV

**ARMOTE-CV** (Automated Regression with Multi-Objective Tuning and Evaluation via Cross-Validation) is a nested cross-validation workflow that:

1. For each outer fold, runs multi-objective Bayesian hyperparameter optimization (Optuna, TPE sampler) on the outer training data only — the outer test fold is never seen during tuning.
2. Retrains the final model with the fold-specific best hyperparameters on the full outer training set.
3. Evaluates on the held-out outer test fold and aggregates metrics across all folds.

Objectives optimized simultaneously: minimize MSE and maximize R². Best trial selected from the Pareto front by highest R².

**Models included:** Linear Regression, Bayesian Ridge, SVR, Decision Tree, Random Forest, XGBoost, Gaussian Process Regressor (GPR), Neural Network Regressor (NNR).

## Repository Structure

```
├── armote_cv.py                  # Core workflow: ARMOTE-CV engine
├── run_models.py                 # Entry point: data loading, model config, CLI
├── data/
│   ├── inputs.csv                # Dataset
│   └── inputs_feature_manifest.csv  # Maps feature columns to blocks (S1–S4)
├── requirements.txt
└── {S1..S4}_{YS|HV}_Results_{cv}_Fold_CV/   # Auto-generated output directories
    ├── models/    # Saved model + scaler .pkl files per fold
    ├── plots/     # Y-Y plots, Optuna optimization plots per fold
    ├── studies/   # Optuna study objects per fold
    └── {target}_results_{cv}_fold_CV.csv  # Summary metrics table
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

```bash
# All feature sets, both targets, all models (defaults)
python run_models.py

# Specific feature sets and target
python run_models.py -f S1 S2 -t YS

# Specific models only
python run_models.py -f S3 S4 -t HV -m SVR XGBoost GPR

# Override CV folds and Optuna trials
python run_models.py --cv 10 --n-trials 100

# Print valid choices and exit
python run_models.py --list
```

### CLI Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `-f`, `--feature-sets` | all (S1–S4) | Feature sets to run |
| `-t`, `--targets` | `YS HV` | Target properties |
| `-m`, `--models` | all 8 models | Models to run |
| `--cv` | `5` | Number of CV folds |
| `--n-trials` | `50` | Optuna trials per inner fold |
| `--list` | — | Print choices and exit |

> **Note:** Total Optuna trials per model = `--cv × --n-trials`.
