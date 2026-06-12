# run_models.py
#
# Runs ARMOTE-CV workflow on grain-size-based HEA data with cumulative feature sets.
#
# Feature set logic (cumulative):
#   S1 : S1_grain
#   S2 : S1_grain + S2_wen
#   S3 : S1_grain + S2_wen + S3_proc
#   S4 : S1_grain + S2_wen + S3_proc + S4_comp + S4_sss
#
# Targets: YS (MPa) and HV — each target handled independently with per-target NaN drop.
# Outputs saved to: {feat_set}_{target}_Results_5_Fold_CV/
#
# Usage examples:
#   python run_models.py                              # all feature sets, all targets, all models
#   python run_models.py -f S1 S2 -t YS              # S1+S2, YS only
#   python run_models.py -f S3 -t HV -m SVR XGBoost  # S3, HV, two models only
#   python run_models.py --list                       # print valid choices and exit

import argparse
import sys
import warnings

import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression, BayesianRidge
from sklearn.svm import SVR
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, RationalQuadratic
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

try:
    from armote_cv import run_workflow
except ImportError:
    print("=" * 80)
    print("ERROR: 'armote_cv.py' not found in the current directory.")
    print("=" * 80)
    raise

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# ── Data loading ───────────────────────────────────────────────────────────────
df = pd.read_csv("data/inputs.csv")
manifest = pd.read_csv("data/inputs_feature_manifest.csv")


# ── Feature sets (built from manifest, cumulative) ────────────────────────────
def cols_for_blocks(manifest_df, blocks):
    return manifest_df.loc[manifest_df["block"].isin(blocks), "column"].tolist()


s1 = cols_for_blocks(manifest, ["S1_grain"])
s2 = cols_for_blocks(manifest, ["S2_wen"])
s3 = cols_for_blocks(manifest, ["S3_proc"])
s4 = cols_for_blocks(manifest, ["S4_comp", "S4_sss"])

ALL_FEATURE_SETS = {
    "S1": s1,
    "S2": s1 + s2,
    "S3": s1 + s2 + s3,
    "S4": s1 + s2 + s3 + s4,
}

ALL_TARGETS = ["YS", "HV"]

# ── Models ─────────────────────────────────────────────────────────────────────
ALL_MODELS = {
    "LinearRegression": LinearRegression(),
    "BayesianRidge": BayesianRidge(),
    "SVR": SVR(),
    "DecisionTree": DecisionTreeRegressor(random_state=42),
    "RandomForest": RandomForestRegressor(random_state=42),
    "XGBoost": XGBRegressor(random_state=42, n_jobs=-1),
    "GPR": GaussianProcessRegressor(random_state=42, n_restarts_optimizer=9),
    "NNR": None,
}

# ── GPR kernel map ─────────────────────────────────────────────────────────────
gpr_kernel_map = {
    "RBF_default": RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2)),
    "Matern_nu_0.5": Matern(length_scale=1.0, nu=0.5, length_scale_bounds=(1e-2, 1e2)),
    "Matern_nu_1.5": Matern(length_scale=1.0, nu=1.5, length_scale_bounds=(1e-2, 1e2)),
    "Matern_nu_2.5": Matern(length_scale=1.0, nu=2.5, length_scale_bounds=(1e-2, 1e2)),
    "RationalQuadratic": RationalQuadratic(
        length_scale=1.0,
        alpha=0.1,
        length_scale_bounds=(1e-2, 1e2),
        alpha_bounds=(1e-2, 1e2),
    ),
}

# ── Hyperparameter search spaces ───────────────────────────────────────────────
param_spaces = {
    "LinearRegression": {},
    "BayesianRidge": {
        "max_iter": ("int", 100, 500),
        "alpha_1": ("float", 1e-7, 1e-5, "log"),
        "alpha_2": ("float", 1e-7, 1e-5, "log"),
        "lambda_1": ("float", 1e-7, 1e-5, "log"),
        "lambda_2": ("float", 1e-7, 1e-5, "log"),
    },
    "SVR": {
        "C": ("float", 0.1, 1e4, "log"),
        "gamma": ("float", 1e-4, 1.0, "log"),
        "epsilon": ("float", 1e-3, 0.5, "log"),
    },
    "DecisionTree": {
        "max_depth": ("int", 5, 50),
        "min_samples_split": ("int", 2, 20),
        "min_samples_leaf": ("int", 1, 20),
    },
    "RandomForest": {
        "n_estimators": ("int", 50, 200),
        "max_depth": ("int", 5, 50),
        "min_samples_split": ("int", 2, 20),
        "min_samples_leaf": ("int", 1, 10),
    },
    "XGBoost": {
        "n_estimators": ("int", 50, 200),
        "learning_rate": ("float", 0.01, 0.5, "log"),
        "max_depth": ("int", 3, 10),
        "subsample": ("float", 0.6, 1.0),
        "colsample_bytree": ("float", 0.6, 1.0),
    },
    "GPR": {
        "kernel": ("categorical", list(gpr_kernel_map.keys())),
        "alpha": ("float", 1e-10, 1e-1, "log"),
    },
    "NNR": {
        "hidden_layers": ("int", 1, 4),
        "units": ("int", 32, 128),
        "activation": ("categorical", ["relu", "tanh", "selu"]),
        "learning_rate": ("float", 1e-5, 1e-2, "log"),
    },
}

# ── CLI argument parsing ───────────────────────────────────────────────────────
parser = argparse.ArgumentParser(
    description="Run ARMOTE-CV nested-CV workflow.",
    formatter_class=argparse.RawTextHelpFormatter,
)
parser.add_argument(
    "-f",
    "--feature-sets",
    nargs="+",
    choices=list(ALL_FEATURE_SETS.keys()),
    default=list(ALL_FEATURE_SETS.keys()),
    metavar="FEAT",
    help=f"Feature sets to run. Choices: {list(ALL_FEATURE_SETS.keys())} (default: all)",
)
parser.add_argument(
    "-t",
    "--targets",
    nargs="+",
    choices=ALL_TARGETS,
    default=ALL_TARGETS,
    metavar="TARGET",
    help=f"Targets to run. Choices: {ALL_TARGETS} (default: all)",
)
parser.add_argument(
    "-m",
    "--models",
    nargs="+",
    choices=list(ALL_MODELS.keys()),
    default=list(ALL_MODELS.keys()),
    metavar="MODEL",
    help=f"Models to run. Choices: {list(ALL_MODELS.keys())} (default: all)",
)
parser.add_argument(
    "--n-trials",
    type=int,
    default=50,
    help="Optuna trials per inner fold (default: 50)",
)
parser.add_argument(
    "--cv",
    type=int,
    default=5,
    help="Number of CV folds (default: 5)",
)
parser.add_argument(
    "--list",
    action="store_true",
    help="Print valid choices for -f / -t / -m and exit.",
)

args = parser.parse_args()

if args.list:
    print("Feature sets :", list(ALL_FEATURE_SETS.keys()))
    print("Targets      :", ALL_TARGETS)
    print("Models       :", list(ALL_MODELS.keys()))
    sys.exit(0)

# Build filtered selections
feature_sets = {k: ALL_FEATURE_SETS[k] for k in args.feature_sets}
targets = args.targets
models_to_run = {k: ALL_MODELS[k] for k in args.models}

# ── Main loop ──────────────────────────────────────────────────────────────────
total_runs = len(feature_sets) * len(targets)
print("=" * 80)
print("STARTING ARMOTE-CV NESTED CV WORKFLOWS")
print(f"Feature sets : {list(feature_sets.keys())}")
print(f"Targets      : {targets}")
print(f"Models       : {list(models_to_run.keys())}")
print(f"CV folds     : {args.cv}")
print(
    f"Optuna trials: {args.n_trials} per inner fold  ({args.cv * args.n_trials} total per model)"
)
print(f"Total runs   : {total_runs}")
print("=" * 80)

for feat_name, feat_cols in feature_sets.items():
    for target_name in targets:
        print(f"\n{'=' * 30} {feat_name} | Target: {target_name} {'=' * 30}")

        subset = df[feat_cols + [target_name]].dropna()
        n_dropped = len(df) - len(subset)
        print(
            f"Samples: {len(subset)} / {len(df)} ({n_dropped} NaN rows dropped for {target_name})"
        )
        print(f"Features ({len(feat_cols)}): {feat_cols}")

        X = subset[feat_cols]
        y = subset[target_name].values

        output_folder = f"{feat_name}_{target_name}_Results_{args.cv}_Fold_CV"

        run_workflow(
            X,
            y,
            models_to_run,
            param_spaces,
            cv=args.cv,
            output_name=target_name,
            output_folder_name=output_folder,
            gpr_kernel_map=gpr_kernel_map,
            nn_epochs=100,
            nn_batch_size=8,
            colors=["#EE6677", "#228833", "#4477AA", "#CCBB44", "#66CCEE"],
            refit_scaler_per_fold=True,
            n_trials=args.n_trials,
            cv_random_state=42,
            nn_model_names=("NNR",),
            gpr_model_names=("GPR",),
        )

        print(f"{'=' * 30} Completed: {feat_name} | {target_name} {'=' * 30}\n")

print("\n" + "=" * 80)
print("ALL WORKFLOWS COMPLETE.")
print("=" * 80)
