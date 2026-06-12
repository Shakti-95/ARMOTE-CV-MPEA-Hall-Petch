# run_models.py
#
# Runs ARMOTE-CV workflow on grain-size-based HEA data with cumulative feature sets.
#
# Feature set logic (cumulative):
#   S1_models : S1_grain
#   S2_models : S1_grain + S2_wen
#   S3_models : S1_grain + S2_wen + S3_proc
#   S4_models : S1_grain + S2_wen + S3_proc + S4_comp + S4_sss
#
# Targets: YS (MPa) and HV — each target handled independently with per-target NaN drop.
# Outputs saved to: {feat_set}_{target}_Results_5_Fold_CV/

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

feature_sets = {
    "S1": s1,
    "S2": s1 + s2,
    "S3": s1 + s2 + s3,
    "S4": s1 + s2 + s3 + s4,
}

targets = ["YS", "HV"]

# ── Models ─────────────────────────────────────────────────────────────────────
models_to_run = {
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
    "LinearRegression": {},  # no tuning — {} is explicit; None also works but less clear
    "BayesianRidge": {
        "max_iter": ("int", 100, 500),
        "alpha_1": ("float", 1e-7, 1e-5, "log"),
        "alpha_2": ("float", 1e-7, 1e-5, "log"),
        "lambda_1": ("float", 1e-7, 1e-5, "log"),
        "lambda_2": ("float", 1e-7, 1e-5, "log"),
    },
    "SVR": {
        "C": ("float", 0.1, 1e4, "log"),  # note: 10002 in old notebook was a typo
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

# ── Main loop ──────────────────────────────────────────────────────────────────
print("=" * 80)
print("STARTING ALL WORKFLOWS")
print(f"Feature sets : {list(feature_sets.keys())}")
print(f"Targets      : {targets}")
print(f"Total runs   : {len(feature_sets) * len(targets)}")
print("=" * 80)

for feat_name, feat_cols in feature_sets.items():
    for target_name in targets:
        print(f"\n{'=' * 30} {feat_name} | Target: {target_name} {'=' * 30}")

        # Drop NaN rows independently per target — preserves max samples for each
        subset = df[feat_cols + [target_name]].dropna()
        n_dropped = len(df) - len(subset)
        print(
            f"Samples: {len(subset)} / {len(df)} ({n_dropped} NaN rows dropped for {target_name})"
        )
        print(f"Features ({len(feat_cols)}): {feat_cols}")

        X = subset[feat_cols]
        y = subset[target_name].values

        output_folder = f"{feat_name}_{target_name}_Results_5_Fold_CV"

        run_workflow(
            X,
            y,
            models_to_run,
            param_spaces,
            cv=5,
            output_name=target_name,
            output_folder_name=output_folder,
            gpr_kernel_map=gpr_kernel_map,
            nn_epochs=100,
            nn_batch_size=8,
            colors=["#EE6677", "#228833", "#4477AA", "#CCBB44", "#66CCEE"],
            refit_scaler_per_fold=True,
            n_trials=100,
            cv_random_state=42,
            nn_model_names=("NNR",),
            gpr_model_names=("GPR",),
        )

        print(f"{'=' * 30} Completed: {feat_name} | {target_name} {'=' * 30}\n")

print("\n" + "=" * 80)
print("ALL WORKFLOWS COMPLETE.")
print("=" * 80)
