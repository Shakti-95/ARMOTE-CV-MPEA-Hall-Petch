# recover_loo_models.py
#
# Recovers results for LOO models that finished ALL outer folds but were never
# written to the results CSV because a run was killed before run_workflow's
# single end-of-loop to_csv() call (armote_cv.py:992 writes once, after every
# model in the run finishes -- not incrementally per model).
#
# Nothing is re-fit or re-tuned. LeaveOneOut() is deterministic (fold i is
# always the same held-out row), so for any model with a complete set of
# per-fold artifacts (best_model_fold_i.pkl, x/y_scaler_fold_i.pkl,
# study_fold_i.pkl) this script reloads them, reapplies the saved model to its
# held-out row, and reconstructs the exact aggregation armote_cv.py would have
# produced (armote_cv.py:909-986).
#
# Not recoverable (timing metadata only, no effect on any metric): per-fold
# optimization/retraining wall-clock times -- these were never persisted
# anywhere and are written as NaN.
#
# Usage: python recover_loo_models.py -f S1 -t YS
#        python recover_loo_models.py -f S1 -t YS --models Ridge Lasso  # restrict

import argparse
import glob
import json
import os
import re

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, r2_score
from sklearn.model_selection import LeaveOneOut


def compute_metrics(y_true, y_pred):
    r2 = r2_score(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred) * 100
    return r2, mse, mape


def cols_for_blocks(manifest_df, blocks):
    return manifest_df.loc[manifest_df["block"].isin(blocks), "column"].tolist()


def feature_cols_for(feat_name, manifest):
    s1 = cols_for_blocks(manifest, ["S1_grain"])
    s2 = cols_for_blocks(manifest, ["S2_wen"])
    s3 = cols_for_blocks(manifest, ["S3_proc"])
    s4 = cols_for_blocks(manifest, ["S4_comp", "S4_sss"])
    sets = {"S1": s1, "S2": s1 + s2, "S3": s1 + s2 + s3, "S4": s1 + s2 + s3 + s4}
    return sets[feat_name]


def detect_complete_models(models_dir, target, n_samples, only_models=None):
    pattern = os.path.join(models_dir, f"*_{target}_best_model_fold_*.pkl")
    by_model = {}
    for path in glob.glob(pattern):
        m = re.match(rf"(.+)_{re.escape(target)}_best_model_fold_(\d+)\.pkl$", os.path.basename(path))
        if not m:
            continue
        by_model.setdefault(m.group(1), set()).add(int(m.group(2)))

    complete = {}
    for name, folds in by_model.items():
        if only_models and name not in only_models:
            continue
        if folds == set(range(n_samples)):
            complete[name] = True
        else:
            print(f"  Skipping {name}: {len(folds)}/{n_samples} folds present (incomplete)")
    return sorted(complete)


def recover_model(name, target, X, y_numpy, models_dir, studies_dir):
    n_samples = len(y_numpy)
    fold_train_metrics_list, fold_test_metrics_list = [], []
    fold_best_params_list = []
    oof_y_true, oof_y_pred = [], []
    all_train_y_true, all_train_y_pred = [], []

    for fold, (train_idx, test_idx) in enumerate(LeaveOneOut().split(X)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train_numpy, y_test_numpy = y_numpy[train_idx], y_numpy[test_idx]

        x_scaler = joblib.load(os.path.join(models_dir, f"{name}_{target}_x_scaler_fold_{fold}.pkl"))
        y_scaler = joblib.load(os.path.join(models_dir, f"{name}_{target}_y_scaler_fold_{fold}.pkl"))
        final_model = joblib.load(os.path.join(models_dir, f"{name}_{target}_best_model_fold_{fold}.pkl"))

        X_train_scaled = x_scaler.transform(X_train)
        X_test_scaled = x_scaler.transform(X_test)

        y_pred_train = y_scaler.inverse_transform(
            final_model.predict(X_train_scaled).reshape(-1, 1)
        )
        y_pred_test = y_scaler.inverse_transform(
            final_model.predict(X_test_scaled).reshape(-1, 1)
        )

        train_metrics = compute_metrics(y_train_numpy, y_pred_train)
        fold_mse = mean_squared_error(y_test_numpy.ravel(), y_pred_test.ravel())
        fold_mape = mean_absolute_percentage_error(y_test_numpy.ravel(), y_pred_test.ravel()) * 100
        test_metrics = (np.nan, fold_mse, fold_mape)

        study_path = os.path.join(studies_dir, f"{name}_{target}_study_fold_{fold}.pkl")
        if os.path.exists(study_path):
            study = joblib.load(study_path)
            best_trial = max(study.best_trials, key=lambda t: t.values[1])
            fold_best_params_list.append(best_trial.params)
        else:
            fold_best_params_list.append({})

        fold_train_metrics_list.append(train_metrics)
        fold_test_metrics_list.append(test_metrics)
        oof_y_true.append(y_test_numpy)
        oof_y_pred.append(y_pred_test)
        all_train_y_true.append(y_train_numpy)
        all_train_y_pred.append(y_pred_train)

    train_metrics_df = pd.DataFrame(fold_train_metrics_list, columns=["R2", "MSE", "MAPE"])
    test_metrics_df = pd.DataFrame(fold_test_metrics_list, columns=["R2", "MSE", "MAPE"])
    avg_train_metrics = train_metrics_df.mean().values

    pooled_true = np.concatenate([a.ravel() for a in oof_y_true])
    pooled_pred = np.concatenate([a.ravel() for a in oof_y_pred])
    avg_test_metrics = compute_metrics(pooled_true, pooled_pred)

    return {
        "Model": name,
        "Total Optimization Time (s)": np.nan,  # not recoverable; timing metadata only
        "Avg Optimization Time per Fold (s)": np.nan,
        "Avg Retraining Time (s)": np.nan,
        "Best Params Per Fold": json.dumps(fold_best_params_list),
        "Avg Train R2": avg_train_metrics[0],
        "Std Train R2": train_metrics_df["R2"].std(),
        "Avg Train MSE (original units)": avg_train_metrics[1],
        "Std Train MSE (original units)": train_metrics_df["MSE"].std(),
        "Avg Train MAPE (%)": avg_train_metrics[2],
        "Std Train MAPE (%)": train_metrics_df["MAPE"].std(),
        "Avg Test R2": avg_test_metrics[0],
        "Std Test R2": np.nan,
        "Avg Test MSE (original units)": avg_test_metrics[1],
        "Std Test MSE (original units)": np.nan,
        "Avg Test MAPE (%)": avg_test_metrics[2],
        "Std Test MAPE (%)": np.nan,
        "All Fold Test R2": json.dumps([None] * n_samples),
        "All Fold Test MSE (original units)": json.dumps(test_metrics_df["MSE"].tolist()),
        "All Fold Test MAPE (%)": json.dumps(test_metrics_df["MAPE"].tolist()),
        "All Fold Train R2": json.dumps(train_metrics_df["R2"].tolist()),
        "All Fold Train MSE (original units)": json.dumps(train_metrics_df["MSE"].tolist()),
        "All Fold Train MAPE (%)": json.dumps(train_metrics_df["MAPE"].tolist()),
        "NNR_Epochs": np.nan,
        "NNR_Batch_Size": np.nan,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-f", "--feature-set", required=True, choices=["S1", "S2", "S3", "S4"])
    parser.add_argument("-t", "--target", required=True, choices=["YS", "HV"])
    parser.add_argument("-m", "--models", nargs="+", default=None, help="Restrict to these model names")
    args = parser.parse_args()

    manifest = pd.read_csv(os.path.join("data", "inputs_feature_manifest.csv"))
    feat_cols = feature_cols_for(args.feature_set, manifest)

    df = pd.read_csv(os.path.join("data", "inputs.csv"))
    subset = df[feat_cols + [args.target]].dropna()
    X = subset[feat_cols]
    y_numpy = subset[args.target].values
    n_samples = len(y_numpy)

    result_dir = f"{args.feature_set}_{args.target}_Results_LOO_CV"
    models_dir = os.path.join(result_dir, "models")
    studies_dir = os.path.join(result_dir, "studies")
    csv_path = os.path.join(result_dir, f"{args.target}_results_{n_samples}_fold_CV.csv")

    print(f"{args.feature_set} | {args.target}: n_samples={n_samples}")
    print("Scanning for complete models...")
    complete_models = detect_complete_models(models_dir, args.target, n_samples, args.models)

    existing_df = pd.read_csv(csv_path) if os.path.exists(csv_path) else None
    already_done = set(existing_df["Model"].tolist()) if existing_df is not None else set()
    to_recover = [m for m in complete_models if m not in already_done]

    if not to_recover:
        print("Nothing to recover (complete models are either absent or already in the CSV).")
        return

    print(f"Recovering: {to_recover}")
    rows = [recover_model(name, args.target, X, y_numpy, models_dir, studies_dir) for name in to_recover]
    new_df = pd.DataFrame(rows)

    combined = pd.concat([existing_df, new_df], ignore_index=True) if existing_df is not None else new_df
    combined.to_csv(csv_path, index=False)
    print(f"Wrote {csv_path} ({len(combined)} rows total, {len(new_df)} recovered)")


if __name__ == "__main__":
    main()
