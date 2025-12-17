# ARMOTE: Automated Regression workflow with Multi-Objective hyperparameter optimization using Tree-Parzen Estimator algorithm
#
# Version: 1.1.0
# Author: Shakti P. Padhy
# Date: 2023-11-06
#
# Description:
# This script provides a comprehensive, end-to-end framework for training, optimizing,
# and evaluating multiple regression models. It leverages Optuna for advanced
# multi-objective hyperparameter tuning (minimizing MSE, maximizing R²) and now
# includes computational time tracking for both optimization and final model training.
#
# Workflow Steps:
# 1.  Data Scaling: Features and targets are standardized for modeling.
# 2.  Multi-Objective Optimization: Optuna efficiently searches for the Pareto front
#     of non-dominated hyperparameter solutions.
# 3.  Automated Model Selection: The model with the highest R-squared from the Pareto
#     front is selected as the champion model.
# 4.  Comprehensive Evaluation: The final model is evaluated using R², MSE, and MAPE.
# 5.  Time Tracking: The workflow measures and reports the time spent on optimization
#     and final model training.
# 6.  Artifact Generation & Saving: All important outputs are saved, including trained
#     models, data scalers, Optuna studies, visualization plots, and a final
#     summary CSV with performance metrics and computational times.
# ---

# --- 1. Imports ---
# Core libraries for data manipulation, file operations, numerical processing, and timing.
import os
import numpy as np
import pandas as pd
import joblib
import time
import matplotlib.pyplot as plt

# Scikit-learn modules for modeling, metrics, and data preprocessing.
from sklearn.model_selection import train_test_split, KFold, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_percentage_error
from sklearn.gaussian_process.kernels import RBF

# Keras (TensorFlow backend) for building the neural network.
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras import backend as K

# Optuna for advanced hyperparameter optimization.
import optuna


# --- 2. Global Variables ---
# Define global scalers to ensure consistent data transformation across the entire workflow.
x_scaler = StandardScaler()
y_scaler = StandardScaler()


# --- 3. Core Helper Functions ---


def compute_metrics(y_true, y_pred):
    """
    Computes and returns a standard set of regression metrics.

    Args:
        y_true (array-like): The ground truth (actual) target values.
        y_pred (array-like): The values predicted by the model.

    Returns:
        tuple: A tuple containing the (R-squared, MSE, MAPE) scores.
    """
    r2 = r2_score(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred) * 100
    return r2, mse, mape


def create_nn(hidden_layers=1, units=64, activation="relu", learning_rate=0.001):
    """
    Creates, configures, and compiles a Keras Sequential neural network for regression
    where the number of units is halved in each subsequent hidden layer.

    Args:
        hidden_layers (int): The number of hidden layers in the network.
        units (int): The number of neurons in each hidden layer.
        activation (str): The activation function for hidden layers.
        learning_rate (float): The learning rate for the Adam optimizer.

    Returns:
        keras.Model: A compiled Keras model instance ready for training.
    """
    input_dim = x_scaler.n_features_in_
    output_dim = y_scaler.n_features_in_
    # --- Model Creation ---
    model = Sequential()

    # Add the input layer and the first hidden layer
    model.add(
        Dense(
            units, activation=activation, input_dim=input_dim, name="input_hidden_layer"
        )
    )

    # Initialize a variable to track the number of units for the next layer
    current_units = units

    # Add the remaining hidden layers with halving units
    # This loop runs for the 2nd, 3rd, ... nth hidden layer
    for i in range(hidden_layers - 1):
        # Halve the number of units, ensuring it's at least 1
        current_units = max(1, current_units // 2)
        model.add(
            Dense(current_units, activation=activation, name=f"hidden_layer_{i + 1}")
        )

    # Add the output layer
    model.add(Dense(output_dim, activation="linear"))

    # --- Model Compilation ---
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss="mse")

    return model


def cross_val_nn(X, y, build_fn, params, cv=5, epochs=100, batch_size=8):
    """
    Performs manual K-Fold cross-validation for a Keras model for multi-objective evaluation.

    Args:
        X (np.array): The scaled feature data.
        y (np.array): The scaled target data.
        build_fn (function): The function used to construct the Keras model.
        params (dict): The dictionary of hyperparameters to pass to the `build_fn`.
        cv (int): The number of folds for cross-validation.

    Returns:
        tuple: Mean MSE (objective 1) and mean R-squared (objective 2) across all folds.
    """
    kf = KFold(n_splits=cv, shuffle=True, random_state=42)
    r2_scores, mse_scores = [], []
    for train_idx, val_idx in kf.split(X):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        model = build_fn(**params)
        model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)
        y_pred_val = model.predict(X_val)
        r2_scores.append(r2_score(y_val, y_pred_val))
        mse_scores.append(mean_squared_error(y_val, y_pred_val))
    return np.mean(mse_scores), np.mean(r2_scores)


def plot_yy(
    y_train_list,
    y_pred_train_list,
    y_test_list,
    y_pred_test_list,
    model_name,
    train_metrics,
    test_metrics,
    save_folder="plots",
):
    """
    Generates and saves a Predicted vs. Actual (Y-Y) plot to visualize model performance.

    Args:
        y_train_list (list): A list of 1D arrays, where each array is the true training targets from a fold.
        y_pred_train_list (list): A list of 1D arrays, where each array is the predicted training targets from a fold.
        y_test_list (list): A list of 1D arrays, where each array is the true test targets from a fold.
        y_pred_test_list (list): A list of 1D arrays, where each array is the predicted test targets from a fold.
        model_name (str): The name of the model for the plot title.
        train_metrics, test_metrics (tuple): Tuples of (R², MSE, MAPE) for train/test sets.
        save_folder (str): The directory where the plot image will be saved.
    """
    with plt.style.context("default"):
        os.makedirs(save_folder, exist_ok=True)
        plt.figure(figsize=(14, 6))
        train_r2, train_mse, train_mape = train_metrics
        test_r2, test_mse, test_mape = test_metrics

        # Define 5 distinct colors for the folds
        # Using Tableau 10 color palette
        colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
        n_folds = len(y_train_list)

        ## --- Training Data Subplot (All Folds) ---
        ax1 = plt.subplot(121)

        # Concatenate all fold data just to get the global min/max for the 'Ideal' line
        all_train_true = np.concatenate(y_train_list).ravel()
        all_train_pred = np.concatenate(y_pred_train_list).ravel()
        min_val_train = min(all_train_true.min(), all_train_pred.min())
        max_val_train = max(all_train_true.max(), all_train_pred.max())

        # Plot each fold with a different color
        for i in range(n_folds):
            ax1.scatter(
                y_train_list[i],
                y_pred_train_list[i],
                c=colors[i % len(colors)],
                alpha=0.7,
                label=f"Fold {i}",
            )

        ax1.plot(
            [min_val_train, max_val_train],
            [min_val_train, max_val_train],
            "k--",  # Changed to black dashed line
            label="Ideal",
        )
        ax1.set_xlabel("True Values", fontsize=16)
        ax1.set_ylabel("Predicted Values", fontsize=16)
        ax1.set_title(
            f"{model_name} Y-Y Plot - Train (All Folds)\nAvg R²: {train_r2:.3f} | Avg MSE: {train_mse:.6f} | Avg MAPE: {train_mape:.2f}%",
            fontsize=14,
        )
        ax1.legend()

        # --- Testing Data Subplot (Out-of-Fold) ---
        ax2 = plt.subplot(122)

        # Concatenate all fold data just to get the global min/max for the 'Ideal' line
        all_test_true = np.concatenate(y_test_list).ravel()
        all_test_pred = np.concatenate(y_pred_test_list).ravel()
        min_val_test = min(all_test_true.min(), all_test_pred.min())
        max_val_test = max(all_test_true.max(), all_test_pred.max())

        # Plot each fold with a different color
        for i in range(n_folds):
            ax2.scatter(
                y_test_list[i],
                y_pred_test_list[i],
                c=colors[i % len(colors)],
                alpha=0.7,
                label=f"Fold {i}",
            )

        ax2.plot(
            [min_val_test, max_val_test],
            [min_val_test, max_val_test],
            "k--",
            label="Ideal",
        )
        ax2.set_xlabel("True Values", fontsize=16)
        ax2.set_ylabel("Predicted Values", fontsize=16)
        ax2.set_title(
            f"{model_name} Y-Y Plot - Test (Out-of-Fold)\nAvg R²: {test_r2:.3f} | Avg MSE: {test_mse:.6f} | Avg MAPE: {test_mape:.2f}%",
            fontsize=14,
        )
        ax2.legend()

        plt.tight_layout()
        plt.savefig(os.path.join(save_folder, f"{model_name}_yy_plot_CV.png"))
        plt.show()


def generate_optuna_plots(study, model_name, save_folder="plots"):
    """
    Generates and saves key Optuna visualization plots as static PNG files.

    Args:
        study (optuna.study.Study): The completed Optuna study object.
        model_name (str): The name of the model for file naming.
        save_folder (str): The directory where plot images will be saved.
    """
    try:
        import optuna.visualization.matplotlib as vis
    except ImportError:
        print(
            "Could not import Optuna's matplotlib visualization. Please run 'pip install matplotlib'."
        )
        return

    # Ensure the save folder exists
    os.makedirs(save_folder, exist_ok=True)

    # Generate and save each plot
    # Each Optuna matplotlib plot function returns an 'Axes' object.
    # We get its 'figure' attribute to save and then close it.

    # 1. Pareto Front
    ax = vis.plot_pareto_front(study, target_names=["MSE", "R-squared"])
    fig = ax.figure
    fig.tight_layout()
    fig.savefig(os.path.join(save_folder, f"{model_name}_pareto_front.png"))
    plt.close(fig)

    # 2. Optimization History (MSE)
    ax = vis.plot_optimization_history(
        study, target=lambda t: t.values[0], target_name="MSE"
    )
    fig = ax.figure
    fig.tight_layout()
    fig.savefig(os.path.join(save_folder, f"{model_name}_optimization_history_mse.png"))
    plt.close(fig)

    # 3. Optimization History (R-squared)
    ax = vis.plot_optimization_history(
        study, target=lambda t: t.values[1], target_name="R-squared"
    )
    fig = ax.figure
    fig.tight_layout()
    fig.savefig(os.path.join(save_folder, f"{model_name}_optimization_history_r2.png"))
    plt.close(fig)

    # 4. Parameter Importances (Combined for MSE and R-squared)
    ax = vis.plot_param_importances(study)
    fig = ax.figure
    fig.tight_layout()
    fig.savefig(
        os.path.join(save_folder, f"{model_name}_param_importances_combined.png")
    )
    plt.close(fig)

    # 5. Parameter Importances (Duration)
    ax = vis.plot_param_importances(
        study, target=lambda t: t.duration.total_seconds(), target_name="duration"
    )
    fig = ax.figure
    fig.tight_layout()
    fig.savefig(os.path.join(save_folder, f"{model_name}_duration_importances.png"))
    plt.close(fig)

    print(f"Optuna plots for {model_name} saved as PNGs in '{save_folder}' directory.")


# --- 4. Main Workflow Functions ---
def find_best_hyperparameters(
    model,
    param_space,
    X,
    y,
    is_nn=False,
    is_gpr=False,
    gpr_kernel_map=None,
    nn_epochs=100,
    nn_batch_size=32,
):
    """
    Performs multi-objective optimization using Optuna on the entire dataset.

    Args:
        model: An unfitted scikit-learn model or None for neural networks.
        param_space (dict): The hyperparameter search space for Optuna.
        X, y: The full feature and target datasets.
        is_nn (bool): A flag to handle neural network logic separately.
        is_gpr (bool): A flag to handle Gaussian Process Regressor logic separately.
        gpr_kernel_map (dict): A mapping of kernel names to kernel objects for GPR.
        nn_epochs, nn_batch_size (int): Number of epochs and batch size for NN training
        during cross-validation.

    Returns:
        tuple: Contains best parameters, metrics, predictions, the final model, the study,
               and the computational times for optimization and retraining.
    """
    # --- 1. Transform Data ---
    X_scaled = x_scaler.transform(X)
    y_scaled = y_scaler.transform(y.reshape(-1, 1))

    # Handle GPR kernel mapping requirement
    final_kernel_map = gpr_kernel_map
    if is_gpr and final_kernel_map is None and param_space and "kernel" in param_space:
        raise ValueError(
            "A 'gpr_kernel_map' must be provided to 'run_workflow' "
            "when tuning the 'kernel' parameter for 'GaussianProcess'."
        )

    # --- 2. Handle Models Without Hyperparameter Tuning ---
    if not param_space:
        print("No hyperparameters defined. Returning default settings.")
        # Return empty params, no study, and zero time
        return ({}, None, 0)

    # --- 3. Define the Multi-Objective Function for Optuna ---
    def objective(trial):
        params = {}
        for name, definition in param_space.items():
            if definition[0] == "int":
                params[name] = trial.suggest_int(name, *definition[1:])
            elif definition[0] == "float":
                params[name] = trial.suggest_float(
                    name, definition[1], definition[2], log=("log" in definition)
                )
            elif definition[0] == "categorical":
                params[name] = trial.suggest_categorical(name, definition[1])

        if is_nn:
            K.clear_session()
            return cross_val_nn(
                X_scaled,
                y_scaled,
                create_nn,
                params,
                epochs=nn_epochs,
                batch_size=nn_batch_size,
            )
        else:
            current_model = model
            if is_gpr and "kernel" in params:
                kernel_name = params.pop("kernel")
                params["kernel"] = final_kernel_map[kernel_name]

            current_model.set_params(**params)

            scoring = {"r2": "r2", "neg_mse": "neg_mean_squared_error"}
            scores = cross_validate(
                current_model,
                X_scaled,
                y_scaled.ravel(),
                cv=5,
                scoring=scoring,
                n_jobs=-1,
            )
            return -np.mean(scores["test_neg_mse"]), np.mean(scores["test_r2"])

    # --- 4. Run and Time the Optuna Optimization Study ---
    n_parallel_jobs = 1 if is_nn or is_gpr else -1
    if is_nn:
        print("Using n_jobs=1 for Neural Network to ensure GPU stability.")
    if is_gpr:
        print(
            "Using n_jobs=1 for GaussianProcessRegressor to manage high memory usage."
        )

    study = optuna.create_study(directions=["minimize", "maximize"])
    start_time = time.time()

    study.optimize(
        objective, n_trials=100, n_jobs=n_parallel_jobs, show_progress_bar=True
    )
    optimization_time = time.time() - start_time
    print(f"Hyperparameter optimization completed in {optimization_time:.3f} seconds.")

    # --- 5. Select the Best Model from the Pareto Front ---
    print("Selecting best trial from the Pareto front...")
    best_trial = max(study.best_trials, key=lambda t: t.values[1])
    best_params = best_trial.params
    print(
        f"Selected Trial #{best_trial.number} with MSE={best_trial.values[0]:.4f}, R2={best_trial.values[1]:.4f}"
    )

    return (
        best_params,
        study,
        optimization_time,
    )


def run_workflow(
    X,
    y,
    models,
    param_spaces,
    output_name="workflow_output",
    gpr_kernel_map=None,
    nn_epochs=100,
    nn_batch_size=32,
):
    """
    Executes the end-to-end multi-objective machine learning workflow.
    1. Finds best hyperparameters using Optuna on 100% of the data.
    2. Performs a 5-fold CV using those best hyperparameters for final evaluation.
    3. Saves the model and scalers from every fold.

    Args:
        X (pd.DataFrame or np.array): The complete feature dataset.
        y (pd.Series or np.array): The complete target dataset.
        models (dict): A dictionary of model names to their unfitted instances.
        param_spaces (dict): A dictionary of model names to their hyperparameter search spaces.
        output_base_name (str): The base name for output files and folders.
        gpr_kernel_map (dict, optional): A map of strings to GPR kernel objects.
        nn_epochs, nn_batch_size (int): Number of epochs and batch size for NN training
        during cross-validation.

    Returns:
        pd.DataFrame: A DataFrame summarizing the performance and timings of all models.
    """
    # --- 1. Setup and Data Preparation ---
    # Print the output name
    print(f"Initializing Pre-Optimization CV workflow for target: '{output_name}'...")

    # Define dynamic paths based on the output_name
    base_dir = output_name
    models_dir = os.path.join(base_dir, "models")
    plots_dir = os.path.join(base_dir, "plots")
    studies_dir = os.path.join(base_dir, "studies")

    print("Setting up directories...")
    # Create all new dynamic directories
    for folder in [models_dir, plots_dir, studies_dir]:
        os.makedirs(folder, exist_ok=True)

    # Ensure X and y are pandas objects for easy .iloc indexing
    if not isinstance(X, (pd.DataFrame, pd.Series)):
        X = pd.DataFrame(X)
    if not isinstance(y, (pd.DataFrame, pd.Series)):
        y = pd.Series(y)
    # Ensure y is a 1D Series
    if isinstance(y, pd.DataFrame):
        y = y.iloc[:, 0]

    # Remove any existing indices to prevent misalignment during CV
    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)

    # Store original numpy arrays for y
    y_numpy = y.values

    # --- 2. Model Training and Evaluation Loop ---
    results = []
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    for name, model in models.items():
        print(f"\n--- Starting Workflow for: {name} ---")

        # Flags for model-specific handling
        is_nn = name == "NNR"
        is_gpr = name == "GPR"

        # === PART 1: GLOBAL OPTIMIZATION ===
        print(f"\n--- {name} - Step 1: Finding Best Hyperparameters (on 100% data) ---")

        # Fit global scalers on 100% of data for optimization step
        global x_scaler, y_scaler
        x_scaler = StandardScaler()
        y_scaler = StandardScaler()
        x_scaler.fit(X)
        y_scaler.fit(y_numpy.reshape(-1, 1))

        # Save these "global" scalers
        joblib.dump(x_scaler, os.path.join(models_dir, "x_scaler_global_opt.pkl"))
        joblib.dump(y_scaler, os.path.join(models_dir, "y_scaler_global_opt.pkl"))
        print(f"Scalers (fit on 100% data) saved in '{models_dir}'.")

        (
            best_params,
            study,
            optimization_time,
        ) = find_best_hyperparameters(
            model,
            param_spaces.get(name, {}),
            X,
            y_numpy,
            is_nn=is_nn,
            is_gpr=is_gpr,
            gpr_kernel_map=gpr_kernel_map,
            nn_epochs=nn_epochs,
            nn_batch_size=nn_batch_size,
        )

        # Save optimization artifacts
        if study:
            study_path = os.path.join(studies_dir, f"{name}_study.pkl")
            joblib.dump(study, study_path)
            print(f"Optuna study saved to: {study_path}")
            generate_optuna_plots(study, name, save_folder=plots_dir)

        # === PART 2: 5-FOLD CV FINAL EVALUATION ===
        print(
            f"\n--- {name} - Step 2: 5-Fold CV Final Evaluation (using best params) ---"
        )

        # Lists to store results from each fold
        fold_train_metrics_list = []
        fold_test_metrics_list = []
        fold_retrain_times = []

        # Lists to store predictions for combined plotting
        oof_y_true = []
        oof_y_pred = []
        all_train_y_true = []
        all_train_y_pred = []

        for fold, (train_idx, test_idx) in enumerate(kf.split(X, y)):
            print(f"\n--- {name} - Evaluation Fold {fold + 1}/5 ---")

            # Get data for this fold
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            y_train_numpy, y_test_numpy = y_numpy[train_idx], y_numpy[test_idx]

            # --- 4. Fit and Save Data Scalers (per-fold) ---
            print(f"Fitting scalers on Fold {fold + 1} training data...")

            # Re-fit global scalers *only on this fold's training data*
            x_scaler = StandardScaler()
            y_scaler = StandardScaler()
            x_scaler.fit(X_train)
            y_scaler.fit(y_train_numpy.reshape(-1, 1))

            # Scale data for this fold
            X_train_scaled = x_scaler.transform(X_train)
            y_train_scaled = y_scaler.transform(y_train_numpy.reshape(-1, 1))
            X_test_scaled = x_scaler.transform(X_test)

            # Save scalers for every fold
            joblib.dump(
                x_scaler, os.path.join(models_dir, f"{name}_x_scaler_fold_{fold}.pkl")
            )
            joblib.dump(
                y_scaler, os.path.join(models_dir, f"{name}_y_scaler_fold_{fold}.pkl")
            )
            print(
                f"{name}_x_scaler_fold_{fold}.pkl and {name}_y_scaler_fold_{fold}.pkl saved in '{models_dir}'."
            )

            # --- 5. Train and Time the Final Model (per-fold) ---
            print("Retraining the final model with the best hyperparameters...")
            start_time = time.time()

            # Clear Keras session for NNR
            if is_nn:
                K.clear_session()
                final_model = create_nn(**best_params)
                final_model.fit(
                    X_train_scaled,
                    y_train_scaled,
                    epochs=nn_epochs,
                    batch_size=nn_batch_size,
                    verbose=0,
                )
            elif is_gpr:
                final_params = best_params.copy()
                if "kernel" in final_params:
                    kernel_name = final_params.pop("kernel")
                    final_params["kernel"] = gpr_kernel_map[kernel_name]
                elif not param_spaces.get(name, {}):
                    final_params["kernel"] = RBF(1.0)
                    final_params["alpha"] = 1e-10
                final_model = model.set_params(**final_params)
                final_model.fit(X_train_scaled, y_train_scaled.ravel())
            else:
                final_model = model.set_params(**best_params)
                final_model.fit(X_train_scaled, y_train_scaled.ravel())

            retraining_time = time.time() - start_time
            print(
                f"Fold {fold} model training complete in {retraining_time:.3f} seconds."
            )

            # --- 6. Final Evaluation (per-fold) ---
            y_pred_train_scaled = final_model.predict(X_train_scaled)
            y_pred_test_scaled = final_model.predict(X_test_scaled)
            y_pred_train = y_scaler.inverse_transform(
                y_pred_train_scaled.reshape(-1, 1)
            )
            y_pred_test = y_scaler.inverse_transform(y_pred_test_scaled.reshape(-1, 1))

            train_metrics = compute_metrics(y_train_numpy, y_pred_train)
            test_metrics = compute_metrics(y_test_numpy, y_pred_test)

            # --- 7. Store Fold Results ---
            fold_train_metrics_list.append(train_metrics)
            fold_test_metrics_list.append(test_metrics)
            fold_retrain_times.append(retraining_time)

            # Store predictions for combined plot
            oof_y_true.append(y_test_numpy)
            oof_y_pred.append(y_pred_test)
            all_train_y_true.append(y_train_numpy)
            all_train_y_pred.append(y_pred_train)

            # --- 8. Save model for every fold ---
            print(f"Saving model for {name} from Fold {fold}...")
            model_path = os.path.join(
                models_dir,
                f"{name}_best_model_fold_{fold}.{'keras' if name == 'NNR' else 'pkl'}",
            )
            if name == "NNR":
                final_model.save(model_path)
            else:
                joblib.dump(final_model, model_path)
            print(f"Best model (Fold {fold}) saved to: {model_path}")

        # --- 9. Collate Results After All Folds ---
        print(f"\n--- Aggregating 5-Fold CV results for: {name} ---")

        # Calculate average metrics and times
        train_metrics_df = pd.DataFrame(
            fold_train_metrics_list, columns=["R2", "MSE", "MAPE"]
        )
        test_metrics_df = pd.DataFrame(
            fold_test_metrics_list, columns=["R2", "MSE", "MAPE"]
        )

        avg_train_metrics = train_metrics_df.mean().values
        avg_test_metrics = test_metrics_df.mean().values
        avg_retrain_time = np.mean(fold_retrain_times)

        # Generate combined plot
        plot_yy(
            all_train_y_true,
            all_train_y_pred,
            oof_y_true,
            oof_y_pred,
            name,
            avg_train_metrics,
            avg_test_metrics,
            save_folder=plots_dir,
        )

        # Store results in the main list
        results.append(
            {
                "Model": name,
                "Optimization Time (s)": round(optimization_time, 3),
                "Avg Retraining Time (s)": round(avg_retrain_time, 3),
                "Best Params": best_params,
                "Avg Train R2": avg_train_metrics[0],
                "Avg Train MSE": avg_train_metrics[1],
                "Avg Train MAPE (%)": avg_train_metrics[2],
                "Avg Test R2": avg_test_metrics[0],
                "Avg Test MSE": avg_test_metrics[1],
                "Avg Test MAPE (%)": avg_test_metrics[2],
                "All Fold Test R2": test_metrics_df["R2"].tolist(),
                "All Fold Test MSE": test_metrics_df["MSE"].tolist(),
                "All Fold Test MAPE (%)": test_metrics_df["MAPE"].tolist(),
                "All Fold Train R2": train_metrics_df["R2"].tolist(),
                "All Fold Train MSE": train_metrics_df["MSE"].tolist(),
                "All Fold Train MAPE (%)": train_metrics_df["MAPE"].tolist(),
                "NNR_Epochs": nn_epochs if name == "NNR" else np.nan,
                "NNR_Batch_Size": nn_batch_size if name == "NNR" else np.nan,
            }
        )

    # --- 10. Final Summary ---
    results_df = pd.DataFrame(results)
    results_csv_path = os.path.join(base_dir, "results_5_fold_CV.csv")
    results_df.to_csv(results_csv_path, index=False)
    print("\n--- Workflow Complete ---")
    print(f"Final 5-fold CV results summary saved to {results_csv_path}")
    return results_df
