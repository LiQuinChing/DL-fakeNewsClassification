from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from joblib import dump


def train_svm(X_train, y_train, model_path="model/svm_model.pkl", quick_tune=True):
    """
    Train a LinearSVC (SVM) model with memory-safe hyperparameter tuning and save it.
    """

    # Scale features
    print("Scaling features...")
    scaler = StandardScaler(with_mean=False)
    X_train_scaled = scaler.fit_transform(X_train)

    # Define the hyperparameter grid for GridSearchCV
    if quick_tune:
        param_grid = {
            "C": [0.001, 0.005, 0.01, 0.05, 0.1],  # Regularization strength
            "class_weight": [None, "balanced"],  # Handles class imbalance
            "max_iter": [10000],  # Iterations for convergence
        }
    else:
        param_grid = {
            "C": [0.001, 0.01, 0.05, 0.1, 1],
            "class_weight": [None, "balanced"],
            "max_iter": [5000, 10000],
            "dual": [False],
            "loss": ["squared_hinge"],  # SVM loss function
        }

    # Hyperparameter Optimization using Grid Search
    print("Running Grid Search (safe mode, n_jobs=1)...")
    grid_search = GridSearchCV(
        LinearSVC(),
        param_grid,
        cv=3,  # 3-fold CV for less memory usage
        scoring="accuracy",
        verbose=2,
        n_jobs=1,  # Run sequentially to avoid Windows crashes
    )

    # Fit the model
    grid_search.fit(X_train_scaled, y_train)

    # Get the best model
    best_svm = grid_search.best_estimator_
    print("\nBest Hyperparameters:", grid_search.best_params_)
    print("Best Cross-validation Score:", grid_search.best_score_)

    # Save model and scaler
    dump((best_svm, scaler), model_path)
    print(f"Model and scaler saved to {model_path}")

    return best_svm, scaler
