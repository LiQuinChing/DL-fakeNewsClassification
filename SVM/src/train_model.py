from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import hinge_loss
from joblib import dump


def train_svm(X, y, model_path="model/svm_model.pkl", quick_tune=True):

    print("Splitting data (80/10/10)...")
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )

    # Scale features
    print("Scaling features...")
    scaler = StandardScaler(with_mean=False)
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    # Define hyperparameter grid
    if quick_tune is True:
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

    # Hyperparameter tuning
    print("Running Grid Search (safe mode, n_jobs=1)...")
    grid_search = GridSearchCV(
        LinearSVC(),
        param_grid,
        cv=3,  # 3-fold cross-validation
        scoring="accuracy",
        verbose=2,
        n_jobs=1,  # Avoid potential issues with multi-threading
    )

    # Fit model
    grid_search.fit(X_train_scaled, y_train)

    # Get best model from grid search
    best_svm = grid_search.best_estimator_
    print("\nBest Hyperparameters:", grid_search.best_params_)
    print("Best Cross-validation Score:", grid_search.best_score_)

    # Evaluate validation loss
    y_val_num = [0 if label == "fake" else 1 for label in y_val]  # convert to numeric
    y_val_decision = best_svm.decision_function(X_val_scaled)
    val_loss = hinge_loss(y_val_num, y_val_decision)
    print(f"Validation Loss: {val_loss:.4f}")

    # Save model and scaler
    dump((best_svm, scaler), model_path)
    print(f"Model and scaler saved to {model_path}")

    # Return model, scaler, val_loss, and all splits
    return (
        best_svm,
        scaler,
        val_loss,
        (X_train_scaled, y_train, X_val_scaled, y_val, X_test_scaled, y_test),
    )
