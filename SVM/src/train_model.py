from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV
from joblib import dump


def train_svm(X_train, y_train, model_path="models/svm_model.pkl"):
    """Train the SVM model with hyperparameter tuning and save it"""

    # Define hyperparameter grid
    param_grid = {
        "C": [0.01, 0.1, 1, 10, 100],
        "class_weight": [None, "balanced"],
        "max_iter": [1000, 5000, 10000],
    }

    # Grid search with 5-fold cross-validation
    grid_search = GridSearchCV(
        LinearSVC(), param_grid, cv=5, scoring="accuracy", verbose=2
    )
    grid_search.fit(X_train, y_train)

    # Best model
    best_svm = grid_search.best_estimator_
    print("Best Hyperparameters:", grid_search.best_params_)

    # Save the best model
    dump(best_svm, model_path)
    print(f"Model saved to {model_path}")

    return best_svm
