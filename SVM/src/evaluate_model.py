import os
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns


def evaluate_model(model, X_test, y_test, model_name="SVM", results_dir="results"):
    """Evaluate model performance, save metrics and confusion matrix"""

    y_pred = model.predict(X_test)

    os.makedirs(results_dir, exist_ok=True)

    # Calculate metrics
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)

    # Save metrics to CSV
    metrics_df = pd.DataFrame(report).transpose()
    metrics_df["accuracy"] = acc
    metrics_path = os.path.join(results_dir, f"{model_name}_metrics.csv")
    metrics_df.to_csv(metrics_path)
    print(f"Metrics saved to {metrics_path}")

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Fake", "Real"],
        yticklabels=["Fake", "Real"],
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"Confusion Matrix - {model_name}")

    # Save plot
    plt.savefig(os.path.join(results_dir, f"{model_name}_confusion_matrix.png"))
    plt.close()

    return acc
