import os
from src.data_preprocessing import load_data, preprocess_data
from src.train_model import train_svm
from src.evaluate_model import evaluate_model
from src.predict import predict_text

DATA_PATH = "data/news.csv"


def main():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Dataset not found at {DATA_PATH}")

    print("Loading data...")
    df = load_data(DATA_PATH)

    print("Preprocessing data...")
    X_train, X_test, y_train, y_test, vectorizer = preprocess_data(df)

    print("Training SVM model...")
    # Unpack the returned tuple
    model, scaler = train_svm(X_train, y_train)

    print("Evaluating model...")
    # Transform both train and test sets using the same scaler
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Get all metrics dictionary from evaluate_model
    metrics = evaluate_model(model, X_train_scaled, y_train, X_test_scaled, y_test)

    # Print results
    print("\n-----------------------------")
    print(f"Training Accuracy: {metrics['train_accuracy']:.4f}")
    print(f"Testing Accuracy:  {metrics['test_accuracy']:.4f}")
    print(f"Overall Accuracy:  {metrics['overall_accuracy']:.4f}")
    print("-----------------------------")

    # Interactive Prediction
    while True:
        print("\nEnter a news article to test (or type 'exit' to quit):")
        user_input = input("> ")

        if user_input.lower() == "exit":
            print("Exiting program...")
            break

        result = predict_text(user_input, vectorizer=vectorizer)
        print(f"Prediction: This news is *{result.upper()}*")


if __name__ == "__main__":
    main()
