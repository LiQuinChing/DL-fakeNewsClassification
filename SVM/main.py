import os
from src.data_preprocessing import load_data, preprocess_data
from src.train_model import train_svm
from src.evaluate_model import evaluate_model
from src.predict import predict_text

DATA_PATH = ""  # Need to add the correct one


def main():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Dataset not found at {DATA_PATH}")

    print("Loading data...")
    df = load_data(DATA_PATH)

    print("Preprocessing data...")
    X_train, X_test, y_train, y_test, vectorizer = preprocess_data(df)

    print("Training SVM model...")
    model = train_svm(X_train, y_train)

    print("Evaluating model...")
    evaluate_model(model, X_test, y_test)

    # --- Interactive Prediction ---
    while True:
        print("\n Enter a news article to test (or type 'exit' to quit):")
        user_input = input("> ")

        if user_input.lower() == "exit":
            print("Exiting program...")
            break

        result = predict_text(user_input, vectorizer=vectorizer)
        print(f"Prediction: This news is *{result.upper()}*")


if __name__ == "__main__":
    main()
