import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer


def load_data(path):
    """Load dataset from CSV file"""
    return pd.read_csv(path)


def preprocess_data(df):
    """Preprocess text and split into train/test sets"""
    # Drop rows where text or label is missing
    df = df.dropna(subset=["text", "label"])

    # Convert labels to lowercase for consistency
    df["label"] = df["label"].str.lower()

    # Keep only real/fake labels (optional safety step)
    df = df[df["label"].isin(["real", "fake"])]

    X = df["text"]
    y = df["label"]

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Vectorize text using TF-IDF
    vectorizer = TfidfVectorizer(stop_words="english", max_df=0.7)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    return X_train_tfidf, X_test_tfidf, y_train, y_test, vectorizer
