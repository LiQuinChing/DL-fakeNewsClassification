from joblib import load

def predict_text(text, model_path="models/svm_model.pkl", vectorizer=None):
    """Predict if a news article is real or fake"""
    model = load(model_path)
    X = vectorizer.transform([text])
    prediction = model.predict(X)[0]
    return prediction
