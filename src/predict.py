import os
import sys

import joblib
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler

from . import config
from .features import transform_with_tfidf
from .preprocess import clean_text


def predict_abstract_labels(
    abstract: str,
    model_name: str = "sbert_logistic_regression.pkl",
    method: str = "sbert",
) -> list:
    """
    Predict one or more scientific categories for a given abstract.

    Args:
        abstract (str): Abstract text
        model_name (str): Model filename
        method (str): 'tfidf' or 'sbert'

    Returns:
        list[str]: List of predicted labels
    """
    cleaned = clean_text(abstract)

    if method == "sbert":
        sbert_model = SentenceTransformer("all-MiniLM-L6-v2")
        X = sbert_model.encode([cleaned])
    else:
        vectorizer = joblib.load(config.TFIDF_PATH)
        X = transform_with_tfidf(vectorizer, [cleaned])
        if "svm" in model_name:
            svd = TruncatedSVD(n_components=300, random_state=42)
            scaler = StandardScaler()
            X = scaler.fit_transform(svd.fit_transform(X))

    model = joblib.load(os.path.join(config.MODEL_DIR, model_name))
    mlb = joblib.load(os.path.join(config.MODEL_DIR, "multi_label_binarizer.pkl"))

    y_pred = model.predict(X)
    labels = mlb.inverse_transform(y_pred)
    return list(labels[0])


if __name__ == "__main__":
    if len(sys.argv) > 2:
        method = sys.argv[1]
        abstract = " ".join(sys.argv[2:])
    else:
        method = "sbert"
        abstract = "We propose a model for ad-hoc cooperation between human players and AI agents in the imperfect-information game Hanabi."

    model_map = {
        "tfidf": "tfidf_logistic_regression.pkl",
        "sbert": "sbert_logistic_regression.pkl",
    }

    model_name = model_map.get(method, "tfidf_logistic_regression.pkl")
    predicted_labels = predict_abstract_labels(
        abstract, model_name=model_name, method=method
    )
    print(f"\n📄 Predicted Categories: {predicted_labels}")
