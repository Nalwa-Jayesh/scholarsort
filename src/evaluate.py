import os

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import classification_report, multilabel_confusion_matrix
from sklearn.preprocessing import StandardScaler

from . import config
from .data_loader import load_arxiv_data
from .features import transform_with_tfidf
from .preprocess import preprocess_dataframe


def evaluate_model(df, model_name="sbert_logistic_regression.pkl", method="sbert"):
    # Load components
    model = joblib.load(os.path.join(config.MODEL_DIR, model_name))
    mlb = joblib.load(os.path.join(config.MODEL_DIR, "multi_label_binarizer.pkl"))

    if method == "sbert":
        sbert_model = SentenceTransformer("all-MiniLM-L6-v2")
        X = sbert_model.encode(df["cleaned_abstract"].tolist())
    else:
        vectorizer = joblib.load(config.TFIDF_PATH)
        X = transform_with_tfidf(vectorizer, df["cleaned_abstract"])

        if "svm" in model_name:
            svd = TruncatedSVD(n_components=300, random_state=42)
            scaler = StandardScaler()
            X = scaler.fit_transform(svd.fit_transform(X))

    y_true = mlb.transform(
        df["label"].apply(lambda x: [l.strip() for l in x.split(",")])
    )

    # Predict
    y_pred = model.predict(X)

    # Classification report
    report = classification_report(
        y_true, y_pred, target_names=mlb.classes_, output_dict=True, zero_division=0
    )
    report_df = pd.DataFrame(report).transpose()

    os.makedirs(config.REPORT_DIR, exist_ok=True)
    report_path = os.path.join(config.REPORT_DIR, f"{model_name}_report.csv")
    report_df.to_csv(report_path)
    print(f"✅ Saved classification report to {report_path}")

    # Multilabel confusion matrix (raw heatmap summary)
    cm = multilabel_confusion_matrix(y_true, y_pred)
    cm_summary = pd.DataFrame(
        [[c[1, 1], c[1, 0], c[0, 1], c[0, 0]] for c in cm],
        columns=["TP", "FN", "FP", "TN"],
        index=mlb.classes_,
    )
    cm_path = os.path.join(config.REPORT_DIR, f"{model_name}_confusion.csv")
    cm_summary.to_csv(cm_path)
    print(f"✅ Saved multilabel confusion matrix summary to {cm_path}")

    # Optional heatmap visualization of TP rates
    plt.figure(figsize=(10, 6))
    sns.heatmap(cm_summary[["TP"]], annot=True, fmt="d", cmap="YlGnBu")
    plt.title(f"True Positives per Label - {model_name}")
    plt.tight_layout()
    plt.savefig(os.path.join(config.REPORT_DIR, f"{model_name}_TP_heatmap.png"))
    plt.close()


def main():
    print("🔍 Loading and preprocessing evaluation data...")
    df = load_arxiv_data(
        from_api=True,
        categories=[
            "cs.LG",
            "math.ST",
            "physics.gen-ph",
            "stat.ML",
            "q-bio.BM",
            "eess.SP",
            "q-fin.EC",
            "cs.MA",
            "cs.AI",
        ],
        per_category=300,
    )
    df = preprocess_dataframe(df)
    df["label"] = df["category"].apply(lambda x: x.split() if isinstance(x, str) else x)
    df["label"] = df["label"].apply(lambda x: ",".join(x))

    print("📊 Evaluating sbert_logistic_regression...")
    evaluate_model(df, model_name="sbert_logistic_regression.pkl", method="sbert")

    print("📊 Evaluating tfidf_logistic_regression...")
    evaluate_model(df, model_name="tfidf_logistic_regression.pkl", method="tfidf")


if __name__ == "__main__":
    main()
