import os

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.svm import LinearSVC

from . import config
from .augment import augment_abstract
from .data_loader import load_arxiv_data
from .features import extract_features
from .preprocess import preprocess_dataframe


def train_models(
    df: pd.DataFrame, method: str = "tfidf", output_dir: str = config.MODEL_DIR
):
    # Encode multi-label targets
    mlb = MultiLabelBinarizer()
    y = mlb.fit_transform(df["label"])

    # Feature extraction
    X, _ = extract_features(df["cleaned_abstract"], method=method)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Define models
    models = {}
    if method == "tfidf":
        models["logistic_regression"] = OneVsRestClassifier(
            LogisticRegression(max_iter=1000, class_weight="balanced")
        )
        models["svm"] = OneVsRestClassifier(
            LinearSVC(max_iter=2000, class_weight="balanced")
        )
    elif method == "sbert":
        models["logistic_regression"] = OneVsRestClassifier(
            LogisticRegression(max_iter=1000)
        )
        models["mlp"] = OneVsRestClassifier(
            MLPClassifier(hidden_layer_sizes=(256,), max_iter=300)
        )
    else:
        raise ValueError("Unknown method. Use 'tfidf' or 'sbert'")

    # Train and evaluate
    for name, model in models.items():
        print(f"\n🔧 Training {name}...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        print(f"\n📊 Evaluation for {name}:")
        report = classification_report(
            y_test, y_pred, target_names=mlb.classes_, zero_division=0
        )
        print(report)

        # Save model
        os.makedirs(output_dir, exist_ok=True)
        model_path = os.path.join(output_dir, f"{method}_{name}.pkl")
        joblib.dump(model, model_path)
        print(f"✅ Saved model to {model_path}")

    # Save label binarizer
    label_path = os.path.join(output_dir, "multi_label_binarizer.pkl")
    joblib.dump(mlb, label_path)
    print(f"✅ Saved label binarizer to {label_path}")


def main(method="tfidf"):
    print("🌐 Fetching abstracts from arXiv API...")
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
    print("🧹 Preprocessing data...")
    df = preprocess_dataframe(df)

    # Convert 'category' column to list (multi-label support)
    df["label"] = df["category"].apply(lambda x: x.split() if isinstance(x, str) else x)

    print("📈 Augmenting underrepresented classes...")
    label_counts = pd.Series(
        [label for sublist in df["label"] for label in sublist]
    ).value_counts()
    underrepresented = label_counts[label_counts < 300].index.tolist()

    augmented_rows = []
    for _, row in df.iterrows():
        for label in row["label"]:
            if label in underrepresented:
                aug_texts = augment_abstract(row["cleaned_abstract"], n=2)
                for aug in aug_texts:
                    augmented_rows.append({"cleaned_abstract": aug, "label": [label]})

    df = pd.concat([df, pd.DataFrame(augmented_rows)], ignore_index=True)
    print(f"🆙 Added {len(augmented_rows)} augmented samples.")
    print(f"🧠 Final dataset: {df.shape[0]} samples.")

    train_models(df, method=method)
