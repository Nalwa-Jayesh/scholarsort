import os

import joblib
import streamlit as st
from lime.lime_text import LimeTextExplainer
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler

from src import config
from src.features import transform_with_tfidf
from src.preprocess import clean_text

# === Setup ===
st.set_page_config(page_title="Scientific Paper Categorizer", layout="centered")
st.title("🧠 Scientific Paper Categorizer")
st.markdown("Paste a scientific abstract below to predict its relevant domains.")

# === Input ===
abstract = st.text_area("📝 Abstract", height=200)

model_choice = st.selectbox(
    "🤖 Choose a model",
    options=["tfidf_logistic_regression", "tfidf_svm", "sbert_logistic_regression"],
)

# Disable LIME for SBERT models
show_explanation = False
if "sbert" not in model_choice:
    show_explanation = st.checkbox("🔍 Show explanation with LIME")


# === Load Models and Vectorizer ===
@st.cache_resource
def load_components(model_name):
    model = joblib.load(os.path.join(config.MODEL_DIR, model_name + ".pkl"))
    binarizer = joblib.load(os.path.join(config.MODEL_DIR, "multi_label_binarizer.pkl"))
    vectorizer = joblib.load(config.TFIDF_PATH) if "tfidf" in model_name else None
    return model, binarizer, vectorizer


model, label_binarizer, vectorizer = load_components(model_choice)


# === Prediction Logic ===
def predict_labels(abstract_text, model_name):
    cleaned = clean_text(abstract_text)

    if "sbert" in model_name:
        sbert_model = SentenceTransformer("all-MiniLM-L6-v2")
        X = sbert_model.encode([cleaned])
    else:
        X = transform_with_tfidf(vectorizer, [cleaned])
        if "svm" in model_name:
            svd = TruncatedSVD(n_components=300, random_state=42)
            scaler = StandardScaler()
            X = scaler.fit_transform(svd.fit_transform(X))

    y_pred = model.predict(X)
    labels = label_binarizer.inverse_transform(y_pred)
    return list(labels[0])


# === LIME Explanation (TF-IDF only) ===
def lime_explanation(abstract_text, model_name):
    class_names = list(label_binarizer.classes_)

    def prediction_fn(texts):
        cleaned = [clean_text(t) for t in texts]
        X = transform_with_tfidf(vectorizer, cleaned)
        if "svm" in model_name:
            svd = TruncatedSVD(n_components=300, random_state=42)
            scaler = StandardScaler()
            X = scaler.fit_transform(svd.fit_transform(X))
        return (
            model.decision_function(X)
            if hasattr(model, "decision_function")
            else model.predict_proba(X)
        )

    explainer = LimeTextExplainer(class_names=class_names)
    exp = explainer.explain_instance(abstract_text, prediction_fn, num_features=10)
    st.pyplot(exp.as_pyplot_figure())


# === Predict Button ===
if st.button("🎯 Predict"):
    if not abstract.strip():
        st.warning("Please enter an abstract.")
    else:
        labels = predict_labels(abstract, model_choice)
        if labels:
            st.success("📄 **Predicted Categories:**")
            st.write("  ")
            for label in labels:
                st.markdown(f"`{label}`")
        else:
            st.warning("No category predicted.")

        if show_explanation:
            st.subheader("🔍 LIME Explanation")
            lime_explanation(abstract, model_choice)
