import os

import joblib
import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler

from src import config
from src.features import transform_with_tfidf, get_sbert_embeddings
from src.preprocess import clean_text
from src.explain import create_explainer, display_explanation_streamlit, explain_with_lime, display_lime_explanation_streamlit

# === Setup ===
st.set_page_config(page_title="Scientific Paper Categorizer", layout="centered")
st.title("🧠 Scientific Paper Categorizer")
st.markdown("Paste a scientific abstract below to predict its relevant domains.")

# === Input ===
abstract = st.text_area("📝 Abstract", height=200)

model_choice = st.selectbox(
    "🤖 Choose a model",
    options=["tfidf_logistic_regression", "tfidf_svm", "sbert_logistic_regression", "sbert_mlp"],
)

# SHAP explanation option
show_explanation = st.checkbox("🔍 Show SHAP explanation", help="Generate feature importance explanations using SHAP")

# === Load Models and Vectorizer ===
@st.cache_resource
def load_components(model_name):
    model = joblib.load(os.path.join(config.MODEL_DIR, model_name + ".pkl"))
    binarizer = joblib.load(os.path.join(config.MODEL_DIR, "multi_label_binarizer.pkl"))
    vectorizer = joblib.load(config.TFIDF_PATH) if "tfidf" in model_name else None
    
    # Load preprocessing components for SVM
    svd = None
    scaler = None
    if "svm" in model_name:
        svd_path = os.path.join(config.MODEL_DIR, model_name + "_svd.pkl")
        scaler_path = os.path.join(config.MODEL_DIR, model_name + "_scaler.pkl")
        if os.path.exists(svd_path) and os.path.exists(scaler_path):
            svd = joblib.load(svd_path)
            scaler = joblib.load(scaler_path)
    
    return model, binarizer, vectorizer, svd, scaler


model, label_binarizer, vectorizer, svd, scaler = load_components(model_choice)


# === Prediction Logic ===
def predict_labels(abstract_text, model_name):
    cleaned = clean_text(abstract_text)

    if "sbert" in model_name:
        sbert_model = SentenceTransformer("all-MiniLM-L6-v2")
        X = sbert_model.encode([cleaned])
    else:
        X = transform_with_tfidf(vectorizer, [cleaned])
        if "svm" in model_name and svd is not None and scaler is not None:
            # Use the fitted preprocessing components
            X = scaler.transform(svd.transform(X))
        elif "svm" in model_name:
            # Fallback if components not found
            st.warning("⚠️ SVM preprocessing components not found, using default transformation")
            svd_fallback = TruncatedSVD(n_components=300, random_state=42)
            scaler_fallback = StandardScaler()
            X = scaler_fallback.fit_transform(svd_fallback.fit_transform(X))

    y_pred = model.predict(X)
    labels = label_binarizer.inverse_transform(y_pred)
    return list(labels[0])


# === SHAP Explanation ===
def generate_explanation(abstract_text, model_name, svd=None, scaler=None):
    """Generate explanation using SHAP (TF-IDF) or LIME (SBERT)"""
    try:
        method_type = "tfidf" if "tfidf" in model_name else "sbert"
        if method_type == "sbert":
            explanation = explain_with_lime(
                abstract_text,
                model,
                label_binarizer,
                get_sbert_embeddings_func=get_sbert_embeddings,
                num_features=10
            )
            display_lime_explanation_streamlit(explanation)
            return explanation
        else:
            explainer = create_explainer(
                model=model,
                vectorizer=vectorizer,
                label_binarizer=label_binarizer,
                method=method_type,
                svd=svd,
                scaler=scaler
            )
            explanation = explainer.explain_with_shap(abstract_text, num_features=10)
            display_explanation_streamlit(explanation)
            return explanation
    except Exception as e:
        st.error(f"❌ Explanation error: {str(e)}")
        st.info("💡 SHAP and LIME require installation: `pip install shap lime`.")
        return None


# === Predict Button ===
if st.button("🎯 Predict"):
    if not abstract.strip():
        st.warning("Please enter an abstract.")
    else:
        try:
            labels = predict_labels(abstract, model_choice)
            if labels:
                st.success("📄 **Predicted Categories:**")
                st.write("  ")
                for label in labels:
                    st.markdown(f"`{label}`")
            else:
                st.warning("No category predicted.")

            # Generate SHAP explanation if requested
            if show_explanation:
                st.write("---")
                generate_explanation(abstract, model_choice, svd=svd, scaler=scaler)
                
        except Exception as e:
            st.error(f"❌ Prediction error: {str(e)}")
            st.info("💡 Try selecting a different model or check if all models are trained.")

# === Model Information ===
with st.sidebar:
    st.header("📊 Model Information")
    
    if "sbert_mlp" in model_choice:
        st.success("🏆 **Best Model: SBERT MLP**")
        st.write("• 93.48% Exact Match Accuracy")
        st.write("• 95.61% Macro F1-Score")
        st.write("• Best for semantic understanding")
    elif "tfidf_svm" in model_choice:
        st.info("🥈 **Runner-up: TF-IDF SVM**")
        st.write("• 82.38% Exact Match Accuracy")
        st.write("• 91.41% Macro F1-Score")
        st.write("• Fast and interpretable")
    elif "tfidf_logistic_regression" in model_choice:
        st.info("🥉 **TF-IDF Logistic Regression**")
        st.write("• 79.15% Exact Match Accuracy")
        st.write("• 89.92% Macro F1-Score")
        st.write("• Good baseline performance")
    else:
        st.info("📈 **SBERT Logistic Regression**")
        st.write("• Good semantic understanding")
        st.write("• Moderate performance")
    
    st.write("---")
    st.write("**SHAP Explanation:**")
    st.write("• State-of-the-art explainability")
    st.write("• Works with all models")
    st.write("• Mathematically sound")
    
    st.write("---")
    st.write("**Supported Categories:**")
    categories = ["cs.AI", "cs.LG", "cs.MA", "eess.SP", "math.ST", 
                 "physics.gen-ph", "q-bio.BM", "q-fin.EC", "stat.ML"]
    for cat in categories:
        st.write(f"• {cat}")
