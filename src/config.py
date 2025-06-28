import os

# Base project root: go one level up from src/
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Data paths
DATA_DIR = os.path.join(BASE_DIR, "data")
DATA_RAW = os.path.join(DATA_DIR, "raw", "arXiv_scientific dataset.csv")

# Models and results
MODEL_DIR = os.path.join(BASE_DIR, "models")
REPORT_DIR = os.path.join(BASE_DIR, "reports")
TFIDF_PATH = os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl")
LABEL_ENCODER_PATH = os.path.join(MODEL_DIR, "label_encoder.pkl")


# SBERT model local path
def sbert_model_path(name="all-MiniLM-L6-v2"):
    return os.path.join(MODEL_DIR, f"sbert_{name.replace('/', '_')}")
