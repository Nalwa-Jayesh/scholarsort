import os
from typing import List, Tuple, Union

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

from . import config


def extract_tfidf(
    texts: List[str], max_features=10000, ngram_range=(1, 2), save_path: str = None
) -> Tuple:
    """
    Extract TF-IDF features and optionally save the Vectorizer.

    Args:
        texts (List[str]): Cleaned text List
        max_features (int): Max vocab size.
        ngram_range (tuple): N-gram range.
        save_path (str): If given, saves vectorizer as .pkl
    """
    vectorizer = TfidfVectorizer(
        max_features=max_features, ngram_range=ngram_range, stop_words="english"
    )
    X = vectorizer.fit_transform(texts)

    if save_path:
        joblib.dump(vectorizer, save_path)

    return X, vectorizer


def transform_with_tfidf(vectorizer, texts: List[str]):
    """Transforms texts using a fitted TF-IDF vectorizer"""
    return vectorizer.transform(texts)


def load_vectorizer(path: str) -> TfidfVectorizer:
    """Loads a previously saved TF-IDF Vectorizer"""
    return joblib.load(path)


def extract_sbert(
    texts: List[str],
    model_name: str = "all-MiniLM-L6-v2",
    models_dir: str = config.MODEL_DIR,
    device: str = "cuda",
) -> Tuple:
    """
    Extract BERT embeddings, auto-download/save if needed.

    Args:
        texts (List[str]): Cleaned text List
        model_name (str): HF  Model name.
        models_dir (str): Directory to check/save model.
        device (str): "cuda" or "device"

    Returns:
        embeddings (ndarray), model
    """
    from sentence_transformers import SentenceTransformer

    local_model_path = config.sbert_model_path(model_name)
    # Load from local if exists, else download and save
    if os.path.exists(local_model_path):
        print(f"[INFO] Loading SBERT from local path: {local_model_path}")
        model = SentenceTransformer(local_model_path, device=device)
    else:
        print(f"[INFO] Downloading SBERT model: {model_name}")
        model = SentenceTransformer(model_name, device=device)
        print(f"[INFO] Saving SBERT to local path: {local_model_path}")
        model.save(local_model_path)

    embeddings = model.encode(texts, show_progress_bar=True, convert_to_tensor=False)
    return embeddings, model


def encode_with_sbert(model, texts: List[str]):
    """Encode new texts using an existing SBERT model"""
    return model.encode(texts, show_progress_bar=True, convert_to_tensor=False)


def extract_features(
    texts: List[str],
    method: str = "tfidf",
    tfidf_path: str = config.TFIDF_PATH,
    model_name: str = "all-MiniLM-L6-v2",
    models_dir: str = config.MODEL_DIR,
    device: str = "cuda",
    max_features: int = 10000,
    ngram_range: Tuple[int, int] = (1, 2),
) -> Tuple[Union[any, list], Union[TfidfVectorizer, any]]:
    """
    Wrapper for feature extraction via TF-IDF or SBERT

    Args:
        texts List([str]): Cleaned abstracts
        method (str): "tfidf" or "sbert"
        tfidf_path (str): Save/load path of TF-IDF vectorizer
        model_name (str): HF model name for SBERT
        models_dir (str): Local model storage
        device (str): "cpu" or "cuda"

    Returns:
        features, fitted vectorizer / model
    """
    if method == "tfidf":
        return extract_tfidf(texts, max_features, ngram_range, save_path=tfidf_path)
    elif method == "sbert":
        return extract_sbert(texts, model_name, models_dir, device)
    else:
        raise ValueError("Method must be 'tfidf' or 'sbert'")


def get_sbert_embeddings(texts: List[str], model_name: str = "all-MiniLM-L6-v2") -> np.ndarray:
    """Get SBERT embeddings for texts using the local model"""
    from sentence_transformers import SentenceTransformer
    
    local_model_path = config.sbert_model_path(model_name)
    if os.path.exists(local_model_path):
        model = SentenceTransformer(local_model_path, device="cpu")
    else:
        model = SentenceTransformer(model_name, device="cpu")
    
    return model.encode(texts, show_progress_bar=False, convert_to_tensor=False)
