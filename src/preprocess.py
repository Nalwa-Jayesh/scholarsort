import re

import nltk
import spacy
from nltk.corpus import stopwords
from spacy.cli import download
from spacy.util import is_package

# Download stopwords if not ready
nltk.download("stopwords")

# Load English stopwords
STOPWORDS = set(stopwords.words("english"))

# Load spaCy English model for lemmatization
MODEL_NAME = "en_core_web_sm"
try:
    if not is_package(MODEL_NAME):
        download(MODEL_NAME)
    nlp = spacy.load(MODEL_NAME, disable=["ner", "parser"])
except Exception as e:
    raise RuntimeError(f"Failed to load or download {MODEL_NAME}: {e}")


def clean_text(text: str) -> str:
    """
    Clean a single abstract: lowercase, remove special chars, stopwords, and lemmatize.

    Args:
        text (str): Raw abstract

    Returns:
        str: Cleaned abstract
    """
    # lowercase
    text = text.lower()

    # Remove digits and punctuation
    text = re.sub(r"[^a-z\s]", " ", text)

    # Tokenize and lemmatize using spaCy
    doc = nlp(text)
    tokens = [
        token.lemma_
        for token in doc
        if token.lemma_ not in STOPWORDS and len(token.lemma_) > 2
    ]

    return " ".join(tokens)


def preprocess_dataframe(df, text_col="abstract"):
    """
    Apply Cleaning to a pandas DataFrame column.

    Args:
        df (pd.DataFrame): Input DataFrame with 'abstract' column
        text_col (str): Name of the column to clean

    Returns:
        pd.DataFrame: DataFrame with new column 'cleaned_abstract'
    """
    df["cleaned_abstract"] = df[text_col].apply(clean_text)
    return df
