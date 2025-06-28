# 🧠 ScholarSort

> A multi-label scientific paper classifier powered by TF-IDF and SBERT embeddings. Built to categorize arXiv abstracts into relevant research domains.

[![Streamlit App](https://img.shields.io/badge/Live%20App-Streamlit-ff4b4b?logo=streamlit&logoColor=white)](https://scholarsort-jsn.streamlit.app/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/)

---

## 🚀 Features

- ✅ TF-IDF and SBERT-based training
- ✅ SBERT-powered semantic predictions
- ✅ Real-time prediction with Streamlit UI
- ✅ LIME explainability for TF-IDF models
- ✅ Automatic data fetching from arXiv API
- ✅ Augmentation for class balancing
- ✅ Clean modular structure with `run.py`

---

## 🧱 Project Structure

```
scifi-classifier/
├── app.py               # Streamlit web UI
├── run.py               # CLI runner (train/predict/evaluate)
├── requirements.txt     # Dependencies
├── README.md
│
├── src/
│   ├── config.py
│   ├── train.py
│   ├── predict.py
│   ├── evaluate.py
│   ├── preprocess.py
│   ├── features.py
│   ├── data_loader.py
│   └── augment.py
│
├── models/              # Trained model files
└── reports/             # Evaluation outputs
```

---

## 📦 Setup

1. **Install requirements**

```bash
pip install -r requirements.txt
```

2. **(Optional)** Download NLTK stopwords:

```python
import nltk
nltk.download("stopwords")
```

---

## 🧠 Supported Categories

| Label                   | arXiv Code       |
| ----------------------- | ---------------- |
| Computer Science        | `cs.LG`          |
| Mathematics             | `math.ST`        |
| Statistics              | `stat.ML`        |
| Physics                 | `physics.gen-ph` |
| Quant Finance           | `q-fin.EC`       |
| Bioinformatics          | `q-bio.BM`       |
| Signal Processing       | `eess.SP`        |
| Multi-Agent Systems     | `cs.MA`          |
| Artificial Intelligence | `cs.AI`          |

---

## 🧪 Run from CLI

### 🔧 Train:

```bash
python run.py --task train --method tfidf
python run.py --task train --method sbert
```

### 📊 Evaluate:

```bash
python run.py --task evaluate
```

### 📄 Predict:

```bash
python run.py --task predict --method sbert --abstract "We propose a new approach to human-AI coordination..."
```

---

## 🌐 Run the Web App

```bash
streamlit run app.py
```

### Web UI Features:

- Paste abstract
- Choose model
- Toggle LIME explanations (TF-IDF only)

---

## 🧑‍💻 Author

Jayesh Nalwa
Built as a full-stack AI pipeline challenge with SBERT, LIME, and arXiv integration.
