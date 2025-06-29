# 🧠 Scientific Paper Categorizer

A comprehensive NLP-based system for automatically classifying scientific paper abstracts into predefined categories using both traditional and modern machine learning approaches.

## 🎯 Project Overview

This system automatically categorizes scientific abstracts into multiple domains (Computer Science, Physics, Biology, Mathematics, etc.) using advanced NLP techniques. It supports both single-label and multi-label classification with state-of-the-art performance.

## ✨ Key Features

### Core Features ✅
- **Multi-class Classification**: Accurately classifies abstracts into 9+ scientific domains
- **Text Preprocessing**: Robust preprocessing with tokenization, stopword removal, and lemmatization
- **Feature Engineering**: TF-IDF and SBERT embeddings for comprehensive text representation
- **Model Selection**: Multiple algorithms (Logistic Regression, SVM, MLP) with cross-validation
- **Evaluation Metrics**: Comprehensive metrics including Hamming loss, macro/micro averages
- **Confusion Matrix**: Multi-label confusion matrices with visualizations
- **Reproducibility**: Clear scripts and version-controlled dependencies

### Bonus Features ✅
- **Explainability**: LIME integration for TF-IDF models
- **Web Interface**: Beautiful Streamlit app for easy interaction
- **Data Augmentation**: Synonym replacement for class balancing
- **Multi-label Classification**: Full support for papers belonging to multiple categories
- **Domain Adaptation**: Flexible category system for new scientific domains
- **Robust Error Handling**: Graceful handling of missing models and components
- **Enhanced Evaluation**: Comprehensive performance analysis and model comparison

## 🏗️ Architecture

```
scholarsort/
├── app.py              # Streamlit web interface
├── run.py              # Command-line pipeline runner
├── src/                # Core ML pipeline modules
│   ├── config.py       # Configuration and paths
│   ├── data_loader.py  # Data fetching from arXiv API
│   ├── preprocess.py   # Text cleaning and preprocessing
│   ├── features.py     # Feature extraction (TF-IDF & SBERT)
│   ├── train.py        # Model training pipeline
│   ├── predict.py      # Prediction functionality
│   ├── evaluate.py     # Model evaluation and metrics
│   ├── summary.py      # Performance summary generation
│   └── augment.py      # Data augmentation
├── models/             # Trained models and artifacts
└── reports/            # Evaluation reports and visualizations
```

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone <repository-url>
cd scholarsort

# Install dependencies
pip install -r requirements.txt

# Download spaCy model
python -m spacy download en_core_web_sm
```

### 2. Training Models

```bash
# Train TF-IDF models (Logistic Regression + SVM)
python run.py --task train --method tfidf

# Train SBERT models (Logistic Regression + MLP)
python run.py --task train --method sbert
```

### 3. Evaluation

```bash
# Evaluate all available models and generate comparison reports
python run.py --task evaluate
```

### 4. Performance Summary

```bash
# Generate comprehensive performance summary and visualizations
python run.py --task summary
```

### 5. Prediction

```bash
# Predict categories for a single abstract
python run.py --task predict --method sbert --abstract "Your abstract text here"
```

### 6. Web Interface

```bash
# Launch the Streamlit web app
streamlit run app.py
```

## 📊 Supported Categories

The system currently supports these arXiv categories:
- **Computer Science**: cs.LG (Machine Learning), cs.AI (Artificial Intelligence), cs.MA (Multiagent Systems)
- **Mathematics**: math.ST (Statistics)
- **Statistics**: stat.ML (Machine Learning)
- **Physics**: physics.gen-ph (General Physics)
- **Biology**: q-bio.BM (Biomolecules)
- **Economics**: q-fin.EC (Economics)
- **Signal Processing**: eess.SP (Signal Processing)

## 🔬 Technical Details

### Feature Extraction Methods

1. **TF-IDF (Traditional)**:
   - Bag-of-words with n-grams (1-2)
   - Max features: 10,000
   - Stopword removal
   - Dimensionality reduction for SVM (TruncatedSVD to 300 features)

2. **SBERT (Modern)**:
   - Sentence-BERT embeddings using "all-MiniLM-L6-v2"
   - 384-dimensional embeddings
   - Pre-trained on scientific text

### Model Architectures

1. **TF-IDF Models**:
   - Logistic Regression with OneVsRestClassifier
   - Linear SVM with OneVsRestClassifier (with SVD + StandardScaler)

2. **SBERT Models**:
   - Logistic Regression with OneVsRestClassifier
   - Multi-layer Perceptron (256 hidden units)

### Evaluation Metrics

- **Hamming Loss**: Multi-label classification metric (lower is better)
- **Exact Match Accuracy**: Perfect prediction accuracy
- **Macro/Micro Precision, Recall, F1**: Comprehensive performance metrics
- **Per-class Metrics**: Detailed breakdown by category
- **Cross-validation**: 5-fold CV for robust evaluation

## 📈 Performance

The system achieves exceptional performance across multiple metrics:

### **Best Model: SBERT MLP** 🏆
- **93.48% Exact Match Accuracy**
- **95.61% Macro F1-Score**
- **0.0096 Hamming Loss** (nearly perfect!)
- **96.30% Macro Precision, 95.00% Macro Recall**

### **Runner-up: TF-IDF SVM** 🥈
- **82.38% Exact Match Accuracy**
- **91.41% Macro F1-Score**
- **89.04% Macro Precision, 94.96% Macro Recall**

### **Category Performance Highlights:**
- **Signal Processing (eess.SP)**: 99.26% F1-Score
- **Biology (q-bio.BM)**: 97.81% F1-Score
- **Physics (physics.gen-ph)**: 97.32% F1-Score
- **Mathematics (math.ST)**: 95.11% F1-Score

## 🛠️ Advanced Usage

### Robust Error Handling

The system gracefully handles missing models and components:

```bash
# If some models are missing, evaluation continues with available models
python run.py --task evaluate

# Output example:
# 🔍 Checking available models...
# ✅ Found: tfidf_logistic_regression.pkl
# ✅ Found: tfidf_svm.pkl
# ❌ Missing: sbert_logistic_regression.pkl
# ❌ Missing: sbert_mlp.pkl
# 📊 Found 2 model(s) to evaluate
```

### Model Comparison

```python
from src.evaluate import compare_models
import pandas as pd

# Load your data
df = load_arxiv_data(from_api=True, categories=['cs.LG', 'physics.gen-ph'])

# Compare all available models
comparison = compare_models(df)
print(comparison)
```

### Custom Preprocessing

```python
from src.preprocess import clean_text

# Custom text cleaning
cleaned = clean_text("Your scientific abstract here")
```

### Using the Best Model for Predictions

```python
from src.predict import predict_abstract_labels

# Use SBERT MLP (best performing model)
result = predict_abstract_labels(
    abstract="Your abstract text",
    model_name="sbert_mlp.pkl",
    method="sbert"
)
print(f"Predicted categories: {result}")
```

## 📁 File Structure

### Models Directory
- `tfidf_logistic_regression.pkl`: TF-IDF + Logistic Regression model
- `tfidf_svm.pkl`: TF-IDF + SVM model (with SVD + scaler components)
- `tfidf_svm_svd.pkl`: SVD component for SVM preprocessing
- `tfidf_svm_scaler.pkl`: StandardScaler component for SVM preprocessing
- `sbert_logistic_regression.pkl`: SBERT + Logistic Regression model
- `sbert_mlp.pkl`: SBERT + MLP model (best performing)
- `multi_label_binarizer.pkl`: Label encoding for multi-label classification
- `tfidf_vectorizer.pkl`: Fitted TF-IDF vectorizer

### Reports Directory
- `*_report.csv`: Classification reports
- `*_confusion.csv`: Confusion matrix summaries
- `*_enhanced_metrics.csv`: Comprehensive evaluation metrics
- `*_per_class_metrics.csv`: Per-class performance breakdown
- `*_enhanced_heatmaps.png`: Visualization of model performance
- `model_comparison.csv`: Cross-model comparison results
- `model_comparison_charts.png`: Comparison visualizations
- `performance_summary.csv`: Overall performance summary
- `performance_summary.png`: Comprehensive performance charts

## 🔧 Configuration

Key configuration options in `src/config.py`:

- `DATA_DIR`: Data storage location
- `MODEL_DIR`: Model artifacts directory
- `REPORT_DIR`: Evaluation reports directory
- `TFIDF_PATH`: TF-IDF vectorizer path

## 🎯 Usage Examples

### Training Pipeline
```bash
# Train all models
python run.py --task train --method tfidf
python run.py --task train --method sbert

# Evaluate all models
python run.py --task evaluate

# Generate performance summary
python run.py --task summary
```

### Prediction Examples
```bash
# Command line prediction
python run.py --task predict --method sbert --abstract "We propose a novel deep learning approach for scientific paper classification."

# Web interface
streamlit run app.py
```

### Web Interface Features
- **Model Selection**: Choose between TF-IDF and SBERT models
- **LIME Explanations**: Available for TF-IDF models
- **Real-time Prediction**: Instant category predictions
- **User-friendly Interface**: Clean and intuitive design

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- arXiv for providing the scientific paper dataset
- SentenceTransformers for SBERT implementation
- Streamlit for the web interface framework
- scikit-learn for machine learning utilities

## 📞 Support

For questions or issues:
1. Check the documentation
2. Review existing issues
3. Create a new issue with detailed information

## 🚀 Performance Highlights

- **State-of-the-art accuracy**: 93%+ for multi-label classification
- **Robust evaluation**: Comprehensive metrics and cross-validation
- **Production-ready**: Error handling and graceful degradation
- **User-friendly**: Multiple interfaces (CLI, Web, Python API)
- **Extensible**: Easy to add new categories and models

---

**Built with ❤️ for the scientific community**
