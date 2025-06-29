import os

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import (
    classification_report, 
    multilabel_confusion_matrix,
    hamming_loss,
    accuracy_score,
    precision_recall_fscore_support
)
from sklearn.preprocessing import StandardScaler

from . import config
from .data_loader import load_arxiv_data
from .features import transform_with_tfidf
from .preprocess import preprocess_dataframe


def evaluate_model(df, model_name="sbert_logistic_regression.pkl", method="sbert"):
    # Check if model file exists
    model_path = os.path.join(config.MODEL_DIR, model_name)
    if not os.path.exists(model_path):
        print(f"⚠️  Model {model_name} not found, skipping evaluation.")
        return None
    
    # Load components
    model = joblib.load(model_path)
    mlb = joblib.load(os.path.join(config.MODEL_DIR, "multi_label_binarizer.pkl"))

    if method == "sbert":
        sbert_model = SentenceTransformer("all-MiniLM-L6-v2")
        X = sbert_model.encode(df["cleaned_abstract"].tolist())
    else:
        vectorizer = joblib.load(config.TFIDF_PATH)
        X = transform_with_tfidf(vectorizer, df["cleaned_abstract"])

        if "svm" in model_name:
            # Load preprocessing components for SVM
            svd_path = os.path.join(config.MODEL_DIR, model_name.replace(".pkl", "_svd.pkl"))
            scaler_path = os.path.join(config.MODEL_DIR, model_name.replace(".pkl", "_scaler.pkl"))
            
            if os.path.exists(svd_path) and os.path.exists(scaler_path):
                svd = joblib.load(svd_path)
                scaler = joblib.load(scaler_path)
                X = scaler.transform(svd.transform(X))
            else:
                print(f"⚠️  Preprocessing components not found for {model_name}, using default transformation")
                svd = TruncatedSVD(n_components=300, random_state=42)
                scaler = StandardScaler()
                X = scaler.fit_transform(svd.fit_transform(X))

    y_true = mlb.transform(
        df["label"].apply(lambda x: [l.strip() for l in x.split(",")])
    )

    # Predict
    y_pred = model.predict(X)

    # Enhanced evaluation metrics
    print(f"\n📊 Enhanced Evaluation for {model_name}")
    print("=" * 50)
    
    # Hamming loss (lower is better for multi-label)
    hamming = hamming_loss(y_true, y_pred)
    print(f"Hamming Loss: {hamming:.4f}")
    
    # Exact match accuracy
    exact_accuracy = accuracy_score(y_true, y_pred)
    print(f"Exact Match Accuracy: {exact_accuracy:.4f}")
    
    # Per-class metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
    )
    
    # Create detailed per-class report
    class_metrics = pd.DataFrame({
        'Class': mlb.classes_,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1,
        'Support': support
    })
    
    print(f"\n📈 Per-Class Metrics:")
    print(class_metrics.round(4))
    
    # Macro and Micro averages
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='macro', zero_division=0
    )
    micro_precision, micro_recall, micro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='micro', zero_division=0
    )
    
    print(f"\n📊 Overall Metrics:")
    print(f"Macro Precision: {macro_precision:.4f}")
    print(f"Macro Recall: {macro_recall:.4f}")
    print(f"Macro F1-Score: {macro_f1:.4f}")
    print(f"Micro Precision: {micro_precision:.4f}")
    print(f"Micro Recall: {micro_recall:.4f}")
    print(f"Micro F1-Score: {micro_f1:.4f}")

    # Classification report
    report = classification_report(
        y_true, y_pred, target_names=mlb.classes_, output_dict=True, zero_division=0
    )
    report_df = pd.DataFrame(report).transpose()

    os.makedirs(config.REPORT_DIR, exist_ok=True)
    report_path = os.path.join(config.REPORT_DIR, f"{model_name}_report.csv")
    report_df.to_csv(report_path)
    print(f"✅ Saved classification report to {report_path}")
    
    # Save enhanced metrics
    enhanced_metrics = {
        'hamming_loss': hamming,
        'exact_accuracy': exact_accuracy,
        'macro_precision': macro_precision,
        'macro_recall': macro_recall,
        'macro_f1': macro_f1,
        'micro_precision': micro_precision,
        'micro_recall': micro_recall,
        'micro_f1': micro_f1
    }
    
    metrics_df = pd.DataFrame([enhanced_metrics])
    metrics_path = os.path.join(config.REPORT_DIR, f"{model_name}_enhanced_metrics.csv")
    metrics_df.to_csv(metrics_path, index=False)
    print(f"✅ Saved enhanced metrics to {metrics_path}")
    
    # Save per-class metrics
    class_metrics_path = os.path.join(config.REPORT_DIR, f"{model_name}_per_class_metrics.csv")
    class_metrics.to_csv(class_metrics_path, index=False)
    print(f"✅ Saved per-class metrics to {class_metrics_path}")

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

    # Enhanced visualization
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # True Positives heatmap
    sns.heatmap(cm_summary[["TP"]], annot=True, fmt="d", cmap="YlGnBu", ax=axes[0,0])
    axes[0,0].set_title(f"True Positives per Label - {model_name}")
    
    # F1-Score heatmap
    f1_heatmap = class_metrics.set_index('Class')['F1-Score'].to_frame()
    sns.heatmap(f1_heatmap, annot=True, fmt=".3f", cmap="RdYlGn", ax=axes[0,1])
    axes[0,1].set_title(f"F1-Score per Label - {model_name}")
    
    # Precision heatmap
    precision_heatmap = class_metrics.set_index('Class')['Precision'].to_frame()
    sns.heatmap(precision_heatmap, annot=True, fmt=".3f", cmap="RdYlGn", ax=axes[1,0])
    axes[1,0].set_title(f"Precision per Label - {model_name}")
    
    # Recall heatmap
    recall_heatmap = class_metrics.set_index('Class')['Recall'].to_frame()
    sns.heatmap(recall_heatmap, annot=True, fmt=".3f", cmap="RdYlGn", ax=axes[1,1])
    axes[1,1].set_title(f"Recall per Label - {model_name}")
    
    plt.tight_layout()
    plt.savefig(os.path.join(config.REPORT_DIR, f"{model_name}_enhanced_heatmaps.png"), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved enhanced heatmaps to {config.REPORT_DIR}/{model_name}_enhanced_heatmaps.png")
    
    return enhanced_metrics


def compare_models(df, model_names=None):
    """
    Compare multiple models and generate comprehensive comparison report.
    """
    if model_names is None:
        model_names = [
            "tfidf_logistic_regression.pkl",
            "tfidf_svm.pkl", 
            "sbert_logistic_regression.pkl",
            "sbert_mlp.pkl"
        ]
    
    print("🔍 Comparing Multiple Models...")
    print("=" * 60)
    
    comparison_results = []
    
    for model_name in model_names:
        if not os.path.exists(os.path.join(config.MODEL_DIR, model_name)):
            print(f"⚠️  Model {model_name} not found, skipping...")
            continue
            
        method = "sbert" if "sbert" in model_name else "tfidf"
        print(f"\n📊 Evaluating {model_name}...")
        
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
                # Load preprocessing components for SVM
                svd_path = os.path.join(config.MODEL_DIR, model_name.replace(".pkl", "_svd.pkl"))
                scaler_path = os.path.join(config.MODEL_DIR, model_name.replace(".pkl", "_scaler.pkl"))
                
                if os.path.exists(svd_path) and os.path.exists(scaler_path):
                    svd = joblib.load(svd_path)
                    scaler = joblib.load(scaler_path)
                    X = scaler.transform(svd.transform(X))
                else:
                    print(f"⚠️  Preprocessing components not found for {model_name}, using default transformation")
                    svd = TruncatedSVD(n_components=300, random_state=42)
                    scaler = StandardScaler()
                    X = scaler.fit_transform(svd.fit_transform(X))

        y_true = mlb.transform(
            df["label"].apply(lambda x: [l.strip() for l in x.split(",")])
        )
        y_pred = model.predict(X)

        # Calculate metrics
        hamming = hamming_loss(y_true, y_pred)
        exact_accuracy = accuracy_score(y_true, y_pred)
        macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average='macro', zero_division=0
        )
        micro_precision, micro_recall, micro_f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average='micro', zero_division=0
        )
        
        comparison_results.append({
            'Model': model_name,
            'Method': method,
            'Hamming_Loss': hamming,
            'Exact_Accuracy': exact_accuracy,
            'Macro_Precision': macro_precision,
            'Macro_Recall': macro_recall,
            'Macro_F1': macro_f1,
            'Micro_Precision': micro_precision,
            'Micro_Recall': micro_recall,
            'Micro_F1': micro_f1
        })
    
    # Create comparison DataFrame
    comparison_df = pd.DataFrame(comparison_results)
    
    # Save comparison results
    comparison_path = os.path.join(config.REPORT_DIR, "model_comparison.csv")
    comparison_df.to_csv(comparison_path, index=False)
    print(f"✅ Saved model comparison to {comparison_path}")
    
    # Print comparison table
    print(f"\n📊 Model Comparison Summary:")
    print("=" * 80)
    print(comparison_df.round(4).to_string(index=False))
    
    # Create comparison visualization
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Hamming Loss comparison
    comparison_df.plot(x='Model', y='Hamming_Loss', kind='bar', ax=axes[0,0], color='red')
    axes[0,0].set_title('Hamming Loss Comparison (Lower is Better)')
    axes[0,0].tick_params(axis='x', rotation=45)
    
    # Exact Accuracy comparison
    comparison_df.plot(x='Model', y='Exact_Accuracy', kind='bar', ax=axes[0,1], color='green')
    axes[0,1].set_title('Exact Match Accuracy Comparison')
    axes[0,1].tick_params(axis='x', rotation=45)
    
    # Macro F1 comparison
    comparison_df.plot(x='Model', y='Macro_F1', kind='bar', ax=axes[1,0], color='blue')
    axes[1,0].set_title('Macro F1-Score Comparison')
    axes[1,0].tick_params(axis='x', rotation=45)
    
    # Micro F1 comparison
    comparison_df.plot(x='Model', y='Micro_F1', kind='bar', ax=axes[1,1], color='orange')
    axes[1,1].set_title('Micro F1-Score Comparison')
    axes[1,1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(os.path.join(config.REPORT_DIR, "model_comparison_charts.png"), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved model comparison charts to {config.REPORT_DIR}/model_comparison_charts.png")
    
    return comparison_df


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

    # Check which models exist
    available_models = []
    model_configs = [
        ("sbert_logistic_regression.pkl", "sbert"),
        ("tfidf_logistic_regression.pkl", "tfidf"),
        ("tfidf_svm.pkl", "tfidf"),
        ("sbert_mlp.pkl", "sbert")
    ]
    
    print("\n🔍 Checking available models...")
    for model_name, method in model_configs:
        model_path = os.path.join(config.MODEL_DIR, model_name)
        if os.path.exists(model_path):
            available_models.append((model_name, method))
            print(f"✅ Found: {model_name}")
        else:
            print(f"❌ Missing: {model_name}")
    
    if not available_models:
        print("❌ No models found! Please train models first using:")
        print("   python run.py --task train --method tfidf")
        print("   python run.py --task train --method sbert")
        return
    
    print(f"\n📊 Found {len(available_models)} model(s) to evaluate")
    
    # Individual model evaluation
    for model_name, method in available_models:
        print(f"\n📊 Evaluating {model_name}...")
        result = evaluate_model(df, model_name=model_name, method=method)
        if result is None:
            print(f"⚠️  Skipped evaluation for {model_name}")
    
    # Model comparison (only if we have multiple models)
    if len(available_models) > 1:
        print(f"\n🔍 Comparing {len(available_models)} models...")
        model_names = [model_name for model_name, _ in available_models]
        compare_models(df, model_names=model_names)
    else:
        print(f"\n⚠️  Only {len(available_models)} model found, skipping comparison.")
        print("   Train more models to enable comparison.")


if __name__ == "__main__":
    main()
