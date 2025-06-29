import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from . import config


def generate_performance_summary():
    """
    Generate a comprehensive performance summary report for all models.
    """
    print("📊 Generating Performance Summary Report...")
    print("=" * 60)
    
    # Check if reports directory exists
    if not os.path.exists(config.REPORT_DIR):
        print("❌ Reports directory not found. Please run evaluation first.")
        print("   Use: python run.py --task evaluate")
        return
    
    # Find all model reports
    report_files = list(Path(config.REPORT_DIR).glob("*_enhanced_metrics.csv"))
    
    if not report_files:
        print("❌ No enhanced metrics files found. Please run evaluation first.")
        print("   Use: python run.py --task evaluate")
        print("\n💡 Available models to evaluate:")
        
        # Check which models exist
        model_configs = [
            "sbert_logistic_regression.pkl",
            "tfidf_logistic_regression.pkl", 
            "tfidf_svm.pkl",
            "sbert_mlp.pkl"
        ]
        
        found_models = []
        for model_name in model_configs:
            model_path = os.path.join(config.MODEL_DIR, model_name)
            if os.path.exists(model_path):
                found_models.append(model_name)
                print(f"   ✅ {model_name}")
            else:
                print(f"   ❌ {model_name} (not trained)")
        
        if not found_models:
            print("\n❌ No models found! Please train models first:")
            print("   python run.py --task train --method tfidf")
            print("   python run.py --task train --method sbert")
        return
    
    # Load and combine all metrics
    all_metrics = []
    for file in report_files:
        model_name = file.stem.replace("_enhanced_metrics", "")
        metrics = pd.read_csv(file)
        metrics['Model'] = model_name
        all_metrics.append(metrics)
    
    if not all_metrics:
        print("❌ No metrics data found.")
        return
    
    combined_metrics = pd.concat(all_metrics, ignore_index=True)
    
    # Add method column
    combined_metrics['Method'] = combined_metrics['Model'].apply(
        lambda x: 'SBERT' if 'sbert' in x else 'TF-IDF'
    )
    
    # Create comprehensive summary
    print("\n📈 Model Performance Summary")
    print("=" * 40)
    
    # Best models by different metrics
    best_hamming = combined_metrics.loc[combined_metrics['hamming_loss'].idxmin()]
    best_accuracy = combined_metrics.loc[combined_metrics['exact_accuracy'].idxmax()]
    best_macro_f1 = combined_metrics.loc[combined_metrics['macro_f1'].idxmax()]
    best_micro_f1 = combined_metrics.loc[combined_metrics['micro_f1'].idxmax()]
    
    print(f"\n🏆 Best Models by Metric:")
    print(f"Lowest Hamming Loss: {best_hamming['Model']} ({best_hamming['hamming_loss']:.4f})")
    print(f"Highest Accuracy: {best_accuracy['Model']} ({best_accuracy['exact_accuracy']:.4f})")
    print(f"Best Macro F1: {best_macro_f1['Model']} ({best_macro_f1['macro_f1']:.4f})")
    print(f"Best Micro F1: {best_micro_f1['Model']} ({best_micro_f1['micro_f1']:.4f})")
    
    # Method comparison
    print(f"\n📊 Method Comparison:")
    method_summary = combined_metrics.groupby('Method').agg({
        'hamming_loss': ['mean', 'std'],
        'exact_accuracy': ['mean', 'std'],
        'macro_f1': ['mean', 'std'],
        'micro_f1': ['mean', 'std']
    }).round(4)
    
    print(method_summary)
    
    # Create comprehensive visualization
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 1. Hamming Loss comparison
    sns.barplot(data=combined_metrics, x='Model', y='hamming_loss', ax=axes[0,0], color='red')
    axes[0,0].set_title('Hamming Loss (Lower is Better)')
    axes[0,0].tick_params(axis='x', rotation=45)
    
    # 2. Accuracy comparison
    sns.barplot(data=combined_metrics, x='Model', y='exact_accuracy', ax=axes[0,1], color='green')
    axes[0,1].set_title('Exact Match Accuracy')
    axes[0,1].tick_params(axis='x', rotation=45)
    
    # 3. Macro F1 comparison
    sns.barplot(data=combined_metrics, x='Model', y='macro_f1', ax=axes[0,2], color='blue')
    axes[0,2].set_title('Macro F1-Score')
    axes[0,2].tick_params(axis='x', rotation=45)
    
    # 4. Micro F1 comparison
    sns.barplot(data=combined_metrics, x='Model', y='micro_f1', ax=axes[1,0], color='orange')
    axes[1,0].set_title('Micro F1-Score')
    axes[1,0].tick_params(axis='x', rotation=45)
    
    # 5. Method comparison heatmap
    method_metrics = combined_metrics.groupby('Method')[['hamming_loss', 'exact_accuracy', 'macro_f1', 'micro_f1']].mean()
    sns.heatmap(method_metrics.T, annot=True, fmt='.3f', cmap='RdYlGn', ax=axes[1,1])
    axes[1,1].set_title('Method Performance Comparison')
    
    # 6. Radar chart for best model
    best_model = combined_metrics.loc[combined_metrics['exact_accuracy'].idxmax()]
    metrics_for_radar = ['exact_accuracy', 'macro_f1', 'micro_f1']
    values = [best_model[metric] for metric in metrics_for_radar]
    
    # Create radar chart
    angles = [i * 360 / len(metrics_for_radar) for i in range(len(metrics_for_radar))]
    values += values[:1]  # Close the loop
    angles += angles[:1]
    
    axes[1,2].plot(angles, values, 'o-', linewidth=2)
    axes[1,2].fill(angles, values, alpha=0.25)
    axes[1,2].set_xticks(angles[:-1])
    axes[1,2].set_xticklabels(metrics_for_radar)
    axes[1,2].set_title(f'Best Model: {best_model["Model"]}')
    axes[1,2].set_ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig(os.path.join(config.REPORT_DIR, "performance_summary.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save summary report
    summary_path = os.path.join(config.REPORT_DIR, "performance_summary.csv")
    combined_metrics.to_csv(summary_path, index=False)
    
    print(f"\n✅ Performance summary saved to:")
    print(f"   📄 {summary_path}")
    print(f"   📊 {config.REPORT_DIR}/performance_summary.png")
    
    # Generate recommendations
    print(f"\n💡 Recommendations:")
    print(f"   • Best overall model: {best_accuracy['Model']}")
    print(f"   • Best for multi-label: {best_hamming['Model']}")
    print(f"   • Best balanced performance: {best_macro_f1['Model']}")
    
    if 'sbert' in best_accuracy['Model']:
        print(f"   • SBERT models generally perform better for semantic understanding")
    else:
        print(f"   • TF-IDF models are faster and more interpretable")
    
    return combined_metrics


def main():
    """Main function to run the performance summary."""
    generate_performance_summary()


if __name__ == "__main__":
    main() 