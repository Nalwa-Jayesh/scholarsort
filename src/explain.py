"""
Advanced explainability module for scientific paper classification
Uses SHAP for state-of-the-art model explanations
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from typing import List, Dict, Any, Optional

# Try to import SHAP
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("⚠️  SHAP not available. Install with: pip install shap")

from . import config
from .features import transform_with_tfidf, get_sbert_embeddings
from .preprocess import clean_text, preprocess_dataframe
from .data_loader import load_arxiv_data


class ModelExplainer:
    """
    SHAP-based explainability class for scientific paper classification
    """
    
    def __init__(self, model, vectorizer=None, label_binarizer=None, method="tfidf"):
        self.model = model
        self.vectorizer = vectorizer
        self.label_binarizer = label_binarizer
        self.method = method
        self.shap_explainer = None
        
    def explain_with_shap(self, text: str, num_features: int = 10) -> Dict[str, Any]:
        """
        Generate SHAP explanations (state-of-the-art explainability)
        """
        if not SHAP_AVAILABLE:
            raise ImportError("SHAP not available. Install with: pip install shap")
        
        try:
            if self.method == "tfidf":
                return self._explain_tfidf_shap(text, num_features)
            else:
                return self._explain_sbert_shap(text, num_features)
        except Exception as e:
            print(f"SHAP explanation error: {str(e)}")
            print(f"Model type: {type(self.model)}")
            print(f"Method: {self.method}")
            # Return a simple fallback explanation
            return self._fallback_explanation(text, num_features)
    
    def _fallback_explanation(self, text: str, num_features: int) -> Dict[str, Any]:
        """Fallback explanation when SHAP fails"""
        explanations = {}
        for class_name in self.label_binarizer.classes_:
            explanations[class_name] = {
                'positive': [("SHAP explanation unavailable", 0.0)],
                'negative': [("SHAP explanation unavailable", 0.0)],
                'overall': [("SHAP explanation unavailable", 0.0)]
            }
        return explanations
    
    def _explain_tfidf_shap(self, text: str, num_features: int) -> Dict[str, Any]:
        """SHAP explanation for TF-IDF models"""
        # Clean and vectorize text
        cleaned = clean_text(text)
        X = transform_with_tfidf(self.vectorizer, [cleaned])
        
        # Get feature names
        feature_names = self.vectorizer.get_feature_names_out()
        
        # Process results - handle different SHAP output formats
        explanations = {}
        
        # For OneVsRestClassifier, we need to handle each class separately
        if hasattr(self.model, 'estimators_'):
            # Create explainers for each class
            for i, class_name in enumerate(self.label_binarizer.classes_):
                if i < len(self.model.estimators_):
                    estimator = self.model.estimators_[i]
                    
                    # Choose appropriate explainer based on estimator type
                    if hasattr(estimator, 'coef_') and hasattr(estimator, 'intercept_'):
                        # Linear models (LogisticRegression, LinearSVC)
                        explainer = shap.LinearExplainer(estimator, X)
                    elif hasattr(estimator, 'feature_importances_'):
                        # Tree-based models
                        explainer = shap.TreeExplainer(estimator)
                    else:
                        # Neural networks and other non-linear models
                        # Use a simple approach with background data
                        background = X[:10] if X.shape[0] > 10 else X
                        explainer = shap.KernelExplainer(estimator.predict_proba, background)
                    
                    class_shap = explainer.shap_values(X)
                    
                    # Handle different SHAP output formats
                    if isinstance(class_shap, list):
                        class_shap = class_shap[0]  # Take first class
                    
                    # Ensure we have the right shape and convert to numpy array if needed
                    if hasattr(class_shap, 'ndim') and class_shap.ndim > 1:
                        class_shap = class_shap[0]  # Take first sample
                    
                    # Convert to numpy array and ensure it's 1D
                    if not isinstance(class_shap, np.ndarray):
                        class_shap = np.array(class_shap)
                    if class_shap.ndim > 1:
                        class_shap = class_shap.flatten()
                    
                    # Ensure we have the right number of features
                    if len(class_shap) != len(feature_names):
                        # Truncate or pad as needed
                        if len(class_shap) > len(feature_names):
                            class_shap = class_shap[:len(feature_names)]
                        else:
                            # Pad with zeros if needed
                            padding = np.zeros(len(feature_names) - len(class_shap))
                            class_shap = np.concatenate([class_shap, padding])
                    
                    feature_importance = list(zip(feature_names, class_shap))
                    feature_importance.sort(key=lambda x: abs(float(x[1])), reverse=True)
                    
                    # Before filtering for positive/negative/overall, print the top 10 feature importances for debugging
                    print(f"Class: {class_name}, Top feature importances: {feature_importance[:10]}")
                    
                    explanations[class_name] = {
                        'positive': [(f, float(w)) for f, w in feature_importance[:num_features] if float(w) > 0],
                        'negative': [(f, float(w)) for f, w in feature_importance[:num_features] if float(w) < 0],
                        'overall': [(f, float(w)) for f, w in feature_importance[:num_features]]
                    }
        else:
            # Handle non-OneVsRestClassifier models
            if hasattr(self.model, 'coef_') and hasattr(self.model, 'intercept_'):
                explainer = shap.LinearExplainer(self.model, X)
            elif hasattr(self.model, 'feature_importances_'):
                explainer = shap.TreeExplainer(self.model)
            else:
                background = X[:10] if X.shape[0] > 10 else X
                explainer = shap.KernelExplainer(self.model.predict_proba, background)
            
            shap_values = explainer.shap_values(X)
            
            # Handle different SHAP output structures
            if isinstance(shap_values, list):
                # Multi-output model
                for i, class_name in enumerate(self.label_binarizer.classes_):
                    if i < len(shap_values):
                        class_shap = shap_values[i]
                        # Ensure we have the right shape
                        if hasattr(class_shap, 'ndim') and class_shap.ndim > 1:
                            class_shap = class_shap[0]  # Take first sample
                        
                        # Convert to numpy array and ensure it's 1D
                        if not isinstance(class_shap, np.ndarray):
                            class_shap = np.array(class_shap)
                        if class_shap.ndim > 1:
                            class_shap = class_shap.flatten()
                        
                        # Ensure we have the right number of features
                        if len(class_shap) != len(feature_names):
                            # Truncate or pad as needed
                            if len(class_shap) > len(feature_names):
                                class_shap = class_shap[:len(feature_names)]
                            else:
                                # Pad with zeros if needed
                                padding = np.zeros(len(feature_names) - len(class_shap))
                                class_shap = np.concatenate([class_shap, padding])
                        
                        feature_importance = list(zip(feature_names, class_shap))
                        feature_importance.sort(key=lambda x: abs(float(x[1])), reverse=True)
                        
                        # Before filtering for positive/negative/overall, print the top 10 feature importances for debugging
                        print(f"Class: {class_name}, Top feature importances: {feature_importance[:10]}")
                        
                        explanations[class_name] = {
                            'positive': [(f, float(w)) for f, w in feature_importance[:num_features] if float(w) > 0],
                            'negative': [(f, float(w)) for f, w in feature_importance[:num_features] if float(w) < 0],
                            'overall': [(f, float(w)) for f, w in feature_importance[:num_features]]
                        }
            else:
                # Single output or different structure
                if hasattr(shap_values, 'ndim') and shap_values.ndim > 1:
                    shap_values = shap_values[0]  # Take first sample
                
                # For single output, create explanation for all classes
                for i, class_name in enumerate(self.label_binarizer.classes_):
                    if i < len(shap_values):
                        class_shap = shap_values[i] if isinstance(shap_values[i], (list, np.ndarray)) else shap_values
                    else:
                        class_shap = shap_values
                    
                    # Convert to numpy array and ensure it's 1D
                    if not isinstance(class_shap, np.ndarray):
                        class_shap = np.array(class_shap)
                    if class_shap.ndim > 1:
                        class_shap = class_shap.flatten()
                    
                    # Ensure we have the right number of features
                    if len(class_shap) != len(feature_names):
                        # Truncate or pad as needed
                        if len(class_shap) > len(feature_names):
                            class_shap = class_shap[:len(feature_names)]
                        else:
                            # Pad with zeros if needed
                            padding = np.zeros(len(feature_names) - len(class_shap))
                            class_shap = np.concatenate([class_shap, padding])
                    
                    feature_importance = list(zip(feature_names, class_shap))
                    feature_importance.sort(key=lambda x: abs(float(x[1])), reverse=True)
                    
                    # Before filtering for positive/negative/overall, print the top 10 feature importances for debugging
                    print(f"Class: {class_name}, Top feature importances: {feature_importance[:10]}")
                    
                    explanations[class_name] = {
                        'positive': [(f, float(w)) for f, w in feature_importance[:num_features] if float(w) > 0],
                        'negative': [(f, float(w)) for f, w in feature_importance[:num_features] if float(w) < 0],
                        'overall': [(f, float(w)) for f, w in feature_importance[:num_features]]
                    }
        
        return explanations
    
    def _explain_sbert_shap(self, text: str, num_features: int) -> Dict[str, Any]:
        """SHAP explanation for SBERT models"""
        # Get embeddings
        embeddings = get_sbert_embeddings([text])
        # Use a robust background for KernelExplainer
        background_embeddings = get_or_create_sbert_background_embeddings()
        # Process results - handle different SHAP output formats
        explanations = {}
        # For OneVsRestClassifier, we need to handle each class separately
        if hasattr(self.model, 'estimators_'):
            # Create explainers for each class
            for i, class_name in enumerate(self.label_binarizer.classes_):
                if i < len(self.model.estimators_):
                    estimator = self.model.estimators_[i]
                    # Choose appropriate explainer based on estimator type
                    if hasattr(estimator, 'coef_') and hasattr(estimator, 'intercept_'):
                        # Linear models (LogisticRegression, LinearSVC)
                        explainer = shap.LinearExplainer(estimator, embeddings)
                    elif hasattr(estimator, 'feature_importances_'):
                        # Tree-based models
                        explainer = shap.TreeExplainer(estimator)
                    else:
                        # Neural networks and other non-linear models
                        explainer = shap.KernelExplainer(estimator.predict_proba, background_embeddings)
                    class_shap = explainer.shap_values(embeddings)
                    
                    # Handle different SHAP output formats
                    if isinstance(class_shap, list):
                        class_shap = class_shap[0]  # Take first class
                    
                    # Ensure we have the right shape and convert to numpy array if needed
                    if hasattr(class_shap, 'ndim') and class_shap.ndim > 1:
                        class_shap = class_shap[0]  # Take first sample
                    
                    # Convert to numpy array and ensure it's 1D
                    if not isinstance(class_shap, np.ndarray):
                        class_shap = np.array(class_shap)
                    if class_shap.ndim > 1:
                        class_shap = class_shap.flatten()
                    
                    # Create feature names for embeddings
                    feature_names = [f'embedding_{j}' for j in range(len(class_shap))]
                    
                    # Ensure we have the right number of features
                    if len(class_shap) != len(feature_names):
                        # Truncate or pad as needed
                        if len(class_shap) > len(feature_names):
                            class_shap = class_shap[:len(feature_names)]
                        else:
                            # Pad with zeros if needed
                            padding = np.zeros(len(feature_names) - len(class_shap))
                            class_shap = np.concatenate([class_shap, padding])
                    
                    feature_importance = list(zip(feature_names, class_shap))
                    feature_importance.sort(key=lambda x: abs(float(x[1])), reverse=True)
                    
                    # Before filtering for positive/negative/overall, print the top 10 feature importances for debugging
                    print(f"Class: {class_name}, Top feature importances: {feature_importance[:10]}")
                    
                    explanations[class_name] = {
                        'positive': [(f, float(w)) for f, w in feature_importance[:num_features] if float(w) > 0],
                        'negative': [(f, float(w)) for f, w in feature_importance[:num_features] if float(w) < 0],
                        'overall': [(f, float(w)) for f, w in feature_importance[:num_features]]
                    }
        else:
            # Handle non-OneVsRestClassifier models
            if hasattr(self.model, 'coef_') and hasattr(self.model, 'intercept_'):
                explainer = shap.LinearExplainer(self.model, embeddings)
            elif hasattr(self.model, 'feature_importances_'):
                explainer = shap.TreeExplainer(self.model)
            else:
                background = embeddings[:10] if embeddings.shape[0] > 10 else embeddings
                explainer = shap.KernelExplainer(self.model.predict_proba, background_embeddings)
            
            shap_values = explainer.shap_values(embeddings)
            
            # Handle different SHAP output structures
            if isinstance(shap_values, list):
                # Multi-output model
                for i, class_name in enumerate(self.label_binarizer.classes_):
                    if i < len(shap_values):
                        class_shap = shap_values[i]
                        # Ensure we have the right shape
                        if hasattr(class_shap, 'ndim') and class_shap.ndim > 1:
                            class_shap = class_shap[0]  # Take first sample
                        
                        # Convert to numpy array and ensure it's 1D
                        if not isinstance(class_shap, np.ndarray):
                            class_shap = np.array(class_shap)
                        if class_shap.ndim > 1:
                            class_shap = class_shap.flatten()
                        
                        # Create feature names for embeddings
                        feature_names = [f'embedding_{j}' for j in range(len(class_shap))]
                        
                        # Ensure we have the right number of features
                        if len(class_shap) != len(feature_names):
                            # Truncate or pad as needed
                            if len(class_shap) > len(feature_names):
                                class_shap = class_shap[:len(feature_names)]
                            else:
                                # Pad with zeros if needed
                                padding = np.zeros(len(feature_names) - len(class_shap))
                                class_shap = np.concatenate([class_shap, padding])
                        
                        feature_importance = list(zip(feature_names, class_shap))
                        feature_importance.sort(key=lambda x: abs(float(x[1])), reverse=True)
                        
                        # Before filtering for positive/negative/overall, print the top 10 feature importances for debugging
                        print(f"Class: {class_name}, Top feature importances: {feature_importance[:10]}")
                        
                        explanations[class_name] = {
                            'positive': [(f, float(w)) for f, w in feature_importance[:num_features] if float(w) > 0],
                            'negative': [(f, float(w)) for f, w in feature_importance[:num_features] if float(w) < 0],
                            'overall': [(f, float(w)) for f, w in feature_importance[:num_features]]
                        }
            else:
                # Single output or different structure
                if hasattr(shap_values, 'ndim') and shap_values.ndim > 1:
                    shap_values = shap_values[0]  # Take first sample
                
                # For single output, create explanation for all classes
                for i, class_name in enumerate(self.label_binarizer.classes_):
                    if i < len(shap_values):
                        class_shap = shap_values[i] if isinstance(shap_values[i], (list, np.ndarray)) else shap_values
                    else:
                        class_shap = shap_values
                    
                    # Convert to numpy array and ensure it's 1D
                    if not isinstance(class_shap, np.ndarray):
                        class_shap = np.array(class_shap)
                    if class_shap.ndim > 1:
                        class_shap = class_shap.flatten()
                    
                    # Create feature names for embeddings
                    feature_names = [f'embedding_{j}' for j in range(len(class_shap))]
                    
                    # Ensure we have the right number of features
                    if len(class_shap) != len(feature_names):
                        # Truncate or pad as needed
                        if len(class_shap) > len(feature_names):
                            class_shap = class_shap[:len(feature_names)]
                        else:
                            # Pad with zeros if needed
                            padding = np.zeros(len(feature_names) - len(class_shap))
                            class_shap = np.concatenate([class_shap, padding])
                    
                    feature_importance = list(zip(feature_names, class_shap))
                    feature_importance.sort(key=lambda x: abs(float(x[1])), reverse=True)
                    
                    # Before filtering for positive/negative/overall, print the top 10 feature importances for debugging
                    print(f"Class: {class_name}, Top feature importances: {feature_importance[:10]}")
                    
                    explanations[class_name] = {
                        'positive': [(f, float(w)) for f, w in feature_importance[:num_features] if float(w) > 0],
                        'negative': [(f, float(w)) for f, w in feature_importance[:num_features] if float(w) < 0],
                        'overall': [(f, float(w)) for f, w in feature_importance[:num_features]]
                    }
        
        return explanations


def create_explainer(model, vectorizer=None, label_binarizer=None, method="tfidf"):
    """Factory function to create SHAP explainer"""
    return ModelExplainer(model, vectorizer, label_binarizer, method)


def display_explanation_streamlit(explanation: Dict[str, Any]):
    """Display SHAP explanation in Streamlit"""
    
    st.subheader("🔍 SHAP Explanation")
    
    if isinstance(explanation, dict) and 'positive' in list(explanation.values())[0]:
        # TF-IDF SHAP explanation
        for class_name, class_explanation in explanation.items():
            st.write(f"**{class_name}:**")
            
            if class_explanation['positive']:
                st.write("🟢 **Positive features:**")
                for feature, weight in class_explanation['positive'][:5]:
                    st.write(f"   • {feature}: {weight:.3f}")
            
            if class_explanation['negative']:
                st.write("🔴 **Negative features:**")
                for feature, weight in class_explanation['negative'][:5]:
                    st.write(f"   • {feature}: {weight:.3f}")
            
            st.write("---")
    
    else:
        # SBERT SHAP explanation
        for class_name, class_explanation in explanation.items():
            st.write(f"**{class_name}:** {class_explanation['importance']:.3f}")


def generate_explanation(text: str, model, vectorizer=None, label_binarizer=None, method="tfidf"):
    """Generate SHAP explanation for any model"""
    try:
        explainer = create_explainer(model, vectorizer, label_binarizer, method)
        return explainer.explain_with_shap(text, num_features=10)
    except Exception as e:
        return f"Error generating explanation: {str(e)}"


def get_or_create_sbert_background_embeddings(path="models/sbert_background_embeddings.npy", n_samples=100):
    if os.path.exists(path):
        print(f"[SHAP] Loading background embeddings from {path}")
        return np.load(path)
    else:
        print("[SHAP] Generating new background embeddings...")
        df = load_arxiv_data(from_api=True, per_category=20)
        df = preprocess_dataframe(df)
        texts = df['cleaned_abstract'].tolist()[:n_samples]
        embeddings = get_sbert_embeddings(texts)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.save(path, embeddings)
        print(f"[SHAP] Saved background embeddings to {path}")
        return embeddings 