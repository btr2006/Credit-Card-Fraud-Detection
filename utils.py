import os
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, Any, Dict
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve, precision_recall_curve
from config import MODELS_DIR, OUTPUTS_DIR

def load_data(path: str) -> pd.DataFrame:
    """Loads the dataset from a given path."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found at {path}")
    return pd.read_csv(path)

def save_model(model: Any, path: str) -> None:
    """Saves a machine learning model to disk."""
    joblib.dump(model, path)

def load_model(path: str) -> Any:
    """Loads a machine learning model from disk."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model not found at {path}")
    return joblib.load(path)

def save_scaler(scaler: Any, path: str) -> None:
    """Saves a scaler to disk."""
    joblib.dump(scaler, path)

def load_scaler(path: str) -> Any:
    """Loads a scaler from disk."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Scaler not found at {path}")
    return joblib.load(path)

def evaluate_model(y_true, y_pred, y_prob) -> Dict[str, float]:
    """Calculates various evaluation metrics."""
    metrics = {
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred),
        'Recall': recall_score(y_true, y_pred),
        'F1 Score': f1_score(y_true, y_pred),
        'ROC AUC': roc_auc_score(y_true, y_prob) if y_prob is not None else None
    }
    return metrics

def plot_confusion_matrix(y_true, y_pred, title="Confusion Matrix", save_path=None):
    """Plots and optionally saves a confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.close()

def plot_roc_curve(y_true, y_prob, title="ROC Curve", save_path=None):
    """Plots and optionally saves an ROC curve."""
    if y_prob is None:
        return
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)
    plt.figure(figsize=(6,4))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.close()

def plot_precision_recall_curve(y_true, y_prob, title="Precision-Recall Curve", save_path=None):
    """Plots and optionally saves a Precision-Recall curve."""
    if y_prob is None:
        return
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    plt.figure(figsize=(6,4))
    plt.plot(recall, precision, color='blue', lw=2)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(title)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.close()

def plot_model_comparison(metrics_df, save_path=None):
    """Plots a comparison of different models."""
    metrics_df.plot(kind='bar', figsize=(10, 6))
    plt.title('Model Comparison')
    plt.ylabel('Score')
    plt.xticks(rotation=45)
    plt.legend(loc='lower right')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.close()
