from __future__ import annotations
from typing import Dict, Tuple, List
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import StratifiedKFold, cross_val_score

def compute_metrics(y_true, y_pred) -> Dict[str, float]:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
    }

def print_metrics_table(title: str, metrics: Dict[str, float]) -> None:
    print(f"\n=== {title} ===")
    for k, v in metrics.items():
        print(f"{k:>16}: {v:.4f}")

def save_confusion_matrix(y_true, y_pred, labels: List[str], out_path: str, title: str) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    fig, ax = plt.subplots(figsize=(5, 4), dpi=120)
    disp.plot(ax=ax, colorbar=False)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)

def cv_stability(estimator, X, y, cv_folds: int = 5, scoring: str = "f1_macro", random_state: int = 42) -> Tuple[float, float]:
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    scores = cross_val_score(estimator, X, y, cv=skf, scoring=scoring, n_jobs=-1)
    return float(np.mean(scores)), float(np.std(scores))

def plot_staged_curve(model, X_train, y_train, out_path: str, metric: str = "f1_macro") -> None:
    """
    For GradientBoostingClassifier: plot metric vs n_estimators using staged predictions.
    """
    if not hasattr(model, "staged_predict"):
        return
    from sklearn.metrics import f1_score, accuracy_score
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    preds = []
    for y_hat in model.staged_predict(X_train):
        if metric == "f1_macro":
            preds.append(f1_score(y_train, y_hat, average="macro", zero_division=0))
        else:
            preds.append(accuracy_score(y_train, y_hat))
    fig, ax = plt.subplots(figsize=(6,4), dpi=120)
    ax.plot(range(1, len(preds)+1), preds)
    ax.set_xlabel("n_estimators")
    ax.set_ylabel(metric)
    ax.set_title(f"Boosting staged {metric} vs n_estimators")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)