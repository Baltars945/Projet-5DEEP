from __future__ import annotations
from os import name
from pathlib import Path
import json
import matplotlib.pyplot as plt
import numpy as np
from pymupdf import name
import torch
from matplotlib.figure import Figure
from sklearn.metrics import (
    accuracy_score, 
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)

from src.config import CLASSES, DEVICE, FIGURES_DIR

def evaluateModel(model, loader, device=DEVICE, class_names=None) -> dict:
    class_names = class_names or CLASSES
    labels = list(range(len(class_names)))

    model.to(device)
    model.eval()

    all_preds = []
    all_targets = []

    total_loss = 0.0
    n_samples = 0

    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = torch.nn.functional.cross_entropy(outputs, targets)

            total_loss += loss.item() * inputs.size(0)
            n_samples += inputs.size(0)

            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    avg_loss = total_loss / n_samples if n_samples > 0 else 0

    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_targets)

    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, labels=labels, average='weighted')
    report = classification_report(y_true, y_pred, labels=labels, target_names=class_names, output_dict=True)
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    per_class = {
        name: {
                "precision": report[name]["precision"],
                "recall": report[name]["recall"],
                "f1": report[name]["f1"],
                "support": report[name]["support"]
        }
        for name in class_names
        }

    return {
        "y_true": y_true,
        "y_pred": y_pred,
        "loss": avg_loss,
        "accuracy": accuracy,
        "per_class": per_class,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "classification_report": report,
        "confusion_matrix": cm
    }

def plotHistory(history: dict, title: str | None = None, save_path: Path | None = None) -> Figure:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    ax_loss, ax_acc = axes

    ax_acc.plot(history['train_acc'], label='Train Accuracy')
    ax_acc.plot(history['val_acc'], label='Validation Accuracy')
    ax_acc.set_xlabel('Epochs')
    ax_acc.set_ylabel('Accuracy')
    ax_acc.set_title(title or 'Training and Validation Accuracy')
    ax_acc.legend()
    ax_acc.grid()
    
    ax_loss.plot(history['train_loss'], label='Train Loss')
    ax_loss.plot(history['val_loss'], label='Validation Loss')
    ax_loss.set_xlabel('Epochs')
    ax_loss.set_ylabel('Loss')
    ax_loss.set_title(title or 'Training and Validation Loss')
    ax_loss.legend()
    ax_loss.grid()

    if save_path:
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path)

    return fig

def pltConfusionMatrix(cm: np.ndarray, class_names: list[str], title: str | None = None, save_path: Path | None = None) -> Figure:
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=class_names,
           yticklabels=class_names,
           ylabel='True label',
           xlabel='Predicted label',
           title=title or 'Confusion Matrix')

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    fmt = 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    
    fig.tight_layout()

    if save_path:
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path)

    return fig

def metricsSummary(name: str, metrics: dict) -> str:
    summary = f"Model name: {name}\n"
    summary = f"Loss: {metrics['loss']:.4f}\n"
    summary += f"Accuracy: {metrics['accuracy']:.4f}\n"
    summary += f"Precision: {metrics['precision']:.4f}\n"
    summary += f"Recall: {metrics['recall']:.4f}\n"
    summary += f"F1 Score: {metrics['f1']:.4f}\n\n"

    summary += "Per Class Metrics:\n"
    for class_name, class_metrics in metrics['per_class'].items():
        summary += f"{class_name} - Precision: {class_metrics['precision']:.4f}, "
        summary += f"Recall: {class_metrics['recall']:.4f}, "
        summary += f"F1 Score: {class_metrics['f1']:.4f}, "
        summary += f"Support: {class_metrics['support']}\n"

    return summary

def saveMetrics(metrics: dict, save_path: Path) -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    return save_path

def loadMetrics(load_path: Path) -> dict:
    with open(load_path, 'r') as f:
        metrics = json.load(f)
    return metrics


