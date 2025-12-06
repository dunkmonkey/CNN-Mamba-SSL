"""
Evaluation metrics for heart sound classification.
"""

import torch
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)
from typing import Dict, Optional, Tuple


def compute_binary_metrics(
    predictions: np.ndarray,
    targets: np.ndarray,
    probabilities: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """
    Compute comprehensive metrics for binary classification.
    
    Args:
        predictions: Predicted class labels (0 or 1)
        targets: True class labels (0 or 1)
        probabilities: Optional prediction probabilities for AUROC
        
    Returns:
        Dictionary of metric names and values
    """
    metrics = {}
    
    # Basic metrics
    metrics['accuracy'] = accuracy_score(targets, predictions)
    metrics['precision'] = precision_score(targets, predictions, zero_division=0)
    metrics['recall'] = recall_score(targets, predictions, zero_division=0)
    metrics['f1'] = f1_score(targets, predictions, zero_division=0)
    
    # AUROC if probabilities provided
    if probabilities is not None:
        try:
            metrics['auroc'] = roc_auc_score(targets, probabilities)
        except ValueError:
            metrics['auroc'] = 0.0
    
    # Confusion matrix components
    tn, fp, fn, tp = confusion_matrix(targets, predictions).ravel()
    metrics['true_positives'] = int(tp)
    metrics['true_negatives'] = int(tn)
    metrics['false_positives'] = int(fp)
    metrics['false_negatives'] = int(fn)
    
    # Specificity
    metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    
    # Sensitivity (same as recall)
    metrics['sensitivity'] = metrics['recall']
    
    return metrics


def compute_multiclass_metrics(
    predictions: np.ndarray,
    targets: np.ndarray,
    probabilities: Optional[np.ndarray] = None,
    num_classes: Optional[int] = None
) -> Dict[str, float]:
    """
    Compute metrics for multiclass classification.
    
    Args:
        predictions: Predicted class labels
        targets: True class labels
        probabilities: Optional prediction probabilities for each class
        num_classes: Number of classes
        
    Returns:
        Dictionary of metric names and values
    """
    metrics = {}
    
    # Basic metrics
    metrics['accuracy'] = accuracy_score(targets, predictions)
    
    # Macro-averaged metrics
    metrics['precision_macro'] = precision_score(
        targets, predictions, average='macro', zero_division=0
    )
    metrics['recall_macro'] = recall_score(
        targets, predictions, average='macro', zero_division=0
    )
    metrics['f1_macro'] = f1_score(
        targets, predictions, average='macro', zero_division=0
    )
    
    # Weighted-averaged metrics
    metrics['precision_weighted'] = precision_score(
        targets, predictions, average='weighted', zero_division=0
    )
    metrics['recall_weighted'] = recall_score(
        targets, predictions, average='weighted', zero_division=0
    )
    metrics['f1_weighted'] = f1_score(
        targets, predictions, average='weighted', zero_division=0
    )
    
    # Per-class metrics
    precision_per_class = precision_score(
        targets, predictions, average=None, zero_division=0
    )
    recall_per_class = recall_score(
        targets, predictions, average=None, zero_division=0
    )
    f1_per_class = f1_score(
        targets, predictions, average=None, zero_division=0
    )
    
    for i in range(len(precision_per_class)):
        metrics[f'precision_class_{i}'] = precision_per_class[i]
        metrics[f'recall_class_{i}'] = recall_per_class[i]
        metrics[f'f1_class_{i}'] = f1_per_class[i]
    
    # Multiclass AUROC
    if probabilities is not None and num_classes is not None:
        try:
            metrics['auroc_ovr'] = roc_auc_score(
                targets, probabilities, multi_class='ovr', average='macro'
            )
        except ValueError:
            metrics['auroc_ovr'] = 0.0
    
    return metrics


def print_classification_report(
    predictions: np.ndarray,
    targets: np.ndarray,
    class_names: Optional[list] = None
):
    """
    Print detailed classification report.
    
    Args:
        predictions: Predicted class labels
        targets: True class labels
        class_names: Optional class names for display
    """
    print("\nClassification Report:")
    print("=" * 60)
    print(classification_report(
        targets, predictions, target_names=class_names, zero_division=0
    ))
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(targets, predictions))


def compute_contrastive_metrics(
    similarities: torch.Tensor,
    temperature: float = 0.07
) -> Dict[str, float]:
    """
    Compute metrics for contrastive learning.
    
    Args:
        similarities: Similarity matrix of shape (batch, batch + queue_size)
                     where positive pairs are at index 0
        temperature: Temperature parameter
        
    Returns:
        Dictionary of metrics
    """
    metrics = {}
    
    # Apply temperature
    logits = similarities / temperature
    
    # Compute top-1 accuracy (positive should be ranked first)
    predictions = logits.argmax(dim=1)
    correct = (predictions == 0).float().sum()
    metrics['top1_acc'] = (correct / logits.size(0)).item()
    
    # Compute top-5 accuracy
    _, top5_pred = logits.topk(5, dim=1)
    top5_correct = (top5_pred == 0).any(dim=1).float().sum()
    metrics['top5_acc'] = (top5_correct / logits.size(0)).item()
    
    # Mean positive similarity
    metrics['mean_pos_sim'] = similarities[:, 0].mean().item()
    
    # Mean negative similarity
    metrics['mean_neg_sim'] = similarities[:, 1:].mean().item()
    
    # Similarity gap (positive - negative)
    metrics['sim_gap'] = metrics['mean_pos_sim'] - metrics['mean_neg_sim']
    
    return metrics


class MetricsTracker:
    """
    Track metrics across batches and compute aggregated statistics.
    """
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset all tracked metrics."""
        self.predictions = []
        self.targets = []
        self.probabilities = []
    
    def update(
        self,
        preds: torch.Tensor,
        targets: torch.Tensor,
        probs: Optional[torch.Tensor] = None
    ):
        """
        Update with batch predictions and targets.
        
        Args:
            preds: Predicted class labels
            targets: True class labels
            probs: Optional prediction probabilities
        """
        self.predictions.extend(preds.cpu().numpy().tolist())
        self.targets.extend(targets.cpu().numpy().tolist())
        
        if probs is not None:
            self.probabilities.extend(probs.cpu().numpy().tolist())
    
    def compute(self, binary: bool = True) -> Dict[str, float]:
        """
        Compute aggregated metrics.
        
        Args:
            binary: Whether classification is binary
            
        Returns:
            Dictionary of metrics
        """
        predictions = np.array(self.predictions)
        targets = np.array(self.targets)
        probabilities = np.array(self.probabilities) if self.probabilities else None
        
        if binary:
            return compute_binary_metrics(predictions, targets, probabilities)
        else:
            return compute_multiclass_metrics(predictions, targets, probabilities)
