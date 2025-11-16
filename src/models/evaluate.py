"""
Model evaluation functions
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    average_precision_score
)
from typing import Dict, Union


def find_optimal_threshold(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    metric: str = 'balanced_precision',
    target_precision: float = 0.70
) -> float:
    """
    Find optimal threshold targeting ~70% precision for both classes
    
    Parameters:
    -----------
    y_true : array
        True labels
    y_pred_proba : array
        Predicted probabilities
    metric : str
        - 'balanced_precision': Target ~70% precision for both classes, maximize recall
        - 'f1': F1 score
        - 'f2': F2 score (emphasizes recall)
        - 'balanced': Simple average of precision and recall
    target_precision : float
        Target precision for both classes (default 0.70)
    
    Returns:
    --------
    float: Optimal threshold
    """
    print(f"\nFinding optimal threshold for {metric} (target precision: {target_precision:.2f})...")
    
    best_score = -np.inf
    best_threshold = 0.5
    best_metrics = {}
    
    # Search wider range of thresholds
    thresholds = np.arange(0.1, 0.95, 0.005)
    
    for threshold in thresholds:
        y_pred = (y_pred_proba >= threshold).astype(int)
        
        # Calculate per-class precision
        cm = confusion_matrix(y_true, y_pred)
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            
            precision_0 = tn / (tn + fp) if (tn + fp) > 0 else 0
            precision_1 = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall_0 = tn / (tn + fn) if (tn + fn) > 0 else 0
            recall_1 = tp / (tp + fn) if (tp + fn) > 0 else 0
        else:
            precision_0 = precision_score(y_true, y_pred, pos_label=0, zero_division=0)
            precision_1 = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
            recall_0 = recall_score(y_true, y_pred, pos_label=0, zero_division=0)
            recall_1 = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
        
        if metric == 'balanced_precision':
            precision_0_diff = abs(precision_0 - target_precision)
            precision_1_diff = abs(precision_1 - target_precision)
            precision_penalty = precision_0_diff + precision_1_diff
            recall_reward = recall_0 * 0.3 + recall_1 * 0.7
            score = recall_reward - precision_penalty
            
            if 0.65 <= precision_0 <= 0.75 and 0.65 <= precision_1 <= 0.75:
                score += 0.5
            if precision_0_diff < 0.05 and precision_1_diff < 0.05:
                score += 0.3
        elif metric == 'f1':
            score = f1_score(y_true, y_pred, zero_division=0)
        elif metric == 'f2':
            precision = precision_1
            recall = recall_1
            if precision + recall > 0:
                score = (1 + 2**2) * (precision * recall) / ((2**2 * precision) + recall)
            else:
                score = 0
        elif metric == 'balanced':
            precision = precision_1
            recall = recall_1
            score = (precision + recall) / 2
        else:
            score = f1_score(y_true, y_pred, zero_division=0)
        
        if score > best_score:
            best_score = score
            best_threshold = threshold
            best_metrics = {
                'precision_0': precision_0,
                'precision_1': precision_1,
                'recall_0': recall_0,
                'recall_1': recall_1,
                'score': score
            }
    
    print(f"Optimal threshold: {best_threshold:.3f}")
    print(f"  Class 0 - Precision: {best_metrics['precision_0']:.4f}, Recall: {best_metrics['recall_0']:.4f}")
    print(f"  Class 1 - Precision: {best_metrics['precision_1']:.4f}, Recall: {best_metrics['recall_1']:.4f}")
    return best_threshold


def evaluate_model(
    model: Union,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    threshold: float = 0.5,
    model_name: str = "Model"
) -> Dict:
    """Comprehensive model evaluation with per-class precision"""
    print("\n" + "="*80)
    print(f"EVALUATING {model_name}")
    print("="*80)
    
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_proba >= threshold).astype(int)
    
    # Calculate metrics
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    # Per-class precision
    precision_0 = tn / (tn + fp) if (tn + fp) > 0 else 0
    precision_1 = tp / (tp + fn) if (tp + fn) > 0 else 0
    recall_0 = tn / (tn + fn) if (tn + fn) > 0 else 0
    recall_1 = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    # Overall metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    pr_auc = average_precision_score(y_test, y_pred_proba)
    
    print(f"\nThreshold: {threshold:.3f}")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"\nPer-Class Metrics:")
    print(f"  Class 0 (No Default):")
    print(f"    Precision: {precision_0:.4f}")
    print(f"    Recall:    {recall_0:.4f}")
    print(f"  Class 1 (Default):")
    print(f"    Precision: {precision_1:.4f}")
    print(f"    Recall:    {recall_1:.4f}")
    print(f"\nOverall Metrics:")
    print(f"  Precision (Class 1): {precision:.4f}")
    print(f"  Recall (Class 1):    {recall:.4f}")
    print(f"  F1-Score:            {f1:.4f}")
    print(f"  ROC-AUC:             {roc_auc:.4f}")
    print(f"  PR-AUC:              {pr_auc:.4f}")
    
    print("\nConfusion Matrix:")
    print("                 Predicted")
    print("              No Default  Default")
    print(f"Actual No Default    {tn:6d}    {fp:6d}")
    print(f"Actual Default       {fn:6d}    {tp:6d}")
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'precision_0': precision_0,
        'precision_1': precision_1,
        'recall': recall,
        'recall_0': recall_0,
        'recall_1': recall_1,
        'f1': f1,
        'roc_auc': roc_auc,
        'pr_auc': pr_auc,
        'y_pred_proba': y_pred_proba,
        'y_pred': y_pred,
        'confusion_matrix': cm
    }

