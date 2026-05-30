from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)


# ============================================================
# GLOBAL OOD CONVENTION
# ============================================================

"""
GLOBAL OOD STANDARD USED EVERYWHERE IN THIS CODEBASE:

    ID  = 0
    OOD = 1

IMPORTANT:
    Higher score -> MORE OOD-LIKE

This convention MUST NEVER change.

Examples of valid OOD scores:
    - entropy
    - energy
    - negative max softmax probability

Examples:
    low score  -> likely ID
    high score -> likely OOD
"""

# ============================================================
# CLASSIFICATION METRICS
# ============================================================

def compute_classification_metrics(
    labels: np.ndarray,
    preds: np.ndarray,
):
    """
    Compute classification metrics for ID classification.

    IMPORTANT:
        OOD samples must already be excluded before
        calling this function.

    Args:
        labels:
            Ground-truth ID labels.

        preds:
            Predicted ID labels.

    Returns:
        Dictionary with classification metrics.
    """

    return {

        # ====================================================
        # STANDARD ACCURACY
        # ====================================================

        "accuracy":
            accuracy_score(
                labels,
                preds,
            ),

        # ====================================================
        # BALANCED ACCURACY
        #
        # Useful for class imbalance.
        # ====================================================

        # "balanced_accuracy":
        #     balanced_accuracy_score(
        #         labels,
        #         preds,
        #     ),

        # ====================================================
        # MACRO F1
        #
        # Equal weight per class.
        # Most important metric for imbalanced NLP tasks.
        # ====================================================

        "macro_f1":
            f1_score(
                labels,
                preds,
                average="macro",
                zero_division=0,
            ),

        # ====================================================
        # WEIGHTED F1
        #
        # Weighted by class frequency.
        # ====================================================

        # "weighted_f1":
        #     f1_score(
        #         labels,
        #         preds,
        #         average="weighted",
        #         zero_division=0,
        #     ),

        # ====================================================
        # MACRO PRECISION
        # ====================================================

        "macro_precision":
            precision_score(
                labels,
                preds,
                average="macro",
                zero_division=0,
            ),

        # ====================================================
        # MACRO RECALL
        # ====================================================

        "macro_recall":
            recall_score(
                labels,
                preds,
                average="macro",
                zero_division=0,
            ),
    }

# ============================================================
# OOD METRICS
# ============================================================

def compute_ood_metrics(
    id_scores: np.ndarray,
    ood_scores: np.ndarray,
):
    """
    Compute OOD detection metrics.

    GLOBAL OOD CONVENTION:
    ----------------------

        ID  = 0
        OOD = 1

        Higher score -> MORE OOD-like

    Args:
        id_scores:
            OOD scores for ID samples.

        ood_scores:
            OOD scores for OOD samples.

    Returns:
        Dictionary with OOD metrics.
    """

    # ========================================================
    # BUILD LABELS
    #
    # ID  -> 0
    # OOD -> 1
    # ========================================================

    y_true = np.concatenate([
        np.zeros(len(id_scores)),
        np.ones(len(ood_scores)),
    ])

    scores = np.concatenate([
        id_scores,
        ood_scores,
    ])

    # ========================================================
    # AUROC
    #
    # Main OOD metric.
    # Measures ranking quality.
    # ========================================================

    auroc = roc_auc_score(
        y_true,
        scores,
    )

    # ========================================================
    # AUPR
    #
    # Precision-Recall for OOD class.
    # More informative for imbalance.
    # ========================================================

    aupr = average_precision_score(
        y_true,
        scores,
    )

    # ========================================================
    # FPR95
    #
    # False Positive Rate when:
    # TPR(OOD) = 95%
    #
    # Lower is better.
    # ========================================================

    precision, recall, thresholds = (
        precision_recall_curve(
            y_true,
            scores,
        )
    )

    fpr95 = np.nan

    try:

        idx = np.argmin(
            np.abs(recall - 0.95)
        )

        threshold = thresholds[
            max(0, idx - 1)
        ]

        pred_ood = scores >= threshold

        fp = np.logical_and(
            pred_ood,
            y_true == 0,
        ).sum()

        tn = np.logical_and(
            ~pred_ood,
            y_true == 0,
        ).sum()

        fpr95 = fp / (fp + tn + 1e-12)

    except Exception:
        pass

    return {

        "auroc": float(auroc),

        "aupr": float(aupr),

        "fpr95": float(fpr95),
    }
