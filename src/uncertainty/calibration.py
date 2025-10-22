"""
Uncertainty Calibration Module
===============================

Provides calibration metrics and conformal prediction intervals.

Methods:
    - Reliability diagrams
    - Brier score
    - Expected Calibration Error (ECE)
    - Conformal prediction intervals

Sacred Geometry:
    - 9 calibration bins for reliability diagrams
    - 95% confidence intervals (aligned with 369 philosophy)

Author: C.C.R.O.P-PhenoHunt Team
Version: 1.0.0

References:
    - Guo et al. (2017). On Calibration of Modern Neural Networks.
    - Angelopoulos & Bates (2021). A Gentle Introduction to Conformal Prediction and Distribution-Free Uncertainty Quantification.
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, Tuple, Optional
from sklearn.metrics import brier_score_loss

logger = logging.getLogger(__name__)


def compute_calibration_metrics(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    n_bins: int = 9  # Sacred number
) -> Dict:
    """
    Compute calibration metrics for probabilistic predictions.

    Args:
        y_true: True binary labels (0 or 1)
        y_pred_proba: Predicted probabilities
        n_bins: Number of bins for reliability diagram (default: 9)

    Returns:
        Dictionary with calibration metrics:
            - brier_score: Brier score (lower is better)
            - ece: Expected Calibration Error
            - reliability_diagram: Data for plotting

    Example:
        >>> y_true = np.array([0, 1, 1, 0, 1])
        >>> y_pred = np.array([0.2, 0.8, 0.9, 0.3, 0.7])
        >>> metrics = compute_calibration_metrics(y_true, y_pred)
        >>> print(f"Brier Score: {metrics['brier_score']:.4f}")
    """
    logger.info(f"Computing calibration metrics with {n_bins} bins")

    # Brier score
    brier = brier_score_loss(y_true, y_pred_proba)

    # Reliability diagram data
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    bin_accuracy = []
    bin_confidence = []
    bin_counts = []

    for i in range(n_bins):
        mask = (y_pred_proba >= bin_edges[i]) & (y_pred_proba < bin_edges[i + 1])
        if i == n_bins - 1:  # Include 1.0 in last bin
            mask = (y_pred_proba >= bin_edges[i]) & (y_pred_proba <= bin_edges[i + 1])

        if np.sum(mask) > 0:
            bin_accuracy.append(np.mean(y_true[mask]))
            bin_confidence.append(np.mean(y_pred_proba[mask]))
            bin_counts.append(np.sum(mask))
        else:
            bin_accuracy.append(np.nan)
            bin_confidence.append(bin_centers[i])
            bin_counts.append(0)

    # Expected Calibration Error (ECE)
    bin_accuracy = np.array(bin_accuracy)
    bin_confidence = np.array(bin_confidence)
    bin_counts = np.array(bin_counts)

    valid_bins = ~np.isnan(bin_accuracy)
    ece = np.sum(
        bin_counts[valid_bins] / np.sum(bin_counts[valid_bins]) *
        np.abs(bin_accuracy[valid_bins] - bin_confidence[valid_bins])
    )

    metrics = {
        'brier_score': float(brier),
        'ece': float(ece),
        'n_bins': n_bins,
        'reliability_diagram': {
            'bin_centers': bin_centers.tolist(),
            'bin_accuracy': [float(x) if not np.isnan(x) else None for x in bin_accuracy],
            'bin_confidence': bin_confidence.tolist(),
            'bin_counts': bin_counts.tolist()
        }
    }

    logger.info(f"Calibration metrics: Brier={brier:.4f}, ECE={ece:.4f}")

    return metrics


def conformal_prediction_intervals(
    y_calibration: np.ndarray,
    y_pred_calibration: np.ndarray,
    y_pred_test: np.ndarray,
    alpha: float = 0.05  # 95% confidence
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute conformal prediction intervals.

    Provides distribution-free prediction intervals with guaranteed coverage.

    Args:
        y_calibration: True values for calibration set
        y_pred_calibration: Predicted values for calibration set
        y_pred_test: Predicted values for test set
        alpha: Significance level (default: 0.05 for 95% CI)

    Returns:
        Tuple of (lower_bounds, upper_bounds) for test predictions

    Example:
        >>> y_cal = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        >>> y_pred_cal = np.array([1.1, 2.2, 2.9, 4.1, 4.8])
        >>> y_pred_test = np.array([3.5, 6.0])
        >>> lower, upper = conformal_prediction_intervals(y_cal, y_pred_cal, y_pred_test)
        >>> print(f"Prediction intervals: [{lower[0]:.2f}, {upper[0]:.2f}]")

    References:
        Angelopoulos & Bates (2021). Conformal Prediction: A Gentle Introduction.
    """
    logger.info(f"Computing conformal prediction intervals (alpha={alpha})")

    # Compute residuals on calibration set
    residuals = np.abs(y_calibration - y_pred_calibration)

    # Find quantile of absolute residuals
    quantile_level = np.ceil((1 - alpha) * (len(residuals) + 1)) / len(residuals)
    quantile_level = min(quantile_level, 1.0)  # Cap at 1.0

    interval_width = np.quantile(residuals, quantile_level)

    # Construct prediction intervals for test set
    lower_bounds = y_pred_test - interval_width
    upper_bounds = y_pred_test + interval_width

    logger.info(f"Conformal interval width: {interval_width:.4f}")

    return lower_bounds, upper_bounds


def regression_uncertainty_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_std: Optional[np.ndarray] = None
) -> Dict:
    """
    Compute uncertainty metrics for regression.

    Args:
        y_true: True values
        y_pred: Predicted values
        y_std: Predicted standard deviations (optional)

    Returns:
        Dictionary with uncertainty metrics

    Example:
        >>> y_true = np.array([1.0, 2.0, 3.0])
        >>> y_pred = np.array([1.1, 2.2, 2.9])
        >>> y_std = np.array([0.2, 0.3, 0.2])
        >>> metrics = regression_uncertainty_metrics(y_true, y_pred, y_std)
    """
    residuals = y_true - y_pred

    metrics = {
        'rmse': float(np.sqrt(np.mean(residuals ** 2))),
        'mae': float(np.mean(np.abs(residuals))),
        'r2': float(1 - np.sum(residuals ** 2) / np.sum((y_true - np.mean(y_true)) ** 2))
    }

    if y_std is not None:
        # Check if true values fall within predicted intervals
        coverage_95 = np.mean(np.abs(residuals) <= 1.96 * y_std)
        metrics['coverage_95'] = float(coverage_95)
        metrics['mean_predicted_std'] = float(np.mean(y_std))

        # Calibration: compare empirical std to predicted std
        metrics['std_calibration_ratio'] = float(np.std(residuals) / np.mean(y_std))

    return metrics


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    print("Uncertainty Calibration Demo")
    print("=" * 60)

    # Classification calibration
    np.random.seed(369)
    y_true_class = np.random.randint(0, 2, 100)
    y_pred_proba = np.random.beta(2, 2, 100)  # Somewhat calibrated

    print("\nClassification Calibration:")
    cal_metrics = compute_calibration_metrics(y_true_class, y_pred_proba, n_bins=9)
    print(f"  - Brier Score: {cal_metrics['brier_score']:.4f}")
    print(f"  - ECE: {cal_metrics['ece']:.4f}")

    # Conformal prediction
    print("\nConformal Prediction Intervals:")
    y_cal = np.random.randn(50)
    y_pred_cal = y_cal + np.random.randn(50) * 0.5
    y_pred_test = np.array([0.5, 1.0, -0.5])

    lower, upper = conformal_prediction_intervals(y_cal, y_pred_cal, y_pred_test)
    print(f"  - Test predictions with 95% intervals:")
    for i, (pred, lo, hi) in enumerate(zip(y_pred_test, lower, upper)):
        print(f"    {i+1}. {pred:.2f} [{lo:.2f}, {hi:.2f}]")

    print("\n✓ Uncertainty calibration demo complete")
