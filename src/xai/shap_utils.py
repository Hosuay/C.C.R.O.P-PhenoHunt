"""
SHAP Utilities for Explainable AI
==================================

Provides SHAP (SHapley Additive exPlanations) analysis for model interpretability.

Sacred Geometry:
    - Top 9 features displayed by default
    - 27 features for extended analysis

Author: C.C.R.O.P-PhenoHunt Team
Version: 1.0.0

References:
    - Lundberg & Lee (2017). A Unified Approach to Interpreting Model Predictions.
"""

import numpy as np
import pandas as pd
import logging
import json
from typing import Dict, List, Optional
from pathlib import Path

logger = logging.getLogger(__name__)

# Try to import SHAP
try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False
    logger.warning("SHAP not available. Install with: pip install shap")


def explain_predictions(
    model,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    top_k: int = 9  # Sacred number
) -> Dict:
    """
    Generate SHAP explanations for model predictions.

    Args:
        model: Trained sklearn-compatible model
        X_train: Training features (for background)
        X_test: Test features to explain
        top_k: Number of top features to return (default: 9)

    Returns:
        Dictionary with SHAP values and feature importance

    Example:
        >>> from sklearn.ensemble import RandomForestRegressor
        >>> model = RandomForestRegressor().fit(X_train, y_train)
        >>> explanations = explain_predictions(model, X_train, X_test)
    """
    if not HAS_SHAP:
        logger.warning("SHAP not available, using fallback feature importance")
        return _fallback_feature_importance(model, X_train.columns, top_k)

    logger.info(f"Computing SHAP values for {len(X_test)} samples")

    # Create explainer
    explainer = shap.TreeExplainer(model)  # For tree-based models

    # Compute SHAP values
    shap_values = explainer.shap_values(X_test)

    # Calculate global feature importance
    mean_abs_shap = np.mean(np.abs(shap_values), axis=0)

    # Get top features
    top_indices = np.argsort(mean_abs_shap)[::-1][:top_k]

    results = {
        'shap_values': shap_values.tolist() if isinstance(shap_values, np.ndarray) else shap_values,
        'feature_names': X_test.columns.tolist(),
        'top_features': {
            X_test.columns[idx]: float(mean_abs_shap[idx])
            for idx in top_indices
        },
        'mean_abs_shap': mean_abs_shap.tolist()
    }

    logger.info(f"Top {top_k} features: {list(results['top_features'].keys())}")

    return results


def _fallback_feature_importance(model, feature_names: List[str], top_k: int) -> Dict:
    """Fallback to built-in feature importance if SHAP unavailable."""
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    elif hasattr(model, 'coef_'):
        importances = np.abs(model.coef_)
    else:
        logger.warning("Model has no feature importance, returning empty")
        return {'top_features': {}, 'method': 'unavailable'}

    top_indices = np.argsort(importances)[::-1][:top_k]

    return {
        'top_features': {
            feature_names[idx]: float(importances[idx])
            for idx in top_indices
        },
        'method': 'model_builtin',
        'all_importances': importances.tolist()
    }


def export_shap_summary(
    shap_results: Dict,
    output_path: Path,
    format: str = 'json'
) -> None:
    """
    Export SHAP results to file.

    Args:
        shap_results: Results from explain_predictions()
        output_path: Output file path
        format: 'json' or 'csv'
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if format == 'json':
        # Remove full shap_values array for cleaner JSON
        export_data = {k: v for k, v in shap_results.items() if k != 'shap_values'}
        with open(output_path, 'w') as f:
            json.dump(export_data, f, indent=2)

    elif format == 'csv':
        df = pd.DataFrame({
            'feature': list(shap_results['top_features'].keys()),
            'importance': list(shap_results['top_features'].values())
        })
        df.to_csv(output_path, index=False)

    logger.info(f"SHAP summary exported to {output_path}")


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    print("SHAP Utils Demo")
    print("=" * 60)

    from sklearn.ensemble import RandomForestRegressor
    from sklearn.datasets import make_regression

    X, y = make_regression(n_samples=100, n_features=20, random_state=369)
    X_train = pd.DataFrame(X[:80], columns=[f'feature_{i}' for i in range(20)])
    X_test = pd.DataFrame(X[80:], columns=[f'feature_{i}' for i in range(20)])
    y_train = y[:80]

    model = RandomForestRegressor(random_state=369).fit(X_train, y_train)

    results = explain_predictions(model, X_train, X_test, top_k=9)

    print(f"\nTop 9 features by importance:")
    for feat, imp in results['top_features'].items():
        print(f"  {feat}: {imp:.4f}")

    print("\n✓ SHAP utils demo complete")
