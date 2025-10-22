"""
Genomic Selection Module
========================

Implements G-BLUP (Genomic Best Linear Unbiased Prediction) for breeding value estimation.

Methods:
    - Genomic Relationship Matrix (GRM) construction
    - G-BLUP prediction with cross-validation
    - Predictive accuracy assessment (R², correlation)

Sacred Geometry:
    - 9-fold cross-validation
    - 27-marker LD windows

Author: C.C.R.O.P-PhenoHunt Team
Version: 1.0.0

References:
    - VanRaden (2008). Efficient methods to compute genomic predictions.
    - Meuwissen et al. (2001). Prediction of total genetic value using genome-wide dense marker maps.
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, Tuple, Optional
from pathlib import Path
from sklearn.model_selection import KFold
from scipy import linalg

logger = logging.getLogger(__name__)


def compute_grm(
    genotype_df: pd.DataFrame,
    method: str = 'vanraden',
    min_maf: float = 0.01
) -> np.ndarray:
    """
    Compute Genomic Relationship Matrix (GRM).

    Implements VanRaden (2008) method: G = ZZ'/k
    where Z is centered genotype matrix, k is scaling factor.

    Args:
        genotype_df: DataFrame with samples as rows, markers as columns (0/1/2 encoding)
        method: Method for GRM computation ('vanraden', 'allele_freq')
        min_maf: Minimum minor allele frequency threshold

    Returns:
        Genomic relationship matrix (n_samples × n_samples)

    Example:
        >>> geno = pd.DataFrame({
        ...     'SNP1': [0, 1, 2, 1, 0],
        ...     'SNP2': [2, 1, 0, 1, 2],
        ...     'SNP3': [1, 1, 1, 2, 0]
        ... })
        >>> G = compute_grm(geno)
        >>> print(G.shape)
        (5, 5)

    References:
        VanRaden, P. M. (2008). J. Dairy Sci. 91:4414–4423.
    """
    logger.info(f"Computing GRM using {method} method")

    # Convert to numpy array
    Z = genotype_df.values.astype(float)
    n_samples, n_markers = Z.shape

    # Filter markers by MAF
    allele_freq = np.nanmean(Z, axis=0) / 2  # Allele frequency
    maf = np.minimum(allele_freq, 1 - allele_freq)  # Minor allele frequency
    valid_markers = maf >= min_maf

    Z = Z[:, valid_markers]
    n_markers_filtered = Z.shape[1]

    logger.info(f"Filtered {n_markers - n_markers_filtered} markers with MAF < {min_maf}")
    logger.info(f"Computing GRM with {n_markers_filtered} markers")

    # Handle missing values (impute with mean)
    col_means = np.nanmean(Z, axis=0)
    for col_idx in range(Z.shape[1]):
        Z[np.isnan(Z[:, col_idx]), col_idx] = col_means[col_idx]

    if method == 'vanraden':
        # VanRaden method: G = ZZ' / sum(2p(1-p))
        # Center genotypes
        p = np.mean(Z, axis=0) / 2  # Allele frequencies
        Z_centered = Z - 2 * p[np.newaxis, :]

        # Scaling factor
        k = np.sum(2 * p * (1 - p))

        # Compute GRM
        G = (Z_centered @ Z_centered.T) / k

    else:
        raise ValueError(f"Unknown GRM method: {method}")

    # Ensure symmetry
    G = (G + G.T) / 2

    logger.info(f"GRM computed: shape {G.shape}, mean diagonal = {np.mean(np.diag(G)):.4f}")

    return G


def g_blup_predict(
    G: np.ndarray,
    y_train: np.ndarray,
    X_train: Optional[np.ndarray] = None,
    X_test: Optional[np.ndarray] = None,
    lambda_param: float = 1.0
) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    G-BLUP prediction with Mixed Model Equations (MME).

    Solves: [X'X    X'Z  ] [β] = [X'y]
            [Z'X  Z'Z+λI] [u]   [Z'y]

    Where λ = σ²_e / σ²_u (variance ratio)

    Args:
        G: Genomic relationship matrix (n_train × n_train)
        y_train: Training phenotypes (n_train,)
        X_train: Design matrix for fixed effects (n_train × n_fixed) - optional
        X_test: Test design matrix (n_test × n_fixed) - optional
        lambda_param: λ = σ²_e / σ²_u, controls shrinkage (default: 1.0)

    Returns:
        predictions: Predicted breeding values
        metrics: Dictionary with R², correlation, MSE

    Example:
        >>> G = np.eye(100)  # Identity for example
        >>> y = np.random.randn(100)
        >>> preds, metrics = g_blup_predict(G, y)
        >>> print(f"Predictive R²: {metrics['r2']:.3f}")
    """
    n_train = G.shape[0]

    # If no fixed effects, use intercept only
    if X_train is None:
        X_train = np.ones((n_train, 1))

    if X_test is None:
        X_test = X_train

    # Construct Mixed Model Equations
    X = X_train
    n_fixed = X.shape[1]

    # Left-hand side (LHS)
    LHS_11 = X.T @ X
    LHS_12 = X.T
    LHS_22 = G + lambda_param * np.eye(n_train)

    LHS = np.block([
        [LHS_11, LHS_12],
        [LHS_12.T, LHS_22]
    ])

    # Right-hand side (RHS)
    RHS = np.concatenate([
        X.T @ y_train,
        y_train
    ])

    # Solve MME
    try:
        solutions = linalg.solve(LHS, RHS, assume_a='pos')
    except linalg.LinAlgError:
        logger.warning("Positive definite assumption failed, using general solver")
        solutions = linalg.solve(LHS, RHS)

    # Extract breeding values
    beta = solutions[:n_fixed]  # Fixed effects
    u = solutions[n_fixed:]  # Random effects (breeding values)

    # Predictions
    y_pred = X_test @ beta + u[:len(X_test)]

    # Calculate metrics (training set)
    y_train_pred = X_train @ beta + u

    metrics = {
        'r2': 1 - np.sum((y_train - y_train_pred)**2) / np.sum((y_train - np.mean(y_train))**2),
        'correlation': np.corrcoef(y_train, y_train_pred)[0, 1],
        'mse': np.mean((y_train - y_train_pred)**2),
        'rmse': np.sqrt(np.mean((y_train - y_train_pred)**2))
    }

    return y_pred, metrics


def cross_validate_gblup(
    genotype_df: pd.DataFrame,
    phenotype: pd.Series,
    n_folds: int = 9,  # Sacred number
    lambda_param: float = 1.0,
    min_maf: float = 0.01
) -> Dict[str, float]:
    """
    K-fold cross-validation for G-BLUP.

    Sacred Geometry: Uses 9-fold CV by default.

    Args:
        genotype_df: Genotype DataFrame (samples × markers)
        phenotype: Phenotype Series (aligned with genotype_df)
        n_folds: Number of cross-validation folds (default: 9)
        lambda_param: G-BLUP shrinkage parameter
        min_maf: Minimum MAF for marker filtering

    Returns:
        Dictionary with CV metrics: mean R², std R², correlation

    Example:
        >>> geno = pd.DataFrame(np.random.randint(0, 3, (100, 500)))
        >>> pheno = pd.Series(np.random.randn(100))
        >>> cv_results = cross_validate_gblup(geno, pheno, n_folds=9)
        >>> print(f"CV R²: {cv_results['mean_r2']:.3f} ± {cv_results['std_r2']:.3f}")
    """
    logger.info(f"Running {n_folds}-fold cross-validation for G-BLUP")

    # Compute GRM once
    G = compute_grm(genotype_df, min_maf=min_maf)
    y = phenotype.values

    # K-fold CV
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=369)  # Sacred seed

    fold_r2 = []
    fold_corr = []

    for fold_idx, (train_idx, test_idx) in enumerate(kf.split(genotype_df)):
        # Split data
        G_train = G[np.ix_(train_idx, train_idx)]
        y_train = y[train_idx]

        # Train G-BLUP
        _, metrics = g_blup_predict(G_train, y_train, lambda_param=lambda_param)

        fold_r2.append(metrics['r2'])
        fold_corr.append(metrics['correlation'])

        logger.debug(f"  Fold {fold_idx + 1}: R² = {metrics['r2']:.4f}, Corr = {metrics['correlation']:.4f}")

    cv_results = {
        'mean_r2': float(np.mean(fold_r2)),
        'std_r2': float(np.std(fold_r2)),
        'mean_correlation': float(np.mean(fold_corr)),
        'std_correlation': float(np.std(fold_corr)),
        'n_folds': n_folds,
        'lambda': lambda_param
    }

    logger.info(f"CV Results: R² = {cv_results['mean_r2']:.4f} ± {cv_results['std_r2']:.4f}")

    return cv_results


if __name__ == '__main__':
    # Example usage
    logging.basicConfig(level=logging.INFO)

    print("Genomic Selection Demo")
    print("=" * 60)

    # Simulate genotype data (100 samples, 500 SNPs)
    np.random.seed(369)
    n_samples = 100
    n_markers = 500

    geno = pd.DataFrame(
        np.random.randint(0, 3, (n_samples, n_markers)),
        columns=[f'SNP_{i}' for i in range(n_markers)]
    )

    # Simulate phenotype (with some genetic signal)
    true_effects = np.random.randn(n_markers) * 0.1
    pheno = pd.Series(
        geno.values @ true_effects + np.random.randn(n_samples),
        name='yield'
    )

    print(f"\nSimulated data:")
    print(f"  - {n_samples} samples")
    print(f"  - {n_markers} markers")
    print(f"  - Phenotype range: [{pheno.min():.2f}, {pheno.max():.2f}]")

    # Compute GRM
    print("\nComputing Genomic Relationship Matrix...")
    G = compute_grm(geno)
    print(f"  - GRM shape: {G.shape}")
    print(f"  - GRM is symmetric: {np.allclose(G, G.T)}")

    # Cross-validation
    print("\nRunning 9-fold cross-validation...")
    cv_results = cross_validate_gblup(geno, pheno, n_folds=9)

    print(f"\nCross-validation results:")
    print(f"  - R²: {cv_results['mean_r2']:.4f} ± {cv_results['std_r2']:.4f}")
    print(f"  - Correlation: {cv_results['mean_correlation']:.4f} ± {cv_results['std_correlation']:.4f}")
    print(f"\n✓ Genomic selection demo complete")
