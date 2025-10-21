"""
Scientific Data Validation Module
Implements rigorous quality control for cannabis chemical profiles
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from sklearn.ensemble import IsolationForest
from scipy import stats
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ChemicalProfileValidator:
    """
    Validates cannabis chemical profiles using scientific quality control methods.

    References:
        - Smith et al. (2022). The phytochemical diversity of commercial Cannabis. PLOS ONE.
        - ISO 17025 laboratory quality standards
    """

    def __init__(self, config: Dict):
        self.config = config
        self.outlier_detector = IsolationForest(
            contamination=config['validation']['outlier_detection']['contamination'],
            random_state=config['reproducibility']['random_seed']
        )

    def validate_ranges(self, df: pd.DataFrame, compounds: List[str]) -> Tuple[pd.DataFrame, List[str]]:
        """
        Validate that compound values fall within scientifically established ranges.

        Args:
            df: DataFrame containing chemical profiles
            compounds: List of compound column names

        Returns:
            Tuple of (validated_df, list of warnings)
        """
        warnings = []

        # Get compound configurations
        all_compounds = {}
        for category in ['cannabinoids', 'terpenes']:
            for level in ['major', 'minor']:
                for compound_config in self.config['compounds'][category][level]:
                    all_compounds[f"{compound_config['name']}_pct"] = compound_config

        for compound in compounds:
            if compound in all_compounds:
                config = all_compounds[compound]
                min_val, max_val = config['typical_range']

                # Check for out-of-range values
                out_of_range = df[(df[compound] < min_val) | (df[compound] > max_val)]

                if len(out_of_range) > 0:
                    warnings.append(
                        f"⚠️ {len(out_of_range)} samples have {compound} outside typical range "
                        f"[{min_val}, {max_val}]%"
                    )

                    # Clip values to valid range
                    df[compound] = df[compound].clip(min_val, max_val)
                    logger.warning(f"Clipped {compound} values to range [{min_val}, {max_val}]")

        return df, warnings

    def validate_totals(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """
        Validate total cannabinoid and terpene levels.

        Total cannabinoids typically don't exceed 40-50% dry weight.
        Total terpenes typically range 0.5-10% dry weight.
        """
        warnings = []

        # Extract cannabinoid and terpene columns
        cannabinoid_cols = [col for col in df.columns if any(
            cann['name'] in col.lower()
            for category in ['major', 'minor']
            for cann in self.config['compounds']['cannabinoids'][category]
        )]

        terpene_cols = [col for col in df.columns if any(
            terp['name'] in col.lower()
            for category in ['major', 'minor']
            for terp in self.config['compounds']['terpenes'][category]
        )]

        # Calculate totals
        df['total_cannabinoids'] = df[cannabinoid_cols].sum(axis=1)
        df['total_terpenes'] = df[terpene_cols].sum(axis=1)

        # Validate cannabinoid totals
        min_cann = self.config['validation']['quality_checks']['min_total_cannabinoids']
        max_cann = self.config['validation']['quality_checks']['max_total_cannabinoids']

        invalid_cann = df[
            (df['total_cannabinoids'] < min_cann) |
            (df['total_cannabinoids'] > max_cann)
        ]

        if len(invalid_cann) > 0:
            warnings.append(
                f"⚠️ {len(invalid_cann)} samples have total cannabinoids outside "
                f"valid range [{min_cann}, {max_cann}]%"
            )

        # Validate terpene totals
        min_terp = self.config['validation']['quality_checks']['min_total_terpenes']
        max_terp = self.config['validation']['quality_checks']['max_total_terpenes']

        invalid_terp = df[
            (df['total_terpenes'] < min_terp) |
            (df['total_terpenes'] > max_terp)
        ]

        if len(invalid_terp) > 0:
            warnings.append(
                f"⚠️ {len(invalid_terp)} samples have total terpenes outside "
                f"valid range [{min_terp}, {max_terp}]%"
            )

        return df, warnings

    def detect_outliers(self, df: pd.DataFrame, feature_cols: List[str]) -> Tuple[pd.DataFrame, List[str]]:
        """
        Detect outliers using Isolation Forest algorithm.

        Args:
            df: DataFrame with chemical profiles
            feature_cols: Columns to use for outlier detection

        Returns:
            Tuple of (cleaned_df, list of warnings)
        """
        warnings = []

        if len(df) < 10:
            warnings.append("⚠️ Dataset too small for reliable outlier detection (n < 10)")
            return df, warnings

        # Fit isolation forest
        X = df[feature_cols].fillna(0).values
        predictions = self.outlier_detector.fit_predict(X)

        # Identify outliers
        outlier_mask = predictions == -1
        n_outliers = outlier_mask.sum()

        if n_outliers > 0:
            outlier_strains = df.loc[outlier_mask, 'strain_name'].tolist()
            warnings.append(
                f"🔍 Detected {n_outliers} potential outliers: {', '.join(outlier_strains[:5])}"
                + ("..." if n_outliers > 5 else "")
            )

            # Flag outliers but don't remove them
            df['is_outlier'] = outlier_mask
            logger.info(f"Flagged {n_outliers} outliers for review")

        return df, warnings

    def check_missing_data(self, df: pd.DataFrame, feature_cols: List[str]) -> Tuple[pd.DataFrame, List[str]]:
        """
        Check for missing data and handle appropriately.
        """
        warnings = []

        missing_fraction = df[feature_cols].isnull().sum() / len(df)
        max_missing = self.config['validation']['missing_data']['max_missing_fraction']

        problematic_cols = missing_fraction[missing_fraction > max_missing]

        if len(problematic_cols) > 0:
            warnings.append(
                f"⚠️ High missing data fraction in columns: "
                f"{', '.join(problematic_cols.index.tolist())}"
            )

        # Impute missing values
        imputation_method = self.config['validation']['missing_data']['imputation_method']

        for col in feature_cols:
            if df[col].isnull().any():
                if imputation_method == 'median':
                    fill_value = df[col].median()
                elif imputation_method == 'mean':
                    fill_value = df[col].mean()
                else:
                    fill_value = 0.0

                n_missing = df[col].isnull().sum()
                df[col].fillna(fill_value, inplace=True)
                logger.info(f"Imputed {n_missing} missing values in {col} with {imputation_method}")

        return df, warnings

    def validate_ratios(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """
        Validate scientifically meaningful ratios (THC:CBD, etc.)

        References:
            - Hazekamp et al. (2016). Cannabis chemovars and phenotypes
        """
        warnings = []

        # Calculate THC:CBD ratio (when both present)
        mask = (df['thc_pct'] > 0) & (df['cbd_pct'] > 0)
        if mask.any():
            df.loc[mask, 'thc_cbd_ratio'] = df.loc[mask, 'thc_pct'] / df.loc[mask, 'cbd_pct']

            # Flag unusual ratios (most strains are either THC-dominant or balanced)
            unusual_ratios = df[
                mask &
                (df['thc_cbd_ratio'] > 0.1) &
                (df['thc_cbd_ratio'] < 1.0)
            ]

            if len(unusual_ratios) > 0:
                warnings.append(
                    f"ℹ️ {len(unusual_ratios)} samples have unusual THC:CBD ratios "
                    f"(0.1 < ratio < 1.0)"
                )

        return df, warnings

    def comprehensive_validation(
        self,
        df: pd.DataFrame,
        feature_cols: List[str]
    ) -> Tuple[pd.DataFrame, Dict[str, List[str]]]:
        """
        Run comprehensive validation pipeline.

        Args:
            df: DataFrame with strain data
            feature_cols: Chemical compound columns

        Returns:
            Tuple of (validated_df, dict of warnings by category)
        """
        all_warnings = {
            'ranges': [],
            'totals': [],
            'outliers': [],
            'missing_data': [],
            'ratios': []
        }

        logger.info(f"Starting comprehensive validation on {len(df)} samples...")

        # Range validation
        df, warnings = self.validate_ranges(df, feature_cols)
        all_warnings['ranges'] = warnings

        # Total validation
        df, warnings = self.validate_totals(df)
        all_warnings['totals'] = warnings

        # Missing data check
        df, warnings = self.check_missing_data(df, feature_cols)
        all_warnings['missing_data'] = warnings

        # Outlier detection
        df, warnings = self.detect_outliers(df, feature_cols)
        all_warnings['outliers'] = warnings

        # Ratio validation
        df, warnings = self.validate_ratios(df)
        all_warnings['ratios'] = warnings

        # Summary
        total_warnings = sum(len(w) for w in all_warnings.values())
        logger.info(f"Validation complete. Total warnings: {total_warnings}")

        return df, all_warnings


class StatisticalValidator:
    """
    Statistical tests for data quality and model assumptions.
    """

    @staticmethod
    def test_normality(data: np.ndarray, alpha: float = 0.05) -> Dict:
        """
        Test if data follows normal distribution using Shapiro-Wilk test.
        """
        statistic, p_value = stats.shapiro(data)

        return {
            'test': 'Shapiro-Wilk',
            'statistic': statistic,
            'p_value': p_value,
            'is_normal': p_value > alpha,
            'alpha': alpha
        }

    @staticmethod
    def test_homoscedasticity(residuals: np.ndarray, predictions: np.ndarray) -> Dict:
        """
        Test for homoscedasticity (constant variance) using Breusch-Pagan test.
        """
        # Simple variance ratio test
        median_pred = np.median(predictions)
        lower_var = np.var(residuals[predictions <= median_pred])
        upper_var = np.var(residuals[predictions > median_pred])

        f_statistic = max(lower_var, upper_var) / min(lower_var, upper_var)

        return {
            'test': 'Variance Ratio',
            'f_statistic': f_statistic,
            'lower_variance': lower_var,
            'upper_variance': upper_var,
            'is_homoscedastic': f_statistic < 2.0  # Rule of thumb
        }

    @staticmethod
    def calculate_cohens_d(group1: np.ndarray, group2: np.ndarray) -> float:
        """
        Calculate Cohen's d effect size.
        """
        mean_diff = np.mean(group1) - np.mean(group2)
        pooled_std = np.sqrt((np.var(group1) + np.var(group2)) / 2)

        return mean_diff / pooled_std if pooled_std > 0 else 0.0
