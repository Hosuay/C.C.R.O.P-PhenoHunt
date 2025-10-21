"""
Scientific Effect Prediction Models
Based on peer-reviewed cannabis pharmacology research
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, precision_recall_curve
import logging

logger = logging.getLogger(__name__)


class EntourageEffectModel:
    """
    Models the entourage effect - synergistic interactions between compounds.

    The entourage effect is the hypothesis that cannabis compounds work synergistically
    to produce effects different from isolated compounds.

    References:
        - Russo EB. (2011). Taming THC: potential cannabis synergy. Br J Pharmacol.
        - Ferber et al. (2020). The "Entourage Effect": Terpenes Coupled with
          Cannabinoids for the Treatment of Mood Disorders. Curr Neuropharmacol.
    """

    def __init__(self, config: Dict):
        self.config = config
        self.interaction_features = []

    def create_interaction_terms(
        self,
        df: pd.DataFrame,
        effect_name: str
    ) -> pd.DataFrame:
        """
        Create polynomial interaction features for synergistic effects.

        Args:
            df: DataFrame with chemical profiles
            effect_name: Name of effect to model

        Returns:
            DataFrame with original features + interaction terms
        """
        df_enhanced = df.copy()

        # Get interaction terms from config
        effect_config = self.config['effects'][effect_name]

        if 'interaction_terms' in effect_config:
            for interaction in effect_config['interaction_terms']:
                compounds = interaction['compounds']
                interaction_type = interaction['interaction_type']

                # Create column names
                col1 = f"{compounds[0]}_pct"
                col2 = f"{compounds[1]}_pct"

                if col1 in df.columns and col2 in df.columns:
                    # Create interaction term
                    if interaction_type == 'synergistic':
                        # Multiplicative interaction
                        feature_name = f"{compounds[0]}_{compounds[1]}_interaction"
                        df_enhanced[feature_name] = df[col1] * df[col2]
                        self.interaction_features.append(feature_name)

                        logger.info(
                            f"Created synergistic interaction: {compounds[0]} × {compounds[1]}"
                        )

        return df_enhanced

    def calculate_effect_score(
        self,
        profile: pd.Series,
        effect_name: str
    ) -> Tuple[float, Dict[str, float]]:
        """
        Calculate research-backed effect score.

        Uses weighted linear combination based on published research.

        Args:
            profile: Chemical profile (pd.Series)
            effect_name: Name of therapeutic effect

        Returns:
            Tuple of (total_score, contribution_breakdown)
        """
        effect_config = self.config['effects'][effect_name]
        contributions = {}
        total_score = 0.0

        # Primary compound contributions
        for compound_info in effect_config['primary_compounds']:
            compound = compound_info['compound']
            coefficient = compound_info['coefficient']
            col_name = f"{compound}_pct"

            if col_name in profile.index:
                contribution = profile[col_name] * coefficient
                contributions[compound] = contribution
                total_score += contribution

        # Interaction term contributions
        if 'interaction_terms' in effect_config:
            for interaction in effect_config['interaction_terms']:
                compounds = interaction['compounds']
                coefficient = interaction['coefficient']

                col1 = f"{compounds[0]}_pct"
                col2 = f"{compounds[1]}_pct"

                if col1 in profile.index and col2 in profile.index:
                    interaction_contribution = (
                        profile[col1] * profile[col2] * coefficient
                    )
                    interaction_name = f"{compounds[0]}×{compounds[1]}"
                    contributions[interaction_name] = interaction_contribution
                    total_score += interaction_contribution

        return total_score, contributions


class EnsembleEffectPredictor:
    """
    Ensemble model for therapeutic effect prediction with uncertainty quantification.

    Combines multiple classifiers with different inductive biases for robust predictions.
    """

    def __init__(self, config: Dict, effect_name: str):
        self.config = config
        self.effect_name = effect_name
        self.effect_config = config['effects'][effect_name]

        # Ensemble components
        self.models = {
            'logistic': LogisticRegression(
                max_iter=1000,
                random_state=config['reproducibility']['random_seed']
            ),
            'random_forest': RandomForestClassifier(
                n_estimators=100,
                max_depth=5,
                random_state=config['reproducibility']['random_seed']
            ),
            'gradient_boosting': GradientBoostingClassifier(
                n_estimators=100,
                max_depth=3,
                random_state=config['reproducibility']['random_seed']
            )
        }

        self.scaler = StandardScaler()
        self.entourage_model = EntourageEffectModel(config)
        self.is_fitted = False

    def fit(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        validate: bool = True
    ) -> Dict[str, float]:
        """
        Fit ensemble model with cross-validation.

        Args:
            X: Feature matrix
            y: Target labels
            validate: Whether to perform cross-validation

        Returns:
            Dictionary of validation metrics
        """
        # Create interaction features
        X_enhanced = self.entourage_model.create_interaction_terms(
            X,
            self.effect_name
        )

        # Get feature columns
        feature_cols = [col for col in X_enhanced.columns if col.endswith('_pct') or 'interaction' in col]
        X_features = X_enhanced[feature_cols].fillna(0)

        # Scale features
        X_scaled = self.scaler.fit_transform(X_features)

        metrics = {}

        # Fit each model
        for model_name, model in self.models.items():
            model.fit(X_scaled, y)

            if validate and len(np.unique(y)) > 1:
                # Cross-validation
                cv = StratifiedKFold(
                    n_splits=min(5, len(y) // 2),
                    shuffle=True,
                    random_state=self.config['reproducibility']['random_seed']
                )

                cv_scores = cross_val_score(
                    model,
                    X_scaled,
                    y,
                    cv=cv,
                    scoring='roc_auc'
                )

                metrics[f'{model_name}_cv_auc'] = cv_scores.mean()
                metrics[f'{model_name}_cv_std'] = cv_scores.std()

                logger.info(
                    f"{model_name} - CV AUC: {cv_scores.mean():.3f} "
                    f"(±{cv_scores.std():.3f})"
                )

        self.is_fitted = True
        self.feature_cols = feature_cols

        return metrics

    def predict_proba(
        self,
        X: pd.DataFrame,
        return_uncertainty: bool = True
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Predict effect probability with uncertainty estimation.

        Args:
            X: Feature matrix
            return_uncertainty: Whether to return prediction uncertainty

        Returns:
            Tuple of (mean_probabilities, std_probabilities)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")

        # Create interaction features
        X_enhanced = self.entourage_model.create_interaction_terms(
            X,
            self.effect_name
        )

        X_features = X_enhanced[self.feature_cols].fillna(0)
        X_scaled = self.scaler.transform(X_features)

        # Get predictions from each model
        predictions = []
        for model in self.models.values():
            pred_proba = model.predict_proba(X_scaled)[:, 1]
            predictions.append(pred_proba)

        predictions = np.array(predictions)

        # Ensemble: average probabilities
        mean_proba = predictions.mean(axis=0)

        if return_uncertainty:
            # Uncertainty: standard deviation across models
            std_proba = predictions.std(axis=0)
            return mean_proba, std_proba
        else:
            return mean_proba, None

    def predict_with_confidence(
        self,
        X: pd.DataFrame,
        confidence_level: float = 0.95
    ) -> pd.DataFrame:
        """
        Predict with confidence intervals.

        Args:
            X: Feature matrix
            confidence_level: Confidence level (default 95%)

        Returns:
            DataFrame with predictions and confidence intervals
        """
        mean_proba, std_proba = self.predict_proba(X, return_uncertainty=True)

        # Calculate confidence intervals (assumes normal distribution)
        z_score = 1.96  # For 95% confidence
        if confidence_level != 0.95:
            from scipy import stats
            z_score = stats.norm.ppf((1 + confidence_level) / 2)

        lower_bound = np.clip(mean_proba - z_score * std_proba, 0, 1)
        upper_bound = np.clip(mean_proba + z_score * std_proba, 0, 1)

        results = pd.DataFrame({
            'effect': self.effect_name,
            'probability': mean_proba,
            'uncertainty': std_proba,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound,
            'confidence_level': confidence_level
        })

        # Binary prediction based on threshold
        threshold = self.effect_config['threshold_probability']
        required_confidence = self.effect_config['confidence_requirement']

        # Only predict positive if probability > threshold AND confidence is high
        results['prediction'] = (
            (results['probability'] > threshold) &
            (results['uncertainty'] < (1 - required_confidence))
        ).astype(int)

        results['confidence_met'] = results['uncertainty'] < (1 - required_confidence)

        return results

    def explain_prediction(
        self,
        profile: pd.Series,
        top_n: int = 5
    ) -> Dict:
        """
        Explain prediction using feature importance.

        Args:
            profile: Single chemical profile
            top_n: Number of top contributing features to return

        Returns:
            Dictionary with explanation
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before explanation")

        # Get prediction
        X_df = pd.DataFrame([profile])
        results = self.predict_with_confidence(X_df)

        # Get mechanistic score from entourage model
        mechanistic_score, contributions = self.entourage_model.calculate_effect_score(
            profile,
            self.effect_name
        )

        # Sort contributions
        sorted_contributions = sorted(
            contributions.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )[:top_n]

        # Get research evidence for top contributors
        evidence = {}
        for compound_info in self.effect_config['primary_compounds']:
            compound = compound_info['compound']
            if compound in [c[0] for c in sorted_contributions]:
                evidence[compound] = compound_info['evidence']

        explanation = {
            'effect': self.effect_name,
            'probability': results['probability'].iloc[0],
            'uncertainty': results['uncertainty'].iloc[0],
            'prediction': bool(results['prediction'].iloc[0]),
            'confidence_met': bool(results['confidence_met'].iloc[0]),
            'mechanistic_score': mechanistic_score,
            'top_contributors': sorted_contributions,
            'scientific_evidence': evidence
        }

        return explanation


class MultiEffectPredictor:
    """
    Predict multiple therapeutic effects simultaneously.
    """

    def __init__(self, config: Dict):
        self.config = config
        self.predictors = {}

        # Initialize predictor for each effect
        for effect_name in config['effects'].keys():
            self.predictors[effect_name] = EnsembleEffectPredictor(
                config,
                effect_name
            )

    def fit(self, X: pd.DataFrame, y_dict: Dict[str, np.ndarray]) -> Dict:
        """
        Fit all effect predictors.

        Args:
            X: Feature matrix
            y_dict: Dictionary mapping effect names to target arrays

        Returns:
            Dictionary of all validation metrics
        """
        all_metrics = {}

        for effect_name, predictor in self.predictors.items():
            if effect_name in y_dict:
                logger.info(f"\nTraining predictor for: {effect_name}")
                metrics = predictor.fit(X, y_dict[effect_name])
                all_metrics[effect_name] = metrics

        return all_metrics

    def predict_all_effects(
        self,
        X: pd.DataFrame,
        confidence_level: float = 0.95
    ) -> pd.DataFrame:
        """
        Predict all therapeutic effects with confidence intervals.

        Args:
            X: Feature matrix
            confidence_level: Confidence level for intervals

        Returns:
            DataFrame with all effect predictions
        """
        all_results = []

        for effect_name, predictor in self.predictors.items():
            if predictor.is_fitted:
                results = predictor.predict_with_confidence(X, confidence_level)
                all_results.append(results)

        if all_results:
            return pd.concat(all_results, ignore_index=True)
        else:
            return pd.DataFrame()

    def create_effect_profile(
        self,
        profile: pd.Series
    ) -> Dict:
        """
        Create comprehensive effect profile for a strain.

        Args:
            profile: Chemical profile

        Returns:
            Dictionary with all effect predictions and explanations
        """
        effect_profile = {
            'strain': profile.get('strain_name', 'Unknown'),
            'effects': {}
        }

        for effect_name, predictor in self.predictors.items():
            if predictor.is_fitted:
                explanation = predictor.explain_prediction(profile)
                effect_profile['effects'][effect_name] = explanation

        return effect_profile
