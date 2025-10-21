"""
Unit Tests for Scientific Improvements
Tests validation, models, and breeding strategies
"""

import pytest
import numpy as np
import pandas as pd
import torch
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.validators import ChemicalProfileValidator, StatisticalValidator
from models.vae import VariationalAutoencoder, VAETrainer
from models.effect_predictor import EntourageEffectModel, EnsembleEffectPredictor
from utils.config import ConfigLoader


@pytest.fixture
def sample_config():
    """Load sample configuration."""
    config_path = Path(__file__).parent.parent.parent / "configs" / "scientific_config.yaml"
    return ConfigLoader.load(str(config_path))


@pytest.fixture
def sample_strain_data():
    """Create sample strain data for testing."""
    data = {
        'strain_name': ['TestStrain1', 'TestStrain2', 'TestStrain3'],
        'type': ['hybrid', 'indica', 'sativa'],
        'thc_pct': [20.0, 22.0, 18.0],
        'cbd_pct': [0.5, 1.0, 0.3],
        'cbg_pct': [0.8, 0.6, 0.9],
        'cbc_pct': [0.3, 0.2, 0.25],
        'cbda_pct': [0.2, 0.15, 0.18],
        'myrcene_pct': [1.5, 2.0, 1.2],
        'limonene_pct': [1.2, 0.8, 1.8],
        'pinene_pct': [0.9, 0.7, 1.3],
        'linalool_pct': [0.6, 0.8, 0.4],
        'caryophyllene_pct': [1.3, 1.5, 1.1],
        'humulene_pct': [0.4, 0.5, 0.35]
    }
    return pd.DataFrame(data)


class TestChemicalProfileValidator:
    """Test data validation functionality."""

    def test_range_validation(self, sample_config, sample_strain_data):
        """Test that range validation clips values correctly."""
        validator = ChemicalProfileValidator(sample_config)

        # Add an out-of-range value
        df = sample_strain_data.copy()
        df.loc[0, 'thc_pct'] = 60.0  # Exceeds typical range

        feature_cols = [col for col in df.columns if col.endswith('_pct')]
        validated_df, warnings = validator.validate_ranges(df, feature_cols)

        # Check that value was clipped
        assert validated_df.loc[0, 'thc_pct'] <= 35.0
        assert len(warnings) > 0

    def test_total_validation(self, sample_config, sample_strain_data):
        """Test total cannabinoid and terpene validation."""
        validator = ChemicalProfileValidator(sample_config)

        validated_df, warnings = validator.validate_totals(sample_strain_data)

        # Check that totals were calculated
        assert 'total_cannabinoids' in validated_df.columns
        assert 'total_terpenes' in validated_df.columns

        # Check totals are reasonable
        assert all(validated_df['total_cannabinoids'] > 0)
        assert all(validated_df['total_terpenes'] > 0)

    def test_outlier_detection(self, sample_config, sample_strain_data):
        """Test outlier detection."""
        validator = ChemicalProfileValidator(sample_config)

        # Add more data for meaningful outlier detection
        df = pd.concat([sample_strain_data] * 5, ignore_index=True)

        # Add an outlier
        df.loc[0, 'thc_pct'] = 5.0  # Much lower than others

        feature_cols = [col for col in df.columns if col.endswith('_pct')]
        validated_df, warnings = validator.detect_outliers(df, feature_cols)

        # Check that outlier flag exists
        if 'is_outlier' in validated_df.columns:
            assert validated_df['is_outlier'].any()


class TestVariationalAutoencoder:
    """Test VAE model."""

    def test_vae_initialization(self, sample_config):
        """Test VAE can be initialized."""
        vae = VariationalAutoencoder(
            input_dim=12,
            latent_dim=5,
            hidden_layers=[16, 8]
        )

        assert vae.input_dim == 12
        assert vae.latent_dim == 5

    def test_vae_forward_pass(self):
        """Test VAE forward pass."""
        vae = VariationalAutoencoder(input_dim=12, latent_dim=5)

        # Create dummy input
        x = torch.randn(3, 12)

        # Forward pass
        reconstruction, mu, logvar = vae(x)

        # Check shapes
        assert reconstruction.shape == (3, 12)
        assert mu.shape == (3, 5)
        assert logvar.shape == (3, 5)

    def test_vae_loss_calculation(self):
        """Test VAE loss function."""
        vae = VariationalAutoencoder(input_dim=12, latent_dim=5)

        x = torch.randn(3, 12)
        reconstruction, mu, logvar = vae(x)

        loss, components = vae.loss_function(reconstruction, x, mu, logvar)

        # Check loss components
        assert 'total_loss' in components
        assert 'reconstruction_loss' in components
        assert 'kl_divergence' in components
        assert components['total_loss'] > 0

    def test_vae_offspring_generation(self):
        """Test offspring generation."""
        vae = VariationalAutoencoder(input_dim=12, latent_dim=5)

        parent1 = torch.randn(1, 12)
        parent2 = torch.randn(1, 12)

        mean_offspring, std_offspring = vae.generate_offspring(
            parent1,
            parent2,
            parent1_weight=0.6,
            n_samples=50
        )

        # Check shapes
        assert mean_offspring.shape == (1, 12)
        assert std_offspring.shape == (1, 12)

        # Check uncertainty is non-zero
        assert torch.any(std_offspring > 0)

    def test_vae_training(self, sample_strain_data):
        """Test VAE training process."""
        feature_cols = [col for col in sample_strain_data.columns if col.endswith('_pct')]
        X = sample_strain_data[feature_cols].values
        X_tensor = torch.tensor(X, dtype=torch.float32)

        vae = VariationalAutoencoder(input_dim=len(feature_cols), latent_dim=3)
        trainer = VAETrainer(vae, learning_rate=0.01)

        # Train for a few epochs
        history = trainer.train(X_tensor, epochs=10, early_stopping_patience=5, verbose=False)

        # Check history was recorded
        assert 'train_loss' in history
        assert len(history['train_loss']) > 0


class TestEntourageEffectModel:
    """Test entourage effect modeling."""

    def test_interaction_term_creation(self, sample_config, sample_strain_data):
        """Test creation of interaction features."""
        model = EntourageEffectModel(sample_config)

        df_enhanced = model.create_interaction_terms(
            sample_strain_data,
            'analgesic'
        )

        # Check that interaction features were added
        interaction_cols = [col for col in df_enhanced.columns if 'interaction' in col]
        assert len(interaction_cols) > 0

    def test_effect_score_calculation(self, sample_config, sample_strain_data):
        """Test effect score calculation."""
        model = EntourageEffectModel(sample_config)

        profile = sample_strain_data.iloc[0]
        score, contributions = model.calculate_effect_score(profile, 'analgesic')

        # Check score is valid
        assert isinstance(score, (int, float))
        assert score >= 0

        # Check contributions dict
        assert isinstance(contributions, dict)
        assert len(contributions) > 0


class TestEnsembleEffectPredictor:
    """Test ensemble effect prediction."""

    def test_predictor_initialization(self, sample_config):
        """Test effect predictor initialization."""
        predictor = EnsembleEffectPredictor(sample_config, 'analgesic')

        assert predictor.effect_name == 'analgesic'
        assert len(predictor.models) > 0

    def test_predictor_fitting(self, sample_config, sample_strain_data):
        """Test fitting effect predictor."""
        predictor = EnsembleEffectPredictor(sample_config, 'analgesic')

        feature_cols = [col for col in sample_strain_data.columns if col.endswith('_pct')]
        X = sample_strain_data[feature_cols]

        # Create dummy targets
        y = np.array([1, 0, 1])

        # Fit
        metrics = predictor.fit(X, y, validate=False)

        assert predictor.is_fitted
        assert isinstance(metrics, dict)

    def test_prediction_with_uncertainty(self, sample_config, sample_strain_data):
        """Test prediction with uncertainty quantification."""
        predictor = EnsembleEffectPredictor(sample_config, 'analgesic')

        feature_cols = [col for col in sample_strain_data.columns if col.endswith('_pct')]
        X = sample_strain_data[feature_cols]
        y = np.array([1, 0, 1])

        predictor.fit(X, y, validate=False)

        # Predict
        mean_proba, std_proba = predictor.predict_proba(X, return_uncertainty=True)

        # Check shapes
        assert len(mean_proba) == len(X)
        assert len(std_proba) == len(X)

        # Check values are valid probabilities
        assert all((mean_proba >= 0) & (mean_proba <= 1))
        assert all(std_proba >= 0)


class TestStatisticalValidator:
    """Test statistical validation methods."""

    def test_normality_test(self):
        """Test normality testing."""
        # Create normal and non-normal data
        normal_data = np.random.normal(0, 1, 100)
        uniform_data = np.random.uniform(0, 1, 100)

        result_normal = StatisticalValidator.test_normality(normal_data)
        result_uniform = StatisticalValidator.test_normality(uniform_data)

        assert 'test' in result_normal
        assert 'p_value' in result_normal
        assert 'is_normal' in result_normal

    def test_cohens_d(self):
        """Test Cohen's d effect size calculation."""
        group1 = np.array([1, 2, 3, 4, 5])
        group2 = np.array([2, 3, 4, 5, 6])

        d = StatisticalValidator.calculate_cohens_d(group1, group2)

        assert isinstance(d, float)
        assert d != 0  # Groups are different


class TestConfigLoader:
    """Test configuration loading."""

    def test_config_loading(self, sample_config):
        """Test that config loads correctly."""
        assert 'compounds' in sample_config
        assert 'effects' in sample_config
        assert 'model' in sample_config

    def test_feature_list_extraction(self, sample_config):
        """Test extraction of feature list."""
        features = ConfigLoader.get_feature_list(sample_config)

        assert isinstance(features, list)
        assert len(features) > 0
        assert all('_pct' in f for f in features)

    def test_compound_info_retrieval(self, sample_config):
        """Test retrieving compound information."""
        info = ConfigLoader.get_compound_info(sample_config, 'thc')

        assert info is not None
        assert 'name' in info
        assert info['name'] == 'thc'


# Integration Tests

class TestIntegration:
    """Integration tests combining multiple components."""

    def test_full_pipeline(self, sample_config, sample_strain_data):
        """Test complete pipeline from validation to prediction."""
        # Validate data
        validator = ChemicalProfileValidator(sample_config)
        feature_cols = [col for col in sample_strain_data.columns if col.endswith('_pct')]

        validated_df, warnings = validator.comprehensive_validation(
            sample_strain_data,
            feature_cols
        )

        # Train VAE
        X = validated_df[feature_cols].values
        X_tensor = torch.tensor(X, dtype=torch.float32)

        vae = VariationalAutoencoder(input_dim=len(feature_cols), latent_dim=3)
        trainer = VAETrainer(vae)
        history = trainer.train(X_tensor, epochs=10, verbose=False)

        # Generate offspring
        parent1 = X_tensor[0:1]
        parent2 = X_tensor[1:2]

        mean_offspring, std_offspring = vae.generate_offspring(
            parent1,
            parent2,
            n_samples=20
        )

        # Check everything worked
        assert mean_offspring.shape[1] == len(feature_cols)
        assert len(history['train_loss']) > 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
