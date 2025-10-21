"""
PhenoHunter Scientific Edition - Main Integration Module
Combines all scientific improvements into a cohesive API
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
import logging
from typing import Dict, List, Tuple, Optional
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from data.validators import ChemicalProfileValidator, StatisticalValidator
from models.vae import VariationalAutoencoder, VAETrainer
from models.effect_predictor import MultiEffectPredictor, EntourageEffectModel
from models.breeding_strategy import AdvancedBreedingStrategy, BreedingResult
from utils.config import ConfigLoader
from utils.visualization import ScientificVisualizer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PhenoHunterScientific:
    """
    Scientific Cannabis Breeding Platform.

    Features:
    - Variational Autoencoder for generation with uncertainty
    - Research-backed therapeutic effect prediction
    - Entourage effect modeling
    - Multi-generation breeding simulation (F1, F2, backcross)
    - Comprehensive data validation
    - Uncertainty quantification
    """

    def __init__(self, config_path: str = None):
        """
        Initialize PhenoHunter with scientific configuration.

        Args:
            config_path: Path to YAML configuration file
        """
        # Load configuration
        if config_path is None:
            config_path = Path(__file__).parent.parent / "configs" / "scientific_config.yaml"

        self.config = ConfigLoader.load(str(config_path))
        logger.info("Configuration loaded successfully")

        # Set random seeds for reproducibility
        seed = self.config['reproducibility']['random_seed']
        np.random.seed(seed)
        torch.manual_seed(seed)

        # Initialize components
        self.validator = ChemicalProfileValidator(self.config)
        self.statistical_validator = StatisticalValidator()
        self.visualizer = ScientificVisualizer()

        # Will be initialized after data loading
        self.vae = None
        self.vae_trainer = None
        self.effect_predictor = None
        self.breeding_strategy = None

        # Data storage
        self.strain_data = None
        self.feature_columns = None

        logger.info("PhenoHunter Scientific initialized")

    def load_strain_database(
        self,
        df: pd.DataFrame,
        validate: bool = True
    ) -> Dict[str, List[str]]:
        """
        Load and validate strain database.

        Args:
            df: DataFrame with strain data
            validate: Whether to run validation

        Returns:
            Dictionary of validation warnings
        """
        logger.info(f"Loading strain database with {len(df)} strains")

        # Get feature columns from config
        self.feature_columns = ConfigLoader.get_feature_list(self.config)

        # Ensure all feature columns exist
        for col in self.feature_columns:
            if col not in df.columns:
                df[col] = 0.0
                logger.warning(f"Added missing column: {col}")

        if validate:
            # Run comprehensive validation
            df, warnings = self.validator.comprehensive_validation(df, self.feature_columns)
            logger.info("Validation complete")

            # Print warnings
            for category, warning_list in warnings.items():
                for warning in warning_list:
                    logger.warning(f"[{category}] {warning}")
        else:
            warnings = {}

        self.strain_data = df
        logger.info(f"Loaded {len(df)} strains with {len(self.feature_columns)} features")

        return warnings

    def train_vae(
        self,
        epochs: int = None,
        verbose: bool = True
    ) -> Dict:
        """
        Train Variational Autoencoder on strain database.

        Args:
            epochs: Number of training epochs (None = use config default)
            verbose: Print training progress

        Returns:
            Training history
        """
        if self.strain_data is None:
            raise ValueError("Must load strain database before training")

        # Get data tensor
        X = self.strain_data[self.feature_columns].fillna(0).values
        X_tensor = torch.tensor(X, dtype=torch.float32)

        # Initialize VAE
        vae_config = self.config['model']['vae']
        self.vae = VariationalAutoencoder(
            input_dim=len(self.feature_columns),
            latent_dim=vae_config['latent_dim'],
            hidden_layers=vae_config['hidden_layers'],
            dropout_rate=vae_config['dropout_rate'],
            beta=vae_config['beta']
        )

        # Initialize trainer
        self.vae_trainer = VAETrainer(
            self.vae,
            learning_rate=vae_config['learning_rate'],
            weight_decay=vae_config['weight_decay']
        )

        # Train
        if epochs is None:
            epochs = vae_config['epochs']

        logger.info(f"Training VAE for {epochs} epochs...")
        history = self.vae_trainer.train(
            X_tensor,
            epochs=epochs,
            early_stopping_patience=vae_config['early_stopping_patience'],
            verbose=verbose
        )

        logger.info("VAE training complete")
        return history

    def train_effect_predictors(
        self,
        target_dict: Dict[str, np.ndarray] = None,
        auto_generate_targets: bool = True
    ) -> Dict:
        """
        Train therapeutic effect prediction models.

        Args:
            target_dict: Dictionary mapping effect names to binary target arrays
            auto_generate_targets: Auto-generate targets from chemical profiles

        Returns:
            Validation metrics
        """
        if self.strain_data is None:
            raise ValueError("Must load strain database first")

        # Initialize effect predictor
        self.effect_predictor = MultiEffectPredictor(self.config)

        if target_dict is None and auto_generate_targets:
            # Generate synthetic targets based on chemical profiles
            logger.info("Auto-generating effect targets from chemical profiles")
            target_dict = self._generate_effect_targets()

        if target_dict is None:
            logger.warning("No targets provided, effect predictor not trained")
            return {}

        # Train
        logger.info("Training effect prediction models...")
        metrics = self.effect_predictor.fit(
            self.strain_data[self.feature_columns],
            target_dict
        )

        logger.info("Effect predictor training complete")
        return metrics

    def _generate_effect_targets(self) -> Dict[str, np.ndarray]:
        """
        Generate synthetic effect labels based on chemical thresholds.

        This is a heuristic method for demonstration purposes.
        In practice, labels should come from clinical data or user reports.
        """
        targets = {}
        entourage_model = EntourageEffectModel(self.config)

        for effect_name in self.config['effects'].keys():
            scores = []

            for _, row in self.strain_data.iterrows():
                score, _ = entourage_model.calculate_effect_score(row, effect_name)
                scores.append(score)

            scores = np.array(scores)

            # Threshold at median to create binary labels
            threshold = np.median(scores)
            targets[effect_name] = (scores > threshold).astype(int)

            logger.info(
                f"Generated {targets[effect_name].sum()} positive samples for {effect_name}"
            )

        return targets

    def initialize_breeding_strategy(self):
        """Initialize advanced breeding strategy module."""
        if self.vae is None:
            raise ValueError("Must train VAE before initializing breeding strategy")

        self.breeding_strategy = AdvancedBreedingStrategy(
            self.vae,
            self.effect_predictor,
            self.config
        )

        logger.info("Breeding strategy initialized")

    def generate_f1_hybrid(
        self,
        parent1_name: str,
        parent2_name: str,
        parent1_weight: float = 0.5,
        n_samples: int = 100
    ) -> BreedingResult:
        """
        Generate F1 hybrid between two parent strains.

        Args:
            parent1_name: Name of first parent
            parent2_name: Name of second parent
            parent1_weight: Contribution weight of first parent (0-1)
            n_samples: Number of Monte Carlo samples for uncertainty

        Returns:
            BreedingResult object
        """
        if self.breeding_strategy is None:
            self.initialize_breeding_strategy()

        # Get parent profiles
        parent1_row = self.strain_data[
            self.strain_data['strain_name'] == parent1_name
        ].iloc[0]

        parent2_row = self.strain_data[
            self.strain_data['strain_name'] == parent2_name
        ].iloc[0]

        parent1_profile = torch.tensor(
            parent1_row[self.feature_columns].values,
            dtype=torch.float32
        ).unsqueeze(0)

        parent2_profile = torch.tensor(
            parent2_row[self.feature_columns].values,
            dtype=torch.float32
        ).unsqueeze(0)

        # Generate F1
        result = self.breeding_strategy.generate_f1(
            parent1_profile,
            parent2_profile,
            parent1_weight=parent1_weight,
            parent1_name=parent1_name,
            parent2_name=parent2_name,
            n_samples=n_samples
        )

        logger.info(
            f"Generated F1: {parent1_name} × {parent2_name} "
            f"(Stability: {result.stability_score:.2f}, "
            f"Heterosis: {result.heterosis_score:.2f})"
        )

        return result

    def generate_f2_population(
        self,
        f1_result: BreedingResult,
        n_offspring: int = 10
    ) -> List[BreedingResult]:
        """
        Generate F2 population from F1.

        Args:
            f1_result: F1 breeding result
            n_offspring: Number of F2 offspring to generate

        Returns:
            List of BreedingResult objects
        """
        if self.breeding_strategy is None:
            raise ValueError("Breeding strategy not initialized")

        f2_population = self.breeding_strategy.generate_f2(
            f1_result,
            n_samples=n_offspring
        )

        logger.info(f"Generated F2 population with {len(f2_population)} individuals")
        return f2_population

    def backcross(
        self,
        f1_result: BreedingResult,
        parent_name: str,
        backcross_generation: int = 1
    ) -> BreedingResult:
        """
        Backcross F1 to a parent.

        Args:
            f1_result: F1 breeding result
            parent_name: Name of parent to backcross to
            backcross_generation: Backcross number (1, 2, 3, ...)

        Returns:
            BreedingResult for backcross
        """
        if self.breeding_strategy is None:
            raise ValueError("Breeding strategy not initialized")

        # Get parent profile
        parent_row = self.strain_data[
            self.strain_data['strain_name'] == parent_name
        ].iloc[0]

        parent_profile = torch.tensor(
            parent_row[self.feature_columns].values,
            dtype=torch.float32
        ).unsqueeze(0)

        result = self.breeding_strategy.backcross(
            f1_result,
            parent_profile,
            parent_name,
            backcross_generation=backcross_generation
        )

        logger.info(
            f"Generated BX{backcross_generation}: {parent_name} × F1 "
            f"(Stability: {result.stability_score:.2f})"
        )

        return result

    def visualize_breeding_result(
        self,
        result: BreedingResult,
        show_uncertainty: bool = True
    ):
        """
        Create comprehensive visualizations for breeding result.

        Args:
            result: BreedingResult to visualize
            show_uncertainty: Include uncertainty plots
        """
        # Create offspring profile as Series
        offspring_profile = pd.Series(
            result.offspring_profile,
            index=self.feature_columns
        )

        offspring_std = pd.Series(
            result.offspring_std,
            index=self.feature_columns
        ) if show_uncertainty else None

        # Chemical profile plot
        fig1 = self.visualizer.plot_chemical_profile_with_uncertainty(
            offspring_profile,
            offspring_std,
            title=f"{result.generation}: {result.parent1_name} × {result.parent2_name}",
            config=self.config
        )
        fig1.show()

        # Get parent profiles for comparison
        if result.parent1_name in self.strain_data['strain_name'].values:
            parent1_row = self.strain_data[
                self.strain_data['strain_name'] == result.parent1_name
            ].iloc[0][self.feature_columns]

            if result.parent2_name in self.strain_data['strain_name'].values:
                parent2_row = self.strain_data[
                    self.strain_data['strain_name'] == result.parent2_name
                ].iloc[0][self.feature_columns]

                # Comparison radar chart
                fig2 = self.visualizer.plot_breeding_comparison(
                    parent1_row,
                    parent2_row,
                    offspring_profile,
                    result.parent1_name,
                    result.parent2_name,
                    f"{result.generation} Offspring"
                )
                fig2.show()

        # Effect predictions
        if result.predicted_effects:
            effects_df = pd.DataFrame([{
                'effect': effect,
                'probability': prob,
                'uncertainty': 0.1,  # Placeholder
                'lower_bound': max(0, prob - 0.1),
                'upper_bound': min(1, prob + 0.1),
                'confidence_met': True
            } for effect, prob in result.predicted_effects.items()])

            fig3 = self.visualizer.plot_effect_predictions(
                effects_df,
                f"{result.generation} Offspring"
            )
            fig3.show()

    def export_results(
        self,
        results: List[BreedingResult],
        filename: str = "breeding_results.csv"
    ):
        """
        Export breeding results to CSV.

        Args:
            results: List of BreedingResult objects
            filename: Output filename
        """
        rows = []

        for result in results:
            row = {
                'parent1': result.parent1_name,
                'parent2': result.parent2_name,
                'parent1_weight': result.parent1_weight,
                'generation': result.generation,
                'stability_score': result.stability_score,
                'heterosis_score': result.heterosis_score
            }

            # Add chemical profile
            for i, col in enumerate(self.feature_columns):
                row[f'{col}_mean'] = result.offspring_profile[i]
                row[f'{col}_std'] = result.offspring_std[i]

            # Add effects
            for effect, prob in result.predicted_effects.items():
                row[f'effect_{effect}'] = prob

            rows.append(row)

        df = pd.DataFrame(rows)
        df.to_csv(filename, index=False)
        logger.info(f"Exported {len(results)} results to {filename}")

    def get_summary_report(self) -> str:
        """
        Generate comprehensive summary report.

        Returns:
            Formatted summary string
        """
        report = []
        report.append("=" * 70)
        report.append("PHENOHUNTER SCIENTIFIC - SYSTEM SUMMARY")
        report.append("=" * 70)

        if self.strain_data is not None:
            report.append(f"\n📊 DATABASE")
            report.append(f"  • Total strains: {len(self.strain_data)}")
            report.append(f"  • Features tracked: {len(self.feature_columns)}")

            if 'total_cannabinoids' in self.strain_data.columns:
                report.append(
                    f"  • Avg total cannabinoids: "
                    f"{self.strain_data['total_cannabinoids'].mean():.2f}%"
                )

            if 'total_terpenes' in self.strain_data.columns:
                report.append(
                    f"  • Avg total terpenes: "
                    f"{self.strain_data['total_terpenes'].mean():.2f}%"
                )

        if self.vae is not None:
            report.append(f"\n🧠 VAE MODEL")
            report.append(f"  • Architecture: {self.vae.input_dim} → {self.vae.latent_dim} (latent)")
            report.append(f"  • Training: Complete")

        if self.effect_predictor is not None:
            report.append(f"\n💊 EFFECT PREDICTORS")
            trained_count = sum(1 for p in self.effect_predictor.predictors.values() if p.is_fitted)
            report.append(f"  • Trained effects: {trained_count}")

        if self.breeding_strategy is not None:
            report.append(f"\n🧬 BREEDING STRATEGY")
            report.append(f"  • Status: Initialized")
            report.append(f"  • History: {len(self.breeding_strategy.breeding_history)} crosses")

        report.append("\n" + "=" * 70)

        return "\n".join(report)


# Convenience function
def create_phenohunter(config_path: str = None) -> PhenoHunterScientific:
    """
    Factory function to create PhenoHunter instance.

    Args:
        config_path: Path to configuration file

    Returns:
        Initialized PhenoHunter instance
    """
    return PhenoHunterScientific(config_path)
