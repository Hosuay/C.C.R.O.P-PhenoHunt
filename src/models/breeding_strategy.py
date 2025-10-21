"""
Advanced Breeding Strategy with Genetic Simulation
Implements multi-generation breeding simulation (F1, F2, backcrosses)
"""

import numpy as np
import pandas as pd
import torch
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class BreedingResult:
    """Container for breeding results."""
    offspring_profile: np.ndarray
    offspring_std: np.ndarray
    parent1_name: str
    parent2_name: str
    parent1_weight: float
    generation: str
    stability_score: float
    heterosis_score: float  # Hybrid vigor
    predicted_effects: Dict[str, float]


class GeneticSimulator:
    """
    Simulates Mendelian genetics for cannabis breeding.

    Cannabis is diploid (2n=20 chromosomes). This simulator models:
    - F1 generation (first filial)
    - F2 generation (self-pollinated F1)
    - Backcrosses (F1 × parent)
    - Genetic variance and segregation
    """

    def __init__(self, config: Dict):
        self.config = config
        self.rng = np.random.RandomState(config['reproducibility']['random_seed'])

    def calculate_f1_variance(
        self,
        parent1_profile: np.ndarray,
        parent2_profile: np.ndarray
    ) -> float:
        """
        Calculate expected variance in F1 generation.

        F1 typically shows low variance (uniform heterozygotes).

        Returns:
            Variance factor (0-1)
        """
        # Genetic distance between parents
        genetic_distance = np.linalg.norm(parent1_profile - parent2_profile)

        # F1 variance is relatively low but increases with parent similarity
        # (due to recessive traits expressing)
        f1_variance = 0.05 + 0.02 * (1 / (1 + genetic_distance))

        return f1_variance

    def calculate_f2_variance(self, f1_variance: float) -> float:
        """
        Calculate variance in F2 generation.

        F2 shows increased variance due to segregation:
        - Homozygous recessive traits appear (1/4 for single gene)
        - Phenotypic ratios like 9:3:3:1 for two genes

        Returns:
            Variance factor for F2
        """
        # F2 variance is typically 2-4x F1 variance
        f2_variance = f1_variance * self.rng.uniform(2.0, 4.0)

        return min(f2_variance, 0.25)  # Cap at 25%

    def calculate_backcross_variance(
        self,
        f1_variance: float,
        backcross_generation: int
    ) -> float:
        """
        Calculate variance for backcross (BX) generations.

        Backcrossing to a parent reduces variance and increases
        similarity to that parent.

        Args:
            f1_variance: Variance in F1
            backcross_generation: Number of backcross generations (BX1, BX2, etc.)

        Returns:
            Variance factor
        """
        # Each backcross reduces variance and increases parent contribution
        reduction_factor = 0.5 ** backcross_generation
        bx_variance = f1_variance * reduction_factor

        return bx_variance

    def simulate_trait_segregation(
        self,
        trait_value: float,
        dominance: float = 0.5,
        n_genes: int = 3
    ) -> np.ndarray:
        """
        Simulate polygenic trait segregation.

        Most cannabis traits are polygenic (controlled by multiple genes).

        Args:
            trait_value: Base trait value
            dominance: Dominance coefficient (0=recessive, 0.5=additive, 1=dominant)
            n_genes: Number of genes affecting trait

        Returns:
            Array of trait values in offspring population
        """
        offspring_values = []

        for _ in range(100):  # Sample population
            gene_effects = []

            for _ in range(n_genes):
                # Each gene has two alleles (diploid)
                allele1 = self.rng.choice([0, 1])  # 0 or 1 copy
                allele2 = self.rng.choice([0, 1])

                # Calculate phenotypic effect
                if allele1 + allele2 == 2:  # Homozygous dominant
                    effect = 1.0
                elif allele1 + allele2 == 1:  # Heterozygous
                    effect = dominance
                else:  # Homozygous recessive
                    effect = 0.0

                gene_effects.append(effect)

            # Average effect across all genes
            total_effect = np.mean(gene_effects)
            offspring_value = trait_value * total_effect

            offspring_values.append(offspring_value)

        return np.array(offspring_values)


class AdvancedBreedingStrategy:
    """
    Advanced breeding strategy with multi-generation simulation.
    """

    def __init__(self, vae_model, effect_predictor, config: Dict):
        self.vae = vae_model
        self.effect_predictor = effect_predictor
        self.config = config
        self.genetic_simulator = GeneticSimulator(config)
        self.breeding_history = []

    def generate_f1(
        self,
        parent1_profile: torch.Tensor,
        parent2_profile: torch.Tensor,
        parent1_weight: float = 0.5,
        parent1_name: str = "Parent1",
        parent2_name: str = "Parent2",
        n_samples: int = 100
    ) -> BreedingResult:
        """
        Generate F1 hybrid with realistic genetic variance.

        F1 generation characteristics:
        - Uniform (all offspring genetically similar)
        - May show hybrid vigor (heterosis)
        - Combines traits from both parents

        Args:
            parent1_profile: First parent chemical profile (tensor)
            parent2_profile: Second parent chemical profile (tensor)
            parent1_weight: Contribution weight of first parent
            parent1_name: Name of first parent
            parent2_name: Name of second parent
            n_samples: Number of Monte Carlo samples for uncertainty

        Returns:
            BreedingResult object
        """
        logger.info(f"Generating F1: {parent1_name} × {parent2_name}")

        # Generate offspring using VAE
        mean_offspring, std_offspring = self.vae.generate_offspring(
            parent1_profile,
            parent2_profile,
            parent1_weight=parent1_weight,
            n_samples=n_samples,
            temperature=1.0
        )

        # Calculate genetic metrics
        parent1_np = parent1_profile.detach().cpu().numpy().flatten()
        parent2_np = parent2_profile.detach().cpu().numpy().flatten()
        offspring_np = mean_offspring.detach().cpu().numpy().flatten()

        # Stability score (how uniform is F1)
        f1_variance = self.genetic_simulator.calculate_f1_variance(
            parent1_np,
            parent2_np
        )
        stability_score = 1.0 - f1_variance

        # Heterosis score (hybrid vigor)
        mid_parent_value = (parent1_np + parent2_np) / 2
        heterosis = np.mean(offspring_np - mid_parent_value)
        heterosis_score = np.clip(heterosis, -1, 1)

        # Predict effects
        offspring_df = pd.DataFrame([offspring_np], columns=self._get_feature_names())
        effects = self._predict_effects(offspring_df)

        result = BreedingResult(
            offspring_profile=offspring_np,
            offspring_std=std_offspring.detach().cpu().numpy().flatten(),
            parent1_name=parent1_name,
            parent2_name=parent2_name,
            parent1_weight=parent1_weight,
            generation="F1",
            stability_score=stability_score,
            heterosis_score=heterosis_score,
            predicted_effects=effects
        )

        self.breeding_history.append(result)
        return result

    def generate_f2(
        self,
        f1_result: BreedingResult,
        n_samples: int = 100
    ) -> List[BreedingResult]:
        """
        Generate F2 generation by self-pollinating F1.

        F2 characteristics:
        - High variance (segregation of traits)
        - Phenotypic ratios appear
        - Some offspring may express recessive traits

        Args:
            f1_result: F1 breeding result
            n_samples: Number of F2 offspring to generate

        Returns:
            List of BreedingResult objects (F2 population)
        """
        logger.info(f"Generating F2 from F1: {f1_result.parent1_name} × {f1_result.parent2_name}")

        # Calculate F2 variance
        f1_variance = 1.0 - f1_result.stability_score
        f2_variance = self.genetic_simulator.calculate_f2_variance(f1_variance)

        # Generate F2 population with increased variance
        f1_tensor = torch.tensor(f1_result.offspring_profile, dtype=torch.float32).unsqueeze(0)

        f2_population = []

        for i in range(min(n_samples, 10)):  # Limit to 10 representative individuals
            # Add genetic noise for segregation
            noise_factor = f2_variance * np.random.randn(*f1_result.offspring_profile.shape)
            f2_profile = f1_result.offspring_profile + noise_factor
            f2_profile = np.clip(f2_profile, 0, 50)  # Keep in valid range

            # Calculate stability (F2 is less stable)
            stability_score = 1.0 - f2_variance

            # Predict effects
            offspring_df = pd.DataFrame([f2_profile], columns=self._get_feature_names())
            effects = self._predict_effects(offspring_df)

            result = BreedingResult(
                offspring_profile=f2_profile,
                offspring_std=np.ones_like(f2_profile) * f2_variance,
                parent1_name=f"{f1_result.parent1_name}×{f1_result.parent2_name}",
                parent2_name=f"{f1_result.parent1_name}×{f1_result.parent2_name}",
                parent1_weight=0.5,
                generation=f"F2-{i+1}",
                stability_score=stability_score,
                heterosis_score=0.0,  # Usually reduced in F2
                predicted_effects=effects
            )

            f2_population.append(result)

        return f2_population

    def backcross(
        self,
        f1_result: BreedingResult,
        parent_profile: torch.Tensor,
        parent_name: str,
        backcross_generation: int = 1,
        n_samples: int = 100
    ) -> BreedingResult:
        """
        Backcross F1 to a parent.

        Backcrossing is used to:
        - Introgress specific traits
        - Stabilize desired characteristics
        - Create near-isogenic lines

        Args:
            f1_result: F1 breeding result
            parent_profile: Parent to backcross to
            parent_name: Name of parent
            backcross_generation: Which backcross (BX1, BX2, etc.)
            n_samples: Monte Carlo samples

        Returns:
            BreedingResult for backcross
        """
        logger.info(f"Backcrossing F1 to {parent_name} (BX{backcross_generation})")

        # Backcross shifts offspring toward parent
        # BX1 ≈ 75% parent, 25% other
        # BX2 ≈ 87.5% parent, 12.5% other
        parent_weight = 0.5 + (0.5 * (1 - 0.5 ** backcross_generation))

        f1_tensor = torch.tensor(f1_result.offspring_profile, dtype=torch.float32).unsqueeze(0)

        mean_offspring, std_offspring = self.vae.generate_offspring(
            parent_profile,
            f1_tensor,
            parent1_weight=parent_weight,
            n_samples=n_samples,
            temperature=0.8  # Lower temperature for more stability
        )

        offspring_np = mean_offspring.detach().cpu().numpy().flatten()

        # Calculate variance
        f1_variance = 1.0 - f1_result.stability_score
        bx_variance = self.genetic_simulator.calculate_backcross_variance(
            f1_variance,
            backcross_generation
        )
        stability_score = 1.0 - bx_variance

        # Predict effects
        offspring_df = pd.DataFrame([offspring_np], columns=self._get_feature_names())
        effects = self._predict_effects(offspring_df)

        result = BreedingResult(
            offspring_profile=offspring_np,
            offspring_std=std_offspring.detach().cpu().numpy().flatten(),
            parent1_name=parent_name,
            parent2_name=f"F1-{f1_result.parent1_name}×{f1_result.parent2_name}",
            parent1_weight=parent_weight,
            generation=f"BX{backcross_generation}",
            stability_score=stability_score,
            heterosis_score=0.0,
            predicted_effects=effects
        )

        return result

    def _get_feature_names(self) -> List[str]:
        """Get feature column names from config."""
        features = []

        for category in ['cannabinoids', 'terpenes']:
            for level in ['major', 'minor']:
                for compound in self.config['compounds'][category][level]:
                    features.append(f"{compound['name']}_pct")

        return features

    def _predict_effects(self, profile_df: pd.DataFrame) -> Dict[str, float]:
        """Predict therapeutic effects for a profile."""
        if self.effect_predictor is None:
            return {}

        try:
            results = self.effect_predictor.predict_all_effects(profile_df)
            effects_dict = {}

            for _, row in results.iterrows():
                effects_dict[row['effect']] = row['probability']

            return effects_dict
        except Exception as e:
            logger.warning(f"Could not predict effects: {e}")
            return {}
