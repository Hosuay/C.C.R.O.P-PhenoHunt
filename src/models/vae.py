"""
Variational Autoencoder for Cannabis Strain Generation
Provides uncertainty quantification and probabilistic generation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, Dict
import logging

logger = logging.getLogger(__name__)


class VariationalAutoencoder(nn.Module):
    """
    Variational Autoencoder (VAE) for cannabis chemical profile generation.

    VAE provides:
    1. Probabilistic latent space representation
    2. Uncertainty quantification
    3. Better generalization than standard autoencoders
    4. Controllable generation with variance estimation

    References:
        - Kingma & Welling (2013). Auto-Encoding Variational Bayes
        - Doersch (2016). Tutorial on Variational Autoencoders
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 8,
        hidden_layers: list = [32, 16],
        dropout_rate: float = 0.15,
        beta: float = 1.0
    ):
        """
        Args:
            input_dim: Number of chemical features
            latent_dim: Dimension of latent space
            hidden_layers: List of hidden layer sizes
            dropout_rate: Dropout probability for regularization
            beta: Weight for KL divergence term (beta-VAE)
        """
        super(VariationalAutoencoder, self).__init__()

        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.beta = beta

        # Build encoder
        encoder_layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_layers:
            encoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = hidden_dim

        self.encoder = nn.Sequential(*encoder_layers)

        # Latent space parameters
        self.fc_mu = nn.Linear(prev_dim, latent_dim)
        self.fc_logvar = nn.Linear(prev_dim, latent_dim)

        # Build decoder
        decoder_layers = []
        prev_dim = latent_dim

        for hidden_dim in reversed(hidden_layers):
            decoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = hidden_dim

        # Output layer with Softplus to ensure positive values
        decoder_layers.append(nn.Linear(prev_dim, input_dim))
        decoder_layers.append(nn.Softplus())

        self.decoder = nn.Sequential(*decoder_layers)

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Xavier initialization for better convergence."""
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode input to latent distribution parameters.

        Args:
            x: Input tensor [batch_size, input_dim]

        Returns:
            Tuple of (mu, logvar) for latent distribution
        """
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization trick for backpropagation through sampling.

        z = mu + sigma * epsilon, where epsilon ~ N(0, 1)
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode latent vector to output space.

        Args:
            z: Latent vector [batch_size, latent_dim]

        Returns:
            Reconstructed output [batch_size, input_dim]
        """
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through VAE.

        Returns:
            Tuple of (reconstruction, mu, logvar)
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        reconstruction = self.decode(z)
        return reconstruction, mu, logvar

    def loss_function(
        self,
        reconstruction: torch.Tensor,
        x: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        VAE loss function = Reconstruction Loss + β * KL Divergence

        Args:
            reconstruction: Reconstructed output
            x: Original input
            mu: Mean of latent distribution
            logvar: Log variance of latent distribution

        Returns:
            Tuple of (total_loss, loss_components_dict)
        """
        # Reconstruction loss (MSE)
        recon_loss = F.mse_loss(reconstruction, x, reduction='mean')

        # KL divergence: -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        kl_div = kl_div / x.size(0)  # Normalize by batch size

        # Total loss
        total_loss = recon_loss + self.beta * kl_div

        loss_components = {
            'total_loss': total_loss.item(),
            'reconstruction_loss': recon_loss.item(),
            'kl_divergence': kl_div.item()
        }

        return total_loss, loss_components

    def generate_offspring(
        self,
        parent1_profile: torch.Tensor,
        parent2_profile: torch.Tensor,
        parent1_weight: float = 0.6,
        n_samples: int = 1,
        temperature: float = 1.0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate offspring with uncertainty quantification.

        Args:
            parent1_profile: First parent chemical profile
            parent2_profile: Second parent chemical profile
            parent1_weight: Weight for first parent (0-1)
            n_samples: Number of offspring samples to generate
            temperature: Sampling temperature (higher = more variation)

        Returns:
            Tuple of (mean_offspring, std_offspring)
        """
        self.eval()
        with torch.no_grad():
            # Encode parents
            mu1, logvar1 = self.encode(parent1_profile)
            mu2, logvar2 = self.encode(parent2_profile)

            # Interpolate in latent space
            parent2_weight = 1 - parent1_weight
            mu_combined = parent1_weight * mu1 + parent2_weight * mu2
            logvar_combined = parent1_weight * logvar1 + parent2_weight * logvar2

            # Generate multiple samples
            offspring_samples = []
            for _ in range(n_samples):
                # Sample from latent distribution with temperature
                std_combined = torch.exp(0.5 * logvar_combined) * temperature
                eps = torch.randn_like(std_combined)
                z_sample = mu_combined + eps * std_combined

                # Decode to chemical profile
                offspring = self.decode(z_sample)
                offspring_samples.append(offspring)

            # Stack samples and compute statistics
            offspring_tensor = torch.stack(offspring_samples)
            mean_offspring = offspring_tensor.mean(dim=0)
            std_offspring = offspring_tensor.std(dim=0)

        return mean_offspring, std_offspring

    def get_uncertainty(
        self,
        profile: torch.Tensor,
        n_samples: int = 100
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Estimate reconstruction uncertainty using Monte Carlo sampling.

        Args:
            profile: Input chemical profile
            n_samples: Number of Monte Carlo samples

        Returns:
            Tuple of (mean_reconstruction, std_reconstruction)
        """
        self.eval()
        reconstructions = []

        with torch.no_grad():
            for _ in range(n_samples):
                mu, logvar = self.encode(profile)
                z = self.reparameterize(mu, logvar)
                recon = self.decode(z)
                reconstructions.append(recon)

        recon_tensor = torch.stack(reconstructions)
        mean_recon = recon_tensor.mean(dim=0)
        std_recon = recon_tensor.std(dim=0)

        return mean_recon, std_recon

    def interpolate_latent(
        self,
        start_profile: torch.Tensor,
        end_profile: torch.Tensor,
        n_steps: int = 10
    ) -> torch.Tensor:
        """
        Interpolate between two profiles in latent space.

        Useful for exploring chemical space between parent strains.

        Args:
            start_profile: Starting profile
            end_profile: Ending profile
            n_steps: Number of interpolation steps

        Returns:
            Tensor of interpolated profiles [n_steps, input_dim]
        """
        self.eval()
        with torch.no_grad():
            # Encode to latent space
            mu_start, _ = self.encode(start_profile)
            mu_end, _ = self.encode(end_profile)

            # Create interpolation weights
            alphas = torch.linspace(0, 1, n_steps).unsqueeze(1).to(mu_start.device)

            # Interpolate in latent space
            mu_interp = (1 - alphas) * mu_start + alphas * mu_end

            # Decode interpolated latent vectors
            interpolated = self.decode(mu_interp)

        return interpolated


class VAETrainer:
    """
    Trainer class for VAE with early stopping and monitoring.
    """

    def __init__(
        self,
        model: VariationalAutoencoder,
        learning_rate: float = 0.001,
        weight_decay: float = 1e-4
    ):
        self.model = model
        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=20,
            verbose=True
        )
        self.history = {
            'train_loss': [],
            'recon_loss': [],
            'kl_div': []
        }

    def train_epoch(self, data: torch.Tensor) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        self.optimizer.zero_grad()

        # Forward pass
        reconstruction, mu, logvar = self.model(data)
        loss, components = self.model.loss_function(reconstruction, data, mu, logvar)

        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()

        return components

    def train(
        self,
        data: torch.Tensor,
        epochs: int = 500,
        early_stopping_patience: int = 50,
        verbose: bool = True
    ) -> Dict:
        """
        Train VAE with early stopping.

        Args:
            data: Training data tensor
            epochs: Maximum number of epochs
            early_stopping_patience: Stop if no improvement for this many epochs
            verbose: Print progress

        Returns:
            Training history dictionary
        """
        best_loss = float('inf')
        patience_counter = 0

        for epoch in range(epochs):
            # Train
            loss_components = self.train_epoch(data)

            # Record history
            self.history['train_loss'].append(loss_components['total_loss'])
            self.history['recon_loss'].append(loss_components['reconstruction_loss'])
            self.history['kl_div'].append(loss_components['kl_divergence'])

            # Learning rate scheduling
            self.scheduler.step(loss_components['total_loss'])

            # Early stopping check
            if loss_components['total_loss'] < best_loss:
                best_loss = loss_components['total_loss']
                patience_counter = 0
            else:
                patience_counter += 1

            # Logging
            if verbose and (epoch + 1) % 50 == 0:
                logger.info(
                    f"Epoch {epoch+1}/{epochs} - "
                    f"Loss: {loss_components['total_loss']:.4f} "
                    f"(Recon: {loss_components['reconstruction_loss']:.4f}, "
                    f"KL: {loss_components['kl_divergence']:.4f})"
                )

            # Early stopping
            if patience_counter >= early_stopping_patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break

        logger.info(f"Training complete. Best loss: {best_loss:.4f}")
        return self.history
