import torch
import torch.nn as nn
from torch.distributions import Normal, Independent
from typing import Optional, Any, Tuple
import numpy as np
import os
from datetime import datetime

from .layers import (
    GpvaeEncoder,
    GpvaeDecoder,
    GaussianProcess,
    FactorNet
)
from ....imputation.gp_ae.gp_model import ProbabilisticGP
from pygrinder import mcar, fill_and_get_mask_torch

# Module-level constants.
NOISE_SIGMA = 0.01
DENSITY_SCALE = 1e-2
EPSILON = 1e-3
RECON_WEIGHT = 0.1
DEFAULT_ALPHA = 1.0

# Import plotting functions.
from .plotting_utils import plot_params, plot_losses, plot_latent_series_and_reconstruction


class BackboneGP_VAE(nn.Module):
    """Modified GPVAE model with prior variance proportional to missing values."""
    
    def __init__(
        self,
        input_dim: int,
        time_length: int,
        latent_dim: int,
        encoder_sizes: Tuple[int, ...] = (128, 64),
        decoder_sizes: Tuple[int, ...] = (64, 128),
        beta: float = 1,
        M: int = 1,
        K: int = 1,
        kernel: Optional[str] = None,
        sigma: float = 1.0,
        length_scale: float = 7.0,
        kernel_scales: int = 1,
        window_size: int = 24,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.kernel = kernel
        self.sigma = sigma
        self.length_scale = length_scale
        self.kernel_scales = kernel_scales

        self.input_dim = input_dim
        self.time_length = time_length
        self.latent_dim = latent_dim
        self.beta = beta
        self.gamma = 0.01
        self.alpha = DEFAULT_ALPHA
        self.noise_sigma = NOISE_SIGMA
        self.compensate = True

        self.encoder = GpvaeEncoder(input_dim, latent_dim, encoder_sizes, device=self.device).to(self.device)
        self.decoder = GpvaeDecoder(latent_dim, input_dim, decoder_sizes).to(self.device)
        self.M = M
        self.K = K
        self.forward_passes_counter = 0

        print("Model dimensions is:")
        print(self.encoder)
        print(self.decoder)

        self.p = 0.5
        self.sampling = False
        self.train_decoder = True

        self.loss_history = {"elbo": [], "nll": [], "kl": [], "prior_loss": [], "temporal_loss": []}
        self.monitoring_history = {"z_mu_mean": [], "z_mu_var": [], "z_var_mean": [], "z_var_var": []}

    def encode(self, x: torch.Tensor, missing_mask: Optional[torch.Tensor] = None) -> Any:
        """Encodes the input using the encoder network."""
        try:
            return self.encoder(x, missing_mask)
        except AttributeError:
            return self.encoder(x['X'], missing_mask)

    def decode(self, z: torch.Tensor) -> Any:
        """Decodes the latent variable z using the decoder network."""
        return self.decoder(z)

    def forward(self, X: torch.Tensor, missing_mask: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model.
        
        Performs data preparation, computes the ELBO, logs progress, and validates.
        """
        self.batch_size, self.time_steps, _ = X.size()

        # Prepare data and simulate missing data
        X, missing_mask, X_corrupted, missing_mask_corrupted = self.prepare_and_simulate(X, missing_mask)

        nll, kl, prior_loss, temporal_loss = self.elbo(X, missing_mask, X_corrupted, missing_mask_corrupted)
        elbo = nll + kl * self.alpha + prior_loss * self.beta + temporal_loss * self.gamma

        self.forward_passes_counter += 1

        if self.forward_passes_counter % 50 == 0:  # Log losses every 50 iterations
            self.loss_history["elbo"].append(elbo.item())
            self.loss_history["nll"].append(nll.item() - 0.5 * np.log(2 * np.pi * self.noise_sigma))
            self.loss_history["kl"].append(kl.item())
            self.loss_history["prior_loss"].append(prior_loss.item())
            self.loss_history["temporal_loss"].append(temporal_loss.item())

            qz_x = self.encode(X)
            # Compute statistics for monitoring
            z_mu_mean = qz_x.mean.mean(axis=(0, 1)).detach().cpu().numpy()
            z_mu_var = qz_x.mean.var(axis=(0, 1)).detach().cpu().numpy()
            z_var_mean = qz_x.variance.mean(axis=(0, 1)).detach().cpu().numpy()
            z_var_var = qz_x.variance.var(axis=(0, 1)).detach().cpu().numpy()

            self.monitoring_history["z_mu_mean"].append(z_mu_mean)
            self.monitoring_history["z_mu_var"].append(z_mu_var)
            self.monitoring_history["z_var_mean"].append(z_var_mean)
            self.monitoring_history["z_var_var"].append(z_var_var)

            # Use the external plotting functions:
            plot_losses(self.loss_history)
            plot_params(self.monitoring_history, self.loss_history, self.alpha, self.beta, self.gamma)

            if len(self.loss_history["elbo"]) == 300:
                # Downsample the history if too many points are stored
                for key in self.loss_history:
                    self.loss_history[key] = self.loss_history[key][::2]

        # Validation and optional plotting
        qz_x = self.encode(X, missing_mask)
        z = qz_x.rsample()
        px_z = self.decode(z)
        self.validate_elbo(-elbo, nll, kl, prior_loss, z, qz_x, X, X_corrupted, px_z, temporal_loss)

        return elbo

    def elbo(
        self,
        X: torch.Tensor,
        missing_mask: torch.Tensor,
        X_corrupted: torch.Tensor,
        missing_mask_corrupted: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Computes the evidence lower bound (ELBO) components."""
        qz_x = self.encode(X)
        qz_x_corrupted = self.encode(X_corrupted)
        z = qz_x_corrupted.rsample()
        nll = self.nll(X, X_corrupted, z, qz_x, qz_x_corrupted, mean_z=self.use_mean).mean()
        kl = self.kl(qz_x, qz_x_corrupted, eps=EPSILON).mean()
        prior_loss = self.prior(X, qz_x)
        temporal_loss = torch.ones(1) * 0.01
        return nll, kl, prior_loss, temporal_loss

    def nll(
        self,
        X: torch.Tensor,
        X_corrupted: torch.Tensor,
        z: torch.Tensor,
        qz_x: Any,
        qz_x_corrupted: Any,
        mean_z: bool = False,
        eps: float = EPSILON,
    ) -> torch.Tensor:
        """Computes the reconstruction loss (negative log-likelihood)."""
        
        # Sample z from q(z|x)
        z = qz_x_corrupted.rsample() if self.compensate else qz_x.rsample()
        px_z = self.decode(z)
        mu, sigma = px_z.mean, torch.clamp(px_z.variance, min=EPSILON) + eps

        # Use zeros_like to ensure proper device and shape
        mu = mu + torch.normal(mean=torch.zeros_like(mu), std=self.noise_std)
        sigma = self.noise_sigma

        nll = 0.5 * (torch.log(2 * torch.tensor(np.pi, device=self.device) * sigma) + (X - mu).pow(2) / sigma)

        nll_observed = 0.5 * (torch.log(2 * torch.tensor(np.pi, device=self.device) * sigma) + (X - mu).pow(2) / sigma)
        nll_masked = 0.5 * (torch.log(2 * torch.tensor(np.pi, device=self.device) * sigma*10) + (X - mu).pow(2) / (sigma*10))

        mask, mask_corrupted = (X!=0), (X_corrupted!=0)
        mask_hidden = (mask) & (~mask_corrupted) 
        nll = nll_observed[mask_corrupted].sum()/mask_corrupted.int().sum() + nll_masked[mask_hidden].mean()/mask_hidden.int().sum()

        mask = X != 0
        if self.compensate:
            density_quotient = torch.exp((qz_x.log_prob(z) - qz_x_corrupted.log_prob(z)) * DENSITY_SCALE).unsqueeze(2)
            density_quotient = torch.clamp(density_quotient, min=EPSILON, max=1e3)
            density_quotient = torch.nan_to_num(density_quotient, 1e3)
            compensated_nll = nll * density_quotient

        else:
            compensated_nll = nll.mean()

        mask = X != 0
        condition_mask = torch.isfinite(compensated_nll) & mask
        compensated_nll = torch.where(condition_mask, compensated_nll, torch.zeros_like(compensated_nll))
        compensated_nll = compensated_nll.sum(axis=2) / (condition_mask.sum(axis=2) + eps)

        if mean_z:
            px_mu_z = self.decode(qz_x.mean)
            mu, sigma = px_mu_z.mean, self.noise_sigma
            nll_mean = 0.5 * (torch.log(2 * torch.tensor(np.pi, device=self.device) * sigma) + (X - mu).pow(2) / sigma)
            condition_mask = torch.isfinite(nll_mean) & mask
            nll_mean = torch.where(condition_mask, nll_mean, torch.zeros_like(nll_mean))
            nll_mean = nll_mean.sum(axis=2) / (condition_mask.sum(axis=2) + eps)
            nll = compensated_nll + nll_mean * RECON_WEIGHT
        else:
            nll = compensated_nll

        return nll

    def kl(self, qz_x: Any, qz_x_corrupted: Any, eps: float = EPSILON) -> torch.Tensor:
        """Computes the KL divergence between two Gaussian distributions."""
        mu_z, mu_z_corrupted = qz_x.mean, qz_x_corrupted.mean
        var_z, var_z_corrupted = qz_x.variance, qz_x_corrupted.variance
        if self.detach:
            var_z = var_z.detach()
        kl = 0.5 * (
            (torch.log(var_z_corrupted) - torch.log(var_z)).detach()
            + (var_z + (mu_z_corrupted - mu_z).pow(2)) / (var_z_corrupted + eps))
        return kl.sum

    def prior(self, X: torch.Tensor, qz_x: Any) -> torch.Tensor:
        """Computes a prior loss that encourages latent means to align with observed data."""
        mask = X != 0
        mu_z = qz_x.mean
        mean_prior = (mu_z.mean() - X[mask].mean()).abs()
        loss = mean_prior
        where_no_missing_feats = (mask.sum(axis=2) == 0)
        if where_no_missing_feats.int().sum() > 0:
            where_no_missing_feats = where_no_missing_feats.unsqueeze(2).repeat(1, 1, mu_z.shape[2])
            var_loss = qz_x.variance[where_no_missing_feats].pow(2).mean()
            loss += var_loss
        return loss

    def prepare_and_simulate(
        self, X: torch.Tensor, missing_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Prepares data by repeating it and simulating missing data."""
        X_ori = torch.clone(X.repeat(self.K * self.M, 1, 1))
        missing_mask_ori = missing_mask.repeat(self.K * self.M, 1, 1).type(torch.bool)
        X = mcar(X_ori, p=self.p)
        X, missing_mask = fill_and_get_mask_torch(X)
        missing_mask = X != 0
        missing_mask_ori = X_ori != 0
        return X_ori, missing_mask_ori, X, missing_mask

    def validate_elbo(
        self,
        elbo: torch.Tensor,
        nll_recon: torch.Tensor,
        nll_imputation: torch.Tensor,
        kl: torch.Tensor,
        z: torch.Tensor,
        qz_x: Any,
        X_ori: torch.Tensor,
        X: torch.Tensor,
        px_z: Any,
        tl: torch.Tensor,
    ) -> None:
        """
        Validates the computed ELBO and raises exceptions if the values are unrealistic.
        """
        if (elbo.abs() > 1e8).any():
            raise ValueError(f"ELBO too big: {elbo} with nll: {nll_recon.mean().item()}, "
                             f"nll_imputation: {nll_imputation.mean().item()}, kl: {kl.mean().item()}")
        if (elbo > 50).any():
            raise ValueError(f"ELBO too high: {elbo.item()}, nll: {nll_recon.mean().item()}, "
                             f"nll_imputation: {nll_imputation.mean().item()}, kl: {kl.mean().item()}")
        if torch.isnan(elbo).any():
            raise ValueError(f"ELBO is NaN: {elbo.item()}, nll: {nll_recon.mean().item()}, "
                             f"nll_imputation: {nll_imputation.mean().item()}, kl: {kl.mean().item()}")

        # Every 800 forward passes, trigger plotting via external functions.
        if self.forward_passes_counter % 800 == 0:
            print("Plotting latent series and reconstruction...")
            losses = {"kl": kl.mean().item(), "nll": nll_recon.mean().item(), "temporal": tl.item()}
            # Pass the decoder as an extra argument
            plot_latent_series_and_reconstruction(qz_x, X_ori.detach(), X.detach(), self.latent_dim, losses, self.decoder)
