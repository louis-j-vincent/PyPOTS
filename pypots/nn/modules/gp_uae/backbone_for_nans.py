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
RECON_WEIGHT = 0.0
DEFAULT_ALPHA = 1.0

# Import plotting functions.
from .plotting_utils import plot_params, plot_losses, plot_latent_series_and_reconstruction


class BackboneGP_UAE(nn.Module):
    """Modified GPUAE model with prior variance proportional to missing values."""
    
    def __init__(
        self,
        input_dim: int,
        time_length: int,
        latent_dim: int,
        encoder_sizes: Tuple[int, ...] = (128, 64),
        decoder_sizes: Tuple[int, ...] = (64, 128),
        alpha: float = 1,
        beta: float = 1,
        M: int = 1,
        K: int = 1,
        kernel: Optional[str] = None,
        sigma: float = 1.0,
        p: float = 0.1,
        length_scale: float = 7.0,
        kernel_scales: int = 1,
        window_size: int = 24,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        #self.kernel = kernel
        #self.sigma = sigma
        #self.length_scale = length_scale
        #self.kernel_scales = kernel_scales

        self.input_dim = input_dim
        self.time_length = time_length
        self.latent_dim = latent_dim

        self.alpha = alpha
        self.beta = beta
        self.sigma = sigma
        self.p = p

        self.encoder = GpvaeEncoder(input_dim, latent_dim, encoder_sizes, device=self.device).to(self.device)
        self.decoder = GpvaeDecoder(latent_dim, input_dim, decoder_sizes).to(self.device)
        self.M = M
        self.K = K
        self.forward_passes_counter = 0

        print("Model dimensions is:")
        print(self.encoder)
        print(self.decoder)


        self.loss_history = {"elbo": [], "nll": [], "kl": [], "prior_loss": [], "temporal_loss": []}
        self.monitoring_history = {"z_mu_mean": [], "z_mu_var": [], "z_var_mean": [], "z_var_var": []}

    def encode(self, x: torch.Tensor, missing_mask: Optional[torch.Tensor] = None) -> Any:
        """Encodes the input using the encoder network."""
        try:
            return self.encoder(x, missing_mask)
        except AttributeError:
            return self.encoder(x['X'], x['missing_mask'])

    def decode(self, z: torch.Tensor) -> Any:
        """Decodes the latent variable z using the decoder network."""
        return self.decoder(z)

    def log_losses(self, nll, kl, prior_loss, temporal_loss, elbo, X, missing_mask):
        """
        Log all losses for easy monitoring
        """
        self.loss_history["elbo"].append(elbo.item())
        self.loss_history["nll"].append(nll.item() - 0.5 * np.log(2 * np.pi * self.sigma))
        self.loss_history["kl"].append(kl.item())
        self.loss_history["prior_loss"].append(prior_loss.item())
        self.loss_history["temporal_loss"].append(temporal_loss.item())

        qz_x = self.encode(X, missing_mask)
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
        plot_params(self.monitoring_history, self.loss_history, self.alpha, self.beta, 0.)

        if len(self.loss_history["elbo"]) == 300:
            # Downsample the history if too many points are stored
            for key in self.loss_history:
                self.loss_history[key] = self.loss_history[key][::2]

    def forward(self, X: torch.Tensor, missing_mask: torch.Tensor, training = True) -> torch.Tensor:
        """Forward pass of the model.
        
        Performs data preparation, computes the ELBO, logs progress, and validates.
        """
        self.batch_size, self.time_steps, _ = X.size()

        # Prepare data and simulate missing data
        X, missing_mask, X_corrupted, missing_mask_corrupted = self.prepare_and_simulate(X, missing_mask)

        # compute losses
        nll, kl, prior_loss, temporal_loss = self.elbo(X, missing_mask, X_corrupted, missing_mask_corrupted)
        elbo = nll + kl + prior_loss * self.beta 

        assert not torch.isnan(elbo).any()

        if training:

            self.forward_passes_counter += 1

            # Log losses every 50 iterations
            if self.forward_passes_counter % 50 == 0:  
                self.log_losses(nll, kl, prior_loss, temporal_loss, elbo, X, missing_mask)

            # Validation and optional plotting
            self.validate_elbo(-elbo, nll, kl, X, X_corrupted, missing_mask, temporal_loss)

        return elbo

    def elbo(
        self,
        X: torch.Tensor,
        missing_mask: torch.Tensor,
        X_corrupted: torch.Tensor,
        missing_mask_corrupted: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Computes the evidence lower bound (ELBO) components."""
        qz_x = self.encode(X, missing_mask)
        qz_x_corrupted = self.encode(X_corrupted, missing_mask_corrupted)
        nll_elbo = self.nll(X, X_corrupted, missing_mask, missing_mask_corrupted, qz_x, qz_x_corrupted, mean_z=False).mean()
        nll_mean = self.nll(X, X_corrupted, missing_mask, missing_mask_corrupted, qz_x, qz_x_corrupted, mean_z=True).mean()
        nll = self.alpha * nll_elbo + (1 - self.alpha) * nll_mean
        kl = self.kl(qz_x, qz_x_corrupted, eps=EPSILON).mean()
        prior_loss = self.prior(X, missing_mask, qz_x)
        temporal_loss = torch.ones(1) * 0.01
        #print(nll.mean(), kl.mean(), prior_loss.mean())
        return nll, kl, prior_loss, temporal_loss

    def nll(
        self,
        X: torch.Tensor,
        X_corrupted: torch.Tensor,
        mask: torch.Tensor,
        mask_corrupted: torch.Tensor,
        qz_x: Any,
        qz_x_corrupted: Any,
        mean_z: bool = False,
        eps: float = EPSILON,
    ) -> torch.Tensor:
        """
        Computes the reconstruction loss (negative log-likelihood).
        Uses 2 different sigmas: one for the reconstruction NLL, one for the imputation NLL
        """

        # get noises
        sigma_recon = self.sigma
        
        # Sample z from q(z|x)
        z = qz_x_corrupted.rsample() if not mean_z else qz_x.mean

        # Decode z
        px_z = self.decode(z)
        mu, sigma = px_z.mean, torch.clamp(px_z.variance, min=EPSILON) + eps

        # Compute NLL
        nll_observed = 0.5 * (torch.log(2 * torch.tensor(np.pi, device=self.device) * sigma_recon) + (X - mu).pow(2) / sigma_recon)
        nll_masked = 0.5 * (torch.log(2 * torch.tensor(np.pi, device=self.device) * sigma) + (X - mu).pow(2) / (sigma)) 

        #mask, mask_corrupted = (X!=0), (X_corrupted!=0)
        mask_hidden = (mask) & (~mask_corrupted) 
        nll = nll_observed[mask_corrupted].sum()/mask_corrupted.int().sum() + nll_masked[mask_hidden].mean()/mask_hidden.int().sum()
        
        #print(X[mask_hidden].mean(), X_corrupted[mask_hidden].mean())
        #print(nll)

        return nll.mean()

    def kl(self, qz_x: Any, qz_x_corrupted: Any, eps: float = EPSILON) -> torch.Tensor:
        """Computes the KL divergence between two Gaussian distributions."""
        mu_z, mu_z_corrupted = qz_x.mean, qz_x_corrupted.mean
        var_z, var_z_corrupted = qz_x.variance, qz_x_corrupted.variance

        mu_q_tilde = self.alpha * mu_z_corrupted + (1 - self.alpha) * mu_z
        var_q_tilde = self.alpha**2 * var_z_corrupted

        kl = 0.5 * (
            (torch.log(var_z_corrupted) - torch.log(var_q_tilde)).detach()
            + (var_q_tilde + (mu_z_corrupted - mu_z).pow(2)) / (var_z_corrupted + eps))

        return kl.mean(axis=-1)

    def prior(self, X: torch.Tensor, mask : torch.Tensor, qz_x: Any) -> torch.Tensor:
        """Computes a prior loss that encourages latent means to align with observed data."""
        #mask = (X != 0)
        mask = (X == X)
        mu_z = qz_x.mean
        mean_prior = (mu_z.mean() - X[mask].mean()).abs()
        loss = mean_prior
        return loss

    def prepare_and_simulate(
        self, X: torch.Tensor, missing_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Prepares data by repeating it and simulating missing data."""

        #X[X==0] = torch.nan
        
        # Repeat K * M times
        X_ori = torch.clone(X.repeat(self.K * self.M, 1, 1))
        missing_mask_ori = torch.clone(missing_mask.repeat(self.K * self.M, 1, 1)).bool()

        # add missingness
        X = mcar(X_ori, p=self.p)
        missing_mask = (X==X)

        # get missing masks and replaces nans with 0s
        #X, missing_mask = fill_and_get_mask_torch(X, nan = 0.)
        #X[X==0] = torch.nan
        #missing_mask, missing_mask_ori = (X != 0), (X_ori != 0)

        #print(X[missing_mask].mean(), X[~missing_mask].mean())
        #print(X_ori[missing_mask_ori].mean(), X_ori[~missing_mask_ori].mean())

        return X_ori, missing_mask_ori, X, missing_mask

    def validate_elbo(
        self,
        elbo: torch.Tensor,
        nll: torch.Tensor,
        kl: torch.Tensor,
        X_ori: torch.Tensor,
        X: torch.Tensor,
        missing_mask: torch.Tensor,
        tl: torch.Tensor,
    ) -> None:
        """
        Validates the computed ELBO and raises exceptions if the values are unrealistic.
        """
        
        #missing_mask = (X_ori!=0.)
        qz_x = self.encode(X_ori, missing_mask)
        z = qz_x.rsample()
        px_z = self.decode(z)
        if (elbo.abs() > 1e8).any():
            raise ValueError(f"ELBO too big: {elbo} with nll: {nll.mean().item()}, "
                             f"nll_imputation: {nll.mean().item()}, kl: {kl.mean().item()}")
        if (elbo > 100000).any():
            raise ValueError(f"ELBO too high: {elbo.item()}, nll: {nll.mean().item()}, "
                             f"nll_imputation: {nll.mean().item()}, kl: {kl.mean().item()}")
        if torch.isnan(elbo).any():
            raise ValueError(f"ELBO is NaN: {elbo.item()}, nll: {nll.mean().item()}, kl: {kl.mean().item()}")

        # Every 800 forward passes, trigger plotting via external functions.
        if self.forward_passes_counter % 1000 == 0 and True:
            print("Plotting latent series and reconstruction...")
            losses = {"kl": kl.mean().item(), "nll": nll.mean().item(), "temporal": tl.item()}
            # Pass the decoder as an extra argument
            plot_latent_series_and_reconstruction(qz_x, X_ori.detach(), X.detach(), self.latent_dim, losses, self.decoder)
