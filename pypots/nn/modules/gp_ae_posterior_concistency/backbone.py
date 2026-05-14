import torch
import torch.nn as nn
from torch.distributions import Normal, Independent, kl_divergence
from .layers import (
    SimpleVAEEncoder,
    SimpleVAEDecoder,
    GpvaeEncoder,  # modified versions
    GpvaeDecoder,  # modified versions
    GaussianProcess,
    FactorNet
)
from ....imputation.gp_ae.gp_model import ProbabilisticGP

# for mcar imputation
from pygrinder import mcar, fill_and_get_mask_torch

# for plotting
import os
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import time

class BackboneGP_VAE_posterior_consistency(nn.Module):
    """Modified GPVAE model with prior variance proportional to missing values.

    Parameters
    ----------
    input_dim : int
        The feature dimension of the input.

    time_length : int
        The length of each time series.

    latent_dim : int
        The feature dimension of the latent embedding.

    encoder_sizes : tuple
        The tuple of the network size in encoder.

    decoder_sizes : tuple
        The tuple of the network size in decoder.

    beta : float
        The weight of the KL divergence.

    M : int
        The number of Monte Carlo samples for ELBO estimation.

    K : int
        The number of importance weights for IWAE model.

    kernel : str
        The Gaussian Process kernel ["cauchy", "diffusion", "rbf", "matern"].

    sigma : float
        The scale parameter for a kernel function.

    length_scale : float
        The length scale parameter for a kernel function.

    kernel_scales : int
        The number of different length scales over latent space dimensions.
    """

    def __init__(
        self,
        input_dim,
        time_length,
        latent_dim,
        encoder_sizes=(128, 64),
        decoder_sizes=(64, 128),
        beta=1,
        M=1,
        K=1,
        kernel=None,  # No kernel needed for simple VAE
        sigma=1.0,
        length_scale=7.0,
        kernel_scales=1,
        window_size=24,
        device=None,  # Added device parameter
    ):
        super().__init__()
        # Device setup
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device

        # Remove kernel-related initializations
        self.kernel = kernel
        self.sigma = sigma
        self.length_scale = length_scale
        self.kernel_scales = kernel_scales

        self.input_dim = input_dim
        self.time_length = time_length
        self.latent_dim = latent_dim
        self.beta = beta
        self.gamma = 0.01

        # Ensure that encoder and decoder are on the correct device
        self.encoder = GpvaeEncoder(input_dim, latent_dim, encoder_sizes, device=self.device).to(self.device)
        #self.encoder = FactorNet(input_dim, latent_dim, encoder_sizes).to(self.device)
        self.decoder = GpvaeDecoder(latent_dim, input_dim, decoder_sizes).to(self.device)
        self.M = M
        self.K = K

        self.forward_passes_counter = 0

        #self.gp = GaussianProcess(
        #    time_length=time_length,
        #    latent_dim=latent_dim,
        #    kernel='rbf',  # or any other kernel you prefer
        #    quantile=0.5  # Adjust quantile as needed
        #).to(self.device)

        #self.gp = ProbabilisticGP(self, 
        #                            assemble_data = 1, 
        #                            n_dims = input_dim, latent_size = latent_dim)

        print('Model dimensions is: ')
        print(self.encoder)
        print(self.decoder)

        self.p = 0.5

        self.sampling = False
        self.train_decoder = True

        # For tracking
        self.loss_history = {
            'elbo': [],
            'elbo_q': [],
            'other term': [],
            'nll_q': [],
            'kl_q': [],
            'nll_p': [],
            'kl_p': [],
            'kl_q_p': []

        }

        self.monitoring_history = {
            'z_mu_mean': [],
            'z_mu_var': [],
            'z_var_mean': [],
            'z_var_var': [],
            'num_missing_vals': []
        }

    def encode(self, x, missing_mask=None):
        return self.encoder(x, missing_mask)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, X, missing_mask):

        self.batch_size, self.time_steps, _ = X.size()

        # Merge prepare_data and simulate_missing_data
        X_q, missing_mask_q, X_p, missing_mask_p = self.prepare_and_simulate(X, missing_mask)
        missing_mask_q, missing_mask_p = (X_q!=0), (X_p!=0)

        # commpute elbos for p and q
        nll_q, kl_q = self.elbo(X_q, missing_mask_q)
        nll_p, kl_p = self.elbo(X_p, missing_mask_p)
        elbo_q, elbo_p = nll_q - kl_q, nll_p - kl_p

        # compute additional terms
        nll_p_bar, kl_q_p = self.additional_terms(X_p, X_q, missing_mask_p, missing_mask_q)

        l = self.alpha

        elbo = elbo_q * (1 - l) - l * ( kl_q_p - elbo_p - nll_p_bar )

        if torch.rand((1,)).mean() < 1e-2:
            print(elbo_q.mean().item(), elbo_p.mean().item(), kl_q_p.mean().item(), nll_p_bar.mean().item())

        self.forward_passes_counter += 1

        if self.forward_passes_counter % 50 == 0: #plot losses

            self.loss_history['elbo'].append(elbo.mean().item())
            self.loss_history['elbo_q'].append(elbo_q.mean().item())
            self.loss_history['other term'].append(-( kl_q_p - elbo_p - nll_p_bar ).mean().item())
            self.loss_history['nll_q'].append(nll_q.mean().item())
            self.loss_history['kl_q'].append(kl_q.mean().item())
            self.loss_history['nll_p'].append(nll_p.mean().item())
            self.loss_history['kl_p'].append(kl_p.mean().item())
            self.loss_history['kl_q_p'].append(kl_q_p.mean().item())



            qz_x = self.encode(X)

            # Compute statistics for z_mu and z_var
            z_mu_mean = qz_x.mean.mean(axis=(0,1)).detach().cpu().numpy()
            z_mu_var = qz_x.mean.var(axis=(0,1)).detach().cpu().numpy()
            z_var_mean = qz_x.variance.mean(axis=(0,1)).detach().cpu().numpy()
            z_var_var = qz_x.variance.var(axis=(0,1)).detach().cpu().numpy()
            
            # Append to history
            self.monitoring_history['z_mu_mean'].append(z_mu_mean)
            self.monitoring_history['z_mu_var'].append(z_mu_var)
            self.monitoring_history['z_var_mean'].append(z_var_mean)
            self.monitoring_history['z_var_var'].append(z_var_var)

            self.plot_losses()
            self.plot_params()

            if len(self.loss_history['elbo']) == 300 : # Divide number of points by 2

                for key in self.loss_history:
                    self.loss_history[key] = self.loss_history[key][::2]

        # Validation and optional plotting
        qz_x = self.encode(X, missing_mask)
        z = qz_x.rsample()
        px_z = self.decode(z)
        #self.validate_elbo(-elbo, z, qz_x, X, X_p, px_z)

        return -elbo.clip(min = -1e3, max = 1e3).mean()

    def elbo(self, X, missing_mask):
        qz_x = self.encode(X)
        z = qz_x.rsample()
        nll = self.ll(X, z, qz_x, missing_mask).mean()

        q_dist = Normal(qz_x.mean, qz_x.variance.sqrt())
        p_dist = Normal(
            torch.zeros_like(qz_x.mean, device=qz_x.mean.device),
            torch.ones_like(qz_x.mean,  device=qz_x.mean.device)
        )

        kl = kl_divergence(q_dist, p_dist).mean()

        return nll, kl

    def ll(self, X, z, qz_x, mask, mean_z = False, eps = 1e-3):
        """
        Compute the reconstruction part of the elbo loss
        """

        z = qz_x.rsample()
        px_z = self.decode(z)

        mu, sigma = px_z.mean, px_z.variance.clip(min = 1e-3) + eps

        nll = 0.5 * ( torch.log(2 * torch.tensor(np.pi) * sigma) + (X - mu).pow(2) / sigma) 

        condition_mask = (torch.isfinite(nll)) & (mask)
        nll = torch.where(condition_mask, nll, torch.zeros_like(nll))
        nll = nll.sum(axis=2)  # compute the mean along the second dimension only on observed values

        return - nll

    def additional_terms(self, X_p, X_q, mask_p, mask_q):

        mask_p_bar = mask_q ^ mask_p #(mask_q ^- mask_p)

        qz_x_q = self.encode(X_q)
        qz_x_p = self.encode(X_p)

        z_q = qz_x_q.rsample()

        nll_p_bar = self.ll(X_q, z_q, qz_x_q, mask_p_bar).mean()

        kl_q_p = kl_divergence(qz_x_q, qz_x_p)

        return nll_p_bar, kl_q_p

## sampling loss

    def latent_sampling_error(self, qz_x, qz_x_ori, X_sampled):
        """
        This error forces the AE to learn representation of missing values
        X_sampled (num_samples, batch_size, n_observations, n_dimensions)
        """
        loss = 0
        # For each sample, compute the log prob of its embedding mean belonging to qz_x
        for x_sampled in X_sampled:
            qz_x_sampled = self.encode(x_sampled)
            z = qz_x_sampled.mean.detach() # we don't want to learn or correct this
            #log_prob = qz_x.log_prob(qz_x_sampled).clip(max = 0)
            log_prob_x = qz_x.log_prob(z)
            log_prob_xt = qz_x_sampled.log_prob(z) # CLIP so the value doesn't explode
            log_prob_x_ori = qz_x_ori.log_prob(z)
            entropy = torch.exp(  log_prob_x_ori - log_prob_xt  ) * ( log_prob_xt - log_prob_x )

            loss += entropy.clip(min = -100)

            #print(entropy.mean())

        loss = loss / len(X_sampled)

        return loss #* 1e-8

    def sampling_error(self, qz_x, X_ori, missing_mask_ori):
        """
        Forces the model to learn representations of missing values.
        """
        num_samples = 3
        X_sampled, mask_sampled = self.sample_selected_missing_vals(X_ori.detach(), missing_mask_ori.detach(), num_samples=num_samples)
        X_sampled = torch.permute(X_sampled, (3, 0, 1, 2))
        qz_x_ori = self.encode(X_ori)
        loss = self.latent_sampling_error(qz_x, qz_x_ori, X_sampled).mean()

        if loss < 0:
            print(loss.item())

        loss = loss.clip(min=0)

        return loss

    def sample_selected_missing_vals(self, X, missing_mask, num_samples=1, num_imputed_vals = 10):
        """
        Impute missing values by selecting a specific unobserved feature for each point with
        observed values, and compute distances only with points that have the selected feature
        observed and share other observed features.
        """
        batch_size, seq_len, feature_size = X.shape
        X_flat = X.reshape(-1, feature_size)
        mask_flat = missing_mask.reshape(-1, feature_size).int()
        X_recon = X_flat.clone().unsqueeze(2).repeat(1, 1, num_samples)
        # Ensure mask is on the correct device
        mask = torch.zeros(X_flat.shape, device=self.device)

        selected_indices = np.random.choice(np.arange(X_flat.shape[0]), size=num_imputed_vals, replace=False)

        for i in selected_indices:

            mask_i = mask_flat[i]

            observed_features_i = mask_flat[i]
            unobserved_features_i = torch.where(~mask_flat[i])[0]

            compat_indices_i = mask_flat @ observed_features_i.unsqueeze(1)
            #print(compat_indices_i)

           # check if there are some features to impute
            if len(unobserved_features_i) == 0:
                continue  # Skip if no unobserved features

            candidates_i = torch.where( mask_flat @ observed_features_i.unsqueeze(1) > 2 )[0]

            if len(candidates_i)==0: #if no candidates to sample from
                continue

            mask_candidates_i = mask_flat[candidates_i]

            for j in unobserved_features_i:

                mask_ij = observed_features_i.clone().unsqueeze(1)
                compatible_indices = torch.where (mask_candidates_i[:,j] )[0] # points that have some feats in common + feature j

                #print(len(candidates_ij), len(mask_flat))
                max_candidates = 50
                if len(compatible_indices) > max_candidates:
                    compatible_indices = compatible_indices[torch.randperm(len(compatible_indices))[:max_candidates]]



                if len(compatible_indices) > 0:
                    # Get observed values for feature j from compatible points
                    observed_values_j = X_flat[compatible_indices, j]

                    # Select features common between xi and compatible points
                    xi = X_flat[i]
                    compatible_points = X_flat[compatible_indices]
                    distances = self.compute_selected_distance(xi, compatible_points, observed_features_i)

                    # Sample from the closest compatible points based on distances
                    temperature = 0.1
                    probabilities = torch.softmax(-distances * temperature, dim=0)
                    p = probabilities.detach().cpu().numpy().flatten()

                    sampled_idx = torch.multinomial(probabilities, num_samples=num_samples).squeeze()
                    X_recon[i, j] = observed_values_j[sampled_idx]
                    mask[i, j] = 1

        return X_recon.reshape(batch_size, seq_len, feature_size, num_samples), mask.reshape(batch_size, seq_len, feature_size).bool()

    def compute_selected_distance(self, xi, compatible_points, observed_features):
        """
        Compute distances only for observed features in xi that are also observed in compatible_points.
        """
        common_features_mask = observed_features.unsqueeze(0) & (compatible_points != 0)
        diff = (xi - compatible_points) ** 2 * common_features_mask.float()
        distances = diff.sum(dim=1)

        # When no common features, replace missing distance by a reasonable quantile of the distance
        distances += ((1 - common_features_mask.float())).sum(dim=1) * 0.5

        return distances

## Helpers and plotters

    def prepare_and_simulate(self, X, missing_mask):
        """Prepare data by repeating and simulate missing data."""
        X_ori = torch.clone(X.repeat(self.K * self.M, 1, 1))
        missing_mask_ori = missing_mask.repeat(self.K * self.M, 1, 1).type(torch.bool)
        X = mcar(X_ori, p=self.p)
        X, missing_mask = fill_and_get_mask_torch(X)
        # Compute masks
        missing_mask = (X != 0)
        missing_mask_ori = (X_ori != 0)
        return X_ori, missing_mask_ori, X, missing_mask

    def validate_elbo(self, elbo, z, qz_x, X_q, X_p, px_z):
        """Perform assertions, debugging, and optional plotting."""
        if len(self.loss_history['elbo']) > 20 and len(self.loss_history['elbo'])%10 == 0:
            loss_ratio = torch.tensor(self.loss_history['temporal_loss'][-20:]).mean().item() / torch.tensor(self.loss_history['nll'][-20:]).mean().item()
            #if loss_ratio < 1: 
            #    self.gamma *= loss_ratio

        if self.forward_passes_counter%100 == 0:
            print('plotting')
            ## Let's shope this doesn't mess up with the optimizer too much, but
            ## I want the recon and temporal losses to be the same order of magnitude, so le'ts 
            ## adapt the gamma

            self.plot_latent_series_and_reconstruction(z, px_z, qz_x, X_q.detach(), X_p.detach(), self.latent_dim, -elbo, torch.ones(5), torch.ones(5))

    def plot_params(self):

        max_points = 200
        num_points = len(self.loss_history['elbo'])
        n = max(1, num_points // max_points)  # Ensure n is at least 1

        # Create the iterations range with step size n
        iterations = range(1, num_points + 1)[::n]

        params_to_plot = [ #'z_mu_mean' ,
            'z_mu_var', 
            'z_var_mean', 
            'z_var_var'
            ]

        # Define a decimation factor for plotting
        n = 10  # Plot every nth point to reduce visual clutter

        # Create a figure for the plots
        plt.figure(figsize=(12, 8))

        # Iterate through parameters and plot them
        style = ['-','o','+',':']
        for i,param in enumerate(params_to_plot):
            plt.semilogy(np.abs(np.array(self.monitoring_history[param])[::n]), style[i])
            plt.plot([], style[i], label=param)

        z_var_ratio = np.array(self.monitoring_history['z_var_mean'])[::n] / np.array(self.monitoring_history['z_mu_var'])[::n]
        plt.semilogy(z_var_ratio, linewidth = 2, linestyle = ':')
        plt.plot([], linewidth = 2, label='z var ratio')

        # Add labels, legend, and title
        plt.xlabel('Iterations')
        plt.ylabel('Value')
        plt.title(f'Monitored Parameters Over Training - alpha {self.alpha} - beta {self.beta} - gamma {self.gamma}')
        plt.legend(bbox_to_anchor=[1.2, 0.3])
        plt.grid()

        plot_path = os.path.join('latent_plots/params.png')

        # Save the plot
        plt.tight_layout()
        plt.savefig(plot_path)
        plt.close()

    def plot_losses(self):
        # Calculate the step size n to ensure we have a maximum of 500 points plotted
        max_points = 200
        num_points = len(self.loss_history['elbo'])
        n = max(1, num_points // max_points)  # Ensure n is at least 1

        # Create the iterations range with step size n
        iterations = range(1, num_points + 1)[::n]

        plt.figure(figsize=(10, 6))

        for key in self.loss_history:
            plt.plot(iterations, self.loss_history[key][::n], label=key)

        plt.xlabel('Iteration')
        plt.ylabel('Loss (log scale)')
        plt.title('Loss Components over Iterations')
        plt.legend(bbox_to_anchor=[1.2, 0.3])
        plt.grid(True)

        plot_path = os.path.join('latent_plots/losses.png')

        # Save the plot
        plt.tight_layout()
        plt.savefig(plot_path)
        plt.close()

    def plot_latent_series_and_reconstruction(self, z, px_z, qz_x, X_ori, X, latent_dim, nll, kl, tl, folder='latent_plots'):
        """
        Plots the mean and variance of all latent time series and the original vs reconstructed data.
        """
        print(0)
        
        # Convert to CPU numpy arrays for plotting
        z_mean = qz_x.mean[0].detach().cpu().numpy()
        z_var = qz_x.variance[0].detach().cpu().numpy()

        X_ori_np = torch.clone(X_ori).detach().cpu().numpy()
        X_np = X.detach().cpu().numpy()

        qz_x_ori = self.encode(X_ori)


        z_mean_ori, z_var_ori = qz_x_ori.mean.detach().cpu(), qz_x_ori.variance.detach().cpu()

        # Sample 10 times from the posterior to get 10 reconstructions
        nll_loss = 0
        reconstructions = []
        for i in range(10):
            print(1)
            z_sample = qz_x.rsample()  # Sample from the posterior
            print(qz_x.mean.shape)
            px_z_sample = self.decode(z_sample)  # Reconstruct the data
            print(2)

            X_recon_sample = px_z_sample.mean.detach().cpu().numpy()  # Get the mean of the reconstruction
            # Set first half of non-reconstructed values to NaN
            num_missing_vals = np.sum(X_ori_np == 0)
            print(X_recon_sample.shape, X_ori_np.shape)
            X_recon_sample[X_ori_np == 0][:num_missing_vals // 2] == np.nan

            print(11)
            reconstructions.append(X_recon_sample)

            print(22)
            print(33)

            #print(nll_loss)

        print(3)

        X_ori_np[X_ori_np == 0] = np.nan

        reconstructions = np.array(reconstructions)

        # Create directory if it doesn't exist
        if not os.path.exists(folder):
            os.makedirs(folder)

        # Create a unique filename based on the current timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        plot_path = os.path.join(folder, f'latent_series_and_reconstruction_{timestamp}.png')

        # Plot all latent dimensions' mean and variance
        plt.figure(figsize=(15, 12))

        print(4)

        plt.subplot(4, 1, 1)
        colors = ['purple', 'brown', 'orange', 'b','g','r','y','black','cyan','gray']
        time_steps = z_mean.shape[0]
        for dim in range(latent_dim):
            plt.plot(range(time_steps), z_mean[:, dim], label=f'Latent dim {dim} Mean', color = colors[dim%10])
            plt.fill_between(range(time_steps), z_mean[:, dim] - z_var[:, dim] ** 0.5, z_mean[:, dim] + z_var[:, dim] ** 0.5, alpha=0.2, color = colors[dim%10])
            plt.scatter(range(time_steps), z_mean_ori[0][:, dim].numpy(), label=f'Latent dim {dim} Mean', color = colors[dim%10])
            plt.fill_between(range(time_steps), z_mean_ori[0][:, dim] - z_var_ori[0][:, dim] ** 0.5, z_mean_ori[0][:, dim] + z_var_ori[0][:, dim] ** 0.5, alpha=0.2, color = 'gray')
        #plt.plot([], color = 'gray', alpha = .2, label = 'Z original variance')
        #plt.plot([], color = 'k', alpha = .2, label = 'Z corrupted variance')


        plt.title('Latent Time Series (Mean and Variance) ')
        plt.xlabel('Time Steps')
        plt.ylabel('Latent Values')
        #plt.legend(bbox_to_anchor=[1.2, 0.3])

        print(5)

        # Plot scales for prior and posterior
        plt.subplot(4, 1, 2)
        nb_missing_vals = (X_np[0] == 0).sum(axis=1)
        missing_ratio = (X_np[0] != 0).mean(axis=1)
        prior_scale = (1 - missing_ratio) ** 0.5
        plt.semilogy(z_var, alpha=0.5)
        plt.semilogy((z_mean - z_mean_ori[0].numpy()) ** 2, 'o', alpha=0.5, label = 'l2 errors')
        plt.semilogy(prior_scale, label='Prior scale')
        plt.semilogy(np.linalg.norm(z_var, axis=1), 'r:', label='Posterior scale')
        plt.legend(bbox_to_anchor=[1.2, 0.3])

        # Plot original vs reconstructed data for all 10 reconstructions
        plt.subplot(4, 1, 3)
        X_recon = self.decode(qz_x.mean[0]).mean.detach().cpu().numpy()
        X_np[X_np==0] = np.nan

        print(6)

        errors = np.array( [np.abs(x_recon - X_ori_np) for x_recon in reconstructions]).mean(0)
        std = reconstructions.std(0)
        normalized_errors = errors / std

        error_recon, error_imp = normalized_errors[X_np==X_np].mean(), np.nanmean(normalized_errors[(X_np!=X_np)&(X_ori_np==X_ori_np)])
        error_recon_raw, error_imp_raw = errors[X_np==X_np].mean(), np.nanmean(errors[(X_np!=X_np)&(X_ori_np==X_ori_np)])

        #print(error_recon, error_imp, error_recon_raw, error_imp_raw, (X_np==X_np).mean())


        for dim in range(X_np.shape[2]):
            for i, X_recon_sample in enumerate(reconstructions):
                plt.plot(range(time_steps), X_recon_sample[0, :, dim], color = colors[dim%10], alpha=0.6)
            plt.plot(range(time_steps), X_recon[:, dim], color = colors[dim%10], linewidth = 2, alpha = .5)
            plt.scatter(range(time_steps), X_np[0, :, dim], color = colors[dim%10])
            plt.scatter(range(time_steps), X_ori_np[0, :, dim], color = colors[dim%10], marker = '+', label='Original Data')
        plt.title('Original vs Reconstructed Data (10 Samples)')
        plt.xlabel('Time Steps')
        plt.ylabel('Feature Values')
        y_min, y_max = np.nanmin(X_ori_np[0]), np.nanmax(X_ori_np[0])
        plt.ylim([y_min * 1.1, y_max * 1.1])

        # Save the plot
        plt.tight_layout()
        plt.savefig(plot_path)
        plt.close()

import math

## for gp kernel estimation during training

def log_joint_gaussian(f1, f2, m1, m2, k11, k22, k12):
    """
    Compute the log probability of a bivariate Gaussian evaluated at (f1, f2).

    Parameters
    ----------
    f1, f2 : float
        Observed function values at x_1 and x_2.
    m1, m2 : float
        Mean of the Gaussian process at x_1 and x_2.
    k11, k22, k12 : float
        Covariance terms:
        k11 = k(x1, x1)
        k22 = k(x2, x2)
        k12 = k(x1, x2) = k(x2, x1)

    Returns
    -------
    float
        The log probability of observing (f1, f2) under the given bivariate normal distribution.
    """
    # Differences from the mean
    d1 = f1 - m1
    d2 = f2 - m2

    # Determinant of covariance matrix
    det = k11 * k22 - k12**2
    if det <= 0:
        raise ValueError("Covariance matrix must be positive definite. Determinant is non-positive.")

    # Inverse covariance quadratic form
    # (f - mu)^T Sigma^{-1} (f - mu) =
    # [ d1  d2 ] * (1/det) * [ k22  -k12; -k12  k11 ] * [ d1; d2 ]
    # Expand this expression:
    quad_form = (k22 * d1**2 - 2 * k12 * d1 * d2 + k11 * d2**2) / det

    # Log probability of a bivariate Gaussian
    # log p(f1, f2) = -1/2 * quad_form - 1/2 * log(det) - log(2*pi)
    log_prob = -0.5 * quad_form - 0.5 * math.log(det) - math.log(2 * math.pi)
    return log_prob

def log_joint_gaussian_batch(f1, f2, m1, m2, k11, k22, k12):
    """
    Compute the log probability of a bivariate Gaussian evaluated at multiple (f1, f2) points.

    Parameters
    ----------
    f1, f2 : np.ndarray of shape (N,)
        Observed function values at x_1 and x_2 for each sample in the batch.
    m1, m2 : np.ndarray of shape (N,)
        Mean of the Gaussian process at x_1 and x_2 for each sample in the batch.
    k11, k22, k12 : float
        Covariance terms (scalars):
        k11 = k(x1, x1)
        k22 = k(x2, x2)
        k12 = k(x1, x2) = k(x2, x1)

    Returns
    -------
    np.ndarray of shape (N,)
        The log probability of observing (f1, f2) under the given bivariate normal distribution,
        for each sample in the batch.
    """
    # Differences from the mean
    d1 = f1 - m1
    d2 = f2 - m2

    # Determinant of covariance matrix
    det = k11 * k22 - k12**2
    if det <= 0:
        raise ValueError("Covariance matrix must be positive definite. Determinant is non-positive.")

    # Compute the quadratic form for each element in the batch
    quad_form = (k22 * d1**2 - 2 * k12 * d1 * d2 + k11 * d2**2) / det

    # Log probability of a bivariate Gaussian
    # log p(f1, f2) = -1/2 * quad_form - 1/2 * log(det) - log(2*pi)
    log_prob = -0.5 * quad_form - 0.5 * math.log(det) - math.log(2 * math.pi)

    return log_prob

def log_joint_gaussian_heteroscedastic_from_z(z, qz_x, i, j, k11, k22, k12):
    """
    Compute the log probability of a bivariate Gaussian with heteroscedastic noise 
    for each observation in the batch, using the observed values from z and 
    the mean/variance from qz_x.

    Parameters
    ----------
    z : np.ndarray of shape (N, D)
        Observed function values for N observations, each of dimension D.
    qz_x : object with attributes:
        - qz_x.mean: np.ndarray of shape (N, D)
        - qz_x.variance: np.ndarray of shape (N, D)
        Corresponding to the mean and variance of q(z|x) for each observation.
    
    i, j : int
        Indices specifying which two dimensions (components) to extract from z, qz_x.mean, and qz_x.variance.
    
    k11, k22, k12 : float
        Covariance terms (scalars) for the base kernel:
        k11 = k(x_i, x_i)
        k22 = k(x_j, x_j)
        k12 = k(x_i, x_j) = k(x_j, x_i)

    Returns
    -------
    np.ndarray of shape (N,)
        The log probability for each of the N observations under the specified bivariate normal distribution.
    """
    # Extract the relevant observed values
    f1 = z[:, i]
    f2 = z[:, j]
    
    # Extract the corresponding means and variances from qz_x
    m1 = qz_x.mean[:, i].detach() # we don't want the model to correct these ?
    m2 = qz_x.mean[:, j].detach() # we don't want the model to correct these ?
    sigma_f1_sq = qz_x.variance[:, i]
    sigma_f2_sq = qz_x.variance[:, j]

    # Differences from the mean
    d1 = f1 - m1
    d2 = f2 - m2

    # Effective diagonal terms with heteroscedastic noise
    k11_eff = k11 + sigma_f1_sq  # shape (N,)
    k22_eff = k22 + sigma_f2_sq  # shape (N,)

    # Determinant of the adjusted covariance matrix for each observation
    det = k11_eff * k22_eff - k12**2
    if torch.any(det <= 0):
        raise ValueError("Covariance matrix must be positive definite for all observations. Found non-positive determinant(s).")

    # Compute the quadratic form for each observation
    quad_form = (k22_eff * d1**2 - 2 * k12 * d1 * d2 + k11_eff * d2**2) / det

    # Log probability of a bivariate Gaussian:
    # log p = -1/2 * quad_form - 1/2 * log(det) - log(2*pi)
    log_prob = -0.5 * quad_form - 0.5 * torch.log(det) - math.log(2 * math.pi)

    #print(log_prob)

    return log_prob

def HSIC_loss(A, B, sigma=1.0):
    """
    Loss that computes the covariance between A and B.
    """
    # Flatten the tensors to compute covariance across all elements
    A = A.reshape(-1, A.shape[-1])
    B = B.reshape(-1, B.shape[-1]).float()  # Convert to float for computation

    n = A.size(0)  # Number of samples
    device = A.device

    # Compute Gram matrices
    K = gaussian_kernel(A, sigma)
    L = gaussian_kernel(B, sigma)

    # Center the Gram matrices
    H = centering_matrix(n, device)
    Kc = H @ K @ H
    Lc = H @ L @ H

    # Compute HSIC
    hsic = (1 / (n - 1) ** 2) * torch.trace(Kc @ Lc)

    return hsic, sigma

def gaussian_kernel(X, sigma):
    """
    Computes the Gaussian (RBF) kernel matrix for tensor X.
    """
    pairwise_distances = torch.cdist(X, X) ** 2  # Shape: [n, n]
    K = torch.exp(-pairwise_distances / (2 * sigma ** 2))
    return K

def centering_matrix(n, device):
    """
    Creates a centering matrix of size n x n.
    """
    I = torch.eye(n, device=device)
    ones = torch.ones(n, n, device=device) / n
    H = I - ones
    return H

def compute_kl_divergence(qz_x, prior_variance=0.1):
    """
    Computes the KL divergence between the approximate posterior q(z|x)
    and the prior p(z) with a specified variance.
    """
    mu_q = qz_x.mean  # Mean of q(z|x)
    sigma_q2 = qz_x.variance  # Variance of q(z|x)

    # Prior parameters
    mu_p = torch.zeros_like(mu_q)  # Prior mean is zero
    sigma_p2 = prior_variance  # Prior variance

    # Compute the KL divergence components
    eps = 1e-8
    sigma_q2 = sigma_q2 + eps

    term1 = sigma_q2 / sigma_p2  # (σ_q^2) / (σ_p^2)
    term2 = (mu_q - mu_p).pow(2) / sigma_p2  # (μ_q - μ_p)^2 / (σ_p^2)
    term3 = -1  # -1
    term4 = torch.log(sigma_p2 / sigma_q2)  # ln(σ_p^2 / σ_q^2)

    # Sum over the latent dimensions
    kl_div = 0.5 * torch.sum(term1 + term2 + term3 + term4, dim=1)  # [batch_size]

    # Return the mean KL divergence over the batch
    return kl_div.mean()
