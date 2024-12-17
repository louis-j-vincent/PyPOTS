import torch
import torch.nn as nn
from torch.distributions import Normal, Independent
from .layers import (
    SimpleVAEEncoder,
    SimpleVAEDecoder,
    GpvaeEncoder,  # modified versions
    GpvaeDecoder,  # modified versions
    GaussianProcess,
    FactorNet
)

# for mcar imputation
from pygrinder import mcar, fill_and_get_mask_torch

# for plotting
import os
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import time

class BackboneGP_VAE(nn.Module):
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

        self.gp = GaussianProcess(
            time_length=time_length,
            latent_dim=latent_dim,
            kernel='rbf',  # or any other kernel you prefer
            quantile=0.5  # Adjust quantile as needed
        ).to(self.device)

        print('Model dimensions is: ')
        print(self.encoder)
        print(self.decoder)

        self.p = 0.5

        self.sampling = False
        self.train_decoder = True

        # For tracking
        self.loss_history = {
            'elbo': [],
            'nll': [],
            'kl': [],
            'prior_loss': [],
            'temporal_loss': []
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
        X, missing_mask, X_corrupted, missing_mask_corrupted = self.prepare_and_simulate(X, missing_mask)

        nll, kl, prior_loss, temporal_loss = self.elbo(X, missing_mask, X_corrupted, missing_mask_corrupted)

        elbo = nll + kl * self.alpha + prior_loss * self.beta + temporal_loss * self.gamma

        self.forward_passes_counter += 1

        if self.forward_passes_counter % 50 == 0: #plot losses

            self.loss_history['elbo'].append(elbo.item())
            self.loss_history['nll'].append(nll.item() - .5 * np.log(2 * np.pi * 0.01))
            self.loss_history['kl'].append(kl.item())
            self.loss_history['prior_loss'].append(prior_loss.item())
            self.loss_history['temporal_loss'].append(temporal_loss.item())

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
        self.validate_elbo(-elbo, nll, kl, prior_loss, z, qz_x, X, X_corrupted, px_z, temporal_loss)

        return elbo

    def elbo(self, X, missing_mask, X_corrupted, missing_mask_corrupted):

        qz_x = self.encode(X)
        qz_x_corrupted = self.encode(X_corrupted)

        z = qz_x_corrupted.rsample()

        nll = self.nll(X, z, qz_x, qz_x_corrupted, mean_z = True).mean()

        kl = self.kl(qz_x, qz_x_corrupted, eps = 1e-5).mean()

        prior_loss = self.prior(X, qz_x)

        temporal_loss = self.temporal_loss(X, qz_x)

        sampling_error = self.sampling_error(qz_x_corrupted, X, (X!=0))

        nll + sampling_error

        return nll, kl, prior_loss, temporal_loss

    def nll(self, X, z, qz_x, qz_x_corrupted, mean_z = False, eps = 1e-3):
        """
        Compute the reconstruction part of the elbo loss
        """

        z = qz_x.rsample()
        px_z_corrupted = self.decode(z)

        mu, sigma = px_z_corrupted.mean, px_z_corrupted.variance.clip(min = 1e-3) + eps

        mu = mu + torch.normal(mean = torch.zeros(mu.shape)).to(self.device)*.001


        sigma = .01
        #sigma = qz_x.variance.mean(2).unsqueeze(2)

        nll = 0.5 * ( torch.log(2 * torch.tensor(np.pi) * sigma) + (X - mu).pow(2) / sigma) 

        #print(torch.log(2 * torch.tensor(np.pi) * sigma)[0,:2], ((X - mu).pow(2) / sigma)[0,:2])

        #nll = nll.clip(min = 0, max = 1e5)

        density_quotient = torch.exp( qz_x.log_prob(z) - qz_x_corrupted.log_prob(z) ).unsqueeze(2).clip(1e-3, 1e3)
        density_quotient = 1

        compensated_nll = nll * density_quotient

        # apply mask to select only observed values
        mask = (X != 0)

        condition_mask = (torch.isfinite(compensated_nll)) & (mask)
        compensated_nll = torch.where(condition_mask, compensated_nll, torch.zeros_like(compensated_nll))
        compensated_nll = compensated_nll.sum(axis=2) / (condition_mask.sum(axis=2) + eps) # compute the mean along the second dimension only on observed values

        if mean_z: # no need to multiply by density quotient because we're sampling directly from a proxy of qz_x

            px_mu_z = self.decode(qz_x.mean)

            mu, sigma = px_mu_z.mean, .01

            nll_mean = 0.5 * ( torch.log(2 * torch.tensor(np.pi) * sigma) + (X - mu).pow(2) / sigma)
            #nll_mean = nll_mean.clip(min = 0, max = 1e5)


            condition_mask = (torch.isfinite(nll_mean)) & (mask)
            nll_mean = torch.where(condition_mask, nll_mean, torch.zeros_like(nll_mean))
            nll_mean = nll_mean.sum(axis=2) / (condition_mask.sum(axis=2) + eps) # compute the mean along the second dimension only on observed values

            nll = compensated_nll + nll_mean * .1

        else:

            nll = compensated_nll

        #print(nll)

        return nll

    def kl(self, qz_x, qz_x_corrupted, eps = 1e-3):
        """
        Compute the KL divergence between the 2 gaussians
        """

        mu_z, mu_z_corrupted = qz_x.mean, qz_x_corrupted.mean
        var_z, var_z_corrupted = qz_x.variance, qz_x_corrupted.variance

        kl = .5 * ( ( torch.log(var_z_corrupted) - torch.log(var_z) ).detach() + (var_z + (mu_z_corrupted - mu_z).pow(2) )/ ( var_z_corrupted + eps) )

        return kl

    def prior(self, X, qz_x):

        mask = (X!=0)
        mu_z = qz_x.mean

        #print(mask.shape, mu_z.shape)

        mean_prior = (mu_z.mean() - X[mask].mean()).abs() # the mean of the latent data should be close to the mean of the original data

        loss = mean_prior

        where_no_missing_feats = (mask.sum(axis=2)==0)
        if where_no_missing_feats.int().sum() > 0:
            #print(where_no_missing_feats)
            where_no_missing_feats = where_no_missing_feats.unsqueeze(2).repeat(1,1,mu_z.shape[2])
            #print((mask.sum(axis=2)==0).shape)
            #print(where_no_missing_feats.shape, qz_x.variance.shape)
            var_loss = qz_x.variance[where_no_missing_feats].pow(2).mean()
            #print(qz_x.variance[where_no_missing_feats])
            #print(var_loss)

            loss += var_loss

        
        #variance_prior = (mu_z.var(dim=2).mean() - 1).abs() # the variance of the latent data should be close to 1

        return loss #+ variance_prior

    def temporal_loss(self, X, qz_x, eps = 1e-3):

        batch_size = self.batch_size

        mu = qz_x.mean
        var = qz_x.variance

        mask = (X!=0).int()
        mask_diff = mask[:,1:] - mask[:,:-1]
        X_diff = X[:,1:] - X[:,:-1]
        z_diff = mu[:,1:] - mu[:,:-1]

        std_z = torch.std(mu,axis=(0,1)).unsqueeze(0).unsqueeze(0).detach() + eps
        #var_X = torch.std(X,axis=(0,1)).unsqueeze(0).unsqueeze(0).detach() + eps

        temporal_loss = 0

        for i in range(self.batch_size):

            mask_diff_i_sum = mask_diff[i::self.batch_size].sum(2)
            mask_diff_i_sum[mask_diff_i_sum==0] = 1

            mean_X_diff_i = (X_diff[i::self.batch_size] * mask_diff[i::self.batch_size]).pow(2).sum(2) / mask_diff_i_sum
            mean_z_diff_i = (z_diff[i::self.batch_size] / std_z).pow(2).mean(2)

            #print(mean_z_diff_i.shape, var_z.shape)

            #mean_z_diff_i = mean_z_diff_i / var_z
            #mean_X_diff_i = mean_X_diff_i / var_X

            #temporal_loss += (mean_X_diff_i - mean_z_diff_i).pow(2)
            temporal_loss += (mean_z_diff_i / (mean_X_diff_i + eps))

            # mean diff should be on ratio of diff over variance

        return temporal_loss.clip(min = 1e-3, max = 1e3).mean() / (self.batch_size)

    def temporal_loss(self, X, qz_x, eps = 1e-3):

        batch_size = self.batch_size

        z = qz_x.rsample()
        var = qz_x.variance

        mask = (X!=0).int()
        mask_diff = mask[:,1:] - mask[:,:-1]
        X_diff = X[:,1:] - X[:,:-1]
        z_diff = z[:,1:] - z[:,:-1]
        std_diff = (var[:,1:] + var[:,:-1]).pow(.5)

        z_diff_normalized = z_diff / std_diff

        temporal_loss = 0

        for i in range(self.batch_size):

            #mask_diff_i_sum = mask_diff[i::self.batch_size].sum(2)
            #mask_diff_i_sum[mask_diff_i_sum==0] = 1

            #mean_X_diff_i = (X_diff[i::self.batch_size] * mask_diff[i::self.batch_size]).pow(2).sum(2) / mask_diff_i_sum
            #mean_z_diff_i = (z_diff_normalized[i::self.batch_size]).pow(2).mean(2)

            #temporal_loss += (mean_z_diff_i / (mean_X_diff_i + eps))

            temporal_loss += (z_diff_normalized[i::self.batch_size]).pow(2).mean(2)

        return temporal_loss.clip(min = 1e-3, max = 1e3).mean() #/ (self.batch_size)


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

    def validate_elbo(self, elbo, nll_recon, nll_imputation, kl, z, qz_x, X_ori, X, px_z, tl):
        """Perform assertions, debugging, and optional plotting."""
        assert not (elbo.abs() > 1e8).any(), print('elbo too big', nll_recon.mean().item(), nll_imputation.mean().item(), kl.mean().item(), elbo.mean().item())
        assert not (elbo > 50), print('elbo negative', elbo.item(), nll_recon.mean().item(), nll_imputation.mean().item(), kl.mean().item())
        assert not (torch.isnan(elbo).any()), print('elbo is nan', elbo.item(), nll_recon.mean().item(), nll_imputation.mean().item(), kl.mean().item())

        if len(self.loss_history['elbo']) > 20 and len(self.loss_history['elbo'])%10 == 0:
            loss_ratio = torch.tensor(self.loss_history['temporal_loss'][-20:]).mean().item() / torch.tensor(self.loss_history['nll'][-20:]).mean().item()
            #if loss_ratio < 1: 
            #    self.gamma *= loss_ratio

        if self.forward_passes_counter %800 == 0:
            print('plotting')
            ## Let's shope this doesn't mess up with the optimizer too much, but
            ## I want the recon and temporal losses to be the same order of magnitude, so le'ts 
            ## adapt the gamma

            self.plot_latent_series_and_reconstruction(z, px_z, qz_x, X_ori.detach(), X.detach(), self.latent_dim, -elbo, kl, tl)

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
            plt.semilogy(iterations, self.loss_history[key][::n], label=key)

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
            z_sample = qz_x.rsample()  # Sample from the posterior
            px_z_sample = self.decode(z_sample)  # Reconstruct the data
            X_recon_sample = px_z_sample.mean.detach().cpu().numpy()  # Get the mean of the reconstruction
            # Set first half of non-reconstructed values to NaN
            num_missing_vals = np.sum(X_ori_np == 0)
            X_recon_sample[X_ori_np == 0][:num_missing_vals // 2] == np.nan
            reconstructions.append(X_recon_sample)

            nll_loss += self.nll(X_ori, z_sample, qz_x_ori, qz_x, eps = 1e-3)[0].detach().cpu().numpy()/10.

            #print(nll_loss)
        
        kl_loss = self.kl(qz_x_ori, qz_x, eps = 1e-3)[0].detach().cpu().numpy()

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

        plt.subplot(4, 1, 4)
        #mask, mask_ori = (X != 0).to(self.device), (X_ori != 0).to(self.device)
        #qz_x = self.encode(X_ori)
        #qz_x_corrupted = self.encode(X)
        #z = qz_x.rsample()
        #imputation_error = self.nll(X, z, qz_x, qz_x_corrupted)[0].detach().cpu().numpy()
        #plt.semilogy(imputation_error, label = 'nll error')
        nll_loss = np.array(nll_loss) -  .5 * np.log(2 * np.pi * 0.01)
        plt.semilogy(kl_loss, label = 'kl_loss')
        plt.semilogy(nll_loss, label = 'nll error')
        plt.legend()

        plt.title('Log probability of original z belonging to the corrupted Gaussian')

        losses = f'kl = {kl.mean().item()} - nll = {nll.mean().item()} - temporal {tl.item()}'
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


        plt.title('Latent Time Series (Mean and Variance) ' + losses)
        plt.xlabel('Time Steps')
        plt.ylabel('Latent Values')
        #plt.legend(bbox_to_anchor=[1.2, 0.3])

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
