import torch
import numpy as np
import matplotlib.pyplot as plt

def reconstruct_with_given_kernel_params(X, kernel_params, gpvae, return_variance = False, n_samples=100):
    """
    Reconstructs data using GP-VAE with given kernel parameters, sampling multiple times from the latent distribution.

    Parameters:
    - X: Input data
    - kernel_params: List of kernel parameters
    - gpvae: Trained GP-VAE model
    - n_samples: Number of samples to draw from the latent Gaussian

    Returns:
    - x_input: The original input with NaNs for missing values
    - x_recon_mean: Mean reconstruction across samples
    - x_recon_min: Minimum reconstructed values across samples
    - x_recon_max: Maximum reconstructed values across samples
    """

    # Convert input to tensor
    x_input = torch.tensor(X).float()

    # Encode to obtain latent parameters (mean & variance)
    qz_x = gpvae.model.backbone.encode(x_input)
    z_mu, z_var = np.array(qz_x.mean.detach().cpu().numpy()), np.array(qz_x.variance.detach().cpu().numpy())

    # Update kernel_params to match batch size and latent dimensions
    #kernel_params = torch.tensor(kernel_params).reshape(1, 1, 4).repeat(z_mu.shape[0], z_mu.shape[2], 1)

    # Plot the original latent mean
    plt.plot(z_mu[0], 'o', label="z_mu (mean)")
    plt.gca().set_prop_cycle(None)

    # Apply GP correction to latent space
    if return_variance:

        z_star, z_star_var = gpvae.gp.correct_with_gp(z_mu, z_var, kernel_params, return_variance=True)

        z_star_var = z_star_var.clamp(min=0.)**.5
   
        # Plot corrected latent mean
        plt.plot(z_star[0], label="z_star (corrected)")
        plt.gca().set_prop_cycle(None)
        for i in range(z_star.shape[-1]):
            plt.fill_between(np.arange(30), (z_star-2*z_star_var)[0,:,i].detach(), (z_star+2*z_star_var)[0,:,i].detach(), alpha = .2)
        plt.show()

        # Sample from Gaussian: z ~ N(z_star, z_star_var)
        z_samples = torch.randn((n_samples,) + z_star.shape) * torch.sqrt(z_star_var) + z_star

        # Decode each sampled latent representation
        x_recon_samples = gpvae.model.backbone.decode(z_samples).mean  # Shape: (n_samples, batch, seq, features)

        # Compute mean, min, and max across samples
        x_recon_mean = x_recon_samples.mean(dim=0)  # Average over n_samples
        x_recon_min = x_recon_samples.min(dim=0).values  # Minimum over n_samples
        x_recon_max = x_recon_samples.max(dim=0).values  # Maximum over n_samples

        # Mask missing values in x_input with NaNs
        x_input[x_input == 0] = torch.nan

        return x_input, x_recon_mean, x_recon_min, x_recon_max

    else:

        z_star = gpvae.gp.correct_with_gp(z_mu, z_var, kernel_params)

        # Plot corrected latent mean
        plt.plot(z_star[0].detach(), label="z_star (corrected)")
        plt.show()
        
        x_recon_samples = gpvae.model.backbone.decode(z_star).mean  # Shape: (n_samples, batch, seq, features)

        return x_input, x_recon_samples




import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF

def generate_gp_samples_with_library(n_groups, n_observations_per_group, n_dimensions, time_points, length_scale=1.0):
    """
    Generate multidimensional GP samples using scikit-learn.
    Each dimension has the same underlying process.
    """
    kernel = RBF(length_scale=length_scale)
    gp = GaussianProcessRegressor(kernel=kernel, alpha=1e-8)

    gp_sample_constant = gp.sample_y(time_points.reshape(-1, 1), random_state=None).flatten()

    samples = []
    for _ in range(n_groups):
        group_samples = []
        for _ in tqdm(range(n_observations_per_group)):
            # Generate independent samples for each dimension
            gp_samples_per_dim = []
            for _ in range(n_dimensions):
                gp_sample = gp.sample_y(time_points.reshape(-1, 1), random_state=None).flatten()
                gp_samples_per_dim.append(gp_sample)
            # Stack the samples to form a multidimensional sample
            gp_samples_multidim = np.stack(gp_samples_per_dim, axis=-1)
            group_samples.append(gp_samples_multidim)
        samples.append(np.array(group_samples))

    return samples


if False:
    # Plot example
    for dim in range(n_dimensions):
        for group_idx in range(n_groups):
            plt.figure()
            for observation in gp_samples[group_idx]:
                plt.plot(time_points, observation[:, dim], label=f"Dim {dim + 1}")
            plt.title(f"Group {group_idx + 1} Observations for Dimension {dim + 1}")
            plt.xlabel("Time")
            plt.ylabel("Value")
            plt.legend()
            plt.show()


def plot_reconstruction(X, X_true, gpae):
    """
    plot reconstruction of X and ground truth X_true
    """

    X_torch = torch.tensor(X).float()
    embedding = gpvae_pc.model.backbone.encode(X_torch)

    z_samples = embedding.rsample((100,))

    print(z_samples.shape)

    X_recon = gpae.model.backbone.decode(z_samples).mean.detach()

    X_recon_mean = X_recon.mean(axis=0)
    X_recon_min, X_recon_max = X_recon.min(axis=0).values, X_recon.max(axis=0).values

    i = 2
    X_torch[X_torch==0] = torch.nan
    plt.gca().set_prop_cycle(None)
    plt.plot(X_recon_mean[i].detach(), label = 'recon')
    for j in range(X_recon_mean.shape[-1]):
        plt.fill_between(np.arange(30), X_recon_min[i,:,j], X_recon_max[i,:,j], alpha = .2)
    plt.gca().set_prop_cycle(None)
    plt.plot(X_torch[i].detach(), 'o');
    plt.gca().set_prop_cycle(None)
    plt.plot(X_true[i], '+');

def plot_reconstruction_with_GP(X, X_true, gpae, i = 0):
    
    # Example usage
    X[X!=X] = 0.
    kernel_params = gpae.gp.update_kernel_params(torch.tensor(X).float())
    x_input, x_recon_mean, x_recon_min, x_recon_max = reconstruct_with_given_kernel_params(
        X, kernel_params, gpae, return_variance = True
    )

    x_input[x_input==0] = np.nan
    plt.plot(x_input[i].detach(),'o');
    plt.gca().set_prop_cycle(None);
    plt.plot(x_recon_mean[i].detach());
    plt.gca().set_prop_cycle(None);
    for j in range(x_recon_mean.shape[2]):
        plt.fill_between(np.arange(30),
                        np.array(x_recon_min[i,:,j].detach()),
                        np.array(x_recon_max[i,:,j].detach()),
                        alpha = .2)
    plt.plot(X_true[i],'+')
    y_min, y_max = np.min(X_true[i]) - .5, np.max(X_true[i]) + .5
    plt.ylim([y_min, y_max]);

    MSE = (X_true - x_recon_mean.detach().numpy())**2

    mask_observed = (X!=0)
    mask_hidden = (X_true!=0)&(~mask_observed)
    MSE_observed = MSE[mask_observed].mean()
    MSE_hidden = MSE[mask_hidden].mean()

    return MSE_observed, MSE_hidden


