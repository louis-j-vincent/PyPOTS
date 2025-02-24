import os
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import torch
from typing import Any, Optional, Dict, Callable

def plot_params(
    monitoring_history: Dict[str, Any],
    loss_history: Dict[str, Any],
    alpha: float,
    beta: float,
    gamma: float,
    save_path: Optional[str] = None,
) -> None:
    """
    Plot monitored parameters over training.
    
    Parameters:
        monitoring_history (dict): Dictionary containing monitoring arrays.
        loss_history (dict): Dictionary containing loss history arrays.
        alpha (float): The model's alpha parameter.
        beta (float): The model's beta parameter.
        gamma (float): The model's gamma parameter.
        save_path (str, optional): Where to save the plot; defaults to 'latent_plots/params.png'.
    """
    max_points = 200
    num_points = len(loss_history.get("elbo", []))
    n = max(1, num_points // max_points)
    decimation = 10

    plt.figure(figsize=(12, 8))
    styles = ["-", "o", "+", ":"]
    params_to_plot = ["z_mu_var", "z_var_mean", "z_var_var"]

    for i, param in enumerate(params_to_plot):
        data = np.array(monitoring_history.get(param, []))[::decimation]
        plt.semilogy(np.abs(data), styles[i], label=param)

    # Plot z variance ratio if data is available.
    if "z_mu_var" in monitoring_history and "z_var_mean" in monitoring_history:
        z_mu_var = np.array(monitoring_history["z_mu_var"])[::decimation]
        z_var_mean = np.array(monitoring_history["z_var_mean"])[::decimation]
        with np.errstate(divide="ignore", invalid="ignore"):
            z_var_ratio = np.where(z_mu_var != 0, z_var_mean / z_mu_var, 0)
        plt.semilogy(z_var_ratio, linewidth=2, linestyle=":", label="z var ratio")

    plt.xlabel("Iterations")
    plt.ylabel("Value")
    plt.title(f"Monitored Parameters Over Training - α={alpha}, β={beta}, γ={gamma}")
    plt.legend(bbox_to_anchor=[1.2, 0.3])
    plt.grid()

    if save_path is None:
        save_path = os.path.join("latent_plots", "params.png")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_losses(loss_history: Dict[str, Any], save_path: Optional[str] = None) -> None:
    """
    Plot the loss components over training iterations.
    
    Parameters:
        loss_history (dict): Dictionary containing loss arrays.
        save_path (str, optional): Where to save the plot; defaults to 'latent_plots/losses.png'.
    """
    max_points = 200
    num_points = len(loss_history.get("elbo", []))
    n = max(1, num_points // max_points)
    iterations = list(range(1, num_points + 1))[::n]

    plt.figure(figsize=(10, 6))
    for key, values in loss_history.items():
        plt.semilogy(iterations, np.array(values)[::n], label=key)

    plt.xlabel("Iteration")
    plt.ylabel("Loss (log scale)")
    plt.title("Loss Components over Iterations")
    plt.legend(bbox_to_anchor=[1.2, 0.3])
    plt.grid(True)
    if save_path is None:
        save_path = os.path.join("latent_plots", "losses.png")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_latent_series_and_reconstruction(
    qz_x: Any,
    X_ori: torch.Tensor,
    X: torch.Tensor,
    latent_dim: int,
    losses: Dict[str, float],
    decoder: Callable[[torch.Tensor], Any],
    n_samples: int = 100,
    folder: str = "latent_plots",
) -> None:
    """
    Plots the latent time series and a confidence interval for the reconstructed data.
    
    For each feature (modality) of the first observation in the batch, this function:
      - Computes the mean, lower bound (min), and upper bound (max) of n_samples reconstructions.
      - Plots the mean reconstruction along with a shaded confidence interval (using fill_between with alpha=0.1).
      - Overlays the original data using a line (with missing values replaced by NaN) and scatter markers:
          * Observed points are marked as circles ('o').
          * Unobserved points are marked as crosses ('+').
    
    Parameters:
        qz_x (Any): The q(z|x) object with attributes 'mean' and 'variance'.
        X_ori (torch.Tensor): The original input data (shape: [batch, time, features]).
        X (torch.Tensor): The processed input data (same shape as X_ori).
        latent_dim (int): Dimensionality of the latent space.
        losses (Dict[str, float]): Dictionary of loss components (e.g., kl, nll) for display.
        decoder (Callable): A function to decode latent samples to reconstructed data. It should return an object with a 'mean' attribute.
        n_samples (int): Number of reconstruction samples to generate.
        folder (str): Directory in which to save the plot.
    """
    # --- Latent Time Series Subplot ---
    # Extract latent statistics (assumes qz_x.mean and qz_x.variance are tensors)
    z_mean = qz_x.mean[0].detach().cpu().numpy()       # shape: [time, latent_dim]
    z_var  = qz_x.variance[0].detach().cpu().numpy()    # shape: [time, latent_dim]
    
    # --- Reconstruction Confidence Interval ---
    # Generate multiple reconstructions using the provided decoder.
    reconstructions_list = []
    for i in range(n_samples):
        z_sample = qz_x.rsample()  # Sample from the latent distribution
        px_z_sample = decoder(z_sample)  # Decode the latent sample
        # Assume px_z_sample.mean is a tensor of shape [batch, time, features]
        X_recon_sample = px_z_sample.mean.detach().cpu().numpy()
        reconstructions_list.append(X_recon_sample)
    
    # Convert to a NumPy array: shape [n_samples, batch, time, features]
    reconstructions = np.array(reconstructions_list)
    
    # Compute the mean, min, and max across the reconstruction samples (for each time & feature)
    mean_recon   = reconstructions.mean(axis=0)   # shape: [batch, time, features]
    lower_bound  = reconstructions.min(axis=0)
    upper_bound  = reconstructions.max(axis=0)
    
    # --- Prepare Original Data for Plotting ---
    # Get the original data as a NumPy array without modification
    X_ori_np = X_ori.detach().cpu().numpy().copy()  # shape: [batch, time, features]
    X_ori_np[X_ori_np==0] = np.nan

    # For the line plot, create a version with missing values replaced by NaN
    X_np_line = X.detach().cpu().numpy().copy()
    # Create a boolean mask: observed if value != 0; missing if value == 0.
    observed_mask = (X_np_line != 0)
    X_np_line[~observed_mask] = np.nan
    
    # Time axis (assumes time dimension is axis 1)
    time_steps = X_ori_np.shape[1]
    t = np.arange(time_steps)
    
    # --- Plotting ---
    colors = ["blue", "green", "red", "orange", "purple", "brown", "cyan", "magenta", "yellow", "black"]
    
    plt.figure(figsize=(15, 8))
    
    # Subplot 1: Latent Time Series with Confidence (using fill_between for uncertainty)
    plt.subplot(2, 1, 1)
    for dim in range(latent_dim):
        plt.plot(t, z_mean[:, dim], label=f"Latent dim {dim} Mean", color=colors[dim % len(colors)])
        plt.fill_between(
            t,
            z_mean[:, dim] - np.sqrt(z_var[:, dim]),
            z_mean[:, dim] + np.sqrt(z_var[:, dim]),
            alpha=0.2,
            color=colors[dim % len(colors)]
        )
    plt.xlabel("Time Steps")
    plt.ylabel("Latent Values")
    plt.title("Latent Time Series")
    #plt.legend(loc="upper right")
    plt.grid(True)
    
    # Subplot 2: Original vs. Reconstruction with Confidence Interval and Markers for Observed/Missing
    plt.subplot(2, 1, 2)
    num_features = X_ori_np.shape[2]
    for dim in range(num_features):
        # Plot the mean reconstructed signal for this feature
        plt.plot(t, mean_recon[0, :, dim], label=f"Feature {dim} Mean", color=colors[dim % len(colors)])
        # Plot the confidence interval using fill_between with alpha=0.1
        plt.fill_between(t,
                         lower_bound[0, :, dim],
                         upper_bound[0, :, dim],
                         color=colors[dim % len(colors)],
                         alpha=0.1)
        # Plot the original data line (with missing values as gaps)
        plt.plot(t, X_ori_np[0, :, dim], linestyle="--", color=colors[dim % len(colors)], label=f"Feature {dim} Original")
        # Scatter plot the observed and missing points:
        # Observed points (where observed_mask is True)
        plt.plot(X_np_line[0, :, dim], "o", color=colors[dim % len(colors)], label=f"Feature {dim} Observed")
        # Missing points (where observed_mask is False)
        miss_idx = t[~observed_mask[0, :, dim]]
        # For missing points, we still want to show their predicted original value if available.
        plt.plot(t, X_ori_np[0, :, dim], '+', color=colors[dim % len(colors)], label=f"Feature {dim} Unobserved")
    plt.xlabel("Time Steps")
    plt.ylabel("Feature Values")
    plt.title("Original vs. Reconstruction (Confidence Interval)")
    #plt.legend(loc="upper right", fontsize="small")
    plt.grid(True)
    
    # Save the plot
    os.makedirs(folder, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_path = os.path.join(folder, f"latent_series_and_reconstruction_{timestamp}.png")
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()
