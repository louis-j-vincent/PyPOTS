import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
import multiprocessing

import torch
import os
import glob
from pygrinder import mcar
import json

import gc

import psutil
import os

def log_resources():
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    print(f"Memory usage: {mem_info.rss / 1e6:.2f} MB")
    print(f"CPU usage: {psutil.cpu_percent()}%")


from pypots.imputation import GP_VAE
from pypots.utils.metrics import calc_mae
from pypots.optim.adam import Adam
from torch.optim.lr_scheduler import LRScheduler

import torch
import numpy as np
import matplotlib.pyplot as plt

from pypots.imputation.gp_ae.gp_model import ProbabilisticGP

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


def get_errors(gpvae, X, num_samples=10, p=0.2, plots = False):
    """
    Computes reconstruction errors and normalized errors for original and corrupted data
    using a Gaussian Process Variational Autoencoder (GPVAE).

    Parameters:
    - gpvae: The GPVAE model instance.
    - X: The input data (numpy array or torch tensor).
    - num_samples: Number of samples to draw for the stochastic encoding.
    - p: Probability of missing data (for corruption).

    Returns:
    - A list containing:
        - error: Mean squared error on the original data.
        - error_corrupted: Mean absolute error on the corrupted data.
        - normalized_error_corrupted_imp: Normalized error for imputed (missing) data.
        - normalized_error_corrupted_recon: Normalized error for observed data.
    """

    # Ensure X is a numpy array and handle NaNs by replacing them with zeros
    X = np.nan_to_num(X, nan=0)

    # Introduce missing completely at random (MCAR) corruption
    X_corrupted = mcar(X, p)  # Assuming mcar function is defined elsewhere
    X_corrupted = np.nan_to_num(X_corrupted, nan=0)

    # Convert data to torch tensors
    X_tensor = torch.tensor(X, dtype=torch.float32)
    X_corrupted_tensor = torch.tensor(X_corrupted, dtype=torch.float32)

    # Encode and decode the original data
    z = gpvae.model.backbone.encode(X_tensor).mean
    X_recon = gpvae.model.backbone.decode(z).mean.detach().cpu().numpy()

    # Encode and decode the corrupted data with sampling
    z_corrupted = gpvae.model.backbone.encode(X_corrupted_tensor).rsample(sample_shape=(num_samples,))
    X_recon_corrupted = np.array([gpvae.model.backbone.decode(z).mean.detach().cpu().numpy() for z in z_corrupted])

    # Compute mean squared error for the original data (excluding zeros)
    mask_nonzero = X != 0
    error = np.mean((X[mask_nonzero] - X_recon[mask_nonzero]) ** 2)

    # Compute mean absolute error for the corrupted data
    error_list = []
    for x_recon in X_recon_corrupted:
        mask_observed = X_corrupted != 0
        error_sample = np.mean(np.abs(X[mask_observed] - x_recon[mask_observed]))
        error_list.append(error_sample)
    error_corrupted = np.mean(error_list)

    embedding = gpvae.model.backbone.encode(X_corrupted_tensor)
    imputed_data = gpvae.gp.infer(embedding, X_corrupted_tensor)

    error_gp_imputation = (imputed_data - X_tensor).pow(2)[X_tensor!=0].mean().item()

    X_recon = gpvae.model.backbone.decode(embedding.mean).mean  

    # Compute standard deviation across the reconstructed samples
    X_recon_corrupted_std = np.std(X_recon_corrupted, axis=0)

    # Compute normalized error
    eps = 1e-1  # Small constant to avoid division by zero
    abs_errors = np.mean(np.abs(X_recon_corrupted - X), axis=0)
    normalized_error = abs_errors / (X_recon_corrupted_std + eps)

    # Compute normalized errors for imputed (missing) and observed data
    mask_missing = X_corrupted == 0
    mask_observed = ~mask_missing
    normalized_error_corrupted_imp = normalized_error[mask_missing].mean()
    normalized_error_corrupted_recon = normalized_error[mask_observed].mean()

    error_corrupted_imp = abs_errors[mask_missing].mean()
    error_corrupted_recon = abs_errors[mask_observed].mean()

    if plots:


        plt.plot(X_recon[0])
        plt.plot(X[0],'o')
        plt.plot(X_corrupted[0],'+')
        plt.show()

        # Plot distributions of errors, standard deviations, and normalized errors
        plt.figure(figsize=(12, 6))
        plt.hist((normalized_error * X_recon_corrupted_std).flatten(), bins=50, density=True, alpha=0.5, label='Errors')
        plt.hist(X_recon_corrupted_std.flatten(), bins=50, density=True, alpha=0.5, label='Standard Deviations')
        plt.hist(normalized_error.flatten(), bins=100, density=True, alpha=0.5, label='Normalized Errors')
        plt.legend()
        plt.xlim([0, 30])
        plt.ylim([0, 1.2])
        plt.title('Distribution of Errors and Standard Deviations')
        plt.xlabel('Value')
        plt.ylabel('Density')
        plt.show()
    
        # Plot reconstructed samples and original data for the first instance
        X_plot = X.copy()
        X_plot[X_plot == 0] = np.nan  # Replace zeros with NaN for plotting
        xx = np.arange(X.shape[1])
    
        plt.figure(figsize=(12, 6))
        for i in range(min(10, num_samples)):
            plt.plot(X_recon_corrupted[i, 0], alpha=0.3)
        plt.plot(xx, X_plot[0], '+', label='Original Data')
        plt.plot(xx, X_corrupted[0], 'o', label='Corrupted Data')
        plt.legend()
        plt.title('Reconstruction of the First Sample')
        plt.xlabel('Feature Index')
        plt.ylabel('Value')
        plt.show()
    
        # Detailed plotting for missing data
        indices_missing = np.where(mask_missing.flatten())[0][:100]  # Get indices of missing data
        fig, ax = plt.subplots(2, figsize=(18, 8))
    
        # Plot reconstructed values for missing data
        ax[0].plot(X_recon_corrupted[:, mask_missing].reshape(num_samples, -1)[:, :100].T, color='k', alpha=0.5)
        ax[0].plot(X[mask_missing].flatten()[:100], 'o', label='Original Missing Data')
        ax[0].set_title('Reconstructed Missing Data')
        ax[0].set_xlabel('Sample Index')
        ax[0].set_ylabel('Value')
        ax[0].legend()
    
        # Plot normalized errors for missing data
        ax[1].semilogy(normalized_error[mask_missing].flatten()[:100])
        ax[1].set_title('Normalized Errors for Missing Data')
        ax[1].set_xlabel('Sample Index')
        ax[1].set_ylabel('Normalized Error')
        plt.tight_layout()
        plt.show()

    return [error, error_corrupted, normalized_error_corrupted_imp, normalized_error_corrupted_recon, error_corrupted_imp, error_corrupted_recon, error_gp_imputation]

print(f'Checking cuda availability {torch.cuda.is_available()} and device count {torch.cuda.device_count()}')

# Parameters
n_groups = 1
n_observations_per_group = 5000
n_dimensions = 3
n_time_points = 30
time_points = np.linspace(0, 10, n_time_points)
length_scale = 1.5

# Generate GP samples
gp_samples = generate_gp_samples_with_library(
    n_groups=n_groups,
    n_observations_per_group=n_observations_per_group,
    n_dimensions=n_dimensions,
    time_points=time_points,
    length_scale=length_scale
)

n_new_dims = 5
Proj = np.random.uniform(size = (n_dimensions, n_new_dims))#.reshape(1,1,n_dimensions, n_new_dims)
X_ori = gp_samples[0] @ Proj

# Normalize

X_normalized = (X_ori - X_ori.mean(axis=(0,1)).reshape(1,1,-1)) / X_ori.std(axis=(0,1)).reshape(1,1,-1)#[:100]
X = mcar(X_normalized.copy(), 0.1)  # randomly hold out 10% observed values as ground truth
X[X!=X] = 0
dataset = {'X' : X}

len_dataset = len(dataset['X'])
cut = int(len_dataset*0.7)
dataset_train, dataset_val = {'X':dataset['X'][:cut]}, {'X':dataset['X'][cut:], 'X_ori':X_normalized[cut:]}

#plt.plot(X_ori[1]);

list_p_dataset = [0.1, 0.2, 0.3, 0.4, 0.5]
print(list_p_dataset)
list_alpha = [1]
list_beta = [1]
list_gamma = [1e-10, 1e-5, 1e-1]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Specify the folder path
folder_path = 'latent_plots'

errors = {}

for beta in list_beta:
    for gamma in list_gamma:
        for p_dataset in list_p_dataset:    
            for alpha in list_alpha:

                try:
                    torch.cuda.empty_cache()

                    # Use glob to get a list of image files (you can add more extensions as needed)
                    image_files = glob.glob(os.path.join(folder_path, 'latent_series_*'))  # Matches jpg, png, etc
                    last_image = os.path.basename(image_files[-1])  # Get the last image's name
                    os.rename(os.path.join(folder_path,last_image), f'latent_plots/last_recon_dataset_{p_dataset}-a_{alpha}_b_{beta}_gamma_{gamma}.png')
                    os.rename('latent_plots/losses.png', f'latent_plots/losses_p_dataset_{p_dataset}-a_{alpha}_b_{beta}_gamma_{gamma}.png')
                    os.rename('latent_plots/params.png', f'latent_plots/params_p_dataset_{p_dataset}-a_{alpha}_b_{beta}_gamma_{gamma}.png')
                    for img in image_files[:-1]:
                        os.remove(img)

                    errors[f'p_dataset_{p_dataset}-a_{alpha}_b_{beta}_gamma_{gamma}'] = get_errors(gpvae, X)
                    
                    torch.save(gpvae, f'gvpae_p_dataset_{p_dataset}-a_{alpha}_b_{beta}_gamma_{gamma}')
                    
                    # Call this after each fit
                    log_resources()
    
                    #print(errors)
                    with open('errors4.json','w') as f:
                        json.dump(errors, f)

                    # After each grid search iteration
                    del gpvae
                    gc.collect()
                    torch.cuda.empty_cache()

                    #multiprocessing.resource_tracker.ResourceTracker()._cleanup()

                                    # Call this after each fit
                    log_resources()

                except:
                    print('Couldnt save model')

                print(f'Running xp for p_dataset_{p_dataset}-a_{alpha}_b_{beta}_gamma_{gamma}')

                
                X = (X_ori - X_ori.mean(axis=(0,1)).reshape(1,1,-1)) / X_ori.std(axis=(0,1)).reshape(1,1,-1)
                X = mcar(X, p_dataset)  # randomly hold out 10% observed values as ground truth
                X[X!=X] = 0
                dataset = {'X' : X}

                try:
                    gpvae = torch.load(f'gvpae_p_dataset_{p_dataset}-a_{alpha}_b_{beta}_gamma_{gamma}')
                    print('Loaded model')
                except:
                
                    gpvae = GP_VAE(n_steps = dataset['X'].shape[1], 
                            n_features = dataset['X'].shape[2], 
                            latent_size = 5, 
                            epochs = 100, 
                            batch_size = 64,
                            beta = beta, 
                            K = 5,  
                            encoder_sizes = (128,64,32), 
                            decoder_sizes = (128,32),
                            optimizer = Adam()
                            )

                    gpvae.model.backbone.device = device
                
                    gpvae.model.backbone.gamma = gamma
                    gpvae.model.backbone.alpha = alpha
                    gpvae.model.backbone.p = 0.1
                    # Here I use the whole dataset as the training set because ground truth is not visible to the model, you can also split it into train/val/test sets
                    #with torch.autograd.set_detect_anomaly(False):
                    gpvae.model.backbone.device = device
                    gpvae.fit(dataset)  # train the model on the dataset

                    # Use glob to get a list of image files (you can add more extensions as needed)
                    image_files = glob.glob(os.path.join(folder_path, 'latent_series_*'))  # Matches jpg, png, etc
                    last_image = os.path.basename(image_files[-1])  # Get the last image's name
                    os.rename(os.path.join(folder_path,last_image), f'latent_plots/last_recon_dataset_{p_dataset}-a_{alpha}_b_{beta}_gamma_{gamma}.png')
                    os.rename('latent_plots/losses.png', f'latent_plots/losses_p_dataset_{p_dataset}-a_{alpha}_b_{beta}_gamma_{gamma}.png')
                    os.rename('latent_plots/params.png', f'latent_plots/params_p_dataset_{p_dataset}-a_{alpha}_b_{beta}_gamma_{gamma}.png')
                    for img in image_files[:-1]:
                        os.remove(img)
                    
                gpvae.device = torch.device('cpu')
                gpvae.model.backbone.device = torch.device('cpu')

                gpvae.gp = ProbabilisticGP(gpvae.model.backbone, assemble_data = gpvae._assemble_input_for_training, n_dims = 5)
                gpvae.gp.device = torch.device('cpu')

                gpvae.fit_kernel(dataset)  # train the model on the dataset

                torch.cuda.empty_cache()

                errors[f'p_dataset_{p_dataset}-a_{alpha}_b_{beta}_gamma_{gamma}'] = get_errors(gpvae, X)
                
                torch.save(gpvae, f'gvpae_p_dataset_{p_dataset}-a_{alpha}_b_{beta}_gamma_{gamma}')
                
                # Call this after each fit
                log_resources()
  
                #print(errors)
                with open('errors3.json','w') as f:
                    json.dump(errors, f)

                # After each grid search iteration
                del gpvae
                gc.collect()
                torch.cuda.empty_cache()

                #multiprocessing.resource_tracker.ResourceTracker()._cleanup()

                                # Call this after each fit
                log_resources()

