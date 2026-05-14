## DATA GENERATION ##

import random
import tsdb
import argparse



random.seed(10)

import sys
sys.path.append('/Users/louis/Documents/Phd/Code_pour_manuscript/PyPOTS/ipynbs')
sys.path.append('/Users/louis/Documents/Phd/Code_pour_manuscript/PyPOTS')
print(sys.path)

import os

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from pygrinder import mcar

import torch
import torch.distributions as dist
from pypots.imputation import GP_VAE_posterior_concistency, GP_VAE, GP_UAE
from pypots.optim.adam import Adam

#from hyperparameter_search import evaluate_model as evaluate_model_fullmetrics

import os
import time

from scipy.stats import kstest, norm


# ==========================
# 🚀 Generate GP Samples (New Function)
# ==========================
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

def Generate_samples(n_groups=2, n_observations_per_group=5000, n_dimensions=30, n_time_points=30,
                     time_points=np.linspace(0, 10, 30), length_scale=1.5, n_new_dims=7):
    """Generate synthetic dataset with Gaussian Process and structured missingness."""
    
    gp_samples = generate_gp_samples_with_library(
        n_groups=n_groups, n_observations_per_group=n_observations_per_group,
        n_dimensions=n_dimensions, time_points=time_points, length_scale=length_scale
    )

    mean, var = [0,1], [2,2]
    X_ori = np.vstack([(gp_samples[i] + mean[i])*var[i] for i in range(n_groups)])

    Proj = np.random.uniform(size=(n_dimensions, n_new_dims))
    
    X_ori = X_ori @ Proj

    len0, len1 = gp_samples[0].shape[0] * gp_samples[0].shape[1], gp_samples[1].shape[0] * gp_samples[1].shape[1]
    y = np.vstack([np.zeros(gp_samples[0].shape), np.ones(gp_samples[1].shape)])

    X_normalized = (X_ori - X_ori.mean(axis=(0, 1)).reshape(1, 1, -1)) / X_ori.std(axis=(0, 1)).reshape(1, 1, -1)
    #X = corrupt_by_dim(X_normalized.copy())
    return X_normalized, y[:,:,0]

def generate_missingness(X_original, p_dataset = .1):
    
    X = mcar(X_original.copy(), p_dataset)  # randomly hold out 10% observed values as ground truth
    X[np.isnan(X)] = 0  # Replace NaNs with zeros

    dataset = {'X': X}
    len_dataset = len(dataset['X'])
    cut = int(len_dataset * 0.7)
    dataset_train, dataset_val = {'X': dataset['X'][:cut]}, {'X': dataset['X'][cut:], 'X_ori': X_original[cut:]}

    return X, dataset_train, dataset_val

# ==========================
# 🚀 Utility Functions
# ==========================

def compute_rmse(true_data, imputed_data, missing_mask):
    """Compute RMSE between imputed and true values, considering only missing values."""
    missing_values = missing_mask.astype(bool)
    mse = np.mean((true_data[missing_values] - imputed_data[missing_values]) ** 2)
    return np.sqrt(mse)

def compute_nll(recon_mu, recon_var, true_data, missing_mask):
    """Compute the Negative Log-Likelihood (NLL) for missing values."""
    missing_values = missing_mask.astype(bool)
    recon_var = np.clip(recon_var, 1e-6, None)  # Avoid division by zero
    nll = 0.5 * np.sum(
        (true_data[missing_values] - recon_mu[missing_values])**2 / recon_var[missing_values]
        + np.log(recon_var[missing_values]) + np.log(2 * np.pi)
    )
    return nll

def compute_kl_divergence(q_mu, q_var, p_mu=0, p_var=1):
    """Compute KL divergence between approximate posterior q(z|x) and prior p(z)."""
    q_dist = dist.Normal(q_mu, torch.sqrt(q_var))
    p_dist = dist.Normal(p_mu, torch.sqrt(torch.tensor(p_var)))
    kl_div = torch.distributions.kl_divergence(q_dist, p_dist)
    return kl_div.sum().item()

# ==========================
# 🚀 Benchmarking methods
# ==========================

def mean_impute(X_train, X):
    """Replaces missing values with the mean of observed values per feature."""

    X_flat = np.copy(X.reshape(-1, X.shape[-1]))  # Flatten time-series data for kNN
    X_flat[X_flat==0] = np.nan

    X_train = np.copy(X_train.reshape(-1, X_train.shape[-1]))
    X_train[X_train==0] = np.nan

    mask = (X_train == 0)  # Assuming 0 represents missing values
    #print(np.nanmean(np.where(mask, np.nan, X_imputed), axis=(0, 1)).shape)

    feature_means = np.nanmean(X_train, axis = 0)
    for j in range(X_train.shape[-1]):
        X_flat_j = X_flat[:,j]
        X_flat_j[X_flat_j!=X_flat_j] = feature_means[j]


    #feature_means = np.nanmean(np.where((X_train == 0), np.nan, X_train), axis=(0, 1))[None, None, :]
    #print(feature_means.shape, X_imputed.shape)
    #X_imputed[mask] = np.take(feature_means, np.where(mask)[-1])

    return X_flat.reshape(X.shape)

from sklearn.impute import KNNImputer

def knn_impute(X_train, X, k=5):
    """Impute missing values using k-Nearest Neighbors."""
    X_flat = np.copy(X.reshape(-1, X.shape[-1]))  # Flatten time-series data for kNN
    X_flat[X_flat==0] = np.nan
    imputer = KNNImputer(n_neighbors=k)

    X_train = np.copy(X_train.reshape(-1, X_train.shape[-1]))
    X_train[X_train==0] = np.nan

    imputer.fit(X_train)

    X_imputed_flat = imputer.transform(X_flat)

    #print(X_imputed_flat.reshape(X.shape)[0], X[0])

    return X_imputed_flat.reshape(X.shape)

#from missingpy import MissForest
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

def missforest_impute(X):
    """Impute missing values using MissForest (Random Forest-based Imputation)."""
    X_flat = np.copy(X.reshape(-1, X.shape[-1]))  # Flatten time-series data for kNN
    X_flat[X_flat==0] = np.nan
    #imputer = MissForest()
    imputer = IterativeImputer(max_iter=10, random_state=0)


    X_train = np.copy(X_train.reshape(-1, X_train.shape[-1]))
    X_train[X_train==0] = np.nan

    imputer.fit(X_train)

    X_imputed_flat = imputer.transform(X_flat)

    #print(X_imputed_flat.reshape(X.shape)[0], X[0])

    return X_imputed_flat.reshape(X.shape)

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

def mice_impute(X_train, X):
    """Impute missing values using MissForest (Random Forest-based Imputation)."""
    X_flat = np.copy(X.reshape(-1, X.shape[-1]))  # Flatten time-series data for kNN
    X_flat[X_flat==0] = np.nan
    #imputer = MissForest()
    imputer = IterativeImputer(max_iter=10, random_state=0)


    X_train = np.copy(X_train.reshape(-1, X_train.shape[-1]))
    X_train[X_train==0] = np.nan

    imputer.fit(X_train)

    X_imputed_flat = imputer.transform(X_flat)

    #print(X_imputed_flat.reshape(X.shape)[0], X[0])

    return X_imputed_flat.reshape(X.shape)

from torch.optim.lr_scheduler import ReduceLROnPlateau
from pypots.optim.lr_scheduler import multiplicative_lrs, step_lrs

from sklearn.impute import KNNImputer

# ==========================
# 🚀 Model Training Function
# ==========================
def train_model(X_train, 
    X_test,
    X_original,
    latent_size=8, 
    epochs=300, 
    posterior_consistency=False, 
    alpha = 1e-1,
    beta = 1e0,
    sigma = 1e-2,
    detach_for_kl=True,
    p = 0.1,
    DAE=False,
    sampling=True,
    pre_impute=False,
    patience = 30):
    """Train GP_VAE or GP_VAE_posterior_consistency on given training data."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #lr_scheduler = step_lrs

    optimizer = Adam(weight_decay = 0.001) #ReduceLROnPlateau(patience = 100, factor = 0.8)
    
    model_cls = GP_VAE_posterior_concistency if posterior_consistency else GP_UAE
    model = model_cls(
        n_steps=X_train.shape[1], n_features=X_train.shape[2],
        latent_size=latent_size, epochs=epochs, batch_size=64,
        beta=1.0, K=3, encoder_sizes=(64, 32), decoder_sizes=(32, 64),
        optimizer=optimizer, patience = patience
    )#.to(device)

    # Set training parameters
    model.model.backbone.alpha = alpha #1e-1 if posterior_consistency else 1e0
    model.model.backbone.beta = beta
    model.model.backbone.sampling = sampling
    model.model.backbone.sigma = sigma
    model.model.backbone.detach_for_kl = detach_for_kl
    model.model.backbone.DAE = DAE

    model.model.backbone.device = device
    model.model.backbone.p = p
    model.train_gp = False
    model.model.backbone.r = 0.5

    if pre_impute:
        X = np.array(X_train)
        imputer = KNNImputer(n_neighbors=5)
        imputer.fit(X.reshape(-1,X.shape[-1]))

        model.model.backbone.encoder.pre_imputer = imputer
        model.model.backbone.encoder.pre_impute = True
    else:
        model.model.backbone.encoder.pre_impute = False

    model.fit({'X': X_train}, {'X': X_test, 'X_ori': X_original})
    model.fit({'X': X_train}, {'X': X_test, 'X_ori': X_original}, train_kernel_only=True)


    return model

# ==========================
# 🚀 Model Evaluation (Manual Encode-Decode)
# ==========================
import torch
import torch.distributions as dist

def compute_latent_log_probability(model, X_full, X_missing):
    """
    Compute the log probability of the mean of q(z|x_full) under q(z|x_missing).

    Parameters:
        model: The trained model (GPVAE or GPVAE Posterior Consistency).
        X_full: Fully observed input samples.
        X_missing: Corresponding samples with missing values.
        
    Returns:
        avg_log_prob: Mean log probability over all samples.
    """
    # Convert inputs to tensors
    X_full_tensor = torch.tensor(X_full, dtype=torch.float32)
    X_missing_tensor = torch.tensor(X_missing, dtype=torch.float32)

    # Get latent distributions for both
    qz_x_full = model.model.backbone.encode(X_full_tensor)
    qz_x_missing = model.model.backbone.encode(X_missing_tensor)

    # Extract mean and variance
    mu_full = qz_x_full.mean  # Mean of q(z|x_full)
    mu_missing = qz_x_missing.mean  # Mean of q(z|x_missing)
    var_missing = qz_x_missing.variance  # Variance of q(z|x_missing)

    # Create Normal distribution for q(z|x_missing)
    dist_missing = dist.Normal(mu_missing, torch.sqrt(var_missing))

    # Compute log probability of mu_full under q(z|x_missing)
    log_probs = dist_missing.log_prob(mu_full).clamp(max = 0).sum(dim=-1)  # Sum over latent dimensions

    if False:
        print((mu_missing - 2*var_missing)[0].shape)
        print(mu_missing.shape, var_missing.shape, mu_full.shape)
        plt.plot(mu_full[0].detach().numpy(),'o')
        plt.plot(mu_missing[0].detach().numpy())
        for i in range(mu_full.shape[2]):
            plt.fill_between(np.arange(30), (mu_missing - 2*var_missing)[0,:,i].detach().numpy(),
                            (mu_missing + 2*var_missing)[0,:,i].detach().numpy(),
                            alpha = .3)
        plt.show()

        plt.plot(var_missing[0].detach().numpy())
        plt.show()

        #print(log_probs.shape)

        plt.plot(log_probs[0].detach().numpy())
        plt.show()

        plt.hist(log_probs.detach().numpy().flatten())
        plt.show()

    sum_log_prob = torch.logsumexp(log_probs, dim = (0,1)).item()  # Average over all samples

    avg_log_prob = sum_log_prob - np.log(log_probs.numel())

    return avg_log_prob

def evaluate_model(model, X_test, X_original, missing_mask):
    """Evaluate model performance on test data (Manual Encode-Decode)."""
    print(f"\n=== Evaluating Model ===\n")
    
    try:
        X_tensor = torch.tensor(X_test, dtype=torch.float32)
        qz_x = model.model.backbone.encode(X_tensor)
        z = qz_x.rsample()
        X_recon = model.model.backbone.decode(z).mean.detach().numpy()

    except AttributeError: 
        X_recon = model

    rmse = compute_rmse(X_original, X_recon, missing_mask)
    nll = compute_nll(X_recon, np.ones_like(X_recon) * 0.1, X_original, missing_mask)
    try:
        kl_div = compute_kl_divergence(qz_x.mean.detach(), qz_x.variance.detach())
        elbo = -nll - kl_div
    except:
        elbo = 0.

    print(f"RMSE: {rmse:.4f}, NLL: {nll:.4f}, ELBO: {elbo:.4f}")
    return rmse, nll, elbo

def evaluate_model_fullmetrics(model, X_test, X_original, missing_mask=None):
    """
    Evaluate the model on test data, computing:
      - RMSE, NLL, and ELBO on observed and missing entries.
      - A latent calibration metric using the average KS statistic between
        the latent means and the standard normal distribution.
      - Reconstruction uncertainty metrics: 95% credible coverage and
        the correlation between predicted uncertainty and absolute error.
    
    Returns a dictionary with all these metrics.
    """
    print("\n=== Evaluating Model ===\n")
    use_latent_space = True  # Flag to track if latent space computations are possible

    try:
        X_tensor = torch.tensor(X_test, dtype=torch.float32)
        qz_x = model.model.backbone.encode(X_tensor)
        z_sample = qz_x.rsample()
        X_recon = model.model.backbone.decode(z_sample).mean.detach().cpu().numpy()
        X_recon_mean = model.model.backbone.decode(qz_x.mean).mean.detach().cpu().numpy()
    except AttributeError:
        # Handle non-GP-VAE models (e.g., MICE, KNN, Mean Imputation)
        print("⚠️ Model lacks encode/decode functionality, skipping latent space metrics.")
        try:
            X_flat = X_test.reshape(-1, X.shape[-1])
            X_recon_flat = model.transform(X_flat)
            X_recon = X_recon_flat.reshape(X.shape)
            X_recon_mean = X_recon # Model directly returns an imputed array
        except:
            X_recon = model
            X_recon_mean = model
        use_latent_space = False  # Skip latent space evaluations
    except ValueError:
        print("⚠️ Model has diverged, returning nan for all metrics.")
        return {
            "observed": {"rmse": np.nan, "nll": np.nan, "elbo": np.nan, "coverage": np.nan, "uncertainty_error_corr": np.nan},
            "missing": {"rmse": np.nan, "nll": np.nan, "elbo": np.nan, "coverage": np.nan, "uncertainty_error_corr": np.nan},
            "latent": {"latent_ks": np.nan},
            "reconstruction": {"overall_coverage": np.nan}
        }

    # Create masks for observed (nonzero) and missing (zero) values.
    observed_mask = (X_test != 0)
    missing_mask = (X_test == 0)

    # --- Compute RMSE and NLL ---
    rmse_obs = compute_rmse(X_original, X_recon_mean, observed_mask)
    rmse_miss = compute_rmse(X_original, X_recon_mean, missing_mask)
    nll_obs = compute_nll(X_recon, np.ones_like(X_recon) * 0.1, X_original, observed_mask)
    nll_miss = compute_nll(X_recon, np.ones_like(X_recon) * 0.1, X_original, missing_mask)

    # --- Compute ELBO (if applicable) ---
    if use_latent_space:
        try:
            kl_div = compute_kl_divergence(qz_x.mean.detach(), qz_x.variance.detach())
        except Exception as e:
            print("⚠️ Warning: Could not compute KL divergence.", e)
            kl_div = np.nan
        elbo_obs = -nll_obs - kl_div
        elbo_miss = -nll_miss - kl_div
    else:
        elbo_obs, elbo_miss = np.nan, np.nan  # ELBO is undefined for simple models

    # --- Latent Calibration Metric (if applicable) ---
    latent_ks = np.nan

    # --- Reconstruction Uncertainty Analysis (if applicable) ---
    if use_latent_space:
        num_samples = 100
        recon_samples = []
        for i in range(num_samples):
            z_sample_i = qz_x.rsample()
            recon_sample_i = model.model.backbone.decode(z_sample_i).mean.detach().cpu().numpy()
            recon_samples.append(recon_sample_i)
        recon_samples = np.array(recon_samples)

        recon_mean = recon_samples.mean(axis=0)
        recon_std = recon_samples.std(axis=0)

        # 95% credible intervals
        #z_val = norm.ppf(0.975)
        #lower_bound = recon_mean - z_val * recon_std
        lower_bound = recon_samples.min(axis=0)
        upper_bound = recon_samples.max(axis=0)
        #upper_bound = recon_mean + z_val * recon_std

        # Coverage computation
        overall_coverage = np.mean((X_original >= lower_bound) & (X_original <= upper_bound))
        coverage_obs = np.mean(((X_original >= lower_bound) & (X_original <= upper_bound))[observed_mask])
        coverage_miss = np.mean(((X_original >= lower_bound) & (X_original <= upper_bound))[missing_mask])

        # Uncertainty–error correlation
        abs_error = np.abs(recon_mean - X_original)
        corr_obs = np.corrcoef(recon_std[observed_mask].flatten(), abs_error[observed_mask].flatten())[0, 1] if np.sum(observed_mask) > 0 else np.nan
        corr_miss = np.corrcoef(recon_std[missing_mask].flatten(), abs_error[missing_mask].flatten())[0, 1] if np.sum(missing_mask) > 0 else np.nan
    else:
        # Assign NaN for uncertainty metrics since the model doesn't provide uncertainty estimates
        overall_coverage, coverage_obs, coverage_miss, corr_obs, corr_miss = np.nan, np.nan, np.nan, np.nan, np.nan

    if use_latent_space:
        latent_log_prob = compute_latent_log_probability(model, X_original, X_test)
    else:
        latent_log_prob = np.nan

    # Reporting
    print("Observed Data Metrics:")
    print(f"  RMSE: {rmse_obs:.4f}, NLL: {nll_obs:.4f}, ELBO: {elbo_obs:.4f}")
    print("Missing Data Metrics:")
    print(f"  RMSE: {rmse_miss:.4f}, NLL: {nll_miss:.4f}, ELBO: {elbo_miss:.4f}")
    print("Latent Calibration (Average KS Statistic):")
    print(f"  KS: {latent_ks:.4f}")
    print("Reconstruction Uncertainty vs. Error:")
    print(f"  Overall Coverage: {overall_coverage:.4f}")
    print(f"  Observed Coverage: {coverage_obs:.4f}, Uncertainty-Error Corr: {corr_obs:.4f}")
    print(f"  Missing Coverage: {coverage_miss:.4f}, Uncertainty-Error Corr: {corr_miss:.4f}")
    print(f"  Latent log prob: {latent_log_prob:.4f}")

    return {
        "observed": {"rmse": rmse_obs, "nll": nll_obs, "elbo": elbo_obs, "coverage": coverage_obs, "uncertainty_error_corr": corr_obs},
        "missing": {"rmse": rmse_miss, "nll": nll_miss, "elbo": elbo_miss, "coverage": coverage_miss, "uncertainty_error_corr": corr_miss},
        "latent": {"latent_ks": latent_ks, "latent_log_prob": latent_log_prob},
        "reconstruction": {"overall_coverage": overall_coverage}
    }

import json
import os

def save_results(results, file_path="benchmark_results_march.json"):
    """
    Saves the benchmark results to a file. Chooses JSON format as it is
    more efficient for structured data and easy to reload.
    
    Parameters:
        results (dict): Dictionary containing model performance metrics.
        file_path (str): File path where results will be saved.
    """
    # Ensure the directory exists
    #os.makedirs(file_path, exist_ok=True)

    # Save as JSON (efficient for structured data)
    with open(file_path, "w") as f:
        json.dump(results, f, indent=4)

    print(f"Results saved to {file_path}")

# ==========================
# 🚀 Benchmarking Function
# ==========================
def benchmark_models(missing_rates, X, y, n_epochs=600, results_file="benchmark_results_march.json", dataset_name = None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ## change results file name by adding dataset name
    if dataset_name == 'physionet':
        results_file = f'_{dataset_name}'.join(results_file.split('.'))

    # ✅ Load previous results if file exists
    if os.path.exists(results_file):
        with open(results_file, "r") as f:
            results = json.load(f)
        print(f"📁 Loaded {len(results)} previous results from {results_file}")
    else:
        results = {}
        print("📁 No previous results found, starting fresh.")

    already_ran = set(results.keys())  # 👈 for quick lookup

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    n_runs = 3

    for n in range(n_runs):
        for p in missing_rates:
            print(f"\n=== Training on {p*100:.0f}% Missing Data ===")

            X_corrupted, dataset_train, dataset_val = generate_missingness(X, p_dataset = p)

            X_train, X_test, X_original = torch.tensor(dataset_train['X']).to(device), torch.tensor(dataset_val['X']).to(device), torch.tensor(dataset_val['X_ori']).to(device)
            X_train, X_test, X_original = dataset_train['X'], dataset_val['X'], dataset_val['X_ori']

            missing_mask = np.ones(X_test.shape)

            for alpha  in [0.1, 0.8]:
                for detach_for_kl in [False]:
                    for DAE in [True, False]:
                        for pre_impute in [True, False]:

                            start_time = time.time()
                            #alpha, beta, sigma, model_p, detach_for_kl, DAE, pre_impute = alpha, 0.1, 0.01, 0.1, True, True, False
                            beta, sigma, model_p = 0.01, 0.01, 0.3

                            key = f"GP_UAE_{p}_{n}_alpha_{alpha}_sigma_{sigma}_model_p_{model_p}_detach_{detach_for_kl}_DAE_{DAE}_pre_impute{pre_impute}"
                            if key in already_ran and results[key][1]>30: #already ran and runtime above 30
                                print(f"⏩ Skipping already evaluated: {key}")
                                continue

                            
                            gpuae = train_model(X_train, X_test, X_original, epochs=n_epochs, 
                                                        alpha = alpha, 
                                                        beta = beta, 
                                                        sigma = sigma,
                                                        p = model_p,
                                                        detach_for_kl = detach_for_kl,
                                                        DAE = DAE,
                                                        pre_impute = pre_impute,
                                                        patience = 30)
                            training_time = time.time() - start_time
                            results[key] = evaluate_model_fullmetrics(gpuae, X_test.copy(), X_original.copy(), missing_mask), training_time
                            save_results(results)
                            torch.save(gpuae.state_dict(), key)

                            #save_results(results[list(results.keys())[-1]])



            for alpha in [1, 0.1, 0.01]:

                save_dir = 'model_weights_sampling_from_x'
                os.makedirs(save_dir, exist_ok=True)  # ✅ Crée le dossier s’il n’existe pas

                key = f"Posterior_consistency_{p}_{n}_alpha_{alpha}"

                if key in already_ran and results[key][1]>30: #already ran and runtime above 30
                    print(f"⏩ Skipping already evaluated: {key}")
                    continue

                start_time = time.time()

                gpvae_posterior = train_model(X_train, X_test, X_original, epochs=n_epochs, posterior_consistency=True, alpha = alpha)

                training_time = time.time() - start_time

                results[key] = evaluate_model_fullmetrics(gpvae_posterior, X_test.copy(), X_original.copy(), missing_mask), training_time
                save_results(results)
                torch.save(gpvae_posterior.state_dict(), key)



            # Evaluate Baseline Methods
            key = f"Mean_Imputation_{p}_{n}"
            if not key in already_ran:
                results[key] = evaluate_model_fullmetrics(mean_impute(X_train.copy(), X_test.copy()), X_test.copy(), X_original.copy(), missing_mask), np.nan
                save_results(results)

            key = f"KNN_Imputation_{p}_{n}"
            #imputed = knn_impute(X_train.copy(), X_test.copy())
            #plt.plot(imputed[0])
            #plt.plot(X_test[0],'o')
            #plt.show()

            if not key in already_ran:
                results[key] = evaluate_model_fullmetrics(knn_impute(X_train.copy(), X_test.copy()), X_test.copy(), X_original.copy(), missing_mask), np.nan
                save_results(results)

            key = f"MICE_Imputation_{p}_{n}"
            #if not key in already_ran:
            results[key] = evaluate_model_fullmetrics(mice_impute(X_train.copy(),X_test.copy()), X_test.copy(), X_original.copy(), missing_mask), np.nan
            save_results(results)



            

            # results[f"MissForest_Imputation_{p}"] = evaluate_model_fullmetricsmissforest_impute(X_test), X_test, X_original, missing_mask)

            #save_reconstruction_plots(gpvae_alpha1, gpvae_posterior, X_test.copy(), X_original.copy(), missing_rate=p)

    return results


# ==========================
# 🚀 Save Reconstruction Plots
# ==========================
def save_reconstruction_plots(gpvae1, gpvae2, X, X_normalized, missing_rate, i = 9):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    models = [gpvae1, gpvae2]
    titles = ["GP-VAE", "Posterior Consistency"]

    X_nan = X.copy()
    X[X!=X] = 0.

    for ax, gpvae, title in zip(axes, models, titles):
        qz_x = gpvae.model.backbone.encode(torch.tensor(X[:i+1]).float())

        reconstructions = [gpvae.model.backbone.decode(qz_x.rsample()).mean.detach().numpy()[i] for _ in range(50)]
        reconstructions = np.array(reconstructions)

        recon_min, recon_max = np.min(reconstructions, axis=0), np.max(reconstructions, axis=0)
        z = qz_x.mean
        X_recon_mean = gpvae.model.backbone.decode(z).mean.detach().numpy()[i]

        X_nan[X_nan==0] = np.nan


        colors = ['gold', 'forestgreen', 'navy', 'purple', 'steelblue', 'coral', 'olive', 'rosybrown']
        for feature_idx in range(recon_min.shape[1]):
            ax.fill_between(range(len(X_recon_mean)), recon_min[:, feature_idx], recon_max[:, feature_idx], alpha=0.3, color=colors[feature_idx])
            ax.plot(X_recon_mean[:, feature_idx], color=colors[feature_idx])
            ax.plot(X_normalized[i][:, feature_idx], '+', color=colors[feature_idx])
            ax.plot(X_nan[i][:, feature_idx], 'o', color=colors[feature_idx])

        ax.set_title(title)
        ax.set_xlabel("Time Steps")
        ax.set_ylabel("Feature Value")

    plot_filename = os.path.join(PLOTS_DIR, f"reconstruction_comparison_missing_{int(missing_rate * 100)}.png")
    plt.suptitle(f"Reconstruction Comparison for {int(missing_rate * 100)}% Missing Data")
    plt.tight_layout()
    plt.savefig(plot_filename)
    plt.close()
    print(f"📊 Saved plot: {plot_filename}")

# ==========================
# 📁 Create Directories for Saving Results
# ==========================
RESULTS_DIR = "results"
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

RESULTS_FILE = os.path.join(RESULTS_DIR, "benchmark_results.txt")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark GP-VAE variants on missing data tasks.")

    # Hyperparameters and settings
    parser.add_argument("--dataset", type=str, default="artificial", choices=["artificial", "physionet"])
    parser.add_argument("--epochs", type=int, default=3000)
    parser.add_argument("--latent_size", type=int, default=6)
    parser.add_argument("--results_file", type=str, default="benchmark_results.json")

    args = parser.parse_args()

    print('PARSED ARGS')

    dataset_name = args.dataset
    n_epochs = args.epochs
    latent_size = args.latent_size
    results_file = args.results_file

    print(n_epochs)

    # ============================
    #   Dataset preparation
    # ============================

    #dataset_name = 'artificial'

    # Define Missingness Levels


    if dataset_name == 'artificial':
        missing_rates = [0.1, 0.3, 0.5, 0.7]#[::-1]

        # Generate Synthetic Data
        X, y = Generate_samples(
            n_groups=2,
            n_observations_per_group=500,  # Adjusted for quick testing
            n_dimensions=3,
            n_time_points=30,
            length_scale=1.5,
            n_new_dims=7
        )
    elif dataset_name == 'physionet':
        missing_rates = [0.1]
        physionet2012_dataset = tsdb.load('physionet_2012')
        print(physionet2012_dataset.keys())
        X = physionet2012_dataset['X']

    # Run Benchmarking
    results = benchmark_models(
        missing_rates,
        X,
        y,
        n_epochs=n_epochs,
        dataset_name=dataset_name
    )
    #results = benchmark_models(missing_rates, X, y, n_epochs = 3000, dataset_name = dataset_name)

    # Save Results to File
    with open(RESULTS_FILE, "w") as f:
        for key, metrics_tuple in results.items():
            # Ensure we have the expected tuple format
            if not isinstance(metrics_tuple, tuple) or len(metrics_tuple) != 2:
                print(f"⚠️ Warning: Unexpected data format for {key}, skipping entry.")
                continue
            
            metrics, training_time = metrics_tuple  # Unpack tuple (metrics dictionary, training time)

            # Extract observed data metrics
            observed = metrics.get("observed", {})
            rmse_obs = observed.get("rmse", np.nan)
            nll_obs = observed.get("nll", np.nan)
            elbo_obs = observed.get("elbo", np.nan)
            coverage_obs = observed.get("coverage", np.nan)
            unc_err_corr_obs = observed.get("uncertainty_error_corr", np.nan)

            # Extract missing data metrics
            missing = metrics.get("missing", {})
            rmse_miss = missing.get("rmse", np.nan)
            nll_miss = missing.get("nll", np.nan)
            elbo_miss = missing.get("elbo", np.nan)
            coverage_miss = missing.get("coverage", np.nan)
            unc_err_corr_miss = missing.get("uncertainty_error_corr", np.nan)

            # Extract latent space metrics
            latent = metrics.get("latent", {})
            latent_ks = latent.get("latent_ks", np.nan)
            latent_log_prob = latent.get("latent_log_prob", np.nan)


            # Extract reconstruction uncertainty metrics
            reconstruction = metrics.get("reconstruction", {})
            overall_coverage = reconstruction.get("overall_coverage", np.nan)

            # Format numeric values properly
            format_metric = lambda x: f"{x:.4f}" if isinstance(x, (int, float)) and not np.isnan(x) else "N/A"
            training_time_str = format_metric(training_time) if isinstance(training_time, (int, float)) else "N/A"

            # Write results to file
            f.write(f"{key}:\n")
            f.write(f"  RMSE (Observed): {format_metric(rmse_obs)}, NLL (Observed): {format_metric(nll_obs)}, ELBO (Observed): {format_metric(elbo_obs)}\n")
            f.write(f"  RMSE (Missing): {format_metric(rmse_miss)}, NLL (Missing): {format_metric(nll_miss)}, ELBO (Missing): {format_metric(elbo_miss)}\n")
            f.write(f"  Latent KS: {format_metric(latent_ks)}, Latent Log Prog {format_metric(latent_log_prob)}\n")
            f.write(f"  Coverage (Overall): {format_metric(overall_coverage)}, Coverage (Observed): {format_metric(coverage_obs)}, Coverage (Missing): {format_metric(coverage_miss)}\n")
            f.write(f"  Uncertainty-Error Corr (Observed): {format_metric(unc_err_corr_obs)}, Uncertainty-Error Corr (Missing): {format_metric(unc_err_corr_miss)}\n")
            f.write(f"  Training Time: {training_time_str}\n\n")

    print(f"\n✅ Benchmarking completed! Results saved in {RESULTS_FILE}")
