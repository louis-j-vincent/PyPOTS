import sys
sys.path.append('/Users/louis/Documents/Phd/Code_pour_manuscript/PyPOTS/ipynbs')
sys.path.append('/Users/louis/Documents/Phd/Code_pour_manuscript/PyPOTS')

import numpy as np
import torch
import torch.distributions as dist
import matplotlib.pyplot as plt
from tqdm import tqdm
from pypots.imputation import GP_VAE_posterior_concistency, GP_VAE
from pypots.optim.adam import Adam
from pygrinder import mcar  # For missing data corruption
import os
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF


# ==========================
# 📁 Create Directories for Saving Results
# ==========================
RESULTS_DIR = "results"
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

RESULTS_FILE = os.path.join(RESULTS_DIR, "benchmark_results.txt")

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
# 🚀 Generate GP Samples (New Function)
# ==========================
def generate_gp_samples_with_library(n_groups, n_observations_per_group, n_dimensions, time_points, length_scale=1.0):
    """Generate multidimensional GP samples using scikit-learn."""
    kernel = RBF(length_scale=length_scale)
    gp = GaussianProcessRegressor(kernel=kernel, alpha=1e-8)

    samples = []
    for _ in range(n_groups):
        group_samples = []
        for _ in tqdm(range(n_observations_per_group), desc="Generating GP Samples"):
            gp_samples_per_dim = [gp.sample_y(time_points.reshape(-1, 1), random_state=None).flatten() for _ in range(n_dimensions)]
            group_samples.append(np.stack(gp_samples_per_dim, axis=-1))
        samples.append(np.array(group_samples))

    return samples

def corrupt_by_dim(X):
    """Corrupt by taking away some dimensions fully."""
    X_corr = np.copy(X)
    T, n_dims = X.shape[1], X.shape[2]
    for x in X_corr:
        nb_obs_vals = np.random.randint(0, 5)
        obs_mask = np.random.choice(np.arange(T), size=T - nb_obs_vals, replace=False)
        obs_mask = np.isin(np.arange(T), obs_mask)
        n_dims_missing = np.random.randint(1, 4)
        dims = np.random.choice(np.arange(n_dims), size=n_dims_missing, replace=False)
        for dim in dims:
            x[obs_mask, dim] = np.nan
    return X_corr

def Generate_samples(n_groups=1, n_observations_per_group=5000, n_dimensions=30, n_time_points=30,
                     time_points=np.linspace(0, 10, 30), length_scale=1.5, n_new_dims=7):
    """Generate synthetic dataset with Gaussian Process and structured missingness."""
    
    gp_samples = generate_gp_samples_with_library(
        n_groups=n_groups, n_observations_per_group=n_observations_per_group,
        n_dimensions=n_dimensions, time_points=time_points, length_scale=length_scale
    )

    Proj = np.random.uniform(size=(n_dimensions, n_new_dims))
    X_ori = gp_samples[0] @ Proj

    X_normalized = (X_ori - X_ori.mean(axis=(0, 1)).reshape(1, 1, -1)) / X_ori.std(axis=(0, 1)).reshape(1, 1, -1)
    #X = corrupt_by_dim(X_normalized.copy())
    return X_normalized

def generate_missingness(X, p_dataset = .1):
    
    X = mcar(X.copy(), p_dataset)  # randomly hold out 10% observed values as ground truth
    X[np.isnan(X)] = 0  # Replace NaNs with zeros

    dataset = {'X': X}
    len_dataset = len(dataset['X'])
    cut = int(len_dataset * 0.7)
    dataset_train, dataset_val = {'X': dataset['X'][:cut]}, {'X': dataset['X'][cut:], 'X_ori': X_normalized[cut:]}

    return dataset_train, dataset_val


# ==========================
# 🚀 Model Training Function
# ==========================
def train_model(X_train, latent_size=15, epochs=300, posterior_consistency=False):
    """Train GP_VAE or GP_VAE_posterior_consistency on given training data."""
    
    model_cls = GP_VAE_posterior_concistency if posterior_consistency else GP_VAE
    model = model_cls(
        n_steps=X_train.shape[1], n_features=X_train.shape[2],
        latent_size=latent_size, epochs=epochs, batch_size=64,
        beta=1.0, K=3, encoder_sizes=(128, 64, 32), decoder_sizes=(64, 32),
        optimizer=Adam()
    )

    # Set training parameters
    model.model.backbone.alpha = 1e-1 if posterior_consistency else 1e0
    model.model.backbone.beta = 1e0
    model.model.backbone.gamma = 1e-5
    model.model.backbone.sampling = True
    model.model.backbone.device = 'cpu'

    model.fit({'X': X_train})
    return model

# ==========================
# 🚀 Model Evaluation (Manual Encode-Decode)
# ==========================
def evaluate_model(model, X_test, X_original, missing_mask):
    """Evaluate model performance on test data (Manual Encode-Decode)."""
    print(f"\n=== Evaluating Model ===\n")

    X_tensor = torch.tensor(X_test, dtype=torch.float32)
    qz_x = model.model.backbone.encode(X_tensor)
    z = qz_x.rsample()
    X_recon = model.model.backbone.decode(z).mean.detach().numpy()

    rmse = compute_rmse(X_original, X_recon, missing_mask)
    nll = compute_nll(X_recon, np.ones_like(X_recon) * 0.1, X_original, missing_mask)
    kl_div = compute_kl_divergence(qz_x.mean.detach(), qz_x.variance.detach())
    elbo = -nll - kl_div

    print(f"RMSE: {rmse:.4f}, NLL: {nll:.4f}, ELBO: {elbo:.4f}")
    return rmse, nll, elbo

# ==========================
# 🚀 Benchmarking Function
# ==========================
def benchmark_models(missing_rates, X):
    results = {}

    for p in missing_rates:
        print(f"\n=== Training on {p*100:.0f}% Missing Data ===")

        dataset_train, dataset_val = generate_missingness(X, p_dataset = .1)

        X_train, X_test, X_original = dataset_train['X'], dataset_val['X'], dataset_val['X_ori']
        missing_mask = (X_test == 0)

        gpvae = train_model(X_train, epochs=1)
        gpvae_posterior = train_model(X_train, epochs=1, posterior_consistency=True)

        results[f"GP_VAE_{p}"] = evaluate_model(gpvae, X_test, X_original, missing_mask)
        results[f"Posterior_Consistency_{p}"] = evaluate_model(gpvae_posterior, X_test, X_original, missing_mask)

    return results

if __name__ == "__main__":
    # Define Missingness Levels
    missing_rates = [0.1, 0.3, 0.5, 0.7]

    # Generate Synthetic Data
    X = Generate_samples(
        n_groups=1,
        n_observations_per_group=50,  # Adjusted for quick testing
        n_dimensions=3,
        n_time_points=30,
        length_scale=1.5,
        n_new_dims = 10
    )

    # Run Benchmarking
    results = benchmark_models(missing_rates, X)

    # Save Results to File
    with open(RESULTS_FILE, "w") as f:
        for key, (rmse, nll, elbo) in results.items():
            f.write(f"{key}:\n  RMSE: {rmse:.4f}, NLL: {nll:.4f}, ELBO: {elbo:.4f}\n")

    print(f"\n✅ Benchmarking completed! Results saved in {RESULTS_FILE}")

