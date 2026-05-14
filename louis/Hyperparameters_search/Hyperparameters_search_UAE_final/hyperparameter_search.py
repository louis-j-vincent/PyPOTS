from benchmarking import *  # Import all functions from your benchmarking module

import numpy as np
import torch
import time
import os
import matplotlib.pyplot as plt
from scipy.stats import kstest, norm

# =============================================================================
# Evaluation Function (with latent calibration & reconstruction uncertainty)
# =============================================================================
def evaluate_model(model, X_test, X_original, missing_mask = None):
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
    try:
        X_tensor = torch.tensor(X_test, dtype=torch.float32)
        qz_x = model.model.backbone.encode(X_tensor)
        # For a single reconstruction we use one sample:
        z_sample = qz_x.rsample()
        X_recon = model.model.backbone.decode(z_sample).mean.detach().cpu().numpy()
        X_recon_mean = model.model.backbone.decode(qz_x.mean).mean.detach().cpu().numpy()
    except AttributeError:
        # In case the model is a baseline that returns a numpy array.
        X_recon = model
        X_recon_mean = model
    except ValueError:
        print('Model has diverged')
        return {
            "observed": {
                "rmse": 0.,
                "nll": 0.,
                "elbo": 0.,
                "coverage": 0.,
                "uncertainty_error_corr": 0.
            },
            "missing": {
                "rmse": 0.,
                "nll": 0.,
                "elbo": 0.,
                "coverage": 0.,
                "uncertainty_error_corr": 0.
            },
            "latent": {"latent_ks": 0.},
            "reconstruction": {"overall_coverage": 0.}
        }



    # Create masks for observed (nonzero) and missing (zero) values.
    observed_mask = (X_test != 0)
    missing_mask = (X_test == 0)
    
    # --- Reconstruction Metrics (RMSE, NLL, ELBO) ---
    rmse_obs = compute_rmse(X_original, X_recon_mean, observed_mask)
    rmse_miss = compute_rmse(X_original, X_recon_mean, missing_mask)
    nll_obs = compute_nll(X_recon, np.ones_like(X_recon) * 0.1, X_original, observed_mask)
    nll_miss = compute_nll(X_recon, np.ones_like(X_recon) * 0.1, X_original, missing_mask)
    try:
        kl_div = compute_kl_divergence(qz_x.mean.detach(), qz_x.variance.detach())
    except Exception as e:
        print("Warning: Could not compute KL divergence.", e)
        kl_div = 0.
    elbo_obs = -nll_obs - kl_div
    elbo_miss = -nll_miss - kl_div

    # --- Latent Calibration Metric ---
    # Get latent means; if the output is 3D (batch, time, latent_dim), average over time.
    latent_means = qz_x.mean.detach().cpu().numpy()
    if latent_means.ndim == 3:
        latent_means = latent_means.mean(axis=1)  # Now shape: (batch, latent_dim)
    
    ks_stats = []
    for d in range(latent_means.shape[1]):
        data_d = latent_means[:, d].flatten()  # Ensure 1D array
        result = kstest(data_d, 'norm')
        ks_stats.append(result.statistic)
    latent_ks = np.mean(ks_stats)  # Lower is better.

    # --- Reconstruction Uncertainty vs. Error Analysis ---
    num_samples = 50  # Number of samples for approximation.
    recon_samples = []
    for i in range(num_samples):
        z_sample_i = qz_x.rsample()
        recon_sample_i = model.model.backbone.decode(z_sample_i).mean.detach().cpu().numpy()
        recon_samples.append(recon_sample_i)
    recon_samples = np.array(recon_samples)  # (num_samples, batch, time, features)
    recon_mean = recon_samples.mean(axis=0)
    recon_std = recon_samples.std(axis=0)
    
    # 95% credible intervals (for a Gaussian, z ≈ 1.96)
    z_val = norm.ppf(0.975)
    lower_bound = recon_mean - z_val * recon_std
    upper_bound = recon_mean + z_val * recon_std
    
    # Coverage: fraction of true values within the credible interval.
    overall_coverage = np.mean((X_original >= lower_bound) & (X_original <= upper_bound))
    coverage_obs = np.mean(((X_original >= lower_bound) & (X_original <= upper_bound))[observed_mask])
    coverage_miss = np.mean(((X_original >= lower_bound) & (X_original <= upper_bound))[missing_mask])
    
    # Uncertainty–error correlation.
    abs_error = np.abs(recon_mean - X_original)
    if np.sum(observed_mask) > 0:
        corr_obs = np.corrcoef(recon_std[observed_mask].flatten(), abs_error[observed_mask].flatten())[0, 1]
    else:
        corr_obs = np.nan
    if np.sum(missing_mask) > 0:
        corr_miss = np.corrcoef(recon_std[missing_mask].flatten(), abs_error[missing_mask].flatten())[0, 1]
    else:
        corr_miss = np.nan

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

    return {
        "observed": {
            "rmse": rmse_obs,
            "nll": nll_obs,
            "elbo": elbo_obs,
            "coverage": coverage_obs,
            "uncertainty_error_corr": corr_obs
        },
        "missing": {
            "rmse": rmse_miss,
            "nll": nll_miss,
            "elbo": elbo_miss,
            "coverage": coverage_miss,
            "uncertainty_error_corr": corr_miss
        },
        "latent": {"latent_ks": latent_ks},
        "reconstruction": {"overall_coverage": overall_coverage}
    }

# =============================================================================
# Hyperparameter Search Function
# =============================================================================
def hyperparameter_search(X, missing_rates, alpha_values, n_epochs=500):
    """
    For each missing rate and each alpha value, train a GP-VAE model,
    evaluate it with all metrics, and store results.
    
    Returns a nested dictionary with results.
    """
    results = {}
    for p in missing_rates:
        print(f"\n=== Hyperparameter Search for {p*100:.0f}% Missing Data ===")
        X_corrupted, dataset_train, dataset_val = generate_missingness(X, p_dataset=p)
        X_train = dataset_train['X']
        X_test  = dataset_val['X']
        X_original = dataset_val['X_ori']
        
        results[p] = {}
        for alpha in alpha_values:
            print(f"\nTraining GP-VAE with alpha = {alpha:.4e}")
            start_time = time.time()
            model = train_model(X_train, epochs=n_epochs, alpha=alpha)
            training_time = time.time() - start_time
            
            metrics = evaluate_model(model, X_test.copy(), X_original.copy())
            results[p][alpha] = {
                "metrics": metrics,
                "training_time": training_time
            }
            print(f"Alpha: {alpha:.4e}, Training Time: {training_time:.2f} sec")
    return results

# =============================================================================
# Plotting Function: Create and Save All Metric Plots
# =============================================================================
def plot_all_metrics(search_results, save_dir="results_hyperparameter_search"):
    """
    Generate several figures:
      - Figure 1: RMSE, NLL, ELBO versus alpha (observed & masked on the same graph).
      - Figure 2: Coverage (observed, masked, overall) and uncertainty-error correlation versus alpha.
      - Figure 3: Latent calibration (KS) and training time versus alpha.
    Each figure uses a log-scale for alpha. Colors are fixed per missing rate.
    All plots are saved automatically.
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    
    # Define a color dictionary for fixed missing rate colors.
    colors = {0.1: 'blue', 0.3: 'green', 0.5: 'red', 0.7: 'purple'}
    
    missing_rates = sorted(search_results.keys())
    
    # ---------------------
    # Figure 1: RMSE, NLL, ELBO
    # ---------------------
    fig1, axes1 = plt.subplots(3, 1, figsize=(10, 15))
    for mr in missing_rates:
        alphas = sorted(search_results[mr].keys())
        alphas_float = np.array(alphas, dtype=float)
        rmse_obs = [search_results[mr][alpha]["metrics"]["observed"]["rmse"] for alpha in alphas]
        rmse_miss = [search_results[mr][alpha]["metrics"]["missing"]["rmse"] for alpha in alphas]
        nll_obs = [search_results[mr][alpha]["metrics"]["observed"]["nll"] for alpha in alphas]
        nll_miss = [search_results[mr][alpha]["metrics"]["missing"]["nll"] for alpha in alphas]
        elbo_obs = [search_results[mr][alpha]["metrics"]["observed"]["elbo"] for alpha in alphas]
        elbo_miss = [search_results[mr][alpha]["metrics"]["missing"]["elbo"] for alpha in alphas]
        
        axes1[0].semilogy(alphas_float, rmse_obs, marker='o', color=colors[mr], label=f"Missing {mr} - Obs")
        axes1[0].semilogy(alphas_float, rmse_miss, marker='x', color=colors[mr], linestyle='--', label=f"Missing {mr} - Mask")
        axes1[1].plot(alphas_float, nll_obs, marker='o', color=colors[mr], label=f"Missing {mr} - Obs")
        axes1[1].plot(alphas_float, nll_miss, marker='x', color=colors[mr], linestyle='--', label=f"Missing {mr} - Mask")
        axes1[2].plot(alphas_float, elbo_obs, marker='o', color=colors[mr], label=f"Missing {mr} - Obs")
        axes1[2].plot(alphas_float, elbo_miss, marker='x', color=colors[mr], linestyle='--', label=f"Missing {mr} - Mask")
    
    for ax, title in zip(axes1, ["RMSE", "NLL", "ELBO"]):
        ax.set_xscale("log")
        ax.set_xlabel("Alpha")
        ax.set_ylabel(title)
        ax.set_title(title)
        ax.grid(True)
        ax.legend()
    fig1.tight_layout()
    fig1_path = os.path.join(save_dir, "figure1_RMSE_NLL_ELBO.png")
    fig1.savefig(fig1_path, dpi=300)
    plt.close(fig1)
    
    # ---------------------
    # Figure 2: Coverage and Uncertainty-Error Correlation
    # ---------------------
    fig2, axes2 = plt.subplots(2, 1, figsize=(10, 12))
    for mr in missing_rates:
        alphas = sorted(search_results[mr].keys())
        alphas_float = np.array(alphas, dtype=float)
        cov_obs = [search_results[mr][alpha]["metrics"]["observed"]["coverage"] for alpha in alphas]
        cov_miss = [search_results[mr][alpha]["metrics"]["missing"]["coverage"] for alpha in alphas]
        overall_cov = [search_results[mr][alpha]["metrics"]["reconstruction"]["overall_coverage"] for alpha in alphas]
        
        axes2[0].plot(alphas_float, cov_obs, marker='o', color=colors[mr], label=f"Missing {mr} - Obs")
        axes2[0].plot(alphas_float, cov_miss, marker='x', color=colors[mr], linestyle='--', label=f"Missing {mr} - Mask")
        axes2[0].plot(alphas_float, overall_cov, marker='s', color=colors[mr], linestyle=':', label=f"Missing {mr} - Overall")
    axes2[0].set_xscale("log")
    axes2[0].set_xlabel("Alpha")
    axes2[0].set_ylabel("Coverage")
    axes2[0].set_title("Coverage Metrics")
    axes2[0].grid(True)
    axes2[0].legend()
    
    for mr in missing_rates:
        alphas = sorted(search_results[mr].keys())
        alphas_float = np.array(alphas, dtype=float)
        corr_obs = [search_results[mr][alpha]["metrics"]["observed"]["uncertainty_error_corr"] for alpha in alphas]
        corr_miss = [search_results[mr][alpha]["metrics"]["missing"]["uncertainty_error_corr"] for alpha in alphas]
        
        axes2[1].plot(alphas_float, corr_obs, marker='o', color=colors[mr], label=f"Missing {mr} - Obs")
        axes2[1].plot(alphas_float, corr_miss, marker='x', color=colors[mr], linestyle='--', label=f"Missing {mr} - Mask")
    axes2[1].set_xscale("log")
    axes2[1].set_xlabel("Alpha")
    axes2[1].set_ylabel("Uncertainty-Error Corr")
    axes2[1].set_title("Uncertainty vs. Error Correlation")
    axes2[1].grid(True)
    axes2[1].legend()
    
    fig2.tight_layout()
    fig2_path = os.path.join(save_dir, "figure2_Coverage_UncertaintyErrorCorr.png")
    fig2.savefig(fig2_path, dpi=300)
    plt.close(fig2)
    
    # ---------------------
    # Figure 3: Latent Calibration and Training Time
    # ---------------------
    fig3, axes3 = plt.subplots(2, 1, figsize=(10, 10))
    for mr in missing_rates:
        alphas = sorted(search_results[mr].keys())
        alphas_float = np.array(alphas, dtype=float)
        latent_ks = [search_results[mr][alpha]["metrics"]["latent"]["latent_ks"] for alpha in alphas]
        train_time = [search_results[mr][alpha]["training_time"] for alpha in alphas]
        
        axes3[0].plot(alphas_float, latent_ks, marker='o', color=colors[mr], label=f"Missing {mr}")
        axes3[1].plot(alphas_float, train_time, marker='o', color=colors[mr], label=f"Missing {mr}")
    axes3[0].set_xscale("log")
    axes3[0].set_xlabel("Alpha")
    axes3[0].set_ylabel("Latent KS")
    axes3[0].set_title("Latent Calibration")
    axes3[0].grid(True)
    axes3[0].legend()
    
    axes3[1].set_xscale("log")
    axes3[1].set_xlabel("Alpha")
    axes3[1].set_ylabel("Training Time (sec)")
    axes3[1].set_title("Training Time")
    axes3[1].grid(True)
    axes3[1].legend()
    
    fig3.tight_layout()
    fig3_path = os.path.join(save_dir, "figure3_LatentCalibration_TrainingTime.png")
    fig3.savefig(fig3_path, dpi=300)
    plt.close(fig3)
    
    print("All figures have been saved.")

# =============================================================================
# Main: Run Hyperparameter Search, Save Results, and Generate Plots
# =============================================================================
if __name__ == "__main__":
    # 1. Generate Synthetic Data
    X = Generate_samples(
        n_groups=1,
        n_observations_per_group=1000,  # Adjust for quicker testing if desired
        n_dimensions=30,
        n_time_points=30,
        length_scale=1.5
    )
    
    # 2. Define Hyperparameter Grid (using 7 values instead of 10)
    missing_rates = [0.1, 0.3, 0.5, 0.7]
    alpha_values = np.logspace(-2, 1, num=7)
    
    # 3. Run the Hyperparameter Search (use a small number of epochs for testing)
    search_results = hyperparameter_search(X, missing_rates, alpha_values, n_epochs=500)
    
    # 4. Save the Results to a Text File
    RESULTS_DIR = "results_hyperparameter_search"
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR, exist_ok=True)
    results_file = os.path.join(RESULTS_DIR, "hyperparameter_search_results.txt")
    with open(results_file, "w") as f:
        for p, alpha_dict in search_results.items():
            f.write(f"Missing Rate: {p}\n")
            for alpha, data in alpha_dict.items():
                m = data["metrics"]
                f.write(f"  Alpha: {alpha:.4e}\n")
                f.write("    Observed Metrics:\n")
                f.write(f"      RMSE: {m['observed']['rmse']:.4f}\n")
                f.write(f"      NLL: {m['observed']['nll']:.4f}\n")
                f.write(f"      ELBO: {m['observed']['elbo']:.4f}\n")
                f.write(f"      Coverage: {m['observed']['coverage']:.4f}\n")
                f.write(f"      Uncertainty-Error Corr: {m['observed']['uncertainty_error_corr']:.4f}\n")
                f.write("    Missing Metrics:\n")
                f.write(f"      RMSE: {m['missing']['rmse']:.4f}\n")
                f.write(f"      NLL: {m['missing']['nll']:.4f}\n")
                f.write(f"      ELBO: {m['missing']['elbo']:.4f}\n")
                f.write(f"      Coverage: {m['missing']['coverage']:.4f}\n")
                f.write(f"      Uncertainty-Error Corr: {m['missing']['uncertainty_error_corr']:.4f}\n")
                f.write("    Latent Calibration:\n")
                f.write(f"      Average KS Statistic: {m['latent']['latent_ks']:.4f}\n")
                f.write("    Reconstruction Overall Coverage:\n")
                f.write(f"      Overall Coverage: {m['reconstruction']['overall_coverage']:.4f}\n")
                f.write(f"    Training Time: {data['training_time']:.2f} sec\n")
            f.write("\n")
    print(f"\n✅ Hyperparameter search completed! Results saved in {results_file}")
    
    # 5. Generate and Save All Plots Automatically
    plot_all_metrics(search_results, save_dir=RESULTS_DIR)
