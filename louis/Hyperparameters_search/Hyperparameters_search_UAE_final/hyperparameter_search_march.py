#!/usr/bin/env python
"""
Hyperparameter Search for beta, NOISE_SIGMA, noise_std, use_mean_z, and detach_flag.

This script performs a grid search over the following hyperparameters:
  - beta: weight for the KL divergence.
  - noise_sigma: the fixed noise variance used in the NLL computation.
  - noise_std: the standard deviation used when adding noise to the latent mean.
  - use_mean_z: boolean flag to decide whether to use q(z|x).mean or sample from q(z|x) in the NLL.
  - detach_flag: boolean flag to decide whether to detach certain tensors (e.g. latent statistics) in loss computations.

A fixed dataset corruption level is used (p_dataset), so that comparisons are consistent.
The script trains the model on a (synthetic) dataset, evaluates it, saves the metrics,
and produces summary plots.
"""

import os
import time
import json
import numpy as np
import torch
import matplotlib.pyplot as plt
from itertools import product
from scipy.stats import norm
import gc

import sys

#sys.path.append('/Users/louis/Documents/Phd/Code_pour_manuscript/PyPOTS/ipynbs')
sys.path.append('/Users/louis/Documents/Phd/Code_pour_manuscript/PyPOTS')
print(sys.path)

# Import your benchmarking utilities and model training/evaluation functions.
# These should include functions such as Generate_samples, generate_missingness,
# train_model, evaluate_model, and any metric computations.
from benchmarking import Generate_samples, generate_missingness, train_model
from hyperparameter_search import plot_all_metrics, evaluate_model

# Define the hyperparameter grid.
alpha_values        = [1e-2, 1e-1, .5, 1]
beta_values         = [0,1]
sigma_values        = [1e-4, 1e-3, 1e-2, 1e-1]
model_p_values      = [0.3, 0.5, 0.1]
pre_impute_values = [True]

# Fix the dataset corruption level for consistency.
p_dataset = 0.3

# Define the file to store intermediate results.
INTERMEDIATE_RESULTS_FILE = "intermediate_results_sampling.jsonl"


def append_result_to_disk(result, file_path):
    """
    Append a result (as a dictionary) to a JSON Lines file.
    Each line in the file will be a valid JSON object.
    """
    with open(file_path, "a") as f:
        json.dump(result, f)
        f.write("\n")

def hyperparameter_search_other(
    X,
    alpha_values,
    beta_values,
    sigma_values,
    model_p_values,
    n_epochs=1000
):
    """
    Perform a grid search over beta, noise_sigma, noise_std, use_mean_z, and detach_flag.
    
    Parameters:
        X (np.ndarray): The complete (synthetic) dataset.
        beta_vals (list of float): Values to try for beta.
        noise_sigma_vals (list of float): Values for NOISE_SIGMA.
        noise_std_vals (list of float): Values for the noise std (for additive noise).
        use_mean_z_opts (list of bool): Options for using mean versus sampling.
        detach_opts (list of bool): Options for detaching tensors or not.
        n_epochs (int): Number of training epochs.
        
    Returns:
        results (dict): Nested dictionary keyed by hyperparameter combinations.
    """
    
    # Read the existing intermediate results from the JSON Lines file.
    computed_hp = set()
    if os.path.exists(INTERMEDIATE_RESULTS_FILE):
        with open(INTERMEDIATE_RESULTS_FILE, "r") as f:
            for line in f:
                if line.strip():
                    try:
                        result = json.loads(line)
                        hp_dict = result.get("hyperparameters", {})
                        # Create a tuple (alpha, beta, sigma, p) from the hyperparameters.
                        hp_tuple = (hp_dict.get("alpha"), hp_dict.get("beta"), hp_dict.get("sigma"), hp_dict.get("p"))
                        computed_hp.add(hp_tuple)
                    except json.JSONDecodeError:
                        continue
    
    results = {}
    # Loop over all combinations.
    for alpha, beta, sigma, p, pre_impute in product(
        alpha_values, beta_values, sigma_values, model_p_values, pre_impute_values
    ):
        hp_key = (alpha, beta, sigma, p, pre_impute)
        
        # Skip if this hyperparameter combination has already been computed.
        if hp_key in computed_hp:
            print(f"Skipping hyperparameter combination {hp_key} as it is already computed.")
            continue

        print(f"\n=== Hyperparameter Combination: alpha {alpha} beta {beta}, sigma={sigma}, p={p}")
        
        # Generate a corrupted dataset using a fixed corruption level.
        X_corrupted, dataset_train, dataset_val = generate_missingness(X, p_dataset=p_dataset)
        X_train     = dataset_train['X']
        X_test      = dataset_val['X']
        X_original  = dataset_val['X_ori']
        
        # Train the model.
        start_time = time.time()
        model = train_model(
            X_train,
            X_test,
            X_original,
            latent_size = 8,
            epochs=n_epochs,
            alpha=alpha,
            beta=beta,
            sigma=sigma,
            p=p
        )
        training_time = time.time() - start_time
        
        # Evaluate the model.
        metrics = evaluate_model(model, X_test.copy(), X_original.copy())
        
        # Build a result dictionary with hyperparameters stored as a dict.
        hp_dict = {
            "alpha": alpha,
            "beta": beta,
            "sigma": sigma,
            "p": p,
            "pre_imputation": True
        }
        result = {
            "hyperparameters": hp_dict,
            "metrics": metrics,
            "training_time": training_time
        }
        results[hp_key] = result
        
        print(f"Combination {hp_key}: Training Time = {training_time:.2f} sec")
        
        # Offload the intermediate result to disk immediately.
        append_result_to_disk(result, INTERMEDIATE_RESULTS_FILE)

        # Free resources after finishing this combination.
        del model
        del X_train, X_test, X_original, X_corrupted, dataset_train, dataset_val
        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return results

def save_results_to_file(results, save_path):
    """
    Save the hyperparameter search results to a text file.
    """
    with open(save_path, "w") as f:
        for hp_key, data in results.items():
            beta, noise_sigma, noise_std, use_mean_z, detach_flag = hp_key
            m = data["metrics"]
            f.write(f"Hyperparameters: beta={beta:.4e}, noise_sigma={noise_sigma:.4e}, noise_std={noise_std:.4e}, "
                    f"use_mean_z={use_mean_z}, detach={detach_flag}\n")
            f.write("  Observed Metrics:\n")
            f.write(f"    RMSE: {m['observed']['rmse']:.4f}\n")
            f.write(f"    NLL: {m['observed']['nll']:.4f}\n")
            f.write(f"    ELBO: {m['observed']['elbo']:.4f}\n")
            f.write(f"    Coverage: {m['observed']['coverage']:.4f}\n")
            f.write(f"    Uncertainty-Error Corr: {m['observed']['uncertainty_error_corr']:.4f}\n")
            f.write("  Missing Metrics:\n")
            f.write(f"    RMSE: {m['missing']['rmse']:.4f}\n")
            f.write(f"    NLL: {m['missing']['nll']:.4f}\n")
            f.write(f"    ELBO: {m['missing']['elbo']:.4f}\n")
            f.write(f"    Coverage: {m['missing']['coverage']:.4f}\n")
            f.write(f"    Uncertainty-Error Corr: {m['missing']['uncertainty_error_corr']:.4f}\n")
            f.write("  Latent Calibration:\n")
            f.write(f"    Average KS Statistic: {m['latent']['latent_ks']:.4f}\n")
            f.write("  Reconstruction Overall Coverage:\n")
            f.write(f"    Overall Coverage: {m['reconstruction']['overall_coverage']:.4f}\n")
            f.write(f"  Training Time: {data['training_time']:.2f} sec\n")
            f.write("\n")

def plot_search_results(results, save_dir="results_hyperparameter_search_other"):
    """
    Plot metrics versus hyperparameters. Because we have multiple hyperparameters,
    we can fix all but one and plot the effect of the remaining one. Alternatively,
    you can produce heatmaps or use parallel coordinate plots.
    
    For simplicity, here we illustrate the effect of beta while fixing the others at baseline values.
    """
    # Choose baseline values for the other parameters:
    baseline_noise_sigma = 0.01
    baseline_noise_std   = 0.1
    baseline_use_mean_z  = True
    baseline_detach      = True
    
    # Extract beta values and corresponding metrics from results.
    betas = []
    rmse_obs_list = []
    training_times = []
    for hp_key, data in results.items():
        beta, noise_sigma, noise_std, use_mean_z, detach_flag = hp_key
        if (np.isclose(noise_sigma, baseline_noise_sigma) and 
            np.isclose(noise_std, baseline_noise_std) and 
            use_mean_z == baseline_use_mean_z and 
            detach_flag == baseline_detach):
            betas.append(beta)
            rmse_obs_list.append(data["metrics"]["observed"]["rmse"])
            training_times.append(data["training_time"])
    
    if betas:
        betas = np.array(betas)
        # Plot RMSE vs beta.
        plt.figure(figsize=(8, 6))
        plt.plot(betas, rmse_obs_list, marker='o', linestyle='-')
        plt.xlabel("Beta")
        plt.ylabel("Observed RMSE")
        plt.title("Observed RMSE vs. Beta (Baseline)")
        plt.grid(True)
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, "RMSE_vs_Beta.png"), dpi=300)
        plt.close()
        
        # Plot training time vs beta.
        plt.figure(figsize=(8, 6))
        plt.plot(betas, training_times, marker='o', linestyle='-')
        plt.xlabel("Beta")
        plt.ylabel("Training Time (sec)")
        plt.title("Training Time vs. Beta (Baseline)")
        plt.grid(True)
        plt.savefig(os.path.join(save_dir, "TrainingTime_vs_Beta.png"), dpi=300)
        plt.close()
    else:
        print("No results found for the baseline hyperparameter configuration.")

if __name__ == "__main__":
    # 1. Generate Synthetic Data
    X = Generate_samples(
        n_groups=1,
        n_observations_per_group=800,
        n_dimensions=3,
        n_time_points=30,
        length_scale=1.5,
        n_new_dims=6
    )

    print('Done')
    
    # 2. Run Hyperparameter Search for the other parameters.
    search_results = hyperparameter_search_other(
        X,
        alpha_values,
        beta_values,
        sigma_values,
        model_p_values,
        n_epochs=3000
    )
    
    # 3. Save the final results to a file.
    RESULTS_DIR = "results_hyperparameter_search_other"
    os.makedirs(RESULTS_DIR, exist_ok=True)
    results_file = os.path.join(RESULTS_DIR, "hyperparameter_search_results_other_sampling.txt")
    save_results_to_file(search_results, results_file)
    print(f"\n✅ Hyperparameter search (other parameters) completed! Results saved in {results_file}")
    
    # 4. Generate summary plots.
    plot_search_results(search_results, save_dir=RESULTS_DIR)
