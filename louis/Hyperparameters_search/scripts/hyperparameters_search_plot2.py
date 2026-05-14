#!/usr/bin/env python
"""
This script reads a hyperparameter search result file (in the format produced by your experiments),
parses the metrics for each hyperparameter combination, and produces summary plots.

The expected format in the text file is (example):

Hyperparameters: beta=5.0000e-01, noise_sigma=5.0000e-03, noise_std=5.0000e-02, use_mean_z=True, detach=True
  Observed Metrics:
    RMSE: 0.8527
    NLL: 10485.4562
    ELBO: -14816.0109
    Coverage: 0.2092
    Uncertainty-Error Corr: -0.2498
  Missing Metrics:
    RMSE: 0.9044
    NLL: 12507.3581
    ELBO: -16837.9128
    Coverage: 0.1798
    Uncertainty-Error Corr: -0.1564
  Latent Calibration:
    Average KS Statistic: 0.4655
  Reconstruction Overall Coverage:
    Overall Coverage: 0.1943
  Training Time: 0.46 sec

Each block is separated by a blank line.
"""

import re
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set seaborn style
sns.set(style="whitegrid")

# -------------------------
# Step 1. Parse the text file.
# -------------------------

import pandas as pd
import re

def parse_results_file(filepath):
    """
    Parses the hyperparameter search results file into a DataFrame.
    
    - Automatically extracts all hyperparameters and metrics (no need for manual updates).
    - Supports dynamically added hyperparameters and metric fields.
    - Properly converts numbers, booleans, and strings.
    """
    with open(filepath, "r") as f:
        text = f.read()

    # Split into blocks (each block corresponds to one hyperparameter combination)
    blocks = text.strip().split("\n\n")
    records = []

    for block in blocks:
        lines = block.splitlines()
        if not lines:
            continue

        record = {}

        # **Extract Hyperparameters**
        hp_match = re.match(r"Hyperparameters:\s*(.*)", lines[0])
        if hp_match:
            hp_parts = hp_match.group(1).split(", ")
            for hp in hp_parts:
                key, value = hp.split("=")
                key = key.strip()
                value = value.strip()

                # Convert booleans
                if value.lower() in ["true", "false"]:
                    record[key] = value.lower() == "true"
                # Convert floats (including scientific notation)
                elif re.fullmatch(r"-?\d+(\.\d+)?(e-?\d+)?", value):
                    record[key] = float(value)
                # Convert integers
                elif re.fullmatch(r"-?\d+", value):
                    record[key] = int(value)
                else:
                    record[key] = value  # Keep as string if none of the above

        # **Extract Metrics**
        section = None  # Track which metrics section we are in
        for line in lines[1:]:
            line = line.strip()

            if line.endswith("Metrics:"):
                section = line.split()[0].lower()  # "observed" -> "obs", "missing" -> "miss"
                continue
            elif line.startswith("Latent Calibration:"):
                section = "latent"
                continue
            elif line.startswith("Reconstruction Overall Coverage:"):
                section = "reconstruction"
                continue
            elif line.startswith("Training Time:"):
                match = re.search(r"Training Time:\s*([\d\.]+)", line)
                if match:
                    record["training_time"] = float(match.group(1))
                continue

            # Extract metric values
            parts = line.split(":")
            if len(parts) >= 2:
                metric_name = parts[0].strip().replace(" ", "_").lower()  # Normalize names
                try:
                    metric_value = float(parts[1].strip())  # Convert to float
                    record[f"{section}_{metric_name}"] = metric_value
                except ValueError:
                    continue  # Skip if it’s not a number

        records.append(record)

    # Convert list of records into a DataFrame
    df = pd.DataFrame(records)

    return df

# -------------------------
# Step 2. Produce plots.
# -------------------------
def produce_plots(df, save_dir="plots"):
    os.makedirs(save_dir, exist_ok=True)
    
    # Example 1: Scatter plot of Observed RMSE vs Beta.
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x="beta", y="obs_rmse", hue="noise_std", style="use_mean_z", s=100)
    plt.xscale("log")
    plt.title("Observed RMSE vs Beta")
    plt.savefig(os.path.join(save_dir, "obs_rmse_vs_beta.png"), dpi=300)
    plt.close()
    
    # Example 2: Scatter plot of Observed ELBO vs Beta.
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x="beta", y="obs_elbo", hue="noise_std", style="detach", s=100)
    plt.xscale("log")
    plt.title("Observed ELBO vs Beta")
    plt.savefig(os.path.join(save_dir, "obs_elbo_vs_beta.png"), dpi=300)
    plt.close()
    
    # Example 3: Scatter plot of Observed RMSE vs Noise Std (colored by Beta).
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x="noise_std", y="obs_rmse", hue="beta", style="use_mean_z", palette="coolwarm", s=100)
    plt.title("Observed RMSE vs Noise Std")
    plt.savefig(os.path.join(save_dir, "obs_rmse_vs_noise_std.png"), dpi=300)
    plt.close()
    
    # Example 4: Plot Training Time vs Beta.
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x="beta", y="training_time", hue="noise_std", style="detach", s=100)
    plt.xscale("log")
    plt.title("Training Time vs Beta")
    plt.savefig(os.path.join(save_dir, "training_time_vs_beta.png"), dpi=300)
    plt.close()
    
    # Example 5: Create a pairplot for key hyperparameters and metrics.
    vars_to_plot = ["beta", "noise_sigma", "noise_std", "obs_rmse", "obs_elbo", "latent_ks", "training_time"]
    pp = sns.pairplot(df, vars=vars_to_plot, hue="use_mean_z", palette="Set2")
    pp.fig.suptitle("Pairplot of Hyperparameters and Metrics", y=1.02)
    pp.savefig(os.path.join(save_dir, "pairplot.png"), dpi=300)
    plt.close()
    
    print(f"Plots saved in directory: {save_dir}")

# -------------------------
# Main
# -------------------------
if __name__ == "__main__":
    # Path to your results text file (adjust the path as needed)
    results_filepath = "results_hyperparameter_search_other/hyperparameter_search_results_other.txt"
    results_filepath = "results_hyperparameter_search_other/hyperparameter_search_results_other.txt"
    # Parse the file.
    df = parse_results_file(results_filepath)
    print("Parsed DataFrame:")
    print(df.head())
    
    # Produce plots.
    produce_plots(df, save_dir="plots")
