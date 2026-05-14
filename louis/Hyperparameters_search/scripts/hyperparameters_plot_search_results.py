import os
import re
import numpy as np
import matplotlib.pyplot as plt

def parse_results(file_path):
    """
    Parse the hyperparameter search results text file.
    
    Expected format (indentation is used for structure):
    
    Missing Rate: 0.1
      Alpha: 1.0000e-03
        Observed Metrics:
          RMSE: 0.0421
          NLL: -12710.3046
          ELBO: -645382.7579
          Coverage: 0.3414
          Uncertainty-Error Corr: -0.0271
        Missing Metrics:
          RMSE: 0.5938
          NLL: 9383.4589
          ELBO: -667476.5214
          Coverage: 0.0576
          Uncertainty-Error Corr: 0.1512
        Latent Calibration:
          Average KS Statistic: 0.1985
        Reconstruction Overall Coverage:
          Overall Coverage: 0.3138
        Training Time: 172.52 sec
      Alpha: 4.6416e-03
        ...
    
    Returns a dictionary with the structure:
    {
      missing_rate (float): {
          alpha (float): {
              "observed": {"rmse": float, "nll": float, "elbo": float,
                           "coverage": float, "uncertainty_error_corr": float},
              "missing": { ... },
              "latent": {"latent_ks": float},
              "reconstruction": {"overall_coverage": float},
              "training_time": float
          },
          ...
      },
      ...
    }
    """
    results = {}
    current_missing = None
    current_alpha = None
    current_section = None  # "observed", "missing", "latent", "reconstruction"
    
    # Regular expressions.
    re_missing = re.compile(r"Missing Rate:\s*([\d\.]+)")
    re_alpha = re.compile(r"Alpha:\s*([\deE\+\-\.]+)")
    re_metric = re.compile(r"(RMSE|NLL|ELBO|Coverage|Uncertainty-Error Corr|Average KS Statistic|Overall Coverage):\s*([-\d\.eE]+)")
    re_time = re.compile(r"Training Time:\s*([-\d\.]+)\s*sec")
    
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            # Check for a Missing Rate line.
            m = re_missing.match(line)
            if m:
                current_missing = float(m.group(1))
                results[current_missing] = {}
                continue

            # Check for an Alpha line.
            m = re_alpha.match(line)
            if m:
                current_alpha = float(m.group(1))
                results[current_missing][current_alpha] = {}
                continue

            # Check for section headers.
            if line.startswith("Observed Metrics:"):
                current_section = "observed"
                results[current_missing][current_alpha][current_section] = {}
                continue
            elif line.startswith("Missing Metrics:"):
                current_section = "missing"
                results[current_missing][current_alpha][current_section] = {}
                continue
            elif line.startswith("Latent Calibration:"):
                current_section = "latent"
                results[current_missing][current_alpha][current_section] = {}
                continue
            elif line.startswith("Reconstruction Overall Coverage:"):
                current_section = "reconstruction"
                results[current_missing][current_alpha][current_section] = {}
                continue

            # Check for metric lines.
            m = re_metric.match(line)
            if m and current_section is not None:
                metric_name = m.group(1)
                value = float(m.group(2))
                if metric_name == "RMSE":
                    key = "rmse"
                elif metric_name == "NLL":
                    key = "nll"
                elif metric_name == "ELBO":
                    key = "elbo"
                elif metric_name == "Coverage":
                    key = "coverage"
                elif metric_name == "Uncertainty-Error Corr":
                    key = "uncertainty_error_corr"
                elif metric_name == "Average KS Statistic":
                    key = "latent_ks"
                elif metric_name == "Overall Coverage":
                    key = "overall_coverage"
                else:
                    key = metric_name.lower()
                results[current_missing][current_alpha][current_section][key] = value
                continue

            # Check for Training Time.
            m = re_time.match(line)
            if m:
                results[current_missing][current_alpha]["training_time"] = float(m.group(1))
                continue

    return results

def plot_metrics(results, save_dir="results_hyperparameter_search"):
    """
    Generate plots from the parsed results.
    
    The following figures are produced:
      1. RMSE, NLL, and ELBO vs. alpha (observed and missing metrics together).
      2. Coverage metrics and Uncertainty-Error Correlation vs. alpha.
      3. Latent Calibration and Training Time vs. alpha.
      4. Observed RMSE and Coverage vs. alpha (twin axes).
      5. Missing RMSE and Coverage vs. alpha (twin axes).
      6. Missing RMSE and Latent Calibration vs. alpha (twin axes).
    
    Colors are fixed per missing rate.
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    
    # Fixed colors.
    colors = {0.1: 'blue', 0.3: 'green', 0.5: 'red', 0.7: 'purple'}
    missing_rates = sorted(results.keys())
    
    # ---------------------
    # Figure 1: RMSE, NLL, ELBO (observed vs. missing)
    # ---------------------
    fig1, axes1 = plt.subplots(3, 1, figsize=(10, 15))
    for mr in missing_rates:
        alphas = sorted(results[mr].keys())
        alphas_float = np.array(alphas, dtype=float)
        rmse_obs = [results[mr][alpha]["observed"]["rmse"] for alpha in alphas]
        rmse_miss = [results[mr][alpha]["missing"]["rmse"] for alpha in alphas]
        nll_obs = [results[mr][alpha]["observed"]["nll"] for alpha in alphas]
        nll_miss = [results[mr][alpha]["missing"]["nll"] for alpha in alphas]
        elbo_obs = [results[mr][alpha]["observed"]["elbo"] for alpha in alphas]
        elbo_miss = [results[mr][alpha]["missing"]["elbo"] for alpha in alphas]
        
        axes1[0].plot(alphas_float, rmse_obs, marker='o', color=colors[mr],
                      label=f"Missing {mr} - Obs")
        axes1[0].plot(alphas_float, rmse_miss, marker='x', linestyle='--', color=colors[mr],
                      label=f"Missing {mr} - Mask")
        axes1[1].plot(alphas_float, nll_obs, marker='o', color=colors[mr],
                      label=f"Missing {mr} - Obs")
        axes1[1].plot(alphas_float, nll_miss, marker='x', linestyle='--', color=colors[mr],
                      label=f"Missing {mr} - Mask")
        axes1[2].plot(alphas_float, elbo_obs, marker='o', color=colors[mr],
                      label=f"Missing {mr} - Obs")
        axes1[2].plot(alphas_float, elbo_miss, marker='x', linestyle='--', color=colors[mr],
                      label=f"Missing {mr} - Mask")
    
    for ax, title in zip(axes1, ["RMSE", "NLL", "ELBO"]):
        ax.set_xscale("log")
        ax.set_xlabel("Alpha")
        ax.set_ylabel(title)
        ax.set_title(title)
        ax.grid(True)
        ax.legend()
    fig1.tight_layout()
    fig1.savefig(os.path.join(save_dir, "figure1_RMSE_NLL_ELBO.png"), dpi=300)
    plt.close(fig1)
    
    # ---------------------
    # Figure 2: Coverage and Uncertainty-Error Correlation.
    # ---------------------
    fig2, axes2 = plt.subplots(2, 1, figsize=(10, 12))
    for mr in missing_rates:
        alphas = sorted(results[mr].keys())
        alphas_float = np.array(alphas, dtype=float)
        cov_obs = [results[mr][alpha]["observed"]["coverage"] for alpha in alphas]
        cov_miss = [results[mr][alpha]["missing"]["coverage"] for alpha in alphas]
        overall_cov = [results[mr][alpha]["reconstruction"]["overall_coverage"] for alpha in alphas]
        corr_obs = [results[mr][alpha]["observed"]["uncertainty_error_corr"] for alpha in alphas]
        corr_miss = [results[mr][alpha]["missing"]["uncertainty_error_corr"] for alpha in alphas]
        
        axes2[0].plot(alphas_float, cov_obs, marker='o', color=colors[mr],
                      label=f"Missing {mr} - Obs")
        axes2[0].plot(alphas_float, cov_miss, marker='x', linestyle='--', color=colors[mr],
                      label=f"Missing {mr} - Mask")
        axes2[0].plot(alphas_float, overall_cov, marker='s', linestyle=':', color=colors[mr],
                      label=f"Missing {mr} - Overall")
        
        axes2[1].plot(alphas_float, corr_obs, marker='o', color=colors[mr],
                      label=f"Missing {mr} - Obs")
        axes2[1].plot(alphas_float, corr_miss, marker='x', linestyle='--', color=colors[mr],
                      label=f"Missing {mr} - Mask")
    
    for ax, ylabel, title in zip(axes2, ["Coverage", "Uncertainty-Error Corr"],
                                 ["Coverage Metrics", "Uncertainty vs. Error Correlation"]):
        ax.set_xscale("log")
        ax.set_xlabel("Alpha")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True)
        ax.legend()
    fig2.tight_layout()
    fig2.savefig(os.path.join(save_dir, "figure2_Coverage_UncertaintyErrorCorr.png"), dpi=300)
    plt.close(fig2)
    
    # ---------------------
    # Figure 3: Latent Calibration and Training Time.
    # ---------------------
    fig3, axes3 = plt.subplots(2, 1, figsize=(10, 10))
    for mr in missing_rates:
        alphas = sorted(results[mr].keys())
        alphas_float = np.array(alphas, dtype=float)
        latent_ks = [results[mr][alpha]["latent"]["latent_ks"] for alpha in alphas]
        train_time = [results[mr][alpha]["training_time"] for alpha in alphas]
        
        axes3[0].plot(alphas_float, latent_ks, marker='o', color=colors[mr],
                      label=f"Missing {mr}")
        axes3[1].plot(alphas_float, train_time, marker='o', color=colors[mr],
                      label=f"Missing {mr}")
    
    axes3[0].set_xscale("log")
    axes3[0].set_xlabel("Alpha")
    axes3[0].set_ylabel("Latent KS")
    axes3[0].set_title("Latent Calibration (Average KS Statistic)")
    axes3[0].grid(True)
    axes3[0].legend()
    
    axes3[1].set_xscale("log")
    axes3[1].set_xlabel("Alpha")
    axes3[1].set_ylabel("Training Time (sec)")
    axes3[1].set_title("Training Time")
    axes3[1].grid(True)
    axes3[1].legend()
    
    fig3.tight_layout()
    fig3.savefig(os.path.join(save_dir, "figure3_LatentCalibration_TrainingTime.png"), dpi=300)
    plt.close(fig3)
    
    # ---------------------
    # Figure 4: Observed RMSE and Coverage (twin axes).
    # ---------------------
    fig4, ax4 = plt.subplots(figsize=(10,6))
    ax4_twin = ax4.twinx()
    for mr in missing_rates:
        alphas = sorted(results[mr].keys())
        alphas_float = np.array(alphas, dtype=float)
        rmse_obs = [results[mr][alpha]["observed"]["rmse"] for alpha in alphas]
        cov_obs = [results[mr][alpha]["observed"]["coverage"] for alpha in alphas]
        ax4.plot(alphas_float, rmse_obs, marker='o', color=colors[mr],
                 label=f"Observed {mr} - RMSE")
        ax4_twin.plot(alphas_float, cov_obs, marker='x', linestyle='--', color=colors[mr],
                     label=f"Observed {mr} - Coverage")
    ax4.set_xscale("log")
    ax4_twin.set_xscale("log")
    ax4.set_xlabel("Alpha")
    ax4.set_ylabel("Observed RMSE")
    ax4_twin.set_ylabel("Observed Coverage")
    ax4.set_title("Observed RMSE and Coverage")
    ax4.grid(True)
    lines1, labels1 = ax4.get_legend_handles_labels()
    lines2, labels2 = ax4_twin.get_legend_handles_labels()
    ax4.legend(lines1 + lines2, labels1 + labels2, loc='best')
    fig4.tight_layout()
    fig4.savefig(os.path.join(save_dir, "figure4_Observed_RMSE_Coverage.png"), dpi=300)
    plt.close(fig4)
    
    # ---------------------
    # Figure 5: Missing RMSE and Coverage (twin axes).
    # ---------------------
    fig5, ax5 = plt.subplots(figsize=(10,6))
    ax5_twin = ax5.twinx()
    for mr in missing_rates:
        alphas = sorted(results[mr].keys())
        alphas_float = np.array(alphas, dtype=float)
        rmse_miss = [results[mr][alpha]["missing"]["rmse"] for alpha in alphas]
        cov_miss = [results[mr][alpha]["missing"]["coverage"] for alpha in alphas]
        ax5.plot(alphas_float, rmse_miss, marker='o', color=colors[mr],
                 label=f"Missing {mr} - RMSE")
        ax5_twin.plot(alphas_float, cov_miss, marker='x', linestyle='--', color=colors[mr],
                     label=f"Missing {mr} - Coverage")
    ax5.set_xscale("log")
    ax5_twin.set_xscale("log")
    ax5.set_xlabel("Alpha")
    ax5.set_ylabel("Missing RMSE")
    ax5_twin.set_ylabel("Missing Coverage")
    ax5.set_title("Missing RMSE and Coverage")
    ax5.grid(True)
    lines1, labels1 = ax5.get_legend_handles_labels()
    lines2, labels2 = ax5_twin.get_legend_handles_labels()
    ax5.legend(lines1 + lines2, labels1 + labels2, loc='best')
    fig5.tight_layout()
    fig5.savefig(os.path.join(save_dir, "figure5_Missing_RMSE_Coverage.png"), dpi=300)
    plt.close(fig5)
    
    # ---------------------
    # Figure 6: Missing RMSE and Latent Calibration (twin axes).
    # ---------------------
    fig6, ax6 = plt.subplots(figsize=(10,6))
    ax6_twin = ax6.twinx()
    for mr in missing_rates:
        alphas = sorted(results[mr].keys())
        alphas_float = np.array(alphas, dtype=float)
        rmse_miss = [results[mr][alpha]["missing"]["rmse"] for alpha in alphas]
        latent_ks = [results[mr][alpha]["latent"]["latent_ks"] for alpha in alphas]
        ax6.plot(alphas_float, rmse_miss, marker='o', color=colors[mr],
                 label=f"Missing {mr} - RMSE")
        ax6_twin.plot(alphas_float, latent_ks, marker='x', linestyle='--', color=colors[mr],
                     label=f"Missing {mr} - Latent KS")
    ax6.set_xscale("log")
    ax6_twin.set_xscale("log")
    ax6.set_xlabel("Alpha")
    ax6.set_ylabel("Missing RMSE")
    ax6_twin.set_ylabel("Latent KS")
    ax6.set_title("Missing RMSE and Latent Calibration")
    ax6.grid(True)
    lines1, labels1 = ax6.get_legend_handles_labels()
    lines2, labels2 = ax6_twin.get_legend_handles_labels()
    ax6.legend(lines1 + lines2, labels1 + labels2, loc='best')
    fig6.tight_layout()
    fig6.savefig(os.path.join(save_dir, "figure6_Missing_RMSE_LatentCalibration.png"), dpi=300)
    plt.close(fig6)
    
    print("All figures have been saved.")

def main():
    results_file = os.path.join("results_hyperparameter_search", "hyperparameter_search_results.txt")
    if not os.path.exists(results_file):
        print(f"Results file not found at {results_file}. Please check the file path.")
        return
    
    results = parse_results(results_file)
    print("Parsed results:")
    print(results)
    
    plot_metrics(results, save_dir="results_hyperparameter_search")

if __name__ == "__main__":
    main()
