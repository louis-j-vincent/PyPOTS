import torch
import torch.optim as optim
import torch.distributions as dist

class MultiScaleKernel:
    def __init__(self):
        pass

    def kernel(self, X1, X2, raw_params):
        """
        Compute a multi-scale kernel as the sum of:
         - An RBF kernel for long-term dynamics
         - A Cauchy kernel for short-term dynamics
        
        raw_params is a tensor of shape (5,):
            raw_params[0]: log10(length_scale_rbf)
            raw_params[1]: log10(sigma_rbf)
            raw_params[2]: log10(length_scale_cauchy)
            raw_params[3]: log10(sigma_cauchy)
            raw_params[4]: log10(noise)
        """
        dists = torch.cdist(X1, X2, p=2)
        
        # Transform raw parameters with clipping
        length_scale_rbf = 10.0 ** raw_params[0].clip(min=-5, max=1)
        sigma_rbf = 10.0 ** raw_params[1].clip(min=-3, max=0)
        
        length_scale_cauchy = 10.0 ** raw_params[2].clip(min=-5, max=0)
        sigma_cauchy = 10.0 ** raw_params[3].clip(min=-3, max=0)
        
        noise = 10.0 ** raw_params[4].clip(min=-5, max=-1)
        
        # RBF kernel (long term)
        K_rbf = sigma_rbf * torch.exp(-0.5 * (dists / length_scale_rbf) ** 2)
        
        # Cauchy kernel (short term)
        K_cauchy = sigma_cauchy**2 * (1 + (dists / length_scale_cauchy)**2) ** (-1)
        
        # Sum the two kernels
        K = K_rbf + K_cauchy
        
        return K, noise

    def prior_loss(self, raw_params):
        """
        Compute the prior loss (negative log prior) for the raw kernel parameters.
        We assume independent Normal priors on the log10 parameters.
        
        Suggested priors (adjust based on exploratory analysis):
          raw_params[0] (RBF length scale): mean = -0.3, std = 0.5
          raw_params[1] (RBF sigma):         mean = -1.0, std = 0.5
          raw_params[2] (Cauchy length scale): mean = -1.3, std = 0.5
          raw_params[3] (Cauchy sigma):         mean = -1.0, std = 0.5
          raw_params[4] (noise):              mean = -2.0, std = 0.5
        """
        prior_loss = 0.0
        priors = [
            (raw_params[0], -0.3, 0.5),
            (raw_params[1], -1.0, 0.5),
            (raw_params[2], -1.3, 0.5),
            (raw_params[3], -1.0, 0.5),
            (raw_params[4], -2.0, 0.5),
        ]
        
        for param, mean, std in priors:
            prior_dist = dist.Normal(torch.tensor(mean, device=param.device),
                                     torch.tensor(std, device=param.device))
            prior_loss = prior_loss - prior_dist.log_prob(param)
        return prior_loss.sum()

def log_marginal_likelihood(K, y, noise):
    """
    Compute the log marginal likelihood of the GP.
    
    Parameters:
      K: Kernel matrix (without noise)
      y: Observations (n x 1 tensor)
      noise: Noise variance (scalar)
    """
    n = y.shape[0]
    K_noise = K + noise * torch.eye(n, device=K.device)
    L = torch.linalg.cholesky(K_noise)
    alpha = torch.cholesky_solve(y, L)
    log_det = 2.0 * torch.sum(torch.log(torch.diag(L)))
    ll = -0.5 * (y.T @ alpha) - 0.5 * log_det - 0.5 * n * torch.log(torch.tensor(2 * torch.pi))
    return ll.squeeze()

import torch
import matplotlib.pyplot as plt

def plot_kernel_components(kernel_model, raw_params):
    """
    Plots the RBF and Cauchy kernel components as well as their sum as a function of distance.
    
    Parameters:
      kernel_model: Instance of your MultiScaleKernel (or similar) class.
      raw_params: Tensor containing the raw parameters (in log10-space) that have been optimized.
    """
    # Define a range of distances (assuming observations in [0,1])
    distances = torch.linspace(0, 1, 200).unsqueeze(1)  # shape: (200, 1)
    
    # Extract and transform parameters (using the same transforms as in your kernel function)
    length_scale_rbf = 10.0 ** raw_params[0].clip(min=-5, max=1)
    sigma_rbf = 10.0 ** raw_params[1].clip(min=-3, max=0)
    length_scale_cauchy = 10.0 ** raw_params[2].clip(min=-5, max=0)
    sigma_cauchy = 10.0 ** raw_params[3].clip(min=-3, max=0)
    
    # Compute kernel components
    K_rbf = sigma_rbf * torch.exp(-0.5 * (distances / length_scale_rbf) ** 2)
    K_cauchy = sigma_cauchy**2 * (1 + (distances / length_scale_cauchy) ** 2) ** (-1)
    K_total = K_rbf + K_cauchy
    
    # Convert to NumPy arrays for plotting
    distances_np = distances.squeeze().cpu().detach().numpy()
    K_rbf_np = K_rbf.squeeze().cpu().detach().numpy()
    K_cauchy_np = K_cauchy.squeeze().cpu().detach().numpy()
    K_total_np = K_total.squeeze().cpu().detach().numpy()
    
    # Plot the kernel components and the total kernel
    plt.figure(figsize=(8, 6))
    plt.plot(distances_np, K_rbf_np, label='RBF Component (Long-term)', linewidth=2)
    plt.plot(distances_np, K_cauchy_np, label='Cauchy Component (Short-term)', linewidth=2)
    plt.plot(distances_np, K_total_np, label='Total Kernel', linewidth=2, linestyle='--')
    plt.xlabel('Distance')
    plt.ylabel('Kernel Value')
    plt.title('Kernel Components vs. Distance')
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_kernel_matrix(kernel_model, X_train, raw_params):
    """
    Computes and displays the kernel matrix on the training data as a heatmap.
    
    Parameters:
      kernel_model: Instance of your MultiScaleKernel (or similar) class.
      X_train: Tensor of training inputs (e.g., time points, shape: [n, 1]).
      raw_params: Tensor containing the raw parameters (in log10-space) that have been optimized.
    """
    K, noise = kernel_model.kernel(X_train, X_train, raw_params)
    K_np = K.cpu().detach().numpy()
    
    plt.figure(figsize=(8, 6))
    plt.imshow(K_np, interpolation='nearest', cmap='viridis')
    plt.title('Kernel Matrix on Training Data')
    plt.xlabel('Time Index')
    plt.ylabel('Time Index')
    plt.colorbar(label='Kernel Value')
    plt.show()

import torch

def gp_predict(kernel_model, X_train, y_train, X_test, raw_params):
    """
    Compute the GP predictive mean and variance at test inputs X_test.
    
    Parameters:
      kernel_model: Instance of your kernel class (e.g., MultiScaleKernel).
      X_train: Training inputs, shape (n, 1).
      y_train: Training outputs, shape (n, 1).
      X_test: Test inputs, shape (m, 1).
      raw_params: Tensor of raw kernel parameters (in log10-space).
      
    Returns:
      pred_mean: Predictive mean at X_test (m x 1).
      pred_var: Predictive variance (diagonal) at X_test (m,).
    """
    # Compute training kernel matrix and noise term
    K_train, noise = kernel_model.kernel(X_train, X_train, raw_params)
    n = X_train.shape[0]
    K_train_noise = K_train + noise * torch.eye(n, device=K_train.device)
    
    # Compute kernel between test and training points
    K_star, _ = kernel_model.kernel(X_test, X_train, raw_params)
    # Compute kernel at test points
    K_star_star, _ = kernel_model.kernel(X_test, X_test, raw_params)
    
    # Use Cholesky decomposition for stability
    L = torch.linalg.cholesky(K_train_noise)
    
    # Solve for alpha: (K_train_noise) * alpha = y_train
    alpha = torch.cholesky_solve(y_train, L)
    
    # Predictive mean: K_star * alpha
    pred_mean = K_star @ alpha
    
    # Compute predictive variance:
    # First, solve for v: L * v = K_star.T
    v = torch.linalg.solve(L, K_star.T)
    # Variance: diag(K_star_star) - sum(v**2, dim=0)
    pred_var = torch.diag(K_star_star) - (v**2).sum(dim=0)
    
    return pred_mean, pred_var

import matplotlib.pyplot as plt

def plot_gp_fit(kernel_model, X_train, y_train, raw_params):
    """
    Plots the training data alongside the GP inference fit.
    
    Parameters:
      kernel_model: Instance of your kernel class.
      X_train: Training inputs (n x 1 tensor).
      y_train: Training outputs (n x 1 tensor).
      raw_params: Optimized raw kernel parameters.
    """
    # Create a dense test grid over the range of X_train
    X_test = torch.linspace(X_train.min(), X_train.max(), 200).unsqueeze(1)
    
    # Compute GP predictions
    pred_mean, pred_var = gp_predict(kernel_model, X_train, y_train, X_test, raw_params)
    pred_mean_np = pred_mean.squeeze().detach().cpu().numpy()
    pred_std_np = torch.sqrt(pred_var).detach().cpu().numpy()
    X_test_np = X_test.squeeze().detach().cpu().numpy()
    
    # Convert training data for plotting
    X_train_np = X_train.squeeze().detach().cpu().numpy()
    y_train_np = y_train.squeeze().detach().cpu().numpy()
    
    # Plot the GP mean and confidence intervals
    plt.figure(figsize=(10, 6))
    plt.plot(X_test_np, pred_mean_np, label="GP Mean", linewidth=2)
    plt.fill_between(
        X_test_np,
        pred_mean_np - 2 * pred_std_np,
        pred_mean_np + 2 * pred_std_np,
        alpha=0.3,
        label="Confidence Interval (±2σ)"
    )
    plt.scatter(X_train_np, y_train_np, color="red", label="Training Data", zorder=5)
    plt.xlabel("Input")
    plt.ylabel("Output")
    plt.title("Gaussian Process Fit")
    plt.legend()
    plt.grid(True)
    plt.show()


# Example usage:
if __name__ == '__main__':
    # Create some synthetic training data
    # Let's assume X_train are time points in [0, 1] and y_train are observations in [0, 1]
    n_train = 50  # for example, 50 observations
    X_train = torch.linspace(0, 1, n_train).unsqueeze(1)
    # For demonstration, let y_train be a simple function of time with noise
    true_function = lambda t: 0.5 * torch.sin(2 * torch.pi * t) + 0.5
    y_train = true_function(X_train) + 0.05 * torch.randn_like(X_train)
    
    # Initialize raw parameters (5 parameters in log10-space)
    raw_params = torch.randn(5, requires_grad=True)
    
    kernel_model = MultiScaleKernel()
    optimizer = optim.Adam([raw_params], lr=0.01)
    num_epochs = 1000
    
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        
        K, noise = kernel_model.kernel(X_train, X_train, raw_params)
        ll = log_marginal_likelihood(K, y_train, noise)
        prior = kernel_model.prior_loss(raw_params)
        
        # We minimize the negative log marginal likelihood plus the prior loss
        loss = -ll + prior
        loss.backward()
        optimizer.step()
        
        if (epoch+1) % 10 == 0:
            plot_gp_fit(kernel_model, X_train, y_train, raw_params)

            print(f"Epoch {epoch+1}/{num_epochs}: Loss = {loss.item():.4f}, LogML = {ll.item():.4f}")
    
    print("Optimized raw parameters:", raw_params.detach().numpy())

    plot_gp_fit(kernel_model, X_train, y_train, raw_params)
    plot_kernel_components(kernel_model, raw_params)
    plot_kernel_matrix(kernel_model, X_train, raw_params)


