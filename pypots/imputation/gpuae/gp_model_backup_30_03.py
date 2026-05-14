from .kernels import *

# for probabilistic GP
import torch.nn.functional as F
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

import matplotlib.pyplot as plt

# For kernel parameters estimation
import gpytorch
from .kernels import SparseGPModel, BatchIndependentMultitaskGPModel, ExactGPModel

import numpy as np
import torch
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt

from tqdm import tqdm
from pygrinder import mcar

import torch
import torch.nn as nn
import torch.nn.functional as F
from ...optim.adam import Adam

import time

def make_nn(input_size, output_size, hidden_sizes):
    """This function used to creates fully connected neural network.

    Parameters
    ----------
    input_size : int,
        the dimension of input embeddings

    output_size : int,
        the dimension of out embeddings

    hidden_sizes : tuple,
        the tuple of hidden layer sizes, and the tuple length sets the number of hidden layers

    Returns
    -------
    output: tensor
        the processing embeddings
    """
    layers = []
    for i in range(len(hidden_sizes)):
        if i == 0:
            layers.append(nn.Linear(in_features=input_size, out_features=hidden_sizes[i]))
        else:
            layers.append(nn.Linear(in_features=hidden_sizes[i - 1], out_features=hidden_sizes[i]))
        layers.append(nn.ReLU())
    layers.append(nn.Linear(in_features=hidden_sizes[-1], out_features=output_size))
    #layers.append(nn.ReLU())
    return nn.Sequential(*layers)

class NeuralNetwork(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.latent_dim = latent_dim
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            #nn.ReLU(),
            #nn.Linear(latent_dim, latent_dim),
            nn.Sigmoid()        
            )

    def forward(self, x):
        x_shape = x.shape
        x = x.reshape(-1,self.latent_dim)
        logits = self.linear_relu_stack(x)
        logits = logits.reshape(x_shape)
        return logits

class ProbabilisticGP:

    def __init__(self, model, assemble_data_train, assemble_data_val, n_dims, latent_size):
        n_kernel_params = 5
        #latent_size = 20

        self.kernel_params_estimator = make_nn(input_size = n_dims, 
                                                output_size = latent_size * n_kernel_params,
                                                hidden_sizes = (n_dims * 10, n_dims * 10))
        self.dims_to_train = np.arange(latent_size)
        self.n_dims = n_dims
        self.latent_dim = latent_size
        self.n_kernel_params = n_kernel_params

        # data loader
        self.assemble_data_train = assemble_data_train
        self.assemble_data_val = assemble_data_val

        # get encoder
        self.encode = model.encoder.eval()
        self.decode = model.decoder.eval()
        self.plot_while_training = False

        # init optimizer
        self.optimizer = Adam()
        self.optimizer.init_optimizer(self.kernel_params_estimator.parameters())

        self.p = .3
        self.alpha = 0.01

        kernel_params_history = {'a':[],'b':[],'c':[],'d':[], 'e':[]}

        self.kernel_params_history = {i:{'a':[],'b':[],'c':[],'d':[], 'e':[]} for i in range(self.latent_dim)}

        self.variance_nn = VarianceNN(latent_dim = self.latent_dim, time_steps = 30)

    def update_kernel_params(self, x):
        """
        Estimate kernel parameters given x
        """

        m = (x!=0).float().mean(axis = 1)
        kernel_params = self.kernel_params_estimator(m).reshape(m.shape[0], self.latent_dim, self.n_kernel_params)
        
        return kernel_params

    def kernel(self, X1, X2, raw_params, kernel='multi_scale'):
        """
        Compute a multi-scale kernel as the sum of:
         - A RBF kernel for long-term dynamics
         - A Cauchy kernel for short-term dynamics
        
        raw_params is a tensor of shape (5,):
            raw_params[0]: log10(length_scale_rbf)
            raw_params[1]: log10(sigma_rbf)
            raw_params[2]: log10(length_scale_cauchy)
            raw_params[3]: log10(sigma_cauchy)
            raw_params[4]: log10(noise)
        """
        # Compute pairwise distances between points in X1 and X2
        dists = torch.cdist(X1, X2, p=2)
        
        # Transform raw parameters (clipped to avoid extreme values)
        # RBF kernel parameters (for long term)
        length_scale_rbf = 10.0 ** raw_params[0].clip(min=-2, max=1)  
        #length_scale_rbf = 1
        sigma_rbf = 10.0 ** raw_params[1].clip(min=-3, max=0)
        
        # Cauchy kernel parameters (for short term)
        length_scale_cauchy = 10.0 ** raw_params[2].clip(min=-5, max=0)
        sigma_cauchy = 10.0 ** raw_params[3].clip(min=-3, max=1)
        
        # Variance importance parameter
        var_importance = 10.0 ** raw_params[4].clip(min=-2, max=1)
        #var_importance = var_importance * 0 + 1


        #fixed params:
        var_importance2 = torch.tensor(1.)
        #length_scale_rbf = torch.tensor(1)
        #sigma_rbf = torch.tensor(1)
        #length_scale_cauchy = torch.tensor(1)
        #sigma_cauchy = torch.tensor(1)

        noise = torch.tensor(1e-6)
        
        # Compute RBF kernel for long-term dynamics:
        K_rbf = sigma_rbf * torch.exp(-0.5 * (dists / length_scale_rbf) ** 2)
        
        # Compute Cauchy kernel for short-term dynamics:
        # Note: The Cauchy kernel here is defined as:
        #   k_cauchy(r) = sigma_cauchy^2 / (1 + (r/length_scale_cauchy)^2)
        K_cauchy = sigma_cauchy**2 * (1 + (dists / length_scale_cauchy)**2) ** (-1)
        
        # Sum the kernels
        K = K_rbf + K_cauchy

        #K = K.detach()
        
        return K, noise, var_importance, var_importance2

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

    def negative_log_likelihood_loss(self, qz_x, kernel_params): #USELESS FOR NOW CAN BE DETLEED

        z_mu, z_var = qz_x.mean.detach(), qz_x.variance.clamp(min=1e-5).detach()

        kernel_params = kernel_params#.detach()
        
        T = torch.linspace(0, 1, z_mu.shape[1]).unsqueeze(1).to(z_mu.device)  # Time steps

        z_star = torch.zeros_like(z_mu)  # Corrected latent mean

        self.dims_to_train = np.arange(z_star.shape[2])  # Train over latent dimensions

        if self.use_quantile:
            latent_dim  = z_var.shape[-1]
            latent_variances = torch.zeros(z_var.shape)

            for j in range(latent_dim):
                quantile_j = torch.quantile(z_var[:,:,j].detach().flatten(), q = 0.2)
                quantile_j_up = torch.quantile(z_var[:,:,j].detach().flatten(), q = 0.99)
                latent_variances[:,:,j] = (z_var[:,:,j] > quantile_j).float() #* quantile_j_up

        else:
            latent_variances = self.nn(z_var)
        
        #latent_variances += 1e-6
        
        for j in self.dims_to_train:
            for b in range(z_mu.shape[0]):  #
                                
                # Compute the kernel matrix and noise using the multi-scale kernel.
                # (Here we use 'multi_scale'; you can also experiment with 'cauchy' or 'rbf'.)
                K, noise, var_importance, _ = self.kernel(T, T, kernel_params[b, j], kernel='multi_scale')
                
                # Get the latent trajectory for this sample and dimension.
                # Shape: (time_steps, 1)
                y = z_mu[b, :, j].unsqueeze(1)
                
                # Compute the log marginal likelihood for this trajectory.
                K_noise = K.detach() + var_importance * torch.diag(z_var[b, :, j]) + noise * torch.eye(time_steps, device=K.device)
                
    def loss(self, X, q):
        """
        Compute the loss for the kernel parameters based on the GP marginal likelihood
        and a prior on the kernel parameters.
        
        Parameters:
        X (torch.Tensor): Input data used to update kernel parameters (e.g., the raw observations).
        z_mu (torch.Tensor): Latent mean trajectories of shape (batch, time, latent_dim)
        
        Returns:
        loss_total (torch.Tensor): A scalar loss value.
        """
        z_mu, z_var = q.mean, q.variance

        # Get the kernel parameters from the estimator.
        # Expected shape: (batch, latent_dim, n_kernel_params)
        raw_params = self.update_kernel_params(X)
        
        loss_total = 0.0
        batch_size, time_steps, latent_dim = z_mu.shape
        
        # Loop over each latent dimension and each batch sample.
        # Here, we assume that for each sample and latent dimension,
        # the GP is fitted to the latent trajectory over time.
        for b in range(batch_size):
            for j in range(latent_dim):
                # Create a time grid for the latent trajectory (here we assume the time grid is uniform).
                T = torch.linspace(0, 1, time_steps).unsqueeze(1).to(z_mu.device)
                
                # Get the raw kernel parameters for this sample and latent dimension.
                # raw_params[b, j] has shape (n_kernel_params,)
                params_bj = raw_params[b, j]
                
                # Compute the kernel matrix and noise using the multi-scale kernel.
                # (Here we use 'multi_scale'; you can also experiment with 'cauchy' or 'rbf'.)
                K, noise, var_importance, _ = self.kernel(T, T, params_bj, kernel='multi_scale')
                
                # Get the latent trajectory for this sample and dimension.
                # Shape: (time_steps, 1)
                y = z_mu[b, :, j].unsqueeze(1)
                
                # Compute the log marginal likelihood for this trajectory.
                K_noise = K.detach() + var_importance * torch.diag(z_var[b, :, j]) + noise * torch.eye(time_steps, device=K.device)
                
                if False:

                    print('Both')
                    plt.imshow(np.array(K.detach().numpy()))
                    plt.show()

                    plt.imshow(np.array(K_noise.detach().numpy()))
                    plt.show()


                try:
                    L = torch.linalg.cholesky(K_noise)
                except RuntimeError as e:
                    print("Cholesky failed; adding extra jitter.")
                    L = torch.linalg.cholesky(K_noise + 1e-5 * torch.eye(time_steps, device=K.device))
                alpha = torch.cholesky_solve(y, L)
                log_det = 2.0 * torch.sum(torch.log(torch.diag(L)))
                ll = -0.5 * (y.T @ alpha) - 0.5 * log_det - 0.5 * time_steps * torch.log(torch.tensor(2 * torch.pi))
                # Negative log marginal likelihood (we want to minimize loss)
                nll = -ll.squeeze()
                
                # Compute the prior loss for the kernel parameters.
                # Here we use your defined prior_for_kernel_params function.
                prior_loss = self.prior_for_kernel_params(params_bj)
                
                # Sum up the loss for this latent dimension and batch sample.
                loss_total = loss_total + nll + prior_loss

        # Average the loss over batch and latent dimensions.
        loss_total = loss_total / (batch_size * latent_dim)
        
        return loss_total

    def loss(self, X, q):
        """
        Compute the loss for the kernel parameters based on the GP marginal likelihood
        and a prior on the kernel parameters.
        
        Parameters:
        X (torch.Tensor): Input data used to update kernel parameters (e.g., the raw observations).
        z_mu (torch.Tensor): Latent mean trajectories of shape (batch, time, latent_dim)
        
        Returns:
        loss_total (torch.Tensor): A scalar loss value.
        """
        z_mu, z_var = q.mean, q.variance

        # Get the kernel parameters from the estimator.
        # Expected shape: (batch, latent_dim, n_kernel_params)
        raw_params = self.update_kernel_params(X)
        
        loss_total = 0.0
        batch_size, time_steps, latent_dim = z_mu.shape

        T = torch.linspace(0, 1, z_mu.shape[1]).unsqueeze(1).to(z_mu.device)  # Time steps

        #z_star = torch.zeros_like(z_mu)  # Corrected latent mean

        #self.dims_to_train = np.arange(z_star.shape[2])  # Train over latent dimensions

        if self.use_quantile:
            latent_dim  = z_var.shape[-1]
            latent_variances = torch.zeros(z_var.shape)

            for j in range(latent_dim):
                quantile_j = torch.quantile(z_var[:,:,j].detach().flatten(), q = 0.2)
                quantile_j_up = torch.quantile(z_var[:,:,j].detach().flatten(), q = 0.99)
                latent_variances[:,:,j] = (z_var[:,:,j] > quantile_j).float() #* quantile_j_up

        else:
            latent_variances = self.nn(z_var)
        
        # Loop over each latent dimension and each batch sample.
        # Here, we assume that for each sample and latent dimension,
        # the GP is fitted to the latent trajectory over time.
        for b in range(batch_size):
            for j in range(latent_dim):
                # Create a time grid for the latent trajectory (here we assume the time grid is uniform).
                T = torch.linspace(0, 1, time_steps).unsqueeze(1).to(z_mu.device)
                
                # Get the raw kernel parameters for this sample and latent dimension.
                # raw_params[b, j] has shape (n_kernel_params,)
                params_bj = raw_params[b, j]
                
                # Compute the kernel matrix and noise using the multi-scale kernel.
                # (Here we use 'multi_scale'; you can also experiment with 'cauchy' or 'rbf'.)
                K, noise, var_importance, _ = self.kernel(T, T, params_bj, kernel='multi_scale')
                
                # Get the latent trajectory for this sample and dimension.
                # Shape: (time_steps, 1)
                y = z_mu[b, :, j].unsqueeze(1)
                
                # Compute the log marginal likelihood for this trajectory.
                #K_noise = K.detach() + var_importance * torch.diag(z_var[b, :, j]) + noise * torch.eye(time_steps, device=K.device)

                K_obs = K + torch.diag(latent_variances[b,:,j]) + noise

                
                if False:

                    print('Both')
                    plt.imshow(np.array(K.detach().numpy()))
                    plt.show()

                    plt.imshow(np.array(K_noise.detach().numpy()))
                    plt.show()


                try:
                    L = torch.linalg.cholesky(K_obs)
                except RuntimeError as e:
                    #print("Cholesky failed; adding extra jitter.")
                    L = torch.linalg.cholesky(K_obs + 1e-5 * torch.eye(time_steps, device=K.device))
                alpha = torch.cholesky_solve(y, L)
                log_det = 2.0 * torch.sum(torch.log(torch.diag(L)))
                ll = -0.5 * (y.T @ alpha) - 0.5 * log_det - 0.5 * time_steps * torch.log(torch.tensor(2 * torch.pi))
                # Negative log marginal likelihood (we want to minimize loss)
                nll = -ll.squeeze()
                
                # Compute the prior loss for the kernel parameters.
                # Here we use your defined prior_for_kernel_params function.
                prior_loss = self.prior_for_kernel_params(params_bj)
                
                # Sum up the loss for this latent dimension and batch sample.
                loss_total = loss_total + nll #+ prior_loss

        # Average the loss over batch and latent dimensions.
        loss_total = loss_total / (batch_size * latent_dim)
        
        return loss_total



    def correct_with_gp(self, z_mu, z_var, kernel_params, return_variance=False):
        """
        Perform Gaussian Process regression in the latent space, returning mean and optionally variance.
        
        Parameters:
            z_mu (torch.Tensor): Mean of the latent variables.
            z_var (torch.Tensor): Variance of the latent variables.
            kernel_params (torch.Tensor): Kernel hyperparameters.
            return_variance (bool): Whether to return predictive variance.

        Returns:
            z_star (torch.Tensor): GP-corrected latent mean.
            sigma_star (torch.Tensor, optional): GP-corrected latent variance (if return_variance=True).
        """
        
        z_mu = z_mu.detach()
        z_var = z_var.detach()
        #kernel_params = kernel_params.detach()
        
        T = torch.linspace(0, 1, z_mu.shape[1]).unsqueeze(1).to(z_mu.device)  # Time steps

        z_star = torch.zeros_like(z_mu)  # Corrected latent mean
        sigma_star = torch.zeros_like(z_var) if return_variance else None  # Corrected variance

        self.dims_to_train = np.arange(z_star.shape[2])  # Train over latent dimensions

        #corrected_variance = self.variance_nn(z_mu, z_var)

        latent_variances = self.nn(z_var)
        latent_variances = (z_var>1e-5)


        for j in self.dims_to_train:
            for b in range(z_mu.shape[0]):  # Iterate over batch dimension

                # Compute kernel matrix and observation noise
                K, noise, var_importance, var_importance2 = self.kernel(T, T, kernel_params[b, j])

                # Construct the observed covariance matrix
                #K_obs = K + var_importance * torch.diag(z_var[b, :, j]) ** var_importance2 + noise
                K_obs = K + torch.diag(latent_variances[b,:,j]) + noise
                #K_obs = K + torch.diag((z_var[b,:,j]>1e-5)) + noise


                # Compute predictive mean
                z_star[b, :, j] = torch.matmul(K, torch.linalg.inv(K_obs)) @ z_mu[b, :, j]

                # Compute predictive variance if requested
                if return_variance:
                    # Compute correction matrix
                    eps = 1e-3
                    correction_matrix = torch.linalg.solve(K_obs + eps * torch.eye(K_obs.size(0), device=K_obs.device), K)

                    k_xx = torch.diagonal(K)  # Extract k(t_*, t_*) elements
                    sigma_star[b, :, j] = k_xx - torch.einsum('ij,ji->i', correction_matrix, K.T)

            #print(K.mean(), K.var(),j)

                    

        return (z_star, sigma_star.clip(min=1e-3)) if return_variance else z_star

    def correct_with_gp(self, z_mu, z_var, kernel_params, return_variance=False):
        """
        Perform Gaussian Process regression in the latent space, returning mean and optionally variance.
        
        Parameters:
            z_mu (torch.Tensor): Mean of the latent variables.
            z_var (torch.Tensor): Variance of the latent variables.
            kernel_params (torch.Tensor): Kernel hyperparameters.
            return_variance (bool): Whether to return predictive variance.

        Returns:
            z_star (torch.Tensor): GP-corrected latent mean.
            sigma_star (torch.Tensor, optional): GP-corrected latent variance (if return_variance=True).
        """
        
        z_mu = z_mu.detach()
        z_var = z_var.detach()
        #kernel_params = kernel_params.detach()
        
        T = torch.linspace(0, 1, z_mu.shape[1]).unsqueeze(1).to(z_mu.device)  # Time steps

        z_star = torch.zeros_like(z_mu)  # Corrected latent mean
        sigma_star = torch.zeros_like(z_var) if return_variance else None  # Corrected variance

        self.dims_to_train = np.arange(z_star.shape[2])  # Train over latent dimensions

        #corrected_variance = self.variance_nn(z_mu, z_var)

        if self.use_quantile:
            latent_dim  = z_var.shape[-1]
            #z_var_reshaped = z_var.reshape(-1,latent_dim)
            #z_var_quantiles = torch.quantile(z_var_reshaped, q = 0.1, interpolation = 'lower', dim = 0)[None]
            #print(z_var_reshaped.shape)
            #z_var_reshaped = (z_var_reshaped > self.thresholds[None]).float()
            #print(z_var_reshaped.mean(axis=0))
            #latent_variances = z_var_reshaped.reshape(z_var.shape)
            #print(z_var[0,0])
            #latent_variances = (z_var>=1e-5).float()
            #print(latent_variances.mean(axis=0))
            latent_variances = torch.zeros(z_var.shape)
            #print('GOOOO')
            #for j in range(6):
            #    plt.title(j)
            #    latent_variances[:,:,j] = (z_var[:,:,j] > self.thresholds[j]).float()
            #    plt.hist(z_var[:,:,j].flatten().detach().numpy(), bins = 100, density = True)
            #    plt.vlines(self.thresholds[j].detach().numpy(),0,100, colors = 'r')
            #    plt.show()

            for j in range(latent_dim):
                quantile_j = torch.quantile(z_var[:,:,j].detach().flatten(), q = 0.5)
                quantile_j_up = torch.quantile(z_var[:,:,j].detach().flatten(), q = 0.99)
                latent_variances[:,:,j] = (z_var[:,:,j] > quantile_j).float() #* quantile_j_up
                #print(torch.max(z_var[:,:,j]),j)

        else:
            latent_variances = self.nn(z_var)
        #print(z_var.mean(axis=(0,1)), z_var.var(axis=(0,1)))
        #print(latent_variances.mean(axis=(0,1)), latent_variances.var(axis=(0,1)))

        #latent_variances += 1e-6
        


        for j in self.dims_to_train:
            for b in range(z_mu.shape[0]):  # Iterate over batch dimension

                # Compute kernel matrix and observation noise
                K, noise, var_importance, var_importance2 = self.kernel(T, T, kernel_params[b, j])

                # Construct the observed covariance matrix
                #K_obs = K + var_importance * torch.diag(z_var[b, :, j]) ** var_importance2 + noise

                K_obs = K + torch.diag(latent_variances[b,:,j]) + noise

                # Compute predictive mean
                z_star[b, :, j] = torch.matmul(K, torch.linalg.inv(K_obs)) @ z_mu[b, :, j]

                # Compute predictive variance if requested
                if return_variance:
                    # Compute correction matrix
                    eps = 1e-2
                    correction_matrix = torch.linalg.solve(K_obs + eps * torch.eye(K_obs.size(0), device=K_obs.device), K)

                    k_xx = torch.diagonal(K)  # Extract k(t_*, t_*) elements
                    sigma_star[b, :, j] = k_xx - torch.einsum('ij,ji->i', correction_matrix, K.T)

            #print(K.mean(), K.var(),j)

                    

        return (z_star, sigma_star.clip(min=1e-3)) if return_variance else z_star

    def correct_qz_x_with_gp(self, qz_x, kernel_params):

        z_mu, z_var = qz_x.mean, qz_x.variance.clamp(min=1e-5)

        kernel_params = kernel_params#.detach()

        explicit_kernel = True

        if explicit_kernel:

            z_mu_star, z_var_star = self.correct_with_gp(z_mu, z_var, kernel_params, return_variance = True)

            q_z_star_x_corrected = torch.distributions.Normal(loc=z_mu_star, scale=z_var_star)

        elif False:

            # Step 1: Predict the precision matrix over time using CNN
            precision_matrix = self.precision_nn(z_mu, z_var)  # Shape: (batch_size, latent_dim, time_steps, time_steps)

            # Step 2: Compute the covariance matrix from the precision matrix
            L = torch.linalg.cholesky(precision_matrix)  # Cholesky decomposition
            cov_matrix = torch.cholesky_inverse(L)  # Invert to get covariance

            # Step 3: Compute the corrected mean
            corrected_mean = torch.einsum('bltt,blt->blt', cov_matrix, z_mu)  # Multiply cov matrix over time

            # Step 4: Compute predictive variance (uncertainty)
            corrected_variance = torch.diagonal(cov_matrix, dim1=-2, dim2=-1)  # Extract diagonal variance over time

            q_z_star_x_corrected = torch.distributions.Normal(loc=corrected_mean, scale=corrected_variance)

        #else:

        #    q_z_star_x_corrected = self.cov_precision_nn(z_mu, z_var)

        return q_z_star_x_corrected

    def correct_qz_x_with_gp(self, qz_x, kernel_params):

        z_mu, z_var = qz_x.mean.detach(), qz_x.variance.clamp(min=1e-5).detach()

        kernel_params = kernel_params#.detach()

        z_mu_star, z_var_star = self.correct_with_gp(z_mu, z_var, kernel_params, return_variance = True)

        q_z_star_x_corrected = torch.distributions.Normal(loc=z_mu_star, scale=z_var_star)

        return q_z_star_x_corrected

    def infer(self, qz_x, x_input, return_variance=False):
        """
        Infer the reconstructed data from the latent GP.

        Parameters:
            qz_x (torch.Tensor): Latent distribution (mean & variance).
            x_input (torch.Tensor): Input data.
            return_variance (bool): Whether to return predictive variance.
            
        Returns:
            x_recon_mean (torch.Tensor): Reconstructed mean.
            x_recon_var (torch.Tensor, optional): Reconstructed variance.
        """
        z_mu, z_var = qz_x.mean, qz_x.variance

        # Update kernel parameters
        kernel_params = self.update_kernel_params(x_input)

        # Perform GP regression in latent space
        if return_variance:
            z_star, sigma_star = self.correct_with_gp(z_mu, z_var, kernel_params, return_variance=True)
        else:
            z_star = self.correct_with_gp(z_mu, z_var, kernel_params, return_variance=False)
            sigma_star = None  # Variance is not computed

        # Decode to reconstruct data
        x_recon_mean = self.decode(z_star).mean  # Reconstructed mean
        x_recon_var = sigma_star if return_variance else None  # Uncertainty in the reconstruction

        return (x_recon_mean, x_recon_var) if return_variance else x_recon_mean

    def prior_for_kernel_params(self, params):
        """
        Set a prior on the kernel parameters
        """

        length_scale = 10. ** (params[0]).clip(min = -3, max = 0)
        sigma = 10 ** (params[1]).clip(-2, 0)
        sigma = 1.
        noise = 10. ** (params[2]).clip(min = -5, max = -3)
        var_importance = 0. + torch.sigmoid((params[3]))

        loss_prior = (noise - 0).pow(2) + (var_importance - 0).pow(2)

        return loss_prior.mean()

## plotting utils

    def plot_gp_regression(self, x_input, qz_x):
        """
        Plot the Gaussian Process regression results with confidence intervals.

        Parameters:
            x_input (torch.Tensor): Original input data.
            qz_x (torch.Tensor): Latent distribution (mean & variance).
        """
        # Run inference to get mean and variance
        x_recon_mean, x_recon_var = self.infer(qz_x, x_input, return_variance=True)

        # Convert to numpy for plotting
        x_recon_mean = x_recon_mean.detach().cpu().numpy()
        x_recon_var = x_recon_var.detach().cpu().numpy()
        
        time_steps = np.arange(x_recon_mean.shape[1])  # Time axis
        
        plt.figure(figsize=(10, 5))

        for dim in range(x_recon_mean.shape[2]):  # Iterate over latent dimensions
            plt.subplot(1, x_recon_mean.shape[2], dim + 1)
            plt.plot(time_steps, x_recon_mean[0, :, dim], label="Mean Prediction", color="blue")
            plt.fill_between(time_steps, 
                            x_recon_mean[0, :, dim] - 2 * np.sqrt(x_recon_var[0, :, dim]),
                            x_recon_mean[0, :, dim] + 2 * np.sqrt(x_recon_var[0, :, dim]),
                            color="blue", alpha=0.3, label="95% Confidence Interval")
            
            plt.xlabel("Time")
            plt.ylabel(f"Latent Dimension {dim+1}")
            plt.legend()
            plt.title(f"GP Regression on Latent Dimension {dim+1}")

        plt.tight_layout()
        plt.show()

    def plot_gp_reconstruction(x, z_mu_j, gp_model_j, likelihood_j, j, training_step):
        # Set models to evaluation mode
        gp_model_j.eval()
        likelihood_j.eval()
        
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            # Ensure x is a tensor of shape [num_points]
            test_x = x.float()
            
            # Make predictions
            predictions = likelihood_j(gp_model_j(test_x))
            try:
                mean = predictions.mean[:,0]
            except:
                mean = predictions.mean
            lower, upper = predictions.confidence_region()
        
        # Plotting
        plt.figure(figsize=(8, 4))
        # Plot observed data as black stars
        plt.plot(x.numpy(), z_mu_j.detach().numpy(), 'k*', label='Observed Data')
        # Plot predictive mean as blue line
        plt.plot(test_x.numpy(), mean.numpy(), 'b', label='Predictive Mean')
        # Shade in confidence intervals
        try:
            plt.fill_between(test_x.numpy(), lower.numpy()[:,0], upper.numpy()[:,0], alpha=0.5, label='Confidence Interval')
        except:
            plt.fill_between(test_x.numpy(), lower.numpy(), upper.numpy(), alpha=0.5, label='Confidence Interval')
        plt.ylim([z_mu_j.min().item() - 1, z_mu_j.max().item() + 1])
        plt.legend()
        plt.title(f'GP Reconstruction for Latent Dimension {j} at Step {training_step}')
        plt.xlabel('Time Steps')
        plt.ylabel('Latent Variable Value')

        plt.savefig(f'latent_plots/gp_plots/dim_{j}_iter_{training_step}.png')
        plt.close()
        #plt.show()
        
        # Set models back to training mode
        gp_model_j.train()
        likelihood_j.train()

    def plot_latent_GP_regression(self, q_star, qz_x, X, n_obs = 0):

        """
        Given the corrected and initial latent distribs, plots the infered GP in the altent space
        """

        # plot reconstructions

        if True:
            X = np.array(X[n_obs].detach().cpu().numpy())
            X[X==0] = np.nan

            z_ = qz_x.rsample()
            X_recon_ = np.array(self.decode(z_).mean.detach().cpu().numpy())[n_obs]

            n_samples = 100
            z = q_star.rsample((n_samples,))
            X_recon = np.array(self.decode(z).mean.detach().cpu().numpy())[:,n_obs]

            X_min, X_max = np.min(X_recon, axis=0), np.max(X_recon, axis=0)
            X_mean = np.mean(X_recon, axis = 0)

            plt.plot(X_mean)
            plt.gca().set_prop_cycle(None)
            plt.plot(X,'o')
            plt.gca().set_prop_cycle(None)
            plt.plot(X_recon_,':')


            for j in range(X_mean.shape[-1]):
                plt.fill_between(np.arange(X.shape[0]), X_min[:,j], X_max[:,j], alpha = .3);

            y_min, y_max = np.nanmin(X) * 1.5, np.nanmax(X) * 1.5
            plt.ylim([y_min,y_max])
            plt.show()



        z_star, z_star_var, z_mu, z_var = q_star.mean, q_star.variance, qz_x.mean, qz_x.variance
        
        mu = z_star.detach()[n_obs].detach().cpu().float().numpy()
        var = (z_star_var.detach()[n_obs]).detach().cpu().float().numpy()
        z_true = (z_mu.detach()[n_obs]).detach().cpu().float().numpy()
        #z_var = (z_var.detach()[n_obs]).detach().cpu().float().numpy()
        z_var = np.array(z_var.detach()[0].detach().cpu().float().numpy())

        #plt.scatter(z_true, 'o', alpha = np.array(z_var/torch.max(z_var)))
        plt.plot(mu)
        for j in range(mu.shape[-1]):
            normalized_variance = (z_var[:,j]/z_var[:,j].max())**.5
            plt.scatter(np.arange(z_true.shape[0]), z_true[:,j], s = (1 - normalized_variance)*100 )

        #    plt.fill_between(np.arange(30),z_true[:,j] - 2*z_var[:,j], z_true[:,j] + 2*z_var[:,j], color = 'r', alpha = .05)
        plt.gca().set_prop_cycle(None)
        for j in range(mu.shape[-1]):
            plt.fill_between(np.arange(z_true.shape[0]),mu[:,j] - 2*var[:,j], mu[:,j] + 2*var[:,j], alpha = .2)
        timestamp = str(time.time()).split('.')[0]
        save_name = f'latent_scatterplot_{timestamp}.png'


        y_min, y_max = np.nanmin(z_true) * 1.5, np.nanmax(z_true) * 1.5
        plt.ylim([y_min,y_max])
        plt.show()

        #print(save_name)
        #plt.savefig(save_name)
        plt.show()

        for j in range(mu.shape[-1]):
            normalized_variance = np.array(z_var[:,j]/np.max(z_var[:,j]))**.5
            plt.hist(normalized_variance, alpha = .2, density = True, bins = 10)
        plt.show()



class VarianceNN(nn.Module):
    """
    Neural network to estimate the precision matrix over time.
    """
    def __init__(self, latent_dim, time_steps, hidden_sizes=(128, 128)):
        super().__init__()

        self.latent_dim = latent_dim
        self.time_steps = time_steps

        input_dim = latent_dim * time_steps  # Inputs: both mean and variance

        # Shared feature extraction layers
        self.fc1 = nn.Linear(input_dim, hidden_sizes[0])
        self.fc2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])

        # Output layer for precision matrix over time
        self.fc3 = nn.Linear(hidden_sizes[1], latent_dim * time_steps)  # Predict precision per latent dim

    def forward(self, z_mu, z_var):
        """
        Forward pass: estimates the precision matrix over the time dimension.
        """
        eps = 1e-5

        batch_size = z_mu.shape[0]

        # Concatenate mean and variance as input
        x = torch.cat([z_mu, z_var], dim=-1)  # Shape: (batch_size, time_steps, 2 * latent_dim)
        x = z_var.view(z_var.shape[0], -1)

        # Pass through the neural network
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        z_var_corrected = self.fc3(x)  # Shape: (batch_size, time_steps, latent_dim * time_steps)
        z_var_corrected = z_var_corrected.view(z_var.shape)

        return (z_var + z_var_corrected).clip(min=eps)
