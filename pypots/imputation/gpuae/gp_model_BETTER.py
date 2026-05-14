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

        self.precision_nn = PrecisionMatrixNN(latent_dim = self.latent_dim,
                                time_steps = 30)

        self.precision_cnn = PrecisionMatrixCNN(latent_dim = self.latent_dim,
                                time_steps = 30)

        self.cov_precision_nn = CovariancePrecisionNN(latent_dim = self.latent_dim, time_steps = 30)

        self.variance_nn = VarianceNN(latent_dim = self.latent_dim, time_steps = 30)

    def update_kernel_params(self, x):

        m = (x!=0).float().mean(axis = 1)
        #m = torch.ones(m.shape)
        #m = torch.rand(size = m.shape)
        kernel_params = self.kernel_params_estimator(m).reshape(m.shape[0], self.latent_dim, self.n_kernel_params)
        
        assert not torch.isnan(m).any(), print('m', kernel_params[:,0])
        assert not torch.isnan(kernel_params).any(), print('kernel params', kernel_params[:,0])

        return kernel_params

    def transform_params(self, raw_params):
            
        length_scale = 1e-10 + raw_params[0].clip(1e-10, 1e5)
        a, noise = raw_params[1], (raw_params[2]**2).clip(1e-3,1e-1)
        noise = torch.sigmoid(raw_params[2])
        #print(raw_params, noise)
        var_importance = 0. + torch.sigmoid((raw_params[3]))/100

        length_scale = raw_params[0]
        noise = raw_params[2]
        var_importance = raw_params[3]

        length_scale = 10. ** (raw_params[0]).clip(min = -5, max = 0)
        sigma = 10 ** (raw_params[1]).clip(-3, 0)
        noise = 10. ** (raw_params[2]).clip(min = -5, max = -1)
        var_importance = 0. + torch.sigmoid((raw_params[3]))

        #length_scale = 10. ** (raw_params[0] - 1).clip(min = -3, max = 0)
        #sigma = 10 ** (raw_params[1]).clip(-3, 0)
        #sigma = 1.
        #noise = 10. ** (raw_params[2] - 2).clip(min = -6, max = 1)
        var_importance = 10 * torch.sigmoid((raw_params[3]))
        var_importance2 = 10 * torch.sigmoid(raw_params[0])#.clip(min = , max = 1)


        # Example: ensure parameters are positive using softplus or exponentiation
        length_scale = 10. ** raw_params[0].clip(min=-3, max=0)
        sigma = 10. ** raw_params[1].clip(min=-3, max=0)
        noise = 10. ** raw_params[2].clip(min=-5, max=-1)
        var_importance = 10 * torch.sigmoid(raw_params[3])
        var_importance2 = 10 * torch.sigmoid(raw_params[4])#.clip(min = , max = 1)

        #length_scale = 10. ** ( (torch.sigmoid(raw_params[0]) - .5 - 1) * 2 )
        #sigma = 10. ** ( (torch.sigmoid(raw_params[1]) - .5 - 2) * (2 * 2) )
        #noise = 10. ** ( (torch.sigmoid(raw_params[2]) - .5 - 3) * (2 * 2) )
        var_importance = 10 * torch.sigmoid(raw_params[3])
        var_importance2 = 10 * torch.sigmoid(raw_params[4])#.clip(min = , max = 1)


        #length_scale = 1e-1
        #sigma = 1e-1
        #noise = torch.tensor(1e-2)
        var_importance = torch.tensor(1e0)
        var_importance2 = 1

        return length_scale, sigma, noise, var_importance, var_importance2

    def transform_params(self, raw_params):
        # Ensure raw_params are transformed to valid ranges.
        # For length scale and sigma, we use an exponential transformation (power-of-10)
        length_scale = 10. ** raw_params[0].clip(min=-5, max=0)
        sigma = 10. ** raw_params[1].clip(min=-3, max=0)
        # For noise, also enforce positivity using exponentiation
        noise = 10. ** raw_params[2].clip(min=-5, max=-1)
        # For var_importance, we want it in a range, say [0,10], using sigmoid scaling
        var_importance = 10 * torch.sigmoid(raw_params[3])
        # Similarly for var_importance2
        var_importance2 = 10 * torch.sigmoid(raw_params[4])

        #length_scale = 1e-1
        #sigma = 1e-1
        noise = torch.tensor(1e-2)
        var_importance = torch.tensor(1e0)
        var_importance2 = 1
        
        return length_scale, sigma, noise, var_importance, var_importance2

    def correct_with_nn_precision(self, z_mu, z_var):
        """
        Apply GP correction using a neural network-predicted precision matrix.
        """

        # Step 1: Estimate precision matrix from neural network
        precision_matrix = self.precision_nn(z_mu, z_var)  # Shape: (batch_size, latent_dim, latent_dim)

        # Step 2: Compute covariance matrix from precision matrix
        L = torch.linalg.cholesky(precision_matrix + 1e-5 * torch.eye(precision_matrix.size(-1)).to(precision_matrix.device))
        cov_tril = torch.cholesky_inverse(L)  # Invert Cholesky factor to get covariance

        # Step 3: Compute corrected mean
        corrected_mean = torch.matmul(cov_tril, z_mu.unsqueeze(-1)).squeeze(-1)

        return corrected_mean, cov_tril

    def kernel(self, X1, X2, params, kernel='cauchy'):
        """Compute the kernel function.
        
        For 'rbf', the kernel is:
            k_rbf(r) = sigma * exp(-0.5 * (r / length_scale)^2)
        For 'cauchy', the kernel is:
            k_cauchy(r) = sigma^2 / (1 + (r / length_scale)^2)
        """
        # Compute pairwise distances
        dists = torch.cdist(X1, X2, p=2)
        
        # Extract parameters
        length_scale, sigma, noise, var_importance, var_importance2 = self.transform_params(params)
        
        if kernel == 'rbf':
            K = sigma * torch.exp(-0.5 * (dists / length_scale) ** 2)
        elif kernel == 'cauchy':
            # Cauchy kernel implementation
            K = sigma ** 2 * (1 + (dists / length_scale) ** 2) ** (-1)
        else:
            K = None

        return K, noise, var_importance, var_importance2

    def kernel(self, X1, X2, raw_params, kernel='multi_scale'):
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
        # Compute pairwise distances between points in X1 and X2
        dists = torch.cdist(X1, X2, p=2)
        
        # Transform raw parameters (clipped to avoid extreme values)
        # RBF kernel parameters (for long term)
        length_scale_rbf = 10.0 ** raw_params[0].clip(min=-5, max=0)  
        sigma_rbf = 10.0 ** raw_params[1].clip(min=-3, max=0)
        
        # Cauchy kernel parameters (for short term)
        length_scale_cauchy = 10.0 ** raw_params[2].clip(min=-5, max=0)
        sigma_cauchy = 10.0 ** raw_params[3].clip(min=-3, max=0)
        
        # Noise parameter (for the overall kernel)
        #noise = 10.0 ** raw_params[4].clip(min=-5, max=-1)
        var_importance = 10.0 ** raw_params[4].clip(min=-2, max=1)

        #length_scale_rbf = length_scale_cauchy*0 + 10
        #sigma_rbf = sigma_rbf*0 + 1e-1

        noise = torch.tensor(1e-3)
        
        # Compute RBF kernel for long-term dynamics:
        K_rbf = sigma_rbf * torch.exp(-0.5 * (dists / length_scale_rbf) ** 2)
        
        # Compute Cauchy kernel for short-term dynamics:
        # Note: The Cauchy kernel here is defined as:
        #   k_cauchy(r) = sigma_cauchy^2 / (1 + (r/length_scale_cauchy)^2)
        K_cauchy = sigma_cauchy**2 * (1 + (dists / length_scale_cauchy)**2) ** (-1)
        
        # Sum the kernels
        K = K_rbf + K_cauchy
        
        return K, noise, var_importance, torch.tensor(1.)

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
                K, noise, _, _ = self.kernel(T, T, params_bj, kernel='multi_scale')
                
                # Get the latent trajectory for this sample and dimension.
                # Shape: (time_steps, 1)
                y = z_mu[b, :, j].unsqueeze(1)
                
                # Compute the log marginal likelihood for this trajectory.
                K_noise = K + torch.diag(z_var[b, :, j]) + noise * torch.eye(time_steps, device=K.device)
                
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
        
        T = torch.linspace(0, 1, z_mu.shape[1]).unsqueeze(1).to(z_mu.device)  # Time steps

        z_star = torch.zeros_like(z_mu)  # Corrected latent mean
        sigma_star = torch.zeros_like(z_var) if return_variance else None  # Corrected variance

        self.dims_to_train = np.arange(z_star.shape[2])  # Train over latent dimensions

        corrected_variance = self.variance_nn(z_mu, z_var)

        for j in self.dims_to_train:
            for b in range(z_mu.shape[0]):  # Iterate over batch dimension

                #print('a')

                # Compute kernel matrix and observation noise
                K, noise, var_importance, var_importance2 = self.kernel(T, T, kernel_params[b, j])

                assert not torch.isnan(K).any(), print(noise, var_importance, K[0])
                assert not torch.isnan(noise), print('noise', noise)
                assert not torch.isnan(var_importance), print('var', var_importance)

                #print('b')


                # Construct the observed covariance matrix
                K_obs = K + var_importance2 * torch.diag(z_var[b, :, j]) ** var_importance + noise

                #K_obs = K + torch.diag(corrected_variance[b, :, j]) + noise

                #plt.semilogy(np.array(z_var[b, :, j].detach().numpy()))
                #plt.semilogy(np.array(corrected_variance[b, :, j].detach().numpy()))
                #plt.show()

                #print('c')
                #print('d')

                # Compute predictive mean
                z_star[b, :, j] = torch.matmul(K, torch.linalg.inv(K_obs)) @ z_mu[b, :, j]

                #print('e')

                # Compute predictive variance if requested
                if return_variance:
                    # Compute correction matrix
                    correction_matrix = torch.linalg.solve(K_obs + 1e-2 * torch.eye(K_obs.size(0), device=K_obs.device), K)

                    k_xx = torch.diagonal(K)  # Extract k(t_*, t_*) elements
                    sigma_star[b, :, j] = k_xx - torch.einsum('ij,ji->i', correction_matrix, K.T)
                    

                #print('f')

        return (z_star, sigma_star.clip(min=1e-3)) if return_variance else z_star

    def correct_qz_x_with_gp(self, qz_x, kernel_params):

        z_mu, z_var = qz_x.mean, qz_x.variance.clamp(min=1e-5)

        explicit_kernel = True

        if explicit_kernel:

            z_mu_star, z_var_star = self.correct_with_gp(z_mu, z_var, kernel_params, return_variance = True)

            q_z_star_x_corrected = torch.distributions.Normal(loc=z_mu_star, scale=z_var_star)

        elif False:

            z_mu, z_var = qz_x.mean, qz_x.variance  # Extract latent mean & variance

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

        else:

            q_z_star_x_corrected = self.cov_precision_nn(z_mu, z_var)

            #z_mu_star, z_var_star = self.correct_with_gp(z_mu, z_var, kernel_params, return_variance = True)

            #q_z_star_x_corrected = torch.distributions.Normal(loc=z_mu_star, scale=z_var_star)

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

class PrecisionMatrixNN(nn.Module):
    """
    Neural network to estimate the precision matrix over time.
    """
    def __init__(self, latent_dim, time_steps, hidden_sizes=(128, 128)):
        super().__init__()

        self.latent_dim = latent_dim
        self.time_steps = time_steps

        input_dim = 2 * latent_dim  # Inputs: both mean and variance

        # Shared feature extraction layers
        self.fc1 = nn.Linear(input_dim, hidden_sizes[0])
        self.fc2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])

        # Output layer for precision matrix over time
        self.fc3 = nn.Linear(hidden_sizes[1], time_steps * time_steps * latent_dim)  # Predict precision per latent dim

    def forward(self, z_mu, z_var):
        """
        Forward pass: estimates the precision matrix over the time dimension.
        """
        batch_size = z_mu.shape[0]

        # Concatenate mean and variance as input
        x = torch.cat([z_mu, z_var], dim=-1)  # Shape: (batch_size, time_steps, 2 * latent_dim)

        # Pass through the neural network
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        precision_matrix = self.fc3(x)  # Shape: (batch_size, time_steps, latent_dim * time_steps)

        print(precision_matrix.shape, z_mu.shape)

        # Reshape output to a valid precision matrix over time
        precision_matrix = precision_matrix.view(batch_size, self.latent_dim, self.time_steps, self.time_steps)

        # Ensure positive definiteness using softplus activation
        precision_matrix = torch.exp(precision_matrix)  # Exponentiate to force positive values
        precision_matrix = precision_matrix + torch.eye(self.time_steps).to(x.device).unsqueeze(0).unsqueeze(0) * 1e-3

        return precision_matrix

import torch
import torch.nn as nn

import torch
import torch.nn as nn

class PrecisionMatrixCNN(nn.Module):
    """
    CNN-based model to estimate the precision matrix over time.
    """
    def __init__(self, latent_dim, time_steps, hidden_channels=32, kernel_size=5):
        super().__init__()

        self.latent_dim = latent_dim
        self.time_steps = time_steps

        # Convolutional feature extractor (1D Conv over time dimension)
        self.conv1 = nn.Conv1d(in_channels=2 * latent_dim, out_channels=hidden_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        self.conv2 = nn.Conv1d(in_channels=hidden_channels, out_channels=hidden_channels, kernel_size=kernel_size, padding=kernel_size // 2)

        # Output layer: Produces (time_steps * time_steps) per latent dimension
        self.conv3 = nn.Conv1d(in_channels=hidden_channels, out_channels=latent_dim * time_steps, kernel_size=1)

    def forward(self, z_mu, z_var):
        """
        Compute the precision matrix over time using CNNs.

        Args:
            z_mu: Mean of latent space (batch_size, time_steps, latent_dim)
            z_var: Variance of latent space (batch_size, time_steps, latent_dim)

        Returns:
            precision_matrix: (batch_size, latent_dim, time_steps, time_steps)
        """
        batch_size = z_mu.shape[0]

        # Concatenate mean and variance along the last dim
        x = torch.cat([z_mu, z_var], dim=-1)  # Shape: (batch_size, time_steps, 2 * latent_dim)

        # ✅ Permute to match Conv1D expectation: (batch_size, channels, time_steps)
        x = x.permute(0, 2, 1)  # Shape: (batch_size, 2 * latent_dim, time_steps)

        print(f"Input to conv1: {x.shape}")  # Debug

        # Apply CNN layers along the time dimension
        x = torch.relu(self.conv1(x))
        print(f"After conv1: {x.shape}")  # Debug
        x = torch.relu(self.conv2(x))
        print(f"After conv2: {x.shape}")  # Debug

        # Apply final conv layer
        x = self.conv3(x)  # Shape: (batch_size, time_steps * time_steps, latent_dim)

        print(f"After conv3: {x.shape}")  # Debug

        # ✅ Reshape output to (batch_size, latent_dim, time_steps, time_steps)
        precision_matrix = x.view(batch_size, self.latent_dim, self.time_steps, self.time_steps)

        # ✅ Ensure positive definiteness
        precision_matrix = torch.exp(precision_matrix)  # Exponentiate to force positive values
        precision_matrix = precision_matrix + torch.eye(self.time_steps).to(x.device).unsqueeze(0).unsqueeze(0) * 1e-3

        return precision_matrix

import torch
import torch.nn as nn

class PrecisionMatrixCNN(nn.Module):
    """
    CNN-based model to estimate the precision matrix over time.
    """
    def __init__(self, latent_dim, time_steps, hidden_channels=32, kernel_size=5):
        super().__init__()

        self.latent_dim = latent_dim
        self.time_steps = time_steps

        # Convolutional feature extractor (1D Conv over time dimension)
        self.conv1 = nn.Conv1d(in_channels=2*time_steps, out_channels=hidden_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        self.conv2 = nn.Conv1d(in_channels=hidden_channels, out_channels=hidden_channels, kernel_size=kernel_size, padding=kernel_size // 2)

        # Output layer: Produces a structured lower-triangular precision matrix
        self.conv3 = nn.Conv1d(in_channels=hidden_channels, out_channels=time_steps * (time_steps + 1) // 2, kernel_size=1)

    def forward(self, z_mu, z_var):
        """
        Compute the structured precision matrix over time using CNNs.

        Args:
            z_mu: Mean of latent space (batch_size, time_steps, latent_dim)
            z_var: Variance of latent space (batch_size, time_steps, latent_dim)

        Returns:
            precision_matrix: (batch_size, latent_dim, time_steps, time_steps)
        """
        batch_size = z_mu.shape[0]

        # Concatenate mean and variance along the last dim
        x = torch.cat([z_mu, z_var], dim=1)  # Shape: (batch_size, time_steps, 2 * latent_dim)

        # ✅ Permute to match Conv1D expectation: (batch_size, channels, time_steps)
        #x = x.permute(0, 2, 1)  # Shape: (batch_size, 2 * latent_dim, time_steps)

        print(f"Input to conv1: {x.shape}")  # Debug

        # Apply CNN layers along the time dimension
        x = torch.relu(self.conv1(x))
        print(f"After conv1: {x.shape}")  # Debug
        x = torch.relu(self.conv2(x))
        print(f"After conv2: {x.shape}")  # Debug

        # Apply final conv layer to produce a structured lower-triangular matrix
        x = F.softplus(self.conv3(x))  # Shape: (batch_size, latent_dim * time_steps * (time_steps + 1) // 2, time_steps)
        print(f"After conv3: {x.shape}")  # Debug

        # Reshape to structured lower triangular form
        tril_indices = torch.tril_indices(row=self.time_steps, col=self.time_steps, offset=0)
        precision_matrix = torch.zeros(batch_size, self.latent_dim, self.time_steps, self.time_steps, device=x.device)
        precision_matrix[:, :, tril_indices[0], tril_indices[1]] = x.view(batch_size, self.latent_dim, -1)

        # Convert to full precision matrix
        precision_matrix = precision_matrix + precision_matrix.transpose(-1, -2)

        # Ensure positive definiteness
        precision_matrix = precision_matrix + torch.eye(self.time_steps).to(x.device).unsqueeze(0).unsqueeze(0) * 1e0

        plt.imshow(np.array(precision_matrix.detach().numpy()[0,0]))
        plt.show()

        return precision_matrix


class MeanCovarianceNN(nn.Module):
    """
    Neural network to estimate the mean and covariance matrix of the latent space.
    """
    def __init__(self, input_dim, latent_dim, hidden_sizes=(128, 128)):
        super().__init__()

        self.latent_dim = latent_dim

        # Shared feature extractor
        self.fc1 = nn.Linear(input_dim, hidden_sizes[0])
        self.fc2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])

        # Output layers
        self.mean_layer = nn.Linear(hidden_sizes[1], latent_dim)  # Outputs mean
        self.cov_factor_layer = nn.Linear(hidden_sizes[1], latent_dim * (latent_dim + 1) // 2)  # Outputs covariance factors

    def forward(self, x):
        """
        Forward pass: estimates the latent mean and covariance matrix.
        """
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))

        # Compute mean
        mean = self.mean_layer(x)

        # Compute covariance factor (low-rank factorization)
        cov_factors = self.cov_factor_layer(x)
        cov_matrix = self.construct_covariance_matrix(cov_factors)

        return mean, torch.diagonal(cov_matrix, dim1 = -2, dim2 = -1)

    def construct_covariance_matrix(self, cov_factors):
        """
        Constructs a symmetric positive-definite covariance matrix from factors.
        """
        batch_size = cov_factors.shape[0]

        # Initialize a zero covariance matrix
        cov_matrix = torch.zeros(batch_size, self.latent_dim, self.latent_dim).to(cov_factors.device)

        # Fill the lower triangular part of the matrix
        tril_indices = torch.tril_indices(row=self.latent_dim, col=self.latent_dim, offset=0)
        cov_matrix[:, tril_indices[0], tril_indices[1]] = cov_factors

        # Convert to positive semi-definite covariance matrix using LL^T
        cov_matrix = torch.bmm(cov_matrix, cov_matrix.transpose(-1, -2))

        # Add a small identity term for numerical stability
        cov_matrix += 1e-3 * torch.eye(self.latent_dim).to(cov_factors.device).unsqueeze(0)

        return cov_matrix

class PrecisionMatrixNN(nn.Module):
    def __init__(self, latent_dim, time_steps, hidden_dim=128):
        """
        Args:
            latent_dim: Number of latent variables (e.g., 3)
            time_steps: Number of time steps (e.g., 30)
            hidden_dim: Optional hidden layer size for future extensions.
        """
        super(PrecisionMatrixNN, self).__init__()
        self.latent_dim = latent_dim
        self.time_steps = time_steps
        
        # The number of elements in the lower triangular matrix (including diagonal)
        self.num_tri_elements = time_steps * (time_steps + 1) // 2
        
        # Fully connected layer mapping from 2*time_steps -> num_tri_elements for each latent variable
        self.fc = nn.Linear(2 * time_steps, self.num_tri_elements)
        # (Optional: You could add additional hidden layers if needed.)

    def forward(self, z_mu, z_var):
        """
        Args:
            z_mu: Tensor of shape (batch_size, time_steps, latent_dim) representing the latent mean.
            z_var: Tensor of shape (batch_size, time_steps, latent_dim) representing the latent variance.
        
        Returns:
            precision_matrix: Tensor of shape (batch_size, latent_dim, time_steps, time_steps)
                              representing a positive definite precision matrix for each latent variable.
        """
        batch_size = z_mu.size(0)
        
        # Rearrange to (batch_size, latent_dim, time_steps)
        # Each latent variable will be processed separately.
        z_mu = z_mu.transpose(1, 2)  # (batch, latent_dim, time_steps)
        z_var = z_var.transpose(1, 2)  # (batch, latent_dim, time_steps)
        
        # Concatenate along the last dimension: (batch, latent_dim, 2*time_steps)
        x = torch.cat([z_mu, z_var], dim=-1)
        
        # Flatten the first two dimensions so that we process each latent variable independently.
        x_flat = x.reshape(batch_size * self.latent_dim, 2 * self.time_steps)
        
        # Map through the fully connected layer.
        # Output shape: (batch_size * latent_dim, num_tri_elements)
        fc_out = self.fc(x_flat)
        
        # Reshape to (batch_size, latent_dim, num_tri_elements)
        fc_out = fc_out.reshape(batch_size, self.latent_dim, self.num_tri_elements)
        
        # Create an empty tensor for the lower-triangular factor L
        # Shape: (batch_size, latent_dim, time_steps, time_steps)
        L = torch.zeros(batch_size, self.latent_dim, self.time_steps, self.time_steps, device=z_mu.device)
        
        # Get indices for the lower triangular part of a time_steps x time_steps matrix.
        tril_indices = torch.tril_indices(row=self.time_steps, col=self.time_steps, offset=0)
        
        # Fill in the lower-triangular part with the network outputs.
        # fc_out has the same number of elements as there are lower-triangular entries.
        L[:, :, tril_indices[0], tril_indices[1]] = fc_out
        
        # Enforce positivity on the diagonal by applying softplus.
        # Softplus is generally preferred over sigmoid as it provides an unbounded positive range.
        diag_indices = torch.arange(self.time_steps, device=z_mu.device)
        L[:, :, diag_indices, diag_indices] = F.softplus(L[:, :, diag_indices, diag_indices])
        
        # Compute the precision matrix as the product L * L^T.
        precision_matrix = torch.matmul(L, L.transpose(-1, -2))

        eps = 1e-3
        eye = torch.eye(precision_matrix.size(-1), device=precision_matrix.device).unsqueeze(0).unsqueeze(0)
        precision_matrix_pd = precision_matrix + eps * eye


        return precision_matrix_pd


class CovariancePrecisionNN(nn.Module):
    def __init__(self, latent_dim, time_steps, hidden_dim=128):
        """
        Args:
            latent_dim: Number of latent variables (e.g., 3).
            time_steps: Number of time steps (e.g., 30).
            hidden_dim: Hidden dimension for intermediate processing (optional).
        """
        super(CovariancePrecisionNN, self).__init__()
        self.z_size = latent_dim       # latent_dim
        self.time_steps = time_steps   # T
        
        # We will process each latent variable’s time series (concatenated mean and variance)
        # Input dimension is 2 * time_steps.
        # The mu_layer outputs a vector of length time_steps.
        # The logvar_layer outputs a vector of length 2*time_steps.
        self.mu_layer = nn.Linear(2 * time_steps, time_steps)
        self.logvar_layer = nn.Linear(2 * time_steps, 2 * time_steps)
    
    def forward(self, z_mu, z_var):
        """
        Args:
            z_mu: Tensor of shape (batch_size, time_steps, latent_dim) representing the latent mean.
            z_var: Tensor of shape (batch_size, time_steps, latent_dim) representing the latent variance.
        
        Returns:
            A torch.distributions.MultivariateNormal distribution with batch shape 
            (batch_size, latent_dim) and event shape (time_steps,).
        """
        batch_size, T, latent_dim = z_mu.shape
        assert T == self.time_steps, "Input time_steps do not match."
        assert latent_dim == self.z_size, "Input latent_dim does not match."

        # Process each latent dimension separately.
        # Transpose to get shape: (batch_size, latent_dim, time_steps)
        z_mu_t = z_mu.transpose(1, 2)    # shape: (B, latent_dim, T)
        z_var_t = z_var.transpose(1, 2)  # shape: (B, latent_dim, T)
        
        # Concatenate along the last dimension: now each latent variable gets a vector of size 2*T.
        x = torch.cat([z_mu_t, z_var_t], dim=-1)  # shape: (B, latent_dim, 2*T)
        
        # Flatten the first two dimensions so we can process each latent variable independently.
        x_flat = x.reshape(batch_size * self.z_size, 2 * T)  # shape: (B * latent_dim, 2*T)
        
        # Compute mapped mean and covariance parameters.
        # mapped_mean will become the distribution mean.
        mu_out = self.mu_layer(x_flat)  # shape: (B * latent_dim, T)
        logvar_out = self.logvar_layer(x_flat)  # shape: (B * latent_dim, 2*T)
        
        # Reshape outputs back to (batch_size, latent_dim, *).
        mapped_mean = mu_out.reshape(batch_size, self.z_size, T)  # (B, latent_dim, T)
        mapped_covar = logvar_out.reshape(batch_size, self.z_size, 2 * T)  # (B, latent_dim, 2*T)
        
        print(mapped_mean.shape, mapped_covar.shape)

        # Apply an activation to bound the covariance parameters.
        # We use sigmoid here to constrain the outputs between 0 and 1.
        mapped_covar = torch.sigmoid(mapped_covar)
        
        # We now want to construct a structured sparse matrix for each latent variable.
        # We assume that our network parameterizes a matrix with only 2*T - 1 free parameters.
        # The desired dense shape for each latent variable is (T, T).
        dense_shape = [batch_size, self.z_size, T, T]
        num_entries = 2 * T - 1  # number of free parameters per latent variable
        
        # Build index arrays.
        # For each latent variable in each batch, we need to map a flattened vector of length (2*T - 1)
        # to positions in a (T, T) matrix.
        idxs_1 = np.repeat(np.arange(batch_size), self.z_size * num_entries)
        idxs_2 = np.tile(np.repeat(np.arange(self.z_size), num_entries), batch_size)
        # These indices define the unique positions. Here we choose:
        idxs_3 = np.tile(np.concatenate([np.arange(T), np.arange(T - 1)]), batch_size * self.z_size)
        idxs_4 = np.tile(np.concatenate([np.arange(T), np.arange(1, T)]), batch_size * self.z_size)
        idxs_all = np.stack([idxs_1, idxs_2, idxs_3, idxs_4], axis=1)  # shape: (B*self.z_size*(2*T-1), 4)
        
        # We drop the last element in mapped_covar so that we have exactly (2*T-1) parameters.
        mapped_values = mapped_covar[:, :, :-1].reshape(-1)  # shape: (B*self.z_size*(2*T-1),)
        
        # Create a sparse tensor with these values.
        prec_sparse = torch.sparse_coo_tensor(
            torch.LongTensor(idxs_all).t().to(z_mu.device),
            mapped_values.to(z_mu.device),
            dense_shape,
        )
        prec_sparse = prec_sparse.coalesce()
        prec_tril = prec_sparse.to_dense()  # shape: (B, latent_dim, T, T)
        
        # Add a small identity offset for numerical stability.
        eye = torch.eye(T, device=z_mu.device).unsqueeze(0).unsqueeze(0).repeat(batch_size, self.z_size, 1, 1)
        prec_tril = prec_tril + eye
        
        # Now, solve the triangular system:
        # We treat prec_tril as a triangular matrix and solve: prec_tril @ X = I,
        # so that X = inv(prec_tril). In practice, this yields a matrix that we can treat as the Cholesky factor.
        cov_tril = torch.linalg.solve_triangular(prec_tril, eye, upper=True)
        # Replace non-finite entries (if any) with zeros.
        cov_tril = torch.where(torch.isfinite(cov_tril), cov_tril, torch.zeros_like(cov_tril))
        
        #print(mapped_mean.shape, cov_tril.shape)


        # Transpose to get the lower-triangular Cholesky factor.
        cov_tril_lower = cov_tril.transpose(-1, -2)

        # mapped_mean is currently (batch, latent_dim, time_steps)
        mapped_mean = mapped_mean.transpose(1, 2)  # Now (batch, time_steps, latent_dim)

        # cov_tril_lower is currently (batch, latent_dim, time_steps, time_steps)
        cov_tril_lower = cov_tril_lower.permute(0, 2, 3, 1)  # Now (batch, time_steps
                
        cov_tril_lower += eye.transpose(1,3)

        for i in range(30):
            plt.plot(np.array(torch.diag(cov_tril_lower[0][i]).detach().numpy()))
        plt.show()

        # Create a multivariate normal distribution.
        # The batch shape will be (batch_size, latent_dim) and the event shape is (time_steps,)
        z_dist = torch.distributions.MultivariateNormal(loc=mapped_mean, scale_tril=cov_tril_lower)
        

        return z_dist

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
