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

    def __init__(self, model, assemble_data, n_dims, latent_size):
        n_kernel_params = 4
        #latent_size = 20

        self.kernel_params_estimator = make_nn(input_size = n_dims, 
                                                output_size = latent_size * n_kernel_params,
                                                hidden_sizes = (n_dims * 10, ))
        self.dims_to_train = np.arange(latent_size)
        self.n_dims = n_dims
        self.latent_dim = latent_size
        self.n_kernel_params = n_kernel_params

        # data loader
        self.assemble_data = assemble_data

        # get encoder
        self.encode = model.encoder.eval()
        self.decode = model.decoder.eval()

        # init optimizer
        self.optimizer = Adam()
        self.optimizer.init_optimizer(self.kernel_params_estimator.parameters())

        self.p = .1

        self.kernel_params_history = {'a':[],'b':[],'c':[],'d':[]}

    def kernel(self, X1, X2, params, kernel = 'rbf'):
        """Compute the RBF kernel (Gaussian kernel)."""
        dists = torch.cdist(X1, X2, p=2)  # Pairwise distances

        if kernel == 'rbf':
            length_scale = 1e-10 + params[0].clip(1e-10, 1e5)
            a, noise = params[1], (params[2]**2).clip(1e-3,1e-1)
            noise = torch.sigmoid(params[2])
            #print(params, noise)
            var_importance = 0. + torch.sigmoid((params[3]))/100

            length_scale = params[0]
            noise = params[2]
            var_importance = params[3]

            length_scale = 10. ** (params[0]).clip(min = -3, max = 0)
            sigma = 10 ** (params[1]).clip(-2, 0)
            noise = 10. ** (params[2]).clip(min = -5, max = -3)
            var_importance = 0. + torch.sigmoid((params[3]))

            K = sigma * torch.exp(-0.5 * ((dists) / length_scale) ** 2)

            #print(length_scale)

        else:
            K = None

        #var_importance = 1
            
        return K, noise, var_importance

    def kernel_one_point(self, t_dist, params, kernel = 'rbf'):
        """
        Evaluate the kernel at one point in time
        """

        length_scale = 10. ** (params[0]).clip(min = -3, max = 0)
        sigma = 10 ** (params[1]).clip(-1, 0)
        return sigma * torch.exp(-0.5 * ((t_dist) / length_scale) ** 2)

    def update_kernel_params(self, x):

        m = (x!=0).float().mean(axis = 1)
        m = torch.ones(m.shape)
        m = torch.rand(size = m.shape)
        kernel_params = self.kernel_params_estimator(m).reshape(m.shape[0], self.latent_dim, self.n_kernel_params)
        
        assert not torch.isnan(m).any(), print('m', kernel_params[:,0])
        assert not torch.isnan(kernel_params).any(), print('kernel params', kernel_params[:,0])

        return kernel_params

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

        for j in self.dims_to_train:
            for b in range(z_mu.shape[0]):  # Iterate over batch dimension

                # Compute kernel matrix and observation noise
                K, noise, var_importance = self.kernel(T, T, kernel_params[b, j])

                assert not torch.isnan(K).any(), print(noise, var_importance, K[0])
                assert not torch.isnan(noise), print('noise', noise)
                assert not torch.isnan(var_importance), print('var', var_importance)

                # Construct the observed covariance matrix
                K_obs = K + torch.diag(z_var[b, :, j]) * var_importance + noise

                # Compute correction matrix
                correction_matrix = torch.linalg.solve(K_obs + 1e-5 * torch.eye(K_obs.size(0), device=K_obs.device), K)

                # Compute predictive mean
                z_star[b, :, j] = torch.matmul(K, torch.linalg.inv(K_obs)) @ z_mu[b, :, j]

                # Compute predictive variance if requested
                if return_variance:
                    k_xx = torch.diagonal(K)  # Extract k(t_*, t_*) elements
                    sigma_star[b, :, j] = k_xx - torch.einsum('ij,ji->i', correction_matrix, K.T)

        return (z_star, sigma_star) if return_variance else z_star

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


        def kernel(self, X1, X2, params, kernel='rbf'):
            dists = torch.cdist(X1, X2, p=2)  # Pairwise distances

            if kernel == 'rbf':
                # Batch parameters
                length_scale = 10. ** params[:, 0].clamp(min=-3, max=-1)
                sigma = 10. ** params[:, 1].clamp(min=-1, max=0)
                noise = 10. ** params[:, 2].clamp(min=-5, max=-3)
                var_importance = torch.ones(params.size(0), device=params.device)  # Assuming it's constant for all batches

                # Kernel computation
                K = sigma[:, None, None] * torch.exp(-0.5 * (dists[None, :, :] / length_scale[:, None, None]) ** 2)

                # Broadcasting noise and importance
                noise = noise[:, None, None] * torch.eye(X1.size(0), device=X1.device).unsqueeze(0)
            else:
                K, noise, var_importance = None, None, None

            return K, noise, var_importance

        def correct_with_gp(self, z_mu, z_var, kernel_params):
            T = torch.linspace(0, 1, z_mu.shape[1]).unsqueeze(1).to(z_mu.device)  # Ensure same device
            z_star = torch.zeros_like(z_mu)

            self.dims_to_train = np.arange(z_mu.shape[2])

            for j in self.dims_to_train:
                # Batch-wise computation for all elements in the batch
                K, noise, var_importance = self.kernel(T, T, kernel_params[:, j])

                # Adding batch dimension to T and kernel parameters for broadcasting
                K_obs = K + torch.diag_embed(z_var[:, :, j]) * var_importance[:, None, None] + noise

                #print(var_importance.shape, noise.shape)

                # Batched solve and correction
                correction_matrices = torch.linalg.solve(K_obs + 1e-5 * torch.eye(K_obs.size(-1), device=K_obs.device), K)

                # Batched matrix multiplication
                z_star[:, :, j] = torch.einsum('bij,bi->bj', correction_matrices, z_mu[:, :, j])

            return z_star

        def fit_kernel_(self, training_loader, training_iter=20):
            for i in range(training_iter):
                total_loss = 0.0
                num_samples = 0

                for training_step, data in tqdm(enumerate(training_loader), desc=f"Epoch {i+1}/{training_iter}"):
                    self.optimizer.zero_grad()

                    inputs = self.assemble_data(data)
                    x_input = inputs['X']

                    x_corr = mcar(x_input, p=self.p)

                    # Encode the data
                    qz_x = self.encode(x_input)
                    z_mu, z_var = qz_x.mean.detach(), qz_x.variance.detach()

                    # Update kernel_params
                    kernel_params = self.update_kernel_params(x_input)

                    if training_step%5==0:
                        self.kernel_params_history['a'].append(kernel_params[0,0,0].detach().item())
                        self.kernel_params_history['b'].append(kernel_params[0,1,0].detach().item())
                        self.kernel_params_history['c'].append(kernel_params[0,2,0].detach().item())
                        self.kernel_params_history['d'].append(kernel_params[0,3,0].detach().item())

                    # Correct with GP
                    z_star = self.correct_with_gp(z_mu, z_var, kernel_params)

                    # Reconstruct and compute reconstruction error
                    x_recon = self.decode(z_star).mean
                    x_recon += torch.rand(x_recon.shape) * 1e-2 # add a bit of noise
                    l2_error = (x_recon - x_input).pow(2)[(x_input != 0)].clip(max = 1)

                    l2_error += (x_recon - x_input).pow(2)[(x_corr != 0)].clip(max = 1).mean()

                    # Backprop on loss
                    loss = l2_error.mean()
                    loss.backward()
                    self.optimizer.step()

                    # Accumulate loss for reporting
                    total_loss += loss.item() * x_input.size(0)  # Multiply by batch size
                    num_samples += x_input.size(0)

                # Compute and print average loss for the epoch
                avg_loss = total_loss / num_samples
                print(f"Epoch {i+1}/{training_iter}: Average Loss = {avg_loss:.6f}")

        def correct_with_gp_(self, z_mu, z_var, kernel_params):

            T = torch.linspace(0,1,z_mu.shape[1]).unsqueeze(1)

            z_star = torch.zeros(z_mu.shape)

            self.dims_to_train = np.arange(z_star.shape[2])

            for j in self.dims_to_train:

                for b in range(z_mu.shape[0]): #over batch dim

                    # Compute kernel matrix and observation noise
                    #K, noise, var_importance = self.kernel(T, T, kernel_params[b, j])

                    # Construct the observed covariance matrix
                    #K_obs = K + torch.diag(z_var[b, :, j]) * var_importance + noise

                    # Inspect correction matrix
                    #correction_matrix = torch.linalg.solve(K_obs + 1e-5 * torch.eye(K_obs.size(0), device=K_obs.device), K)

                    # Compare z_star and z_mu
                    #z_star[b, :, j] = correction_matrix.T @ z_mu[b, :, j]

                    K, noise, var_importance = self.kernel(T, T, kernel_params[b,j])

                    K_obs = K + torch.diag(z_var[b,:,j]) * var_importance + noise
                    
                    z_star[b,:,j] = torch.matmul(K ,torch.linalg.inv(K_obs)) @ z_mu[b,:,j]


                    if False:   

                        # Visualize the matrices for debugging
                        plt.figure(figsize=(6, 6))
                        plt.title("Observed Kernel (K)")
                        plt.imshow(K.detach().cpu().numpy(), cmap='viridis', aspect='auto')
                        plt.colorbar()
                        plt.show()

                        # Visualize the matrices for debugging
                        plt.figure(figsize=(6, 6))
                        plt.title("Observed Kernel (K_obs)")
                        plt.imshow(K_obs.detach().cpu().numpy(), cmap='viridis', aspect='auto')
                        plt.colorbar()
                        plt.show()

                        plt.figure(figsize=(6, 6))
                        plt.title("Correction Matrix (K @ inv(K_obs))")
                        plt.imshow((torch.eye(K_obs.size(0), device=K_obs.device) - correction_matrix).detach().cpu().numpy(), cmap='viridis', aspect='auto')
                        plt.colorbar()
                        plt.show()


                        # Update z_star
                        plt.plot((correction_matrix@z_mu[b, :, j]).detach())
                        plt.plot(z_mu[b, :, j].detach(),'o')
                        plt.show()

                    #z_star[b, :, j] = torch.matmul(correction_matrix, z_mu[b, :, j])


            return z_star

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

    def fit_kernel(self, training_loader, training_iter=20):
        for i in range(training_iter):
            total_loss = 0.0
            num_samples = 0

            for training_step, data in tqdm(enumerate(training_loader), desc=f"Epoch {i+1}/{training_iter}"):
                self.optimizer.zero_grad()

                inputs = self.assemble_data(data)
                x_input = torch.clone(inputs['X'])

                x_corr = mcar(x_input, p=self.p)
                x_corr[x_corr!=x_corr] = 0.

                # Encode the data
                qz_x = self.encode(x_corr)
                z_mu, z_var = qz_x.mean.detach(), qz_x.variance.detach()

                # Update kernel_params
                kernel_params = self.update_kernel_params(x_corr)

                if training_step%5==0:
                    self.kernel_params_history['a'].append(kernel_params[0,0,0].detach().item())
                    self.kernel_params_history['b'].append(kernel_params[0,1,0].detach().item())
                    self.kernel_params_history['c'].append(kernel_params[0,2,0].detach().item())
                    self.kernel_params_history['d'].append(kernel_params[0,3,0].detach().item())

                # Correct with GP
                z_star, z_star_var = self.correct_with_gp(z_mu, z_var, kernel_params, return_variance=True)

                z_star_var = z_star_var.clamp(min=0.)**.5

                n_samples = 10
   
                # Sample from Gaussian: z ~ N(z_star, z_star_var)
                z_samples = torch.randn((n_samples,) + z_star.shape) * torch.sqrt(z_star_var) + z_star

                # Reconstruct and compute reconstruction error
                x_recon = self.decode(z_samples).mean.mean(axis=0)
                x_recon += torch.rand(x_recon.shape) * 1e-2 # add a bit of noise
                l2_error = (x_recon - x_input).pow(2)[(x_input != 0)].clip(max = 10)

                #l2_error += (x_recon - x_input).pow(2)[(x_corr != 0)].clip(max = 10).mean()

                # Backprop on loss
                loss = l2_error.mean()
                alpha = .1
                loss = loss + self.prior_for_kernel_params(kernel_params) * alpha
                #plt.plot(l2_error.detach())
                #plt.show()
                loss.backward()
                self.optimizer.step()

                with torch.no_grad():
                    for param in self.kernel_params_estimator.parameters():
                        param.clamp_(-3, 3)

                if training_step%50==0:
                    mu = z_star.detach()[0]
                    var = z_star_var.detach()[0]
                    z_true = z_mu.detach()[0]
                    z_var = z_var.detach()[0]
                    plt.plot(z_true,'o')
                    plt.plot(mu)
                    for j in range(mu.shape[-1]):
                        plt.fill_between(np.arange(30),z_true[:,j] - 2*z_var[:,j], z_true[:,j] + 2*z_var[:,j], color = 'r', alpha = .05)
                    plt.gca().set_prop_cycle(None)
                    for j in range(mu.shape[-1]):
                        plt.fill_between(np.arange(30),mu[:,j] - 2*var[:,j], mu[:,j] + 2*var[:,j], alpha = .2)
                    plt.show()

                # Accumulate loss for reporting
                total_loss += loss.item() * x_input.size(0)  # Multiply by batch size
                num_samples += x_input.size(0)

            # Compute and print average loss for the epoch
            avg_loss = total_loss / num_samples
            print(f"Epoch {i+1}/{training_iter}: Average Loss = {avg_loss:.6f}")

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