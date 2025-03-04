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
        self.assemble_data_train = assemble_data_train
        self.assemble_data_val = assemble_data_val

        # get encoder
        self.encode = model.encoder.eval()
        self.decode = model.decoder.eval()
        self.plot_while_training = False

        # init optimizer
        self.optimizer = Adam(lr=1e-3)
        self.optimizer.init_optimizer(self.kernel_params_estimator.parameters())

        self.p = .3

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

            length_scale = 10. ** (params[0]).clip(min = -5, max = 1)
            sigma = 10 ** (params[1]).clip(-3, 0)
            noise = 10. ** (params[2]).clip(min = -5, max = -1)
            var_importance = 0. + torch.sigmoid((params[3]))

            length_scale = 10. ** (params[0]).clip(min = -5, max = 1)
            sigma = 10 ** (params[1]).clip(-5, 1)
            noise = 10. ** (params[2]).clip(min = -5, max = -1)
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

        return (z_star, sigma_star.clip(min=1e-3)) if return_variance else z_star

    def correct_qz_x_with_gp(self, qz_x, kernel_params):

        z_mu, z_var = qz_x.mean, qz_x.variance

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

    def kernel_(self, X1, X2, params, kernel='rbf'):
        dists = torch.cdist(X1, X2, p=2)  # Pairwise distances

        if kernel == 'rbf':
            # Batch parameters
            length_scale = 10. ** params[:, 0].clamp(min=-3, max=-1)
            sigma = 10. ** params[:, 1].clamp(min=-1, max=0)
            sigma = 1.
            noise = 10. ** params[:, 2].clamp(min=-5, max=-3)
            var_importance = torch.ones(params.size(0), device=params.device)  # Assuming it's constant for all batches

            # Kernel computation
            K = sigma[:, None, None] * torch.exp(-0.5 * (dists[None, :, :] / length_scale[:, None, None]) ** 2)

            # Broadcasting noise and importance
            noise = noise[:, None, None] * torch.eye(X1.size(0), device=X1.device).unsqueeze(0)
        else:
            K, noise, var_importance = None, None, None

        return K, noise, var_importance

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

    def fit_kernel(self, training_loader, val_loader=None, training_iter=20):
        """Train the model using training_loader and validate using val_loader."""
        
        for epoch in range(training_iter):
            total_train_loss = 0.0
            num_train_samples = 0

            # Training Loop
            for training_step, data in tqdm(enumerate(training_loader), desc=f"Epoch {epoch+1}/{training_iter}"):
                self.optimizer.zero_grad()

                inputs = self.assemble_data_train(data)
                x_input = torch.clone(inputs['X'])

                x_corr = mcar(x_input, p=self.p)
                x_corr[x_corr != x_corr] = 0.  # Replace NaNs with 0

                # Encode the data
                qz_x = self.encode(x_corr)
                z_mu, z_var = qz_x.mean.detach(), qz_x.variance.detach()

                # Update kernel parameters
                kernel_params = self.update_kernel_params(x_corr)

                if training_step % 10 == 0:
                    self.kernel_params_history['a'].append(kernel_params[0,0,0].detach().item())
                    self.kernel_params_history['b'].append(kernel_params[0,1,0].detach().item())
                    self.kernel_params_history['c'].append(kernel_params[0,2,0].detach().item())
                    self.kernel_params_history['d'].append(kernel_params[0,3,0].detach().item())

                # Correct with GP
                z_star, z_star_var = self.correct_with_gp(z_mu, z_var, kernel_params, return_variance=True)
                z_star_var = z_star_var.clamp(min=1e-3)#**.5  # Avoid NaN variance

                q_z_star_x_corr = torch.distributions.Normal(loc=z_star, scale=z_star_var)
                
                n_samples = 10
                
                z_samples = q_z_star_x_corr.rsample((n_samples,))
                # Sample from Gaussian: z ~ N(z_star, z_star_var)
                #z_samples = torch.randn((n_samples,) + z_star.shape) * torch.sqrt(z_star_var) + z_star

                # Reconstruct and compute reconstruction error
                x_recon = self.decode(z_samples).mean
                x_recon += torch.rand(x_recon.shape) * 1e-2  # Add small noise
                l2_error = (x_recon - x_input[None]).pow(2)[(x_input != 0).unsqueeze(0).repeat(n_samples,1,1,1)].clip(max=10)

                z = self.encode(x_input).mean
                latent_log_prob = q_z_star_x_corr.log_prob(z).clip(min=1e-3,max=10)#.mean(axis=2).mean()

                # Compute loss
                loss = l2_error.mean()
                #loss += torch.exp(-latent_log_prob).mean()
                alpha = .1
                loss += self.prior_for_kernel_params(kernel_params) * alpha  # Regularization

                assert not torch.isnan(loss), print(x_recon)

                # Backpropagation
                loss.clip(min=0,max=10).backward()
                self.optimizer.step()

                # Gradient Clipping
                with torch.no_grad():
                    for param in self.kernel_params_estimator.parameters():
                        param.clamp_(-3, 3)

                # Training loss accumulation
                total_train_loss += loss.item() * x_input.size(0)
                num_train_samples += x_input.size(0)

            # Compute average training loss
            avg_train_loss = total_train_loss / num_train_samples

            # Validation Step (if val_loader is provided)
            avg_val_loss = None
            if val_loader is not None:
                self.kernel_params_estimator.eval()  # Set to evaluation mode
                total_val_loss = 0.0
                num_val_samples = 0

                with torch.no_grad():  # No gradient updates for validation
                    for val_data in val_loader:
                        val_inputs = self.assemble_data_val(val_data)
                        x_val = torch.clone(val_inputs['X'])
                        x_val_corr = mcar(x_val, p=self.p)
                        x_val_corr[x_val_corr != x_val_corr] = 0.

                        # Encode validation data
                        qz_val = self.encode(x_val_corr)
                        z_mu_val, z_var_val = qz_val.mean.detach(), qz_val.variance.detach()

                        # Update kernel parameters
                        kernel_params_val = self.update_kernel_params(x_val_corr)

                        # Correct with GP
                        z_star_val, z_star_var_val = self.correct_with_gp(z_mu_val, z_var_val, kernel_params_val, return_variance=True)
                        z_star_var_val = z_star_var_val.clamp(min=0.)**.5

                        z_star_var_val = z_star_var_val.clamp(min=1e-3)#**.5  # Avoid NaN variance

                        q_z_star_x_corr = torch.distributions.Normal(loc=z_star_val, scale=z_star_var_val)
                        
                        n_samples = 10
                        
                        z_samples = q_z_star_x_corr.rsample((n_samples,))
                        # Sample from Gaussian: z ~ N(z_star, z_star_var)
                        #z_samples = torch.randn((n_samples,) + z_star.shape) * torch.sqrt(z_star_var) + z_star

                        # Reconstruct and compute reconstruction error
                        x_recon_val = self.decode(z_samples).mean.mean(axis=0)
                        x_recon_val += torch.rand(x_recon_val.shape) * 1e-2  # Add small noise
                        val_l2_error = (x_recon_val - x_val).pow(2)[(x_val != 0)].clip(max=10)

                        z = self.encode(x_val).mean
                        latent_log_prob = q_z_star_x_corr.log_prob(z).mean(axis=2).mean()

                        # Compute loss
                        val_loss = val_l2_error.mean()
                        #val_loss -= latent_log_prob

                        # Accumulate validation loss
                        total_val_loss += val_loss.item() * x_val.size(0)
                        num_val_samples += x_val.size(0)

                # Compute average validation loss
                avg_val_loss = total_val_loss / num_val_samples

                self.kernel_params_estimator.train()  # Set back to training mode

            # Print Training and Validation Loss
            if val_loader:
                print(f"Epoch {epoch+1}/{training_iter}: Training Loss = {avg_train_loss:.6f}, Validation Loss = {avg_val_loss:.6f}")
            else:
                print(f"Epoch {epoch+1}/{training_iter}: Training Loss = {avg_train_loss:.6f}")

            if self.plot_while_training:
                mu = z_star.detach()[0].numpy()
                var = z_star_var.detach()[0].numpy()
                z_true = z_mu.detach()[0].numpy()
                z_var = z_var.detach()[0].numpy()
                #plt.scatter(z_true, 'o', alpha = np.array(z_var/torch.max(z_var)))
                plt.plot(mu)
                for j in range(mu.shape[-1]):
                    normalized_variance = np.array(z_var[:,j]/np.max(z_var[:,j]))**.5
                    plt.scatter(np.arange(z_true.shape[0]), z_true[:,j], s = (1 - normalized_variance)*100 )

                #    plt.fill_between(np.arange(30),z_true[:,j] - 2*z_var[:,j], z_true[:,j] + 2*z_var[:,j], color = 'r', alpha = .05)
                plt.gca().set_prop_cycle(None)
                for j in range(mu.shape[-1]):
                    plt.fill_between(np.arange(z_true.shape[0]),mu[:,j] - 2*var[:,j], mu[:,j] + 2*var[:,j], alpha = .2)
                timestamp = str(time.time()).split('.')[0]
                save_name = f'latent_scatterplot_{timestamp}.png'
                #print(save_name)
                #plt.savefig(save_name)
                plt.show()

                for j in range(mu.shape[-1]):
                    normalized_variance = np.array(z_var[:,j]/np.max(z_var[:,j]))**.5
                    plt.hist(normalized_variance, alpha = .2, density = True, bins = 10)
                plt.show()

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