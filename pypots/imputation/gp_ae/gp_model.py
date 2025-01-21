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

class ProbabilisticGP:
    """
    A sub-module of a VAE that corrects the latent time-series via the Probabilistic GP regression scheme
    """
    def __init__(self, AEmodel, assemble_data):
        self.encoder = AEmodel.encoder
        self.enforce_variance_bias = False
        self.latent_size = AEmodel.latent_dim
        self.gp_models = []
        self.likelihoods = []
        self.mll = []
        self.optimizer = []
        self.assemble_data = assemble_data

    def instantiate_gp_models(self, training_loader):
        self.gp_models = []
        self.likelihoods = []
        self.mll = []
        self.optimizer = []

        for j in range(self.latent_size):
            # Define inducing points for the sparse GP model (batch size 8 assumed here)
            #n_inducing_pts = 48
            #inducing_points = torch.linspace(0, n_inducing_pts - 1, n_inducing_pts).reshape(1, -1).repeat(8, 1)
            
            # Instantiate model and likelihood
            #gp_model = SparseGPModel(inducing_points=inducing_points)
            
            # Ensure noise aligns with the batch and time steps: noise [8, 48] for batched FixedNoiseGaussianLikelihood
            #likelihood = gpytorch.likelihoods.FixedNoiseGaussianLikelihood(
            #    noise=torch.ones(8, 48),  # Adjust to batch_size x num_time_steps
            #    noise_constraint=gpytorch.constraints.GreaterThan(1e-9)
            #)

            batch_training = False
            if batch_training:

                likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=8)

                n = 48
                train_x = torch.linspace(0,n-1,n).detach()
                train_y = train_x.reshape(1,n).repeat(8,1).detach()

                gp_model = BatchIndependentMultitaskGPModel(train_x, train_y, likelihood)

            else:
                    
                likelihood = gpytorch.likelihoods.FixedNoiseGaussianLikelihood(
                    noise=torch.ones(48)*.1,  # Adjust to batch_size x num_time_steps
                    noise_constraint=gpytorch.constraints.GreaterThan(1e-9))

                n = 48
                train_x = torch.linspace(0,n-1,n).detach()
                train_y = train_x.detach()

                gp_model = ExactGPModel(train_x, train_y, likelihood)


            if torch.cuda.is_available():
                gp_model = gp_model.cuda()
                likelihood = likelihood.cuda()

            gp_model.train()
            likelihood.train()

            # Define optimizer and marginal log-likelihood
            optimizer = torch.optim.Adam(gp_model.parameters(), lr=0.1)

            # Define MLL for the Exact Marginal Log Likelihood
            mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, gp_model)

            # Append model and likelihood to lists
            self.gp_models.append(gp_model)
            self.likelihoods.append(likelihood)
            
            # Append optimizer and mll
            self.optimizer.append(optimizer)
            self.mll.append(mll)

# Main function using train_on_batch
    def fit_kernel(self, training_loader, training_iter=50, batch_iter=1):
        # Instantiate parameters
        self.instantiate_gp_models(training_loader)
        self.kernel_params = {j: [] for j in range(self.latent_size)}
        self.dims_to_train = list(np.arange(self.latent_size))

        for i in range(training_iter):

            while len(self.dims_to_train) > 0:

                for training_step, data in enumerate(training_loader):


                    inputs = self.assemble_data(data)
                    x_input = inputs['X']

                    # Encode the data
                    qz_x = self.encoder(x_input)
                    z_mu, z_var = qz_x.mean.detach(), qz_x.variance.detach()

                    # Train on the batch for batch_iter iterations
                    train_on_batch(
                        model=self,
                        gp_models=self.gp_models,
                        mlls=self.mll,
                        optimizers=self.optimizer,
                        batch_x=x_input,
                        z_mu=z_mu,
                        z_var=z_var,
                        n_iter=batch_iter
                    )

                    # Optional: Plotting kernel parameters
                    n_plot = 30
                    if training_step % n_plot == n_plot - 1:
                        for j in range(self.latent_size):

                            # Plot GP reconstruction for the first sample
                            plot_gp_reconstruction(
                                torch.arange(z_mu.size(1)).float(),
                                z_mu[0, :, j],
                                self.gp_models[j],
                                self.likelihoods[j],
                                j,
                                training_step
                            )

                        plt.figure()

                        for j in range(self.latent_size):
                            plt.semilogy(self.kernel_params[j])

                        plt.title(f'Kernel Parameter Progression for Dimension {j}')
                        plt.xlabel('Iteration')
                        plt.ylabel('Lengthscale')

                    
                        plt.savefig('latent_plots/gp_plots/kernels_params.png')
                        plt.close()
                        #plt.show()

            # Optionally print progress after each training iteration
            try:
                print(f"RBF Lengthscale: {self.gp_models[0].covar_module.base_kernel.kernels[0].lengthscale.item():.3f}")
                print(f"Periodic Lengthscale: {self.gp_models[0].covar_module.base_kernel.kernels[1].lengthscale.item():.3f}")
                print(f"Linear variance: {self.gp_models[0].covar_module.base_kernel.kernels[2].lengthscale.item():.3f}")
                print(f"Noise: {self.likelihoods[0].noise.item():.3f}")
            except:
                continue

    def select_values_for_GP_inference(self, z_mu, z_var, q = .8):
        """
        Returns a set of values above a certain quantile
        """
        x = torch.arange(len(z_mu))
        quantile = torch.quantile(z_var, q = q)
        mask = z_var > quantile
        return x[mask], z_mu[mask]

    def batch_select_values_for_GP_inference(self, z_mu, z_var, q = .8):
        """
        Returns a set of values above a certain quantile
        """

        k = 8
        X, Z = [], []
        x = torch.arange(z_mu.shape[1])

        for i in range(z_mu.shape[0]):
            # Sort the indices of z_var[i] in descending order and get top `k` indices
            top_k_indices = torch.topk(z_var[i], k, largest=True).indices

            # Select values of `x` and `z_mu` corresponding to these indices
            X.append(x[top_k_indices])
            Z.append(z_mu[i][top_k_indices])

        # Stack results to ensure same shape
        X_stacked = torch.stack(X)
        Z_stacked = torch.stack(Z)

        #print(X_stacked.shape, Z_stacked.shape)
        
        return X_stacked, Z_stacked

def train_on_batch(model, gp_models, mlls, optimizers, batch_x, z_mu, z_var, n_iter):
    """
    Train the GP models on a single batch for multiple iterations.
    
    Parameters:
    - model: Main model object (self).
    - gp_models: List of GP models for each latent dimension.
    - mlls: List of Marginal Log Likelihood objects for each latent dimension.
    - optimizers: List of optimizers for each latent dimension.
    - batch_x: Input data for the batch.
    - z_mu: Mean of the latent encoding for the batch.
    - z_var: Variance of the latent encoding for the batch.
    - dims_to_train: List of dimensions (indices) currently being trained.
    - n_iter: Number of iterations to run training for each batch.
    """
    batch_size, num_points, latent_size = z_mu.size()
    
    #for _ in range(n_iter):
    # Loop over each dimension to train
    for j in model.dims_to_train:
        cumulative_loss = 0.0  # Reset cumulative loss for dimension j

        # Loop over each sample in the batch
        for b in range(batch_size):

            for _ in range(n_iter):

                # Create time step positions for sample `b`
                x_positions = torch.arange(num_points).float().detach()
                
                # Extract z_mu and z_var for sample `b` and dimension `j`
                z_mu_b_j = z_mu[b, :, j].detach()
                z_var_b_j = z_var[b, :, j].detach()

                model.likelihoods[j].noise = z_var_b_j.detach() / (z_var_b_j.mean().detach() * 100)
                
                # Update GP model’s training data for dimension `j`
                gp_models[j].set_train_data(inputs=x_positions, targets=z_mu_b_j, strict=False)
                
                # Compute GP output and loss for dimension `j` and sample `b`
                output = gp_models[j](x_positions)
                loss = -mlls[j](output, z_mu_b_j)
                cumulative_loss += loss
            
            # Backward pass and optimizer step for dimension `j`
            cumulative_loss.backward(retain_graph=True)
            optimizers[j].step()
            optimizers[j].zero_grad()
            
            # Update kernel parameters
            model.kernel_params[j].append(
                gp_models[j].covar_module.base_kernel.lengthscale.item()
            )

            # Early stopping for dimension `j`
            criterion = np.std(model.kernel_params[j][-20:]) < 1e-4
            if (
                len(model.kernel_params[j]) > 30
                and criterion
            ):
                model.dims_to_train.remove(j)
                print(f'No further training for dim {j}')

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

from tqdm import tqdm
from pygrinder import mcar
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
            var_importance = 0. #+ torch.sigmoid((params[3]))/100

            length_scale = params[0]
            noise = params[2]
            var_importance = params[3]

            length_scale = 10. ** (params[0]).clip(min = -3, max = 0)
            sigma = 10 ** (params[1]).clip(-2, 0)
            noise = 10. ** (params[2]).clip(min = -5, max = -3)
            #noise = .01
            #var_importance = 10. ** (params[3]).clip(-1, 2)
            var_importance = 1

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
        
        return kernel_params

    def fit_kernel(self, training_loader, training_iter=20):
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

    def correct_with_gp(self, z_mu, z_var, kernel_params):

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

    def infer(self, qz_x, x_input):

        z_mu, z_var = qz_x.mean, qz_x.variance

        # Update kernel_params
        kernel_params = self.update_kernel_params(x_input)

        # Correct with GP
        z_star = self.correct_with_gp(z_mu, z_var, kernel_params)

        # Reconstruct and compute reconstruction error
        x_recon = self.decode(z_star).mean

        return x_recon

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




class ProbabilisticGP_:
    """
    A sub-module of a VAE that corrects the latent time-seties via the Probabilistic GP regression scheme
    """
    def __init__(self, encoder):
        self.encoder = encoder
        self.kernel = None
        self.enforce_variance_bias = False


    def correct_embedding_with_gp(self, z_mu, z_sigma):
        """
        Correct the latent embeddings using Gaussian Process regression.

        Args:
            z_mu (Tensor): Mean of the latent variables.
            z_sigma (Tensor): Standard deviation of the latent variables.

        Returns:
            Tensor: Corrected latent variables.
        """
        if self.kernel is None:
            self.fit_gp_kernels(z_mu.detach(), z_sigma.detach())

        n_samples, n_dims = z_mu.shape
        z_corrected = torch.zeros_like(z_mu)

        x = torch.arange(n_samples).reshape(-1, 1)
        v = z_sigma.clone()

        # Enforce variance bias adjustment
        if self.enforce_variance_bias:
            v = self.adjust_variances(v)

        a,b,temp = self.set_kernel(z_mu)

        for j in range(n_dims):
            self.kernel[j].fit(x, z_mu[:,j], temp*v[:,j])
            z_corrected[:,j], var_corrected = self.kernel[j].predict() #predict(x)

        return z_corrected

    def adjust_variances(self, v, window_size=4):
            """
            Adjusts variances by enforcing variance bias based on local variance patterns.

            Args:
                v (torch.Tensor): Variance tensor to adjust.
                window_size (int): Number of elements to include on each side for local averaging.

            Returns:
                torch.Tensor: Adjusted variance tensor.
            """
            kernel_size = 2 * window_size + 1
            kernel = torch.ones(kernel_size, device=v.device)
            kernel[window_size] = 0  # Exclude the current element
            kernel /= (kernel_size - 1)

            # Pad the variance tensor
            v_padded = F.pad(v.unsqueeze(0).unsqueeze(0), (window_size, window_size), mode='reflect')

            # Compute local mean variances
            local_means = F.conv1d(v_padded, kernel.view(1, 1, -1))[0, 0]

            # Calculate variance differences
            v_diff = (local_means - v).clamp(-1e3, 1e3)
            v_diff_std = v_diff.std()

            # Avoid division by zero
            if v_diff_std == 0:
                added_variance_bias = torch.ones_like(v_diff)
            else:
                added_variance_bias = torch.max(torch.tensor(1.0, device=v.device), v_diff / v_diff_std)

            # Apply the adjustment
            v_adjusted = v * added_variance_bias
            return v_adjusted

    def fit_gp_kernels(self, z_mu, z_sigma, noise_ratio = 100):
        """
        Fit Gaussian Process kernels for each latent dimension.

        Args:
            z_mu (Tensor): Mean of the latent variables.
            z_sigma (Tensor): Standard deviation of the latent variables.
        """
        n_samples, n_dims = z_mu.size()
        self.kernel = []

        x = np.arange(n_samples).reshape(-1, 1)

        for dim in range(n_dims):
            y = z_mu[:, dim].numpy()
            y_std = z_sigma[:, dim].numpy()

            # Filter out high uncertainty points
            quantile = np.quantile(y_std, 0.9)
            mask = y_std < quantile
            x_filtered = x[mask]
            y_filtered = y[mask]

            # Define kernel
            kernel = C(1.0, (1e-3, 1e3)) * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2))

            # Define noise level
            noise = y_filtered.std() / noise_ratio

            # Instantiate GP regressor
            gp = GaussianProcessRegressor(kernel=kernel, alpha=noise, n_restarts_optimizer=10, normalize_y=True)

            # Fit the GP model to the observed data
            gp.fit(x_filtered, y_filtered)

            # Set parameters of the ProbabilisticGPRepgressor
            length_scale = gp.kernel_.get_params()['k2__length_scale']
            self.kernel.append(ProbabilisticGaussianProcessRegressor(length_scale = length_scale, noise = noise))

    def set_kernel(self, z):


        out = self.encoder.kernel_params(z.mean(axis=0).reshape(1,z.shape[1]))[0]

        a = torch.abs(out[0])*10
        b = torch.abs(out[1])*10
        temp = torch.exp(out[2])*10

        a, b = torch.tensor(1), torch.tensor(.5)
        #temp = torch.tensor(10)

        return a, b, temp


class ProbabilisticGaussianProcessRegressor: #using torch linalg solve
    def __init__(self, length_scale=1.0, noise=1e-6):
        self.length_scale = length_scale
        self.noise = noise
        print(noise, 'noise')
        self.compute_variance = False
        self.alpha = 1.

    def rbf_kernel(self, X1, X2):
        """Compute the RBF kernel (Gaussian kernel)."""
        dists = torch.cdist(X1, X2, p=2)  # Pairwise distances
        K = torch.exp(-0.5 * (dists / self.length_scale) ** 2)
        return K

    def fit(self, X_train, y_train, variance):
        """Fit the Gaussian Process model with training data and probabilities."""
        # Detach X and y
        X_train = X_train.detach().reshape(-1,1).float()
        y_train = y_train.detach().float()
        
        self.X_train = X_train
        self.y_train = y_train
        #self.probabilities = probabilities

        # Compute the kernel matrix for the training data
        self.K_star = self.rbf_kernel(X_train, X_train)
        
        # Add noise term to the diagonal (regularization)
        #self.alpha = .5

        # Normalize probas
        #self.probabilities = self.probabilities / self.probabilities.max()
        #variance = (1 - self.probabilities)/self.probabilities
        #variance = -1 * torch.log(self.probabilities)
        K_regularisation = self.noise * torch.diag(variance).clone()
        self.K = self.K_star + K_regularisation * self.alpha

        self.bias = torch.sign(y_train) * variance * .05

    def predict(self):
        """Predict the mean and variance for the training points themselves."""
        
        # Mean prediction
        mu_s = self.K_star @ torch.linalg.solve(self.K, self.y_train + self.bias)
        #K_star = K_star / K_star.sum(axis=0)[None,:]
        #mu_s = K_star @ (self.y_train * self.probabilities)
        
        if self.compute_variance:
            # Compute the inverse of the kernel matrix
            self.K_inv = torch.linalg.solve(self.K, torch.eye(self.K.size(0), device=self.K.device))

            # Variance (which will be 0 for exact points)
            K_star = self.K_star
            K_s_s = K_star - K_star @ self.K_inv @ K_star
            sigma_s = K_s_s.diag()

            M = K_star @ self.K_inv
            idx = M.shape[0]//2
            plt.imshow(K_star.detach())
            plt.title('K_star')
            plt.show()
            plt.imshow(M.detach())
            plt.show()
            plt.plot(M[idx].detach(), label = 'M')
            plt.legend()
            plt.show()
            plt.plot((M[idx]*self.probabilities).detach())
            plt.plot((M[idx//2]*self.probabilities).detach())
            plt.show()
            plt.legend()
            plt.show()
        else:
            sigma_s = 1

        
        return mu_s, sigma_s
