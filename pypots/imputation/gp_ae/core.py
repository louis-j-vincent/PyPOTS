"""
The core wrapper assembles the submodules of GP-VAE imputation model
and takes over the forward progress of the algorithm.

"""

# Created by Jun Wang <jwangfx@connect.ust.hk> and Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause


import torch.nn as nn
import torch

from ...nn.modules.gp_ae import BackboneGP_VAE

from pygrinder import mcar


class _GP_VAE(nn.Module):
    """model GPVAE with Gaussian Process prior

    Parameters
    ----------
    input_dim : int,
        the feature dimension of the input

    time_length : int,
        the length of each time series

    latent_dim : int,
        the feature dimension of the latent embedding

    encoder_sizes : tuple,
        the tuple of the network size in encoder

    decoder_sizes : tuple,
        the tuple of the network size in decoder

    beta : float,
        the weight of the KL divergence

    M : int,
        the number of Monte Carlo samples for ELBO estimation

    K : int,
        the number of importance weights for IWAE model

    kernel : str,
        the Gaussian Process kernel ["cauchy", "diffusion", "rbf", "matern"]

    sigma : float,
        the scale parameter for a kernel function

    length_scale : float,
        the length scale parameter for a kernel function

    kernel_scales : int,
        the number of different length scales over latent space dimensions
    """

    def __init__(
        self,
        input_dim,
        time_length,
        latent_dim,
        encoder_sizes=(64, 64),
        decoder_sizes=(64, 64),
        beta=1,
        M=1,
        K=1,
        kernel="cauchy",
        sigma=1.0,
        length_scale=7.0,
        kernel_scales=1,
        window_size=24,
        gp = None
    ):
        super().__init__()

        self.backbone = BackboneGP_VAE(
            input_dim,
            time_length,
            latent_dim,
            encoder_sizes,
            decoder_sizes,
            beta,
            M,
            K,
            kernel,
            sigma,
            length_scale,
            kernel_scales,
            window_size,
        )

        self.gp = gp

    def forward(self, inputs, training=True, n_sampling_times=10, use_GP=False, gp = None):
        X, missing_mask = inputs["X"], inputs["missing_mask"]
        if torch.isnan(X).any():
            X[X!=X] = 0. #replace nans by zeros
        results = {}

        n_sampling_times = n_sampling_times

        missing_mask = (X!=0)

        if use_GP:

            # get embedding for X
            qz_x = self.backbone.encode(X, missing_mask)

            # corrupt X and get emebdding for X corrupted
            X_corrupted = mcar(X, self.p)
            X_corrupted = torch.nan_to_num(X_corrupted, 0)
            missing_mask_corrupted = (X_corrupted!=0)
            qz_x_corrupted = self.backbone.encode(X_corrupted, missing_mask_corrupted)

            # Correct with Gaussian Process Regressor
            kernel_params = gp.update_kernel_params(X_corrupted)
            q_star = gp.correct_qz_x_with_gp(qz_x_corrupted, kernel_params)

            # reconstruct X star
            X_star = self.backbone.decode(q_star.mean).mean

            # Get the log prob
            log_prob_loss = torch.exp(-q_star.log_prob(qz_x.mean)).mean()

            # Get the recon loss
            recon_loss = (X - X_star).pow(2)[missing_mask].mean()

            print(recon_loss, log_prob_loss)

            results['loss'] = recon_loss + log_prob_loss
            results['imputed_data'] = X_star

        else:

                
            if training:

                elbo_loss = self.backbone(X, missing_mask)
                results["loss"] = elbo_loss
        
            else:
                elbo_loss = self.backbone(X, missing_mask)
                results["loss"] = elbo_loss
                #try:
                #    imputed_data = self.backbone.impute(X, missing_mask, n_sampling_times)
                #except:
                #print('Impute not implemented')
                qz_x = self.backbone.encode(X, missing_mask)
                z_samples = qz_x.rsample(torch.tensor([n_sampling_times]))
                #z = qz_x.mean.detach()
                reconstructions = [self.backbone.decode(z).mean for z in z_samples]
                imputed_data = torch.stack(reconstructions).mean(0)
                imputed_data += torch.rand(imputed_data.shape) * 1e-2

                results["imputed_data"] = imputed_data.unsqueeze(1) #because dim 1 is where the samples are

        return results

    def encode(self, data, training=False, n_sampling_times=1):

        return self.backbone.encode(data)