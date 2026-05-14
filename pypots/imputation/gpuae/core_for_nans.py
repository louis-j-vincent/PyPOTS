"""
The core wrapper assembles the submodules of GP-UAE imputation model
and takes over the forward progress of the algorithm.

"""

# Created by Jun Wang <jwangfx@connect.ust.hk> and Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause

import matplotlib.pyplot as plt
import torch.nn as nn
import torch
import numpy as np

from ...nn.modules.gp_uae import BackboneGP_UAE

from pygrinder import mcar
import torch.distributions as dist


class _GP_UAE(nn.Module):
    """model GPUAE with Gaussian Process prior

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
        alpha=1,
        beta=1,
        M=1,
        K=1,
        kernel="cauchy",
        sigma=1.0,
        length_scale=7.0,
        kernel_scales=1,
        window_size=24,
        p=0.1,
        gp = None
    ):
        super().__init__()

        self.backbone = BackboneGP_UAE(
            input_dim,
            time_length,
            latent_dim,
            encoder_sizes,
            decoder_sizes,
            alpha,
            beta,
            M,
            K,
            kernel,
            sigma,
            p,
            length_scale,
            kernel_scales,
            window_size,
        )

        self.gp = gp
        self.p = p

    def forward(self, inputs, training=True, n_sampling_times=10, use_GP=False, gp = None):
        
        X, missing_mask = inputs["X"], inputs["missing_mask"].bool()        
        # Replace nans by zeros
        #if torch.isnan(X).any():
        #    X[X!=X] = 0. 
        results = {}

        #missing_mask = (X!=0)

        if use_GP:

            # get embedding for X
            qz_x = self.backbone.encode(X, missing_mask)

            # corrupt X and get emebdding for X corrupted
            #X_corrupted = torch.nan_to_num(mcar(X, self.p), 0)
            #X_corrupted = torch.nan_to_num(X_corrupted, 0)
            #missing_mask_corrupted = (X_corrupted!=0)
            X_corrupted = mcar(X, self.p)
            missing_mask_corrupted = (X_corrupted==X_corrupted)
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
                #imputed_data += torch.rand(imputed_data.shape) * 1e-2

                results["imputed_data"] = imputed_data.unsqueeze(1) #because dim 1 is where the samples are

        return results

    def forward_for_GP(self, inputs, training=True, n_sampling_times=10, use_GP=False, gp = None):
        
        X, missing_mask = torch.clone(inputs["X"]), inputs["missing_mask"]
        
        # Replace nans by zeros
        #if torch.isnan(X).any():
        #    X[X!=X] = 0. 
        results = {}

        #missing_mask = (X!=0)

        if training:

            # create corrupted version
            X_corr = mcar(X, p=self.p)
            missing_mask_corr = (X_corr!=X_corr)
            #X_corr[X_corr != X_corr] = 0.  # Replace NaNs with 0
            #missing_mask_corr = (X_corr!=0)

            qz_x_corr = self.backbone.encode(X_corr)

            kernel_params = self.gp.update_kernel_params(X_corr)
 
            q_star = self.gp.correct_qz_x_with_gp(qz_x_corr, kernel_params) #THIS TAKES TIME

            z_samples = q_star.rsample((n_sampling_times,))
            #z_samples = q_star.mean.repeat((n_sampling_times,1,1,1))

            X_recon = self.backbone.decode(z_samples).mean

            imputed_data = X_recon.mean(axis=0)

            results["imputed_data"] = imputed_data.unsqueeze(1) #because dim 1 is where the samples are

            #recon_loss = (X[None] - X_recon).pow(2)[missing_mask.unsqueeze(0).repeat(n_sampling_times,1,1,1)].mean()

            prior_loss = self.prior_loss(kernel_params)


            #recon_loss = (X - X_recon.mean(axis=0)).pow(2)[missing_mask].mean()

            mask_hidden = missing_mask & (~missing_mask_corr)
            recon_loss_observed = (X[None] - X_recon).pow(2)[missing_mask_corr.unsqueeze(0).repeat(n_sampling_times,1,1,1)].mean()
            recon_loss_imputed = (X[None] - X_recon).pow(2)[mask_hidden.unsqueeze(0).repeat(n_sampling_times,1,1,1)].mean()

            #results["loss"] = recon_loss_observed + recon_loss_imputed + 0.01 * prior_loss

            recon_loss = self.recon_loss(X, X_corr, X_recon)
            

            results["loss"] = recon_loss + 0.01 * prior_loss #+ 0.01 * self.gp.loss(X, qz_x_corr)

            #results["loss"] = self.gp.loss(X, qz_x_corr)




            print(recon_loss_observed.item(), recon_loss_imputed.item())

            #plt.plot((X[None] - X_recon).pow(2)[missing_mask_corr.unsqueeze(0).repeat(n_sampling_times,1,1,1)].detach().numpy()[:1000])
            #plt.show()

            #print(kernel_params.shape, z_samples.shape)

            for j in range(z_samples.shape[-1]):

                self.gp.kernel_params_history[j]['a'].append(kernel_params[0,j,1].detach().item())
                self.gp.kernel_params_history[j]['b'].append(kernel_params[0,j,2].detach().item())
                self.gp.kernel_params_history[j]['c'].append(kernel_params[0,j,3].detach().item())
                self.gp.kernel_params_history[j]['d'].append(kernel_params[0,j,4].detach().item())
                self.gp.kernel_params_history[j]['e'].append(results["loss"].detach().item())
                #print(kernel_params[0,j,1].detach().item(),j)
                #print(self.gp.kernel_params_history[j])


        else:

            with torch.no_grad():

                # create corrupted version
                X_corr = mcar(X, p=self.p)
                missing_mask_corr = (X_corr!=X_corr)
                #X_corr[X_corr != X_corr] = 0.  # Replace NaNs with 0
                #missing_mask_corr = (X_corr!=0)

                qz_x_corr = self.backbone.encode(X_corr)

                kernel_params = self.gp.update_kernel_params(X_corr)
    
                q_star = self.gp.correct_qz_x_with_gp(qz_x_corr, kernel_params) #THIS TAKES TIME

                z_samples = q_star.rsample((n_sampling_times,))
                #z_samples = q_star.mean.repeat((n_sampling_times,1,1,1))


                X_recon = self.backbone.decode(z_samples).mean

                imputed_data = X_recon.mean(axis=0)

                results["imputed_data"] = imputed_data#.unsqueeze(1) #because dim 1 is where the samples are

                #recon_loss = (X[None] - X_recon).pow(2)[missing_mask.unsqueeze(0).repeat(n_sampling_times,1,1,1)].mean()

                recon_loss = self.recon_loss(X, X_corr, X_recon)

                #recon_loss = (X - X_recon.mean(axis=0)).pow(2)[missing_mask].mean()

                results["loss"] = recon_loss 


        return results

    def recon_loss(self, X, X_corr, X_recon, eps = 1e-5):

        missing_mask = (X!=0)

        #recon_loss = (X - X_recon.mean(axis=0)).pow(2)[missing_mask].clip(max = 1e1).mean()
        recon_loss = (X[None] - X_recon).pow(2)[missing_mask.unsqueeze(0).repeat(X_recon.shape[0],1,1,1)].mean()

        if False:


            missing_mask_corr = (X_corr!=0)

            mask_hidden = missing_mask & (~missing_mask_corr)

            mu, var = X_recon.mean(axis=0), X_recon.var(axis=0)+ eps

            log_prob_obs =  0.5 * (torch.log(2 * torch.tensor(np.pi, device=self.backbone.device) * var) + (X - mu).pow(2) / var)[missing_mask_corr]#.mean()
            log_prob_imputed =  0.5 * (torch.log(2 * torch.tensor(np.pi, device=self.backbone.device) * var) + (X - mu).pow(2) / var)[mask_hidden]#.mean()


            #print(log_prob)
            log_prob = log_prob_obs.mean() + log_prob_imputed.mean()

            var_mean_prior = 0
            var_mean_variance = 0.01
            var_loss = ( (var - var_mean_prior).pow(2) / var_mean_variance) .mean()

            return log_prob + var_loss

        return recon_loss

    def prior_loss(self, params):

        #length_scale = 10. ** (kernel_params[0]).clip(min = -3, max = 0)
        #sigma = 10 ** (kernel_params[1]).clip(-2, 0)
        #noise = 10. ** (kernel_params[2]).clip(min = -5, max = -3)
        #var_importance = 0. + torch.sigmoid((kernel_params[3]))

        length_scale = 10. ** (params[0] - 1).clip(min = -3, max = 0)
        sigma = 10 ** (params[1]).clip(-3, 0)
        #sigma = 1.
        noise = 10. ** (params[2] - 2).clip(min = -6, max = 1)
        var_importance = 10 * torch.sigmoid((params[3]))
        var_importance2 = 10 * torch.sigmoid(params[0])#.clip(min = , max = 1)

        loss_prior = (length_scale - 1e-1).pow(2) + (sigma - 1).pow(2) + (noise - 1e-2).pow(2) + (var_importance - 1).pow(2)


        #loss_prior = (noise - 0).pow(2) + (var_importance - 0).pow(2)

        return loss_prior.mean()

    def prior_loss_(self, raw_params):
        """
        Compute the prior loss (negative log prior) for the raw kernel parameters.
        We assume independent Normal priors on the log10 parameters.
        
        Recommended prior settings:
          raw_params[0] (RBF length scale): mean = -2, std = 1
          raw_params[1] (RBF sigma):         mean = -1, std = 1
          raw_params[2] (Cauchy length scale): mean = -3, std = 1
          raw_params[3] (Cauchy sigma):         mean = -1, std = 1
          raw_params[4] (noise):              mean = -3, std = 1
        """
        prior_loss = 0.0
        
        priors = [
            (raw_params[0], 1., 1.0),
            (raw_params[1], -1.0, 1.0),
            (raw_params[2], -1.0, 1.0),
            (raw_params[3], -1.0, 1.0),
            (raw_params[4], -3.0, 1.0),
        ]
        
        for param, mean, std in priors:
            # Create a Normal distribution for the log10 parameter
            prior_dist = dist.Normal(torch.tensor(mean, device=param.device),
                                     torch.tensor(std, device=param.device))
            # Accumulate the negative log probability
            prior_loss = prior_loss - prior_dist.log_prob(param)
        
        # Return the summed prior loss
        return prior_loss.sum()

    def encode(self, data, training=False, n_sampling_times=1):

        return self.backbone.encode(data)