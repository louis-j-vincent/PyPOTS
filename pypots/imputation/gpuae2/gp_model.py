"""Lightweight latent GP utilities for GP-UAE2."""

import torch
import torch.nn as nn

from ...optim.adam import Adam


def make_nn(input_size, output_size, hidden_sizes):
    layers = []
    for i, hidden in enumerate(hidden_sizes):
        in_features = input_size if i == 0 else hidden_sizes[i - 1]
        layers.append(nn.Linear(in_features=in_features, out_features=hidden))
        layers.append(nn.ReLU())
    layers.append(nn.Linear(in_features=hidden_sizes[-1], out_features=output_size))
    return nn.Sequential(*layers)


class ProbabilisticGP:
    """Simple GP correction in latent space with learned kernel parameters."""

    def __init__(self, model, assemble_data_train, assemble_data_val, n_dims, latent_size):
        self.n_dims = n_dims
        self.latent_dim = latent_size
        self.n_kernel_params = 5

        self.kernel_params_estimator = make_nn(
            input_size=n_dims,
            output_size=latent_size * self.n_kernel_params,
            hidden_sizes=(max(8, n_dims * 10), max(8, n_dims * 10)),
        )

        self.assemble_data_train = assemble_data_train
        self.assemble_data_val = assemble_data_val

        self.encode = model.encoder.eval()
        self.decode = model.decoder.eval()

        self.optimizer = Adam()
        self.optimizer.init_optimizer(self.kernel_params_estimator.parameters())

    def update_kernel_params(self, x, mask=None):
        if mask is None:
            mask = (x != 0).float()
        else:
            mask = mask.float()

        if mask.ndim == 4:
            observed_ratio = mask.mean(dim=(2, 3))
        elif mask.ndim == 3:
            observed_ratio = mask.mean(dim=-1)
        else:
            raise ValueError(f"Unsupported mask shape {tuple(mask.shape)} for kernel parameter estimation.")

        return self.kernel_params_estimator(observed_ratio).reshape(
            observed_ratio.shape[0], self.latent_dim, self.n_kernel_params
        )

    def kernel(self, X1, X2, raw_params):
        dists = torch.cdist(X1, X2, p=2)

        length_scale_rbf = 10.0 ** raw_params[0].clip(min=-2, max=1)
        sigma_rbf = 10.0 ** raw_params[1].clip(min=-3, max=0)

        length_scale_cauchy = 10.0 ** raw_params[2].clip(min=-5, max=0)
        sigma_cauchy = 10.0 ** raw_params[3].clip(min=-3, max=1)

        k_rbf = sigma_rbf * torch.exp(-0.5 * (dists / length_scale_rbf) ** 2)
        k_cauchy = sigma_cauchy**2 * (1 + (dists / length_scale_cauchy) ** 2) ** (-1)

        return k_rbf + k_cauchy

    def correct_with_gp(self, z_mu, z_var, kernel_params, return_variance=False):
        z_mu = z_mu.detach()
        z_var = z_var.detach().clamp(min=1e-5)

        batch_size, time_steps, latent_dim = z_mu.shape
        T = torch.linspace(0, 1, time_steps, device=z_mu.device).unsqueeze(1)

        z_star = torch.zeros_like(z_mu)
        sigma_star = torch.zeros_like(z_var) if return_variance else None

        eye = torch.eye(time_steps, device=z_mu.device)
        jitter = 1e-4

        for j in range(latent_dim):
            for b in range(batch_size):
                K = self.kernel(T, T, kernel_params[b, j])
                K_obs = K + torch.diag(z_var[b, :, j]) + 1e-6 * eye

                alpha = torch.linalg.solve(K_obs + jitter * eye, z_mu[b, :, j])
                z_star[b, :, j] = K @ alpha

                if return_variance:
                    correction = torch.linalg.solve(K_obs + jitter * eye, K)
                    var = torch.diagonal(K - K @ correction).clamp(min=1e-5)
                    sigma_star[b, :, j] = var

        if return_variance:
            return z_star, sigma_star
        return z_star

    def correct_qz_x_with_gp(self, qz_x, kernel_params):
        z_mu = qz_x.mean.detach()
        z_var = qz_x.variance.clamp(min=1e-5).detach()

        z_mu_star, z_var_star = self.correct_with_gp(
            z_mu, z_var, kernel_params, return_variance=True
        )

        return torch.distributions.Normal(loc=z_mu_star, scale=z_var_star.clamp(min=1e-5))

    def infer(self, qz_x, x_input, mask=None, return_variance=False):
        z_mu = qz_x.mean
        z_var = qz_x.variance
        kernel_params = self.update_kernel_params(x_input, mask=mask)

        if return_variance:
            z_star, sigma_star = self.correct_with_gp(
                z_mu, z_var, kernel_params, return_variance=True
            )
        else:
            z_star = self.correct_with_gp(z_mu, z_var, kernel_params, return_variance=False)
            sigma_star = None

        x_recon_mean = self.decode(z_star).mean
        if return_variance:
            return x_recon_mean, sigma_star
        return x_recon_mean
