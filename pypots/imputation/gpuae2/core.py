"""
Core wrapper for GP-UAE2 imputation model.
"""

# Created by Jun Wang <jwangfx@connect.ust.hk> and Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause

import torch
import torch.nn as nn

from ...nn.modules.gp_uae2 import BackboneGP_UAE


class _GP_UAE(nn.Module):
    """Model GP-UAE2 with optional latent Gaussian-process correction."""

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
        deterministic_ae=False,
        recon_loss_type="mse",
        recon_l1_weight=0.0,
        recon_bce_weight=0.0,
        gp=None,
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
            deterministic_ae=deterministic_ae,
            recon_loss_type=recon_loss_type,
            recon_l1_weight=recon_l1_weight,
            recon_bce_weight=recon_bce_weight,
        )

        self.gp = gp
        self.p = p
        self.alpha = alpha
        self.deterministic_ae = deterministic_ae

    def forward(self, inputs, training=True, n_sampling_times=10, use_GP=False, gp=None):
        X = torch.nan_to_num(inputs["X"], nan=0.0)
        missing_mask = inputs.get("mask", inputs.get("missing_mask"))
        if missing_mask is None:
            raise ValueError("`mask` is required in inputs for GP_UAE2.")
        missing_mask = missing_mask.to(X.device).to(torch.float32)
        results = {}

        if use_GP:
            gp = gp if gp is not None else self.gp
            if gp is None:
                raise ValueError("GP correction requested but no GP module was provided.")

            qz_x = self.backbone.encode(X, missing_mask)

            keep = (torch.rand_like(missing_mask) > self.p).to(torch.float32)
            missing_mask_corrupted = missing_mask * keep
            X_corrupted = X * missing_mask_corrupted
            qz_x_corrupted = self.backbone.encode(X_corrupted, missing_mask_corrupted)

            kernel_params = gp.update_kernel_params(X_corrupted, missing_mask_corrupted)
            q_star = gp.correct_qz_x_with_gp(qz_x_corrupted, kernel_params)

            X_star = self.backbone.decode(q_star.mean).mean
            log_prob_loss = torch.exp(-q_star.log_prob(qz_x.mean)).mean()
            recon_loss = (X - X_star).pow(2)[missing_mask > 0.5].mean()

            results["loss"] = recon_loss + log_prob_loss
            results["imputed_data"] = X_star
            return results

        backbone_results = self.backbone(X, missing_mask, training=training, return_components=True)
        results["loss"] = backbone_results["loss"]
        results["nll"] = backbone_results["nll"]
        results["kl"] = backbone_results["kl"]

        if not training:
            qz_x = self.backbone.encode(X, missing_mask)
            if self.deterministic_ae:
                imputed_data = self.backbone.decode(qz_x.mean).mean
            else:
                z_samples = qz_x.rsample(torch.tensor([n_sampling_times], device=X.device))
                reconstructions = [self.backbone.decode(z).mean for z in z_samples]
                imputed_data = torch.stack(reconstructions).mean(0)
            results["imputed_data"] = imputed_data.unsqueeze(1)

        return results

    def forward_for_GP(self, inputs, training=True, n_sampling_times=10, use_GP=False, gp=None):
        del use_GP, gp  # kept for compatibility with existing call sites

        X = torch.nan_to_num(torch.clone(inputs["X"]), nan=0.0)
        missing_mask = inputs.get("mask", inputs.get("missing_mask"))
        if missing_mask is None:
            raise ValueError("`mask` is required in inputs for GP_UAE2.")
        missing_mask = missing_mask.to(X.device).to(torch.float32)

        keep = (torch.rand_like(missing_mask) > self.p).to(torch.float32)
        mask_corr = missing_mask * keep
        X_corr = X * mask_corr
        qz_x_corr = self.backbone.encode(X_corr, mask_corr)
        kernel_params = self.gp.update_kernel_params(X_corr, mask_corr)
        q_star = self.gp.correct_qz_x_with_gp(qz_x_corr, kernel_params)

        z_samples = q_star.rsample((n_sampling_times,))
        X_recon = torch.stack([self.backbone.decode(z).mean for z in z_samples], dim=0)
        imputed_data = X_recon.mean(axis=0)

        recon_loss = self.recon_loss(X, missing_mask, mask_corr, X_recon)
        if training:
            loss = recon_loss + 0.01 * self.prior_loss(kernel_params)
            return {"loss": loss, "imputed_data": imputed_data.unsqueeze(1)}

        return {"loss": recon_loss, "imputed_data": imputed_data}

    def recon_loss(self, X, missing_mask, missing_mask_corr, X_recon):
        mask_obs = missing_mask > 0.5
        mask_obs_corr = missing_mask_corr > 0.5
        mask_hidden = mask_obs & (~mask_obs_corr)

        sq_err = (X[None] - X_recon).pow(2)
        repeated_obs_corr = mask_obs_corr.unsqueeze(0).repeat(X_recon.shape[0], 1, 1, 1, 1)
        repeated_hidden = mask_hidden.unsqueeze(0).repeat(X_recon.shape[0], 1, 1, 1, 1)

        if repeated_obs_corr.any():
            recon_loss_observed = sq_err[repeated_obs_corr].mean()
        else:
            recon_loss_observed = sq_err.mean()

        if repeated_hidden.any():
            recon_loss_imputed = sq_err[repeated_hidden].mean()
        else:
            recon_loss_imputed = recon_loss_observed

        return recon_loss_observed * self.alpha + (1 - self.alpha) * recon_loss_imputed

    def prior_loss(self, params):
        length_scale = 10.0 ** (params[0] - 1).clip(min=-3, max=0)
        sigma = 10.0 ** (params[1]).clip(-3, 0)
        noise = 10.0 ** (params[2] - 2).clip(min=-6, max=1)
        var_importance = 10.0 * torch.sigmoid(params[3])

        loss_prior = (
            (length_scale - 1e-1).pow(2)
            + (sigma - 1).pow(2)
            + (noise - 1e-2).pow(2)
            + (var_importance - 1).pow(2)
        )
        return loss_prior.mean()

    def encode(self, data, training=False, n_sampling_times=1):
        del training, n_sampling_times
        if not isinstance(data, dict):
            raise TypeError("encode expects a dict with keys `X` and `mask`.")
        return self.backbone.encode(data["X"], data["mask"])
