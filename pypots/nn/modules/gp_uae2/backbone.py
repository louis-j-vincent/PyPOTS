import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Independent, Normal, kl_divergence
from typing import Optional, Tuple, Any

from .layers import GpvaeEncoder_cnn, GpvaeDecoder_cnn


class BackboneGP_UAE(nn.Module):
    """GP-UAE2 backbone for image sequences with explicit masks.

    Expected input format:
    - X: (B, T, 28, 28)
    - missing_mask: (B, T, 28, 28), 1 for observed and 0 for missing.
    """

    def __init__(
        self,
        input_dim: int,
        time_length: int,
        latent_dim: int,
        encoder_sizes: Tuple[int, ...] = (128, 64),
        decoder_sizes: Tuple[int, ...] = (64, 128),
        alpha: float = 1.0,
        beta: float = 1.0,
        M: int = 1,
        K: int = 1,
        kernel: Optional[str] = None,
        sigma: float = 1.0,
        p: float = 0.1,
        length_scale: float = 7.0,
        kernel_scales: int = 1,
        window_size: int = 24,
        deterministic_ae: bool = False,
        recon_loss_type: str = "mse",
        recon_l1_weight: float = 0.0,
        recon_bce_weight: float = 0.0,
        device: Optional[torch.device] = None,
    ) -> None:
        del input_dim, kernel, length_scale, kernel_scales, window_size
        super().__init__()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.time_length = time_length
        self.latent_dim = latent_dim
        self.alpha = alpha
        self.beta = beta
        self.sigma = sigma
        self.p = p
        self.M = M
        self.K = K
        self.deterministic_ae = deterministic_ae
        self.recon_loss_type = recon_loss_type
        self.recon_l1_weight = recon_l1_weight
        self.recon_bce_weight = recon_bce_weight

        self.encoder = GpvaeEncoder_cnn(input_size=2, z_size=latent_dim, hidden_sizes=encoder_sizes).to(self.device)
        self.decoder = GpvaeDecoder_cnn(input_size=latent_dim, output_size=1, hidden_sizes=decoder_sizes).to(self.device)

    def encode(self, x: torch.Tensor, missing_mask: Optional[torch.Tensor] = None) -> Any:
        if missing_mask is None:
            raise ValueError("`missing_mask` is required for GP_UAE2 encode.")
        return self.encoder(x, missing_mask)

    def decode(self, z: torch.Tensor) -> Any:
        return self.decoder(z)

    def _check_shapes(self, X: torch.Tensor, missing_mask: torch.Tensor) -> None:
        if X.ndim != 4 or missing_mask.ndim != 4:
            raise ValueError(
                "GP_UAE2 expects X and mask as 4D tensors shaped (B, T, 28, 28). "
                f"Got X{tuple(X.shape)} and mask{tuple(missing_mask.shape)}."
            )
        if X.shape != missing_mask.shape:
            raise ValueError(
                f"X and mask must share shape, got X{tuple(X.shape)} and mask{tuple(missing_mask.shape)}."
            )
        if X.shape[-2:] != (28, 28):
            raise ValueError(f"GP_UAE2 supports only 28x28 images, got {tuple(X.shape)}.")

    def prepare_and_simulate(
        self, X: torch.Tensor, missing_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        X = torch.nan_to_num(X, nan=0.0).to(torch.float32)
        missing_mask = missing_mask.to(X.device).to(torch.float32).clamp(0.0, 1.0)

        # Additional random masking over already observed entries.
        keep = (torch.rand_like(missing_mask) > self.p).to(torch.float32)
        corrupted_mask = missing_mask * keep
        X_corrupted = X * corrupted_mask

        return X, missing_mask, X_corrupted, corrupted_mask

    def kl(self, qz_x: Independent) -> torch.Tensor:
        prior = Independent(
            Normal(
                loc=torch.zeros_like(qz_x.base_dist.loc),
                scale=torch.ones_like(qz_x.base_dist.scale),
            ),
            1,
        )
        return kl_divergence(qz_x, prior).mean()

    def nll(
        self,
        X: torch.Tensor,
        X_corrupted: torch.Tensor,
        px_z: Normal,
        observed_mask: torch.Tensor,
        corrupted_mask: torch.Tensor,
    ) -> torch.Tensor:
        del X_corrupted
        recon = px_z.mean.clamp(min=1e-6, max=1 - 1e-6)
        target = X.clamp(min=0.0, max=1.0)

        if self.recon_loss_type == "mse":
            base_loss = (target - recon).pow(2)
        elif self.recon_loss_type == "l1":
            base_loss = (target - recon).abs()
        elif self.recon_loss_type == "bce":
            base_loss = F.binary_cross_entropy(recon, target, reduction="none")
        elif self.recon_loss_type == "bce_l1":
            bce = F.binary_cross_entropy(recon, target, reduction="none")
            l1 = (target - recon).abs()
            bce_w = self.recon_bce_weight if self.recon_bce_weight > 0 else 1.0
            l1_w = self.recon_l1_weight if self.recon_l1_weight > 0 else 1.0
            base_loss = bce_w * bce + l1_w * l1
        else:
            raise ValueError(
                f"Unsupported recon_loss_type `{self.recon_loss_type}`. "
                "Use one of ['mse', 'l1', 'bce', 'bce_l1']."
            )

        visible_mask = corrupted_mask > 0.5
        hidden_mask = (observed_mask > 0.5) & (~visible_mask)

        if visible_mask.any():
            recon_visible = base_loss[visible_mask].mean()
        else:
            recon_visible = base_loss.mean()

        if hidden_mask.any():
            recon_hidden = base_loss[hidden_mask].mean()
        else:
            recon_hidden = recon_visible

        return self.alpha * recon_visible + (1 - self.alpha) * recon_hidden

    def elbo(
        self,
        X: torch.Tensor,
        missing_mask: torch.Tensor,
        X_corrupted: torch.Tensor,
        missing_mask_corrupted: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        qz_x_corrupted = self.encode(X_corrupted, missing_mask_corrupted)
        if self.deterministic_ae:
            z = qz_x_corrupted.mean
        else:
            z = qz_x_corrupted.rsample()
        px_z = self.decode(z)

        nll = self.nll(X, X_corrupted, px_z, missing_mask, missing_mask_corrupted)
        if self.deterministic_ae:
            kl = torch.zeros((), device=X.device, dtype=X.dtype)
        else:
            kl = self.kl(qz_x_corrupted)
        return nll, kl

    def forward(
        self,
        X: torch.Tensor,
        missing_mask: torch.Tensor,
        training: bool = True,
        return_components: bool = False,
    ):
        del training
        self._check_shapes(X, missing_mask)
        X, missing_mask, X_corrupted, missing_mask_corrupted = self.prepare_and_simulate(X, missing_mask)
        nll, kl = self.elbo(X, missing_mask, X_corrupted, missing_mask_corrupted)
        loss = nll + self.beta * kl
        if return_components:
            return {"loss": loss, "nll": nll, "kl": kl}
        return loss
