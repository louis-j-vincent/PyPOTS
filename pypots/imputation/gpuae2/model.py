"""
The implementation of GP-VAE for the partially-observed time-series imputation task.

"""

# Created by Jun Wang <jwangfx@connect.ust.hk> and Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause


import os
import csv
from typing import Union, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

try:
    import nni
except ImportError:
    pass


from .core import _GP_UAE
from .data import DatasetForGPVAE
from ..base import BaseNNImputer
from ...optim.adam import Adam
from ...optim.base import Optimizer
from ...utils.logging import logger
from .gp_model import ProbabilisticGP

class GP_UAE2(BaseNNImputer):
    """The PyTorch implementation of the GPUAE model :cite:`fortuin2020gpUAE`.

    Parameters
    ----------
    n_steps :
        The number of time steps in the time-series data sample.

    n_features :
        The number of features in the time-series data sample.

    latent_size : int,
        The feature dimension of the latent embedding

    encoder_sizes : tuple,
        The tuple of the network size in encoder

    decoder_sizes : tuple,
        The tuple of the network size in decoder

    beta : float,
        The weight of KL divergence in ELBO.

    M : int,
        The number of Monte Carlo samples for ELBO estimation during training.

    K : int,
        The number of importance weights for IWAE model training loss.

    kernel: str
        The type of kernel function chosen in the Gaussain Process Proir. ["cauchy", "diffusion", "rbf", "matern"]

    sigma : float,
        The scale parameter for a kernel function

    length_scale : float,
        The length scale parameter for a kernel function

    kernel_scales : int,
        The number of different length scales over latent space dimensions

    window_size : int,
        Window size for the inference CNN.

    batch_size : int
        The batch size for training and evaluating the model.

    epochs : int
        The number of epochs for training the model.

    patience : int
        The patience for the early-stopping mechanism. Given a positive integer, the training process will be
        stopped when the model does not perform better after that number of epochs.
        Leaving it default as None will disable the early-stopping.

    optimizer : pypots.optim.base.Optimizer
        The optimizer for model training.
        If not given, will use a default Adam optimizer.

    num_workers : int
        The number of subprocesses to use for data loading.
        `0` means data loading will be in the main process, i.e. there won't be subprocesses.

    device : :class:`torch.device` or list
        The device for the model to run on. It can be a string, a :class:`torch.device` object, or a list of them.
        If not given, will try to use CUDA devices first (will use the default CUDA device if there are multiple),
        then CPUs, considering CUDA and CPU are so far the main devices for people to train ML models.
        If given a list of devices, e.g. ['cuda:0', 'cuda:1'], or [torch.device('cuda:0'), torch.device('cuda:1')] , the
        model will be parallely trained on the multiple devices (so far only support parallel training on CUDA devices).
        Other devices like Google TPU and Apple Silicon accelerator MPS may be added in the future.

    saving_path : str
        The path for automatically saving model checkpoints and tensorboard files (i.e. loss values recorded during
        training into a tensorboard file). Will not save if not given.

    model_saving_strategy : str
        The strategy to save model checkpoints. It has to be one of [None, "best", "better"].
        No model will be saved when it is set as None.
        The "best" strategy will only automatically save the best model after the training finished.
        The "better" strategy will automatically save the model during training whenever the model performs
        better than in previous epochs.

    """

    def __init__(
        self,
        n_steps: int,
        n_features: int,
        latent_size: int,
        encoder_sizes: tuple = (64, 64),
        decoder_sizes: tuple = (64, 64),
        kernel: str = "cauchy",
        alpha: float = 1.0,
        beta: float = 1,
        M: int = 1,
        K: int = 1,
        sigma: float = 1.0,
        length_scale: float = 7.0,
        kernel_scales: int = 1,
        window_size: int = 3,
        batch_size: int = 32,
        epochs: int = 100,
        patience: Optional[int] = 100,
        optimizer: Optional[Optimizer] = Adam(),
        num_workers: int = 0,
        device: Optional[Union[str, torch.device, list]] = None,
        saving_path: str = None,
        model_saving_strategy: Optional[str] = "best",
        verbose: bool = True,
        p: float = 0.1,
        train_gp: bool = False,
        use_gp: bool = False,
        deterministic_ae: bool = False,
        recon_loss_type: str = "mse",
        recon_l1_weight: float = 0.0,
        recon_bce_weight: float = 0.0,
        visualization_dir: Optional[str] = None,
        n_visual_samples: int = 6,
        pretrained_weights_path: Optional[str] = None,
        load_backbone_only: bool = False,
        freeze_loaded_backbone: bool = False,
    ):
        super().__init__(
            batch_size,
            epochs,
            patience,
            num_workers,
            device,
            saving_path,
            model_saving_strategy,
            verbose
        )
        available_kernel_type = ["cauchy", "diffusion", "rbf", "matern"]
        assert kernel in available_kernel_type, f"kernel should be one of {available_kernel_type}, but got {kernel}"

        self.n_steps = n_steps
        self.n_features = n_features
        self.latent_size = latent_size
        self.kernel = kernel
        self.encoder_sizes = encoder_sizes
        self.decoder_sizes = decoder_sizes
        self.alpha = alpha
        self.beta = beta
        self.M = M
        self.K = K
        self.sigma = sigma
        self.length_scale = length_scale
        self.kernel_scales = kernel_scales
        self.p = p
        self.train_gp = train_gp
        self.use_gp = use_gp
        self.deterministic_ae = deterministic_ae
        self.recon_loss_type = recon_loss_type
        self.recon_l1_weight = recon_l1_weight
        self.recon_bce_weight = recon_bce_weight
        self.visualization_dir = visualization_dir
        self.n_visual_samples = n_visual_samples
        self._metrics_history = []
        self.pretrained_weights_path = pretrained_weights_path
        self.load_backbone_only = load_backbone_only
        self.freeze_loaded_backbone = freeze_loaded_backbone

        # Save patience value for early stopping.
        # If patience is None, early stopping is disabled.
        if patience is None:
            self.patience = float("inf")
            self.original_patience = float("inf")
        else:
            self.patience = patience
            self.original_patience = patience

        # set up the model
        self.model = _GP_UAE(
            input_dim=self.n_features,
            time_length=self.n_steps,
            latent_dim=self.latent_size,
            kernel=self.kernel,
            encoder_sizes=self.encoder_sizes,
            decoder_sizes=self.decoder_sizes,
            alpha = self.alpha,
            beta=self.beta,
            M=self.M,
            K=self.K,
            sigma=self.sigma,
            length_scale=self.length_scale,
            kernel_scales=self.kernel_scales,
            window_size=window_size,
            p=p,
            deterministic_ae=deterministic_ae,
            recon_loss_type=recon_loss_type,
            recon_l1_weight=recon_l1_weight,
            recon_bce_weight=recon_bce_weight,
        )
        self._send_model_to_given_device()
        self._print_model_size()

        # set up the optimizer
        self.optimizer = optimizer
        self.optimizer.init_optimizer(self.model.parameters())

        self.model.backbone.to(self.device)

        # set gp
        self.gp = ProbabilisticGP(self.model.backbone, 
                            assemble_data_train = self._assemble_input_for_training, 
                            assemble_data_val = self._assemble_input_for_validating,
                            n_dims = self.n_steps,
                            latent_size = self.latent_size)

        self.model.gp = self.gp

        if self.visualization_dir is not None:
            self.visualization_dir = os.path.abspath(self.visualization_dir)
            self._epoch_img_dir = os.path.join(self.visualization_dir, "epoch_images")
            os.makedirs(self._epoch_img_dir, exist_ok=True)
        else:
            self._epoch_img_dir = None

        if self.pretrained_weights_path is not None:
            self.load_weights(
                self.pretrained_weights_path,
                load_backbone_only=self.load_backbone_only,
                freeze_backbone=self.freeze_loaded_backbone,
                strict=False,
            )

    def _assemble_input_for_training(self, data: list) -> dict:
        # fetch data
        (
            indices,
            X,
            mask,
        ) = self._send_data_to_given_device(data)

        # assemble input data
        inputs = {
            "indices": indices,
            "X": X,
            "mask": mask,
        }

        return inputs

    def _assemble_input_for_validating(self, data: list) -> dict:
        # fetch data
        (
            indices,
            X,
            mask,
            X_ori,
            indicating_mask,
        ) = self._send_data_to_given_device(data)

        # assemble input data
        inputs = {
            "indices": indices,
            "X": X,
            "mask": mask,
            "X_ori": X_ori,
            "indicating_mask": indicating_mask,
        }

        return inputs

    def _assemble_input_for_testing(self, data: list) -> dict:
        return self._assemble_input_for_training(data)

    def _train_model(
        self,
        training_loader: DataLoader,
        val_loader: DataLoader = None,
    ) -> None:
        # each training starts from the very beginning, so reset the loss and model dict here
        self.best_loss = float("inf")
        self.best_model_dict = None
        self._metrics_history = []

        try:
            training_step = 0.
            for epoch in range(1, self.epochs + 1):
                self.model.train()
                self.model.backbone.temperature = epoch / (self.epochs + 1)
                epoch_train_loss_collector = []
                epoch_train_nll_collector = []
                epoch_train_kl_collector = []
                for idx, data in enumerate(training_loader):
                    training_step += 1
                    inputs = self._assemble_input_for_training(data)
                    self.optimizer.zero_grad()

                    if self.train_gp:
                        results = self.model.forward(inputs, use_GP=True, gp = self.gp)
                    else:
                        results = self.model.forward(inputs)

                    # use sum() before backward() in case of multi-gpu training
                    results["loss"].sum().backward()

                    #clip gradients
                    #torch.nn.utils.clip_grad_norm_(v_1, max_norm=1.0, norm_type=2)
                    self.optimizer.step()

                    epoch_train_loss_collector.append(results["loss"].sum().item())
                    if "nll" in results:
                        epoch_train_nll_collector.append(results["nll"].sum().item())
                    if "kl" in results:
                        epoch_train_kl_collector.append(results["kl"].sum().item())

                    # save training loss logs into the tensorboard file for every step if in need
                    if self.summary_writer is not None:
                        self._save_log_into_tb_file(training_step, "training", results)


                # mean training loss of the current epoch
                mean_train_loss = np.mean(epoch_train_loss_collector)
                mean_train_nll = np.mean(epoch_train_nll_collector) if len(epoch_train_nll_collector) else np.nan
                mean_train_kl = np.mean(epoch_train_kl_collector) if len(epoch_train_kl_collector) else np.nan

                #
                if self.train_gp and epoch%10==0:
                    self._train_kernel(training_loader, val_loader)

                if val_loader is not None:
                    self.model.eval()
                    imputation_loss_collector = []
                    val_nll_collector = []
                    val_kl_collector = []
                    with torch.no_grad():
                        for idx, data in enumerate(val_loader):

                            inputs = self._assemble_input_for_validating(data)

                            if self.train_gp:
                                results = self.model.forward(inputs, use_GP=True, gp = self.gp)
                            else:
                                results = self.model.forward(inputs, training=False, n_sampling_times=1)

                            imputation_loss_collector.append(results['loss'].sum().item())
                            if "nll" in results:
                                val_nll_collector.append(results["nll"].sum().item())
                            if "kl" in results:
                                val_kl_collector.append(results["kl"].sum().item())


                            #imputation_loss_collector.append(imputation_mse)

                            #inputs = self._assemble_input_for_validating(data)
                            
                            #elbo_loss_val = self.model.forward(inputs, training=False, n_sampling_times=1)
                            #imputation_loss_collector.append(elbo_loss_val['loss'].sum().item())
                            
                            
                            #imputed_data = results["imputed_data"].mean(axis=1)

                    mean_val_loss = np.mean(imputation_loss_collector)
                    mean_val_nll = np.mean(val_nll_collector) if len(val_nll_collector) else np.nan
                    mean_val_kl = np.mean(val_kl_collector) if len(val_kl_collector) else np.nan

                    # save validation loss logs into the tensorboard file for every epoch if in need
                    if self.summary_writer is not None:
                        val_loss_dict = {
                            "imputation_loss": mean_val_loss,
                        }
                        self._save_log_into_tb_file(epoch, "validating", val_loss_dict)

                    logger.info(
                        f"Epoch {epoch:03d} - "
                        f"training loss: {mean_train_loss:.4f}, "
                        f"validation loss: {mean_val_loss:.4f}, "
                        f"train nll: {mean_train_nll:.4f}, train kl: {mean_train_kl:.4f}, "
                        f"val nll: {mean_val_nll:.4f}, val kl: {mean_val_kl:.4f}"
                    )
                    mean_loss = mean_val_loss
                else:
                    mean_val_nll, mean_val_kl = np.nan, np.nan
                    logger.info(
                        f"Epoch {epoch:03d} - training loss: {mean_train_loss:.4f}, "
                        f"train nll: {mean_train_nll:.4f}, train kl: {mean_train_kl:.4f}"
                    )
                    mean_loss = mean_train_loss

                self._metrics_history.append(
                    {
                        "epoch": epoch,
                        "train_loss": float(mean_train_loss),
                        "val_loss": float(mean_loss if val_loader is not None else np.nan),
                        "train_nll": float(mean_train_nll),
                        "train_kl": float(mean_train_kl),
                        "val_nll": float(mean_val_nll),
                        "val_kl": float(mean_val_kl),
                    }
                )
                self._save_epoch_outputs(epoch, val_loader if val_loader is not None else training_loader, use_val=val_loader is not None)

                if np.isnan(mean_loss):
                    logger.warning(f"‼️ Attention: got NaN loss in Epoch {epoch}. This may lead to unexpected errors.")

                if mean_loss < self.best_loss:
                    self.best_epoch = epoch
                    self.best_loss = mean_loss
                    self.best_model_dict = self.model.state_dict()
                    self.patience = self.original_patience
                else:
                    self.patience -= 1

                # save the model if necessary
                self._auto_save_model_if_necessary(
                    confirm_saving=self.best_epoch == epoch and self.model_saving_strategy == "better",
                    saving_name=f"{self.__class__.__name__}_epoch{epoch}_loss{mean_loss:.4f}",
                )

                if os.getenv("enable_tuning", False):
                    nni.report_intermediate_result(mean_loss)
                    if epoch == self.epochs - 1 or self.patience == 0:
                        nni.report_final_result(self.best_loss)

                if self.patience == 0:
                    logger.info("Exceeded the training patience. Terminating the training procedure...")
                    break

        except KeyboardInterrupt:  # if keyboard interrupt, only warning
            logger.warning("‼️ Training got interrupted by the user. Exist now ...")
        except Exception as e:  # other kind of exception follows below processing
            logger.error(f"❌ Exception: {e}")
            if self.best_model_dict is None:  # if no best model, raise error
                raise RuntimeError(
                    "Training got interrupted. Model was not trained. Please investigate the error printed above."
                )
            else:
                RuntimeWarning(
                    "Training got interrupted. Please investigate the error printed above.\n"
                    "Model got trained and will load the best checkpoint so far for testing.\n"
                    "If you don't want it, please try fit() again."
                )

        if np.isnan(self.best_loss):
            raise ValueError("Something is wrong. best_loss is Nan after training.")

        logger.info(f"Finished training. The best model is from epoch#{self.best_epoch}.")
        self._save_metrics_summary()

    def _save_epoch_outputs(self, epoch: int, data_loader: DataLoader, use_val: bool = True) -> None:
        if self._epoch_img_dir is None:
            return
        self.model.eval()
        with torch.no_grad():
            batch = next(iter(data_loader))
            if use_val and len(batch) >= 5:
                inputs = self._assemble_input_for_validating(batch)
                target = inputs["X_ori"]
            else:
                inputs = self._assemble_input_for_training(batch)
                target = inputs["X"]

            results = self.model.forward(inputs, training=False, n_sampling_times=1)
            recon = results["imputed_data"][:, 0]
            x_masked = inputs["X"]

            n = min(self.n_visual_samples, recon.shape[0])
            fig, axes = plt.subplots(n, 3, figsize=(9, 2.5 * n))
            if n == 1:
                axes = np.expand_dims(axes, axis=0)

            for i in range(n):
                axes[i, 0].imshow(x_masked[i, 0].detach().cpu(), cmap="gray", vmin=0, vmax=1)
                axes[i, 0].set_title("Input")
                axes[i, 1].imshow(recon[i, 0].detach().cpu(), cmap="gray", vmin=0, vmax=1)
                axes[i, 1].set_title("Reconstruction")
                axes[i, 2].imshow(target[i, 0].detach().cpu(), cmap="gray", vmin=0, vmax=1)
                axes[i, 2].set_title("Target")
                for j in range(3):
                    axes[i, j].axis("off")

            plt.tight_layout()
            out_path = os.path.join(self._epoch_img_dir, f"epoch_{epoch:03d}.png")
            fig.savefig(out_path, dpi=120)
            plt.close(fig)

    def _save_metrics_summary(self) -> None:
        if self.visualization_dir is None or len(self._metrics_history) == 0:
            return

        csv_path = os.path.join(self.visualization_dir, "metrics.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["epoch", "train_loss", "val_loss", "train_nll", "train_kl", "val_nll", "val_kl"],
            )
            writer.writeheader()
            writer.writerows(self._metrics_history)

        epochs = [m["epoch"] for m in self._metrics_history]
        train_nll = [m["train_nll"] for m in self._metrics_history]
        val_nll = [m["val_nll"] for m in self._metrics_history]
        train_kl = [m["train_kl"] for m in self._metrics_history]
        val_kl = [m["val_kl"] for m in self._metrics_history]

        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].plot(epochs, train_nll, label="train_nll")
        if np.isfinite(np.array(val_nll)).any():
            axes[0].plot(epochs, val_nll, label="val_nll")
        axes[0].set_title("NLL by Epoch")
        axes[0].set_xlabel("Epoch")
        axes[0].legend()

        axes[1].plot(epochs, train_kl, label="train_kl")
        if np.isfinite(np.array(val_kl)).any():
            axes[1].plot(epochs, val_kl, label="val_kl")
        axes[1].set_title("KL by Epoch")
        axes[1].set_xlabel("Epoch")
        axes[1].legend()

        plt.tight_layout()
        fig.savefig(os.path.join(self.visualization_dir, "nll_kl_curves.png"), dpi=140)
        plt.close(fig)

    def _train_kernel(
        self,
        training_loader: DataLoader,
        val_loader: DataLoader = None,
    ) -> None:
        # each training starts from the very beginning, so reset the loss and model dict here
        self.best_kernel_loss = float("inf")
        self.best_kernel_model_dict = None
        self.kernel_patience = self.original_patience

        try:
            training_step = 0.
            for epoch in range(1, self.epochs + 1):

                #self.model.train()
                #self.model.backbone.temperature = epoch / (self.epochs + 1)
                epoch_train_loss_collector = []
                for idx, data in enumerate(training_loader):
                    training_step += 1

                    inputs = self._assemble_input_for_training(data)

                    self.model.gp.optimizer.zero_grad()

                    results = self.model.forward_for_GP(inputs, n_sampling_times=3)

                    # use sum() before backward() in case of multi-gpu training
                    results["loss"].mean().clip(min=0,max=10).backward()

                    #clip gradients
                    #torch.nn.utils.clip_grad_norm_(v_1, max_norm=1.0, norm_type=2)
                    self.model.gp.optimizer.step()

                    epoch_train_loss_collector.append(results["loss"].sum().item())

                    # save training loss logs into the tensorboard file for every step if in need
                    #if self.summary_writer is not None:
                    #    self._save_log_into_tb_file(training_step, "training", results)

                    # Gradient Clipping
                    with torch.no_grad():
                        for param in self.model.gp.kernel_params_estimator.parameters():
                            param.clamp_(-3, 3)

                # mean training loss of the current epoch
                mean_train_loss = np.mean(epoch_train_loss_collector)

                if val_loader is not None:
                    #self.model.gp.eval()
                    imputation_loss_collector = []
                    with torch.no_grad():
                        for idx, data in enumerate(val_loader):

                            inputs = self._assemble_input_for_validating(data)

                            results = self.model.forward_for_GP(inputs, training=False, n_sampling_times=3)

                            imputation_loss_collector.append(results['loss'].mean().item())

                    mean_val_loss = np.mean(imputation_loss_collector)

                mean_loss = mean_val_loss if val_loader is not None else mean_train_loss

                if val_loader:
                    logger.info(
                        f"Kernel epoch {epoch:03d} - training loss: {mean_train_loss:.6f}, "
                        f"validation loss: {mean_val_loss:.6f}"
                    )
                else:
                    logger.info(f"Kernel epoch {epoch:03d} - training loss: {mean_train_loss:.6f}")


                if mean_loss < self.best_kernel_loss:
                    self.best_kernel_epoch = epoch
                    self.best_kernel_loss = mean_loss
                    self.best_kernel_model_dict = self.model.state_dict()
                    self.kernel_patience = self.original_patience
                else:
                    self.kernel_patience -= 1

                if os.getenv("enable_tuning", False):
                    nni.report_intermediate_result(mean_loss)
                    if epoch == self.epochs - 1 or self.patience == 0:
                        nni.report_final_result(self.best_loss)

                if self.kernel_patience == 0:
                    logger.info("Exceeded the training patience. Terminating the training procedure...")
                    break

        except KeyboardInterrupt:  # if keyboard interrupt, only warning
            logger.warning("‼️ Training got interrupted by the user. Exist now ...")
        except Exception as e:  # other kind of exception follows below processing
            logger.error(f"❌ Exception: {e}")
            if self.best_kernel_model_dict is None:  # if no best model, raise error
                raise RuntimeError(
                    "Training got interrupted. Model was not trained. Please investigate the error printed above."
                )
            else:
                RuntimeWarning(
                    "Training got interrupted. Please investigate the error printed above.\n"
                    "Model got trained and will load the best checkpoint so far for testing.\n"
                    "If you don't want it, please try fit() again."
                )

        if np.isnan(self.best_kernel_loss):
            raise ValueError("Something is wrong. best_loss is Nan after training.")

        logger.info(f"Finished training. The best model is from epoch#{self.best_kernel_epoch}.")

    def fit(
        self,
        train_set: Union[dict, str],
        val_set: Optional[Union[dict, str]] = None,
        file_type: str = "hdf5",
        train_kernel_only = False
    ) -> None:
        if not isinstance(train_set, dict):
            raise TypeError("GP_UAE2.fit expects `train_set` as a dict with keys `X` and `mask`.")
        required_keys = {"X", "mask"}
        missing = required_keys.difference(train_set.keys())
        if missing:
            raise ValueError(f"train_set is missing required keys: {sorted(missing)}")

        # Step 1: wrap the input data with classes Dataset and DataLoader
        training_set = DatasetForGPVAE(train_set, return_X_ori=False, return_y=False, file_type=file_type)
        training_loader = DataLoader(
            training_set,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )
        val_loader = None
        if val_set is not None:
            if not isinstance(val_set, dict):
                raise TypeError("GP_UAE2.fit expects `val_set` as a dict when provided.")
            required_val_keys = {"X", "mask", "X_ori", "indicating_mask"}
            missing_val = required_val_keys.difference(val_set.keys())
            if missing_val:
                raise ValueError(f"val_set is missing required keys: {sorted(missing_val)}")
            val_set = DatasetForGPVAE(val_set, return_X_ori=True, return_y=False, file_type=file_type)
            val_loader = DataLoader(
                val_set,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
            )
        
        if not train_kernel_only:
            # Step 2: train the AE model and freeze it
            self._train_model(training_loader, val_loader)
            self.model.load_state_dict(self.best_model_dict)
            self.model.eval()  # set the model as eval status to freeze it.

            # Step 3: save the model if necessary
            self._auto_save_model_if_necessary(confirm_saving=self.model_saving_strategy == "best")

        # Step 2bis: learn the kernel
        else:
            logger.info("Training GP kernel only...")
            self._train_kernel(training_loader, val_loader)

    def impute_with_gp(self,
                    inputs,
                    add_mcar = False,
                    gp = False,
                    n_sampling_times=1):

        #results = self.model.forward(inputs, training=False, n_sampling_times=n_sampling_times)
        #imputed_data = results["imputed_data"]

        # embed data in latent space
        x = inputs['X']
        mask = inputs["mask"]
        if add_mcar:
            keep = (torch.rand_like(mask) > self.p).to(mask.dtype)
            mask_corr = mask * keep
            x_corr = x * mask_corr
            qz_x = self.model.encode({"X": x_corr, "mask": mask_corr}, training=False, n_sampling_times=n_sampling_times)
            x = x_corr
            #kernel_params = self.update_kernel_params(x_corr)
        else:
            qz_x = self.model.encode({"X": x, "mask": mask}, training=False, n_sampling_times=n_sampling_times)
            #kernel_params = self.update_kernel_params(x)
        #z_mu, z_var = qz_x.mean.detach(), qz_x.variance.detach()

        if self.use_gp or gp:

            imputed_data = self.gp.infer(qz_x, x, mask=mask)

        else:

            #z_star = qz_x.rsample()
            z_star = qz_x.mean
            imputed_data = self.model.backbone.decode(z_star).mean

        return imputed_data

    def predict(
        self,
        test_set: Union[dict, str],
        file_type: str = "hdf5",
        n_sampling_times: int = 1,
        with_gp = False
    ) -> dict:
        """

        Parameters
        ----------
        test_set : dict or str
            The dataset for model validating, should be a dictionary including keys as 'X' and 'y',
            or a path string locating a data file.
            If it is a dict, X should be array-like of shape [n_samples, sequence length (n_steps), n_features],
            which is time-series data for validating, can contain missing values, and y should be array-like of shape
            [n_samples], which is classification labels of X.
            If it is a path string, the path should point to a data file, e.g. a h5 file, which contains
            key-value pairs like a dict, and it has to include keys as 'X' and 'y'.

        file_type :
            The type of the given file if test_set is a path string.

        n_sampling_times:
            The number of sampling times for the model to produce predictions.

        Returns
        -------
        result_dict: dict
            Prediction results in a Python Dictionary for the given samples.
            It should be a dictionary including a key named 'imputation'.

        """
        assert n_sampling_times > 0, "n_sampling_times should be greater than 0."

        self.model.eval()  # set the model as eval status to freeze it.
        test_set = DatasetForGPVAE(test_set, return_X_ori=False, return_y=False, file_type=file_type)
        test_loader = DataLoader(
            test_set,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )
        imputation_collector = []

        with torch.no_grad():
            for idx, data in enumerate(test_loader):

                inputs = self._assemble_input_for_testing(data)
                imputed_data = self.impute_with_gp(inputs, gp=with_gp, n_sampling_times=n_sampling_times)
                imputation_collector.append(imputed_data)

        imputation = torch.cat(imputation_collector).cpu().detach().numpy()
        result_dict = {
            "imputation": imputation,
        }
        return result_dict

    def impute(
        self,
        test_set: Union[dict, str],
        file_type: str = "hdf5",
        with_gp = False
    ) -> np.ndarray:
        """Impute missing values in the given data with the trained model.

        Parameters
        ----------
        test_set :
            The data samples for testing, should be array-like of shape [n_samples, sequence length (n_steps),
            n_features], or a path string locating a data file, e.g. h5 file.

        file_type :
            The type of the given file if X is a path string.

        Returns
        -------
        array-like, shape [n_samples, sequence length (n_steps), n_features],
            Imputed data.
        """

        results_dict = self.predict(test_set, file_type=file_type, with_gp = with_gp)
        return results_dict["imputation"]

    def save_weights(self, path: str, include_optimizer: bool = True) -> None:
        """Save model weights/state-dicts for transfer learning."""
        save_dir = os.path.dirname(path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        payload = {
            "model_state_dict": self.model.state_dict(),
            "backbone_state_dict": self.model.backbone.state_dict(),
            "config": {
                "n_steps": self.n_steps,
                "n_features": self.n_features,
                "latent_size": self.latent_size,
            },
        }
        if include_optimizer and self.optimizer is not None:
            try:
                payload["optimizer_state_dict"] = self.optimizer.torch_optimizer.state_dict()
            except Exception:
                pass
        torch.save(payload, path)
        logger.info(f"Saved GP_UAE2 weights to {path}")

    def load_weights(
        self,
        path: str,
        load_backbone_only: bool = False,
        freeze_backbone: bool = False,
        strict: bool = False,
    ) -> None:
        """Load model weights/state-dicts, optionally only the backbone."""
        map_location = self.device if isinstance(self.device, torch.device) else torch.device("cpu")
        checkpoint = torch.load(path, map_location=map_location)

        if isinstance(checkpoint, dict):
            model_sd = checkpoint.get("model_state_dict")
            backbone_sd = checkpoint.get("backbone_state_dict")
        else:
            model_sd = checkpoint
            backbone_sd = checkpoint

        if load_backbone_only:
            if backbone_sd is None:
                raise ValueError(f"No backbone_state_dict found in {path}")
            self.model.backbone.load_state_dict(backbone_sd, strict=strict)
            logger.info(f"Loaded backbone weights from {path}")
        else:
            if model_sd is None:
                raise ValueError(f"No model_state_dict found in {path}")
            self.model.load_state_dict(model_sd, strict=strict)
            logger.info(f"Loaded full model weights from {path}")

        if freeze_backbone:
            for p in self.model.backbone.parameters():
                p.requires_grad = False
            logger.info("Backbone parameters frozen after loading.")
