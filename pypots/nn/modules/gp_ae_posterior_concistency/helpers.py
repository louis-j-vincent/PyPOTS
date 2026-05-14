import matplotlib.pyplot as plt

class Plotter():

    def __init__(self, model) -> None:
        self.model = model

    def plot_params(self):

        max_points = 500
        num_points = len(self.loss_history['elbo'])
        n = max(1, num_points // max_points)  # Ensure n is at least 1

        # Create the iterations range with step size n
        iterations = range(1, num_points + 1)[::n]

        plt.figure(figsize=(10, 6))

        params_to_plot = [ #'z_mu_mean' ,
            'z_mu_var', 
            'z_var_mean', 
            'z_var_var'
            ]

        # Define a decimation factor for plotting
        n = 10  # Plot every nth point to reduce visual clutter

        # Create a figure for the plots
        plt.figure(figsize=(12, 8))

        # Iterate through parameters and plot them
        style = ['-','o','+',':']
        for i,param in enumerate(params_to_plot):
            plt.semilogy(np.abs(np.array(self.monitoring_history[param])[::n]), style[i])
            plt.plot([], style[i], label=param)

        # Add labels, legend, and title
        plt.xlabel('Iterations')
        plt.ylabel('Value')
        plt.title('Monitored Parameters Over Training')
        plt.legend(bbox_to_anchor=[1.2, 0.3])
        plt.grid()

        plot_path = os.path.join('latent_plots/params.png')

        # Save the plot
        plt.tight_layout()
        plt.savefig(plot_path)
        plt.close()

    def plot_losses(self):
        # Calculate the step size n to ensure we have a maximum of 500 points plotted
        max_points = 500
        num_points = len(self.loss_history['elbo'])
        n = max(1, num_points // max_points)  # Ensure n is at least 1

        # Create the iterations range with step size n
        iterations = range(1, num_points + 1)[::n]

        plt.figure(figsize=(10, 6))

        # Plot each loss component with downsampling
        plt.semilogy(iterations, self.loss_history['elbo'][::n], label='ELBO')
        plt.semilogy(iterations, self.loss_history['nll_recon'][::n], label='NLL Recon')
        plt.semilogy(iterations, self.loss_history['nll_imputation'][::n], label='NLL Imputation')
        plt.semilogy(iterations, self.loss_history['nll_sampling'][::n], label='NLL Sampling')
        plt.semilogy(iterations, self.loss_history['kl'][::n], label='KL Divergence')
        plt.semilogy(iterations, self.loss_history['temporal_loss'][::n], label='Temporal Loss')
        # plt.semilogy(iterations, self.loss_history['dependence_loss'][::n], label='Dependence Loss')

        plt.xlabel('Iteration')
        plt.ylabel('Loss (log scale)')
        plt.title('Loss Components over Iterations')
        plt.legend(bbox_to_anchor=[1.2, 0.3])
        plt.grid(True)

        plot_path = os.path.join('latent_plots/losses.png')

        # Save the plot
        plt.tight_layout()
        plt.savefig(plot_path)
        plt.close()

    def plot_latent_series_and_reconstruction(self, z, px_z, qz_x, X_ori, X, latent_dim, time_steps, nll, kl, tl, folder='latent_plots'):
        """
        Plots the mean and variance of all latent time series and the original vs reconstructed data.
        """
        # Convert to CPU numpy arrays for plotting
        z_mean = qz_x.mean[0].detach().cpu().numpy()
        z_var = qz_x.variance[0].detach().cpu().numpy()

        X_ori_np = torch.clone(X_ori).detach().cpu().numpy()
        X_np = X.detach().cpu().numpy()

        z_mean_ori = self.encode(X_ori).mean.detach().cpu()
        z_var_ori = self.encode(X_ori).variance.detach().cpu()


        # Sample 10 times from the posterior to get 10 reconstructions
        reconstructions = []
        for i in range(10):
            z_sample = qz_x.rsample()  # Sample from the posterior
            px_z_sample = self.decode(z_sample)  # Reconstruct the data
            X_recon_sample = px_z_sample.mean.detach().cpu().numpy()  # Get the mean of the reconstruction
            # Set first half of non-reconstructed values to NaN
            num_missing_vals = np.sum(X_ori_np == 0)
            X_recon_sample[X_ori_np == 0][:num_missing_vals // 2] == np.nan
            reconstructions.append(X_recon_sample)

        X_ori_np[X_ori_np == 0] = np.nan

        # Create directory if it doesn't exist
        if not os.path.exists(folder):
            os.makedirs(folder)

        # Create a unique filename based on the current timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        plot_path = os.path.join(folder, f'latent_series_and_reconstruction_{timestamp}.png')

        # Plot all latent dimensions' mean and variance
        plt.figure(figsize=(15, 12))

        plt.subplot(4, 1, 4)
        mask, mask_ori = (X != 0).to(self.device), (X_ori != 0).to(self.device)
        imputation_error = self.latent_imputation_error(qz_x, X_ori, mask, mask_ori, for_plotting=True)[0].detach().cpu().numpy()
        plt.semilogy(imputation_error)
        plt.title('Log probability of original z belonging to the corrupted Gaussian')

        losses = f'kl = {kl.mean().item()} - nll = {nll.mean().item()} - temporal {tl.item()}'
        plt.subplot(4, 1, 1)
        colors = ['purple', 'brown', 'orange', 'b','g','r','y']
        for dim in range(latent_dim):
            plt.plot(range(time_steps), z_mean[:, dim], label=f'Latent dim {dim} Mean', color = colors[dim])
            plt.fill_between(range(time_steps), z_mean[:, dim] - z_var[:, dim] ** 0.5, z_mean[:, dim] + z_var[:, dim] ** 0.5, alpha=0.2, color = colors[dim])
            plt.scatter(range(time_steps), z_mean_ori[0][:, dim].numpy(), label=f'Latent dim {dim} Mean', color = colors[dim])
            plt.fill_between(range(time_steps), z_mean_ori[0][:, dim] - z_var_ori[0][:, dim] ** 0.5, z_mean_ori[0][:, dim] + z_var_ori[0][:, dim] ** 0.5, alpha=0.2, color = 'gray')
        plt.plot([], color = 'gray', alpha = .2, label = 'Z original variance')
        plt.plot([], color = 'k', alpha = .2, label = 'Z corrupted variance')


        plt.title('Latent Time Series (Mean and Variance) ' + losses)
        plt.xlabel('Time Steps')
        plt.ylabel('Latent Values')
        plt.legend(bbox_to_anchor=[1.2, 0.3])

        # Plot scales for prior and posterior
        plt.subplot(4, 1, 2)
        nb_missing_vals = (X_np[0] == 0).sum(axis=1)
        missing_ratio = (X_np[0] != 0).mean(axis=1)
        prior_scale = (1 - missing_ratio) ** 0.5
        plt.semilogy(z_var, alpha=0.5)
        plt.semilogy((z_mean - z_mean_ori[0].numpy()) ** 2, 'o', alpha=0.5, label = 'l2 errors')
        plt.semilogy(prior_scale, label='Prior scale')
        plt.semilogy(np.linalg.norm(z_var, axis=1), 'r:', label='Posterior scale')
        plt.legend(bbox_to_anchor=[1.2, 0.3])

        # Plot original vs reconstructed data for all 10 reconstructions
        plt.subplot(4, 1, 3)
        for i, X_recon_sample in enumerate(reconstructions):
            plt.plot(range(time_steps), X_recon_sample[0, :, :], alpha=0.6)

        X_np[X_np==0] = np.nan

        plt.gca().set_prop_cycle(None)
        plt.plot(range(time_steps), X_np[0, :, :], 'o')
        plt.gca().set_prop_cycle(None)
        plt.plot(range(time_steps), X_ori_np[0, :, :], '+', label='Original Data')
        plt.title('Original vs Reconstructed Data (10 Samples)')
        plt.xlabel('Time Steps')
        plt.ylabel('Feature Values')

        # Save the plot
        plt.tight_layout()
        plt.savefig(plot_path)
        plt.close()
