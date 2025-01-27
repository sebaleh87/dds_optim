import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import os
import wandb
import re
from sklearn.manifold import TSNE

# Base Class
class EnergyModelClass:

    def __init__(self,EnergyConfig):
        self.config = EnergyConfig
        self.dim_x = EnergyConfig["dim_x"]
        self.has_tractable_distribution = False  # Default to False, override in subclasses

        if("shift" in EnergyConfig.keys()):
            self.shift = EnergyConfig["shift"]
        else:
            self.shift = 0.

        self.x_min = -5 + self.shift
        self.y_min = -5 + self.shift
        self.x_max = 5 + self.shift
        self.y_max = 5 + self.shift
        self.latent_dim = self.dim_x
        self.levels = 100
        self.invariance = False
        self.scaling = self.config["scaling"]*jnp.ones((self.dim_x))
        self.Energy_params = self.init_EnergyParams()
        self.dataset_file = ""
        ### TODO define plot range here

    #TODO why needed?
    def init_EnergyParams(self):
        return {"log_var_x": jnp.log(1.)*jnp.ones((self.dim_x))}

    def energy_function(self, x):       #TODO: check if jit pays off 
        """
        This method should be overridden by subclasses to define
        the specific energy function.
        """
        raise NotImplementedError("Subclasses should implement this method")
    
    def calc_energy(self, diff_samples, energy_params, key):
        y, key = self.scale_samples(diff_samples, energy_params, key)

        return self.energy_function(y), key

    def vmap_calc_energy(self, diff_samples, energy_params, key):
        key, subkey = jax.random.split(key)
        batched_key = jax.random.split(subkey, diff_samples.shape[0])
        energy_value, batched_key = jax.vmap(self.calc_energy, in_axes=(0, None, 0))(diff_samples, energy_params, batched_key)
        return energy_value, key
    
    def scale_samples(self, diff_samples, energy_params, key):
        Y = diff_samples#/self.scaling
        return Y, key


    def calc_log_probs(self, x, T):
        """
        Calculate the log probabilities, where T is the temperature.
        log_probs = -1/T * energy(x)
        """
        energy, _ = self.calc_energy(x, None, None)
        return -1.0 / T * energy
    
    def plot_properties(self):
        if(self.dim_x == 2):
            return self.plot_2_D_properties()
        elif(self.dim_x == 1):
            return self.plot_1d_properties()
        else:
            pass

    def plot_2_D_properties(self, resolution=100, T=1.0 , panel = "fig"):
        """
        Plot both the energy landscape and the log probability landscape.
        """
        # Create a grid of points over the specified range
        x = jnp.linspace(self.x_min, self.x_max, resolution) 
        y = jnp.linspace(self.y_min, self.y_max, resolution)
        X, Y = jnp.meshgrid(x, y)
        
        # Stack X and Y into a 2D array of coordinates for vectorized evaluation
        grid_points = jnp.stack([X.ravel(), Y.ravel()], axis=-1)
        
        # Calculate energy and log-probs for each point in the grid
        # Z_energy = jax.vmap(self.energy_function)(grid_points)
        Z_energy = jax.vmap(self.calc_energy)(grid_points)
        Z_log_probs = jax.vmap(lambda pt: self.calc_log_probs(pt, T))(grid_points)
        
        # Reshape results back into a 2D grid
        Z_energy = Z_energy.reshape(X.shape)
        Z_log_probs = Z_log_probs.reshape(X.shape)

        # Plotting
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # Energy landscape
        energy_plot = ax1.contourf(X, Y, Z_energy, levels=self.levels, cmap='viridis')
        fig.colorbar(energy_plot, ax=ax1)
        ax1.set_title("Energy Landscape")
        ax1.set_xlabel("x")
        ax1.set_ylabel("y")

        # Log probability landscape
        log_probs_plot = ax2.contourf(X, Y, Z_log_probs, levels=self.levels, cmap='plasma')
        fig.colorbar(log_probs_plot, ax=ax2)
        ax2.set_title("Log Probability Landscape")
        ax2.set_xlabel("x")
        ax2.set_ylabel("y")

        current_file_path = os.path.abspath(__file__)

        # Get the parent directory of the current file
        parent_folder = os.path.dirname(os.path.dirname(current_file_path))
        plt.tight_layout()

        wfig = wandb.Image(fig)
        wandb.log({f"{panel}/Energy_Landscape": wfig})
        plt.close()
        return wfig

    def plot_1d_properties(self, x_range=(-10, 10), resolution=100, T=1.0, panel = "fig"):
        """
        Plot both the energy landscape and the log probability landscape for a 1-D energy function.
        """
        # Create a range of points over the specified range
        x = jnp.linspace(x_range[0], x_range[1], resolution)
        
        # Calculate energy and log-probs for each point in the range
        # Z_energy = jax.vmap(self.energy_function)(x)
        Z_energy = jax.vmap(self.calc_energy)(x)
        Z_log_probs = jax.vmap(lambda pt: self.calc_log_probs(pt, T))(x)

        # Plotting
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # Energy landscape
        ax1.plot(x, Z_energy, label="Energy", color='blue')
        ax1.set_title("Energy Landscape")
        ax1.set_xlabel("x")
        ax1.set_ylabel("Energy")
        ax1.legend()

        # Log probability landscape
        ax2.plot(x, Z_log_probs, label="Log Probability", color='red')
        ax2.set_title("Log Probability Landscape")
        ax2.set_xlabel("x")
        ax2.set_ylabel("Log Probability")
        ax2.legend()

        plt.tight_layout()
        wfig = wandb.Image(fig)
        wandb.log({f"{panel}/Energy_Landscape": wfig})
        plt.show()
        return wfig

    def plot_trajectories(self, Xs, panel = "fig"):
        if(self.dim_x == 2):
            return {"trajectories": self.plot_2_D_trajectories(Xs, panel = panel)}
        elif(self.dim_x == 1):
            return {"trajectories": self.plot_1_D_trajectories(Xs, panel = panel)}
        else:
            pass

    def plot_last_samples(self, Xs, panel = "fig"):
        if(self.dim_x == 2):
            return {"last_samples": self.plot_2_D_last_samples(Xs, panel = panel)}
        elif(self.dim_x == 1):
            pass
        elif( hasattr(self, 'n_particles')):
            data = self.load_data()
            return {"interatomic_distances": self.plot_interatomic_distances(Xs, data, panel=panel), "energy_histogram": self.plot_energy_histogram(Xs, data, panel=panel)}
        elif(True):
            return {"tsne": self.plot_tsne_last_samples(Xs, panel=panel)}
        else:
            pass

    def plot_interatomic_distances(self, x, data, panel = "fig"):
        data_samples = data.reshape((data.shape[0], self.n_particles, -1))

        fig = plt.figure()
        x_resh = x.reshape(-1, self.n_particles, self.particle_dim)
        mask = 1 - jnp.repeat(jnp.eye(self.n_particles)[None, ...], x.shape[0], axis = 0)

        data_mask = 1 - jnp.repeat(jnp.eye(self.n_particles)[None, ...], data.shape[0], axis = 0)

        d_ij = np.sqrt(np.sum((x_resh[:, None, :, :] - x_resh[:, :, None, :]) ** 2 , axis=-1))
        d_ij = d_ij[mask == 1]

        data_d_ij = np.sqrt(np.sum((data_samples[:, None, :, :] - data_samples[:, :, None, :]) ** 2 , axis=-1))
        data_d_ij = data_d_ij[data_mask == 1]

        plt.hist(d_ij.flatten(), bins=100, density=True, color='blue', alpha=0.5, edgecolor='blue', label = "model")
        plt.hist(data_d_ij.flatten(), bins=100, density=True, color='red', alpha=0.5, edgecolor='red', label = "data")
        plt.xlabel("Interatomic distance")
        plt.ylabel("Density")
        plt.legend()
        
        wfig = wandb.Image(fig)
        #wandb.log({f"{panel}/Energy_Landscape": wandb.Image(fig)})
        plt.close()
        return wfig


    def load_data(self):
        s = "test"
        el = self.dataset_file
        if(el != ""):
            match = re.search(r'\d+', el)
            if match:
                n_paricles = int(match.group())
            else:
                raise ValueError(f"No number found in string: {el}")

            split_el = np.load(f'/system/user/publicwork/sanokows/Denoising_diff_sampler/Data/{s}_split_{el}.npy')
            print(f"Loaded {s} with shape: {split_el.shape}")
            return split_el

    def plot_energy_histogram(self, x, data, panel="fig"):
        """
        Compute the energy values of x and plot a histogram of these energies.
        """
        # Compute the energy values
        energy_values, _ = self.vmap_calc_energy(x, self.Energy_params, jax.random.PRNGKey(0))
        data_energy_values, _ = self.vmap_calc_energy(data, self.Energy_params, jax.random.PRNGKey(0))

        # Plot the histogram
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(energy_values, bins=100, density=True, color='blue', alpha=0.5, edgecolor='blue', label = "model")
        
        ax.hist(data_energy_values, bins=100, density=True, color='red', alpha=0.5, edgecolor='red', label = "data")
        ax.set_title('Histogram of Energy Values')
        ax.set_xlabel('Energy')
        ax.set_ylabel('Density')
        ax.grid(True)
        plt.legend()

        # Log the figure using wandb
        wfig = wandb.Image(fig)
        wandb.log({f"{panel}/energy_histogram": wfig})
        plt.close()

        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.hist(data_energy_values, bins=100, density=True, color='red', alpha=0.5, edgecolor='red', label = "data")
        ax.set_title('Histogram of Energy Values')
        ax.set_xlabel('Energy')
        ax.set_ylabel('Density')
        ax.grid(True)
        plt.legend()

        # Log the figure using wandb
        wfig = wandb.Image(fig)
        #wandb.log({f"{panel}/energy_gt_histogram": wfig})
        plt.close()
        return wfig




    def plot_histogram(self, Xs, panel = "fig"):
        if(self.dim_x == 2):
            return {"sample_histogram": self.plot_2_D_histogram(Xs, panel = panel)}
        elif(self.dim_x == 1):
            return {"sample_histogram": self.plot_1_D_histogram(Xs, panel = panel)}
        else:
            pass
    
    def plot_1_D_trajectories(self, Xs, panel = "fig"):
        T, B, D = Xs.shape
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.plot(np.linspace(0,1,T), Xs[...,0], label=f'Trajectory')
        
        ax.set_xlabel('Time')
        ax.set_ylabel('Value')
        ax.set_title('1D Trajectories Over Time')
        ax.grid(True)

        #wandb.log({f"{panel}/1d_trajectories": wandb.Image(fig)})
        plt.close()
        return wandb.Image(fig)

    def plot_2_D_trajectories(self, Xs, panel = "fig"):
        T, B, dim = Xs.shape
        if dim != 2:
            raise ValueError("The dimension of the trajectories must be 2.")
        
        fig = plt.figure(figsize=(10, 6))
        # Create a grid of points over the specified range
        x = jnp.linspace(self.x_min, self.x_max, 100) 
        y = jnp.linspace(self.y_min, self.y_max, 100)
        X, Y = jnp.meshgrid(x, y)

        # Stack X and Y into a 2D array of coordinates for vectorized evaluation
        grid_points = jnp.stack([X.ravel(), Y.ravel()], axis=-1)

        # Calculate log probabilities for each point in the grid
        Z_log_probs = jax.vmap(lambda pt: self.calc_log_probs(pt, 1.0))(grid_points)

        # Reshape results back into a 2D grid
        Z_log_probs = Z_log_probs.reshape(X.shape)

        # Plot the log probability landscape in the background
        log_probs_plot = plt.contourf(X, Y, Z_log_probs, levels=self.levels, cmap='Blues', alpha=0.3)
        plt.colorbar(log_probs_plot, label='Log Probability')

        for b in range(B):
            trajectory = Xs[:, b, :]
            plt.plot(trajectory[:, 0], trajectory[:, 1], label=f'Trajectory {b+1}')

        
        
        plt.xlabel('X-axis')
        plt.ylabel('Y-axis')
        plt.title('2D Trajectories')
        plt.legend()
        plt.grid(True)

        wfig = wandb.Image(fig)
        #wandb.log({f"{panel}/trajectories": wfig})
        plt.close()
        return wfig

    def plot_tsne_last_samples(self, Xs, panel = "fig"):
        print("TSNE plot")
        Xs = Xs[:2000]
        if(hasattr(self, 'means')):
            means_reshaped = self.means.reshape(-1, self.dim_x)
            Xs = np.concatenate([Xs, means_reshaped], axis=0)
        tsne = TSNE(n_components=2, random_state=0)
        Xs_reshaped = Xs.reshape(-1, self.dim_x)
        Xs_embedded = tsne.fit_transform(Xs_reshaped)
        fig = plt.figure(figsize=(10, 6))
        plt.scatter(Xs_embedded[:, 0], Xs_embedded[:, 1], alpha=0.5)
        if hasattr(self, 'means'):
            means_embedded = Xs_embedded[-self.means.shape[0]:]
            plt.scatter(means_embedded[:, 0], means_embedded[:, 1], color='orange', alpha=1., label='Means')
        plt.xlabel('t-SNE 1')
        plt.ylabel('t-SNE 2')
        plt.title('t-SNE of Last Samples')
        plt.grid(True)
        wfig = wandb.Image(fig)
        #wandb.log({f"{panel}/tsne_last_samples": wfig})
        plt.close()
        return wfig
    
    def plot_2_D_last_samples(self, Xs, panel = "fig"):
        fig = plt.figure(figsize=(10, 6))

        # Create a grid of points over the specified range
        x = jnp.linspace(self.x_min, self.x_max, 100) 
        y = jnp.linspace(self.y_min, self.x_max, 100)
        X, Y = jnp.meshgrid(x, y)

        # Stack X and Y into a 2D array of coordinates for vectorized evaluation
        grid_points = jnp.stack([X.ravel(), Y.ravel()], axis=-1)

        # Calculate energy for each point in the grid
        Z_energy = jax.vmap(lambda pt: self.calc_log_probs(pt, 1.0))(grid_points)#jax.vmap(self.energy_function)(grid_points)

        # Reshape results back into a 2D grid
        Z_energy = Z_energy.reshape(X.shape)

        # Plot the energy landscape in the background
        plt.plot(Xs[:,0], Xs[:,1], "o", alpha=0.15)
        energy_plot = plt.contourf(X, Y, Z_energy, levels=self.levels, cmap='Reds', alpha=0.3)
        plt.colorbar(energy_plot, label='log probs')

        plt.xlabel('X-axis')
        plt.ylabel('Y-axis')
        plt.title('2D last samples')
        plt.grid(True)
        plt.xlim(self.x_min, self.x_max)
        plt.ylim(self.y_min, self.y_max)

        wfig = wandb.Image(fig)
        #wandb.log({f"{panel}/last_samples": wfig})
        plt.close()
        return wfig

    def plot_1_D_histogram(self, samples, panel = "fig"):
                # Calculate the histogram
        hist, bin_edges = np.histogram(samples, bins=100, density=True)

        # Filter out bins with small likelihood
        threshold = 0.01  # Define a threshold for small likelihood
        filtered_indices = hist > threshold
        filtered_bin_edges = bin_edges[:-1][filtered_indices]
        filtered_hist = hist[filtered_indices]

        # Plot the filtered histogram
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(filtered_bin_edges, filtered_hist, drawstyle='steps-post', color='blue')

        ax.set_title('1D Histogram with Likelihood')
        ax.set_xlabel('Value')
        ax.set_ylabel('Likelihood')
        ax.grid(True)

        # Log the figure using wandb
        wfig = wandb.Image(fig)
        #wandb.log({f"{panel}/1d_histogram": wfig})
        plt.close()
        return wfig


    def plot_2_D_histogram(self, samples, n_bins = 80, panel = "fig"):
        # Filter samples where both coordinates are within [-4, 4]
        filtered_samples = samples[(samples[:, 0] >= self.x_min) & (samples[:, 0] <= self.x_max) & (samples[:, 1] >= self.y_min) & (samples[:, 1] <= self.y_max)]
        # Assuming `samples` is provided and wandb is initialized

        # Create 2D histogram
        x_samples = filtered_samples[:, 0]
        y_samples = filtered_samples[:, 1]

        # Create a grid of points over the specified range
        x = jnp.linspace(self.x_min, self.x_max, n_bins) 
        y = jnp.linspace(self.y_min, self.y_max, n_bins)
        X, Y = jnp.meshgrid(x, y)
        
        # Stack X and Y into a 2D array of coordinates for vectorized evaluation
        grid_points = jnp.stack([X.ravel(), Y.ravel()], axis=-1)
        
        # Calculate energy for each point in the grid
        Z_energy = jax.vmap(lambda pt: self.calc_log_probs(pt, 1.0))(grid_points)#jax.vmap(self.energy_function)(grid_points)
        
        # Reshape results back into a 2D grid
        Z_energy = Z_energy.reshape(X.shape)

        # Plot the energy landscape in the background
        fig, ax = plt.subplots(figsize=(10, 8))
        energy_plot = ax.contourf(X, Y, Z_energy, levels=self.levels, cmap='Reds', alpha=0.6)
        fig.colorbar(energy_plot, ax=ax, label='log probs')
        
        # Plot the zoomed-in heatmap
        hist2d = ax.hist2d(x_samples, y_samples, bins=n_bins, cmap='Blues', alpha=0.7)
        fig.colorbar(hist2d[3], ax=ax, label='Likelihood')
        
        ax.set_xlim(xmin=self.x_min, xmax=self.x_max)
        ax.set_ylim(ymin=self.y_min, ymax=self.y_max)
        ax.set_title('Zoomed 2D Histogram with Energy Landscape')
        ax.set_xlabel('X-axis')
        ax.set_ylabel('Y-axis')
        plt.tight_layout()

        # Log the figure using wandb
        wfig = wandb.Image(fig)
        #wandb.log({f"{panel}/2d_histogram": wfig})
        plt.close()
        return wfig

    def generate_samples(self, key, n_samples):
        """
        Generate samples from the tractable distribution if available.
        Should be implemented by subclasses that have tractable distributions.
        """
        if not self.has_tractable_distribution:
            raise NotImplementedError("This energy function does not have a tractable distribution")
        raise NotImplementedError("Subclasses with tractable distributions should implement this method")