import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import os
import wandb

# Base Class
class EnergyModelClass:

    def __init__(self,EnergyConfig):
        self.dim_x = EnergyConfig["dim_x"]
        pass

    def calc_energy(self, x):
        """
        This method should be overridden by subclasses to define
        the specific energy function.
        """
        raise NotImplementedError("Subclasses should implement this method")

    def calc_log_probs(self, x, T):
        """
        Calculate the log probabilities, where T is the temperature.
        log_probs = -1/T * energy(x)
        """
        energy = self.calc_energy(x)
        return -1.0 / T * energy
    
    def plot_properties(self):
        if(self.dim_x == 2):
            self.plot_2_D_properties()
        elif(self.dim_x == 1):
            self.plot_1d_properties()
        else:
            pass

    def plot_2_D_properties(self, x_range=(-5.12, 5.12), resolution=100, T=1.0):
        """
        Plot both the energy landscape and the log probability landscape.
        """
        # Create a grid of points over the specified range
        x = jnp.linspace(x_range[0], x_range[1], resolution)
        y = jnp.linspace(x_range[0], x_range[1], resolution)
        X, Y = jnp.meshgrid(x, y)
        
        # Stack X and Y into a 2D array of coordinates for vectorized evaluation
        grid_points = jnp.stack([X.ravel(), Y.ravel()], axis=-1)
        
        # Calculate energy and log-probs for each point in the grid
        Z_energy = jax.vmap(self.calc_energy)(grid_points)
        Z_log_probs = jax.vmap(lambda pt: self.calc_log_probs(pt, T))(grid_points)
        
        # Reshape results back into a 2D grid
        Z_energy = Z_energy.reshape(X.shape)
        Z_log_probs = Z_log_probs.reshape(X.shape)

        # Plotting
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # Energy landscape
        energy_plot = ax1.contourf(X, Y, Z_energy, levels=100, cmap='viridis')
        fig.colorbar(energy_plot, ax=ax1)
        ax1.set_title("Energy Landscape")
        ax1.set_xlabel("x")
        ax1.set_ylabel("y")

        # Log probability landscape
        log_probs_plot = ax2.contourf(X, Y, Z_log_probs, levels=100, cmap='plasma')
        fig.colorbar(log_probs_plot, ax=ax2)
        ax2.set_title("Log Probability Landscape")
        ax2.set_xlabel("x")
        ax2.set_ylabel("y")

        current_file_path = os.path.abspath(__file__)

        # Get the parent directory of the current file
        parent_folder = os.path.dirname(os.path.dirname(current_file_path))
        plt.tight_layout()
        wandb.log({"target/Energy_Landscape": wandb.Image(fig)})
        plt.close()

    def plot_1d_properties(self, x_range=(-10, 10), resolution=100, T=1.0):
        """
        Plot both the energy landscape and the log probability landscape for a 1-D energy function.
        """
        # Create a range of points over the specified range
        x = jnp.linspace(x_range[0], x_range[1], resolution)
        
        # Calculate energy and log-probs for each point in the range
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
        wandb.log({"fig/Energy_Landscape": wandb.Image(fig)})
        plt.show()

    def plot_trajectories(self, Xs):
        if(self.dim_x == 2):
            self.plot_2_D_trajectories(Xs)
        elif(self.dim_x == 1):
            self.plot_1_D_trajectories(Xs)

    def plot_last_samples(self, Xs):
        if(self.dim_x == 2):
            self.plot_2_D_last_samples(Xs)
        elif(self.dim_x == 1):
            pass

    def plot_histogram(self, Xs):
        if(self.dim_x == 2):
            self.plot_2_D_histogram(Xs)
        elif(self.dim_x == 1):
            self.plot_1_D_histogram(Xs)
    
    def plot_1_D_trajectories(self, Xs):
        T, B, D = Xs.shape
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.plot(np.linspace(0,1,T), Xs[...,0], label=f'Trajectory')
        
        ax.set_xlabel('Time')
        ax.set_ylabel('Value')
        ax.set_title('1D Trajectories Over Time')
        ax.grid(True)

        wandb.log({"fig/1d_trajectories": wandb.Image(fig)})
        plt.close()

    def plot_2_D_trajectories(self, Xs):
        T, B, dim = Xs.shape
        if dim != 2:
            raise ValueError("The dimension of the trajectories must be 2.")

        fig = plt.figure(figsize=(10, 6))
        for b in range(B):
            trajectory = Xs[:, b, :]
            plt.plot(trajectory[:, 0], trajectory[:, 1], label=f'Trajectory {b+1}')
        
        plt.xlabel('X-axis')
        plt.ylabel('Y-axis')
        plt.title('2D Trajectories')
        plt.legend()
        plt.grid(True)

        wandb.log({"fig/trajectories": wandb.Image(fig)})
        plt.close()
    
    def plot_2_D_last_samples(self, Xs):
        fig = plt.figure(figsize=(10, 6))
        plt.plot(Xs[:,0], Xs[:,1], "x")
        
        plt.xlabel('X-axis')
        plt.ylabel('Y-axis')
        plt.title('2D last samples')
        plt.grid(True)
        plt.xlim(-5, 5)
        plt.ylim(-5, 5)

        wandb.log({"fig/last_samples": wandb.Image(fig)})
        plt.close()

    def plot_1_D_histogram(self, samples):
        
        # Create 1D histogram
        hist, bin_edges = np.histogram(samples, bins=100, density=True)
        
        # Plot the histogram
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(bin_edges[:-1], hist, drawstyle='steps-post', color='blue')
        
        ax.set_title('1D Histogram with Likelihood')
        ax.set_xlabel('Value')
        ax.set_ylabel('Likelihood')
        ax.grid(True)
        
        # Log the figure using wandb
        wandb.log({"fig/1d_histogram": wandb.Image(fig)})
        plt.close()

    def plot_2_D_histogram(self, samples):
        # Filter samples where both coordinates are within [-4, 4]
        filtered_samples = samples[(samples[:, 0] >= -4) & (samples[:, 0] <= 4) & (samples[:, 1] >= -4) & (samples[:, 1] <= 4)]
        # Assuming `samples` is provided and wandb is initialized

        # Create 2D histogram
        heatmap, xedges, yedges = np.histogram2d(filtered_samples[:, 0], filtered_samples[:, 1], bins=100, density=True)

        extent = [-4, 4, -4, 4]

        # Plot the zoomed-in heatmap
        fig, ax = plt.subplots(figsize=(10, 8))
        cax = ax.imshow(heatmap.T, extent=extent, origin='lower', cmap='viridis', aspect='auto')

        ax.set_title('Zoomed 2D Histogram with Likelihood')
        ax.set_xlabel('X-axis')
        ax.set_ylabel('Y-axis')
        fig.colorbar(cax, ax=ax, label='Likelihood')

        # Log the figure using wandb
        wandb.log({"fig/2d_histogram": wandb.Image(fig)})
        plt.close()