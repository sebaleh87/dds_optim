from matplotlib import pyplot as plt
from .BaseEnergy import EnergyModelClass
import jax
import jax.numpy as jnp
import numpy as np
from functools import partial
from jax.scipy.special import logsumexp
import chex
import os
import pickle

class ManyWellClass(EnergyModelClass):
    def __init__(self, config):
        """

        """

        super().__init__(config)
        self.d = self.config["d"]
        self.m = self.config["m"]
        self.b = 1
        self.c = 0.5
        self.dim_x = self.d
        self.has_tractable_distribution = True
        #self.chosen_energy_function = self.energy_function_richter
        self.invariance = False
        self.d_0 = 4.0
        zer = np.array([np.sqrt(self.d_0), -np.sqrt(self.d_0)])
        grids = np.meshgrid(*([zer] * self.m))
        self.means = jnp.array(np.stack(grids, axis=-1).reshape(-1, self.m))
        
    @partial(jax.jit, static_argnums=(0,))
    def energy_function(self, x):    
        """

        """
        #d_0 = 4.0

        energy = self.b *jnp.sum((x[:self.m]**2 - self.d_0)**2) + self.c*jnp.sum(x[self.m:]**2)
        
        return energy
    
    def log_prob(self, x: chex.Array) -> chex.Array:
        """
        Calculate the log probability (negative energy)
        """
        return -self.energy_function(x)
    
    def sample(self, seed: chex.PRNGKey, sample_shape: chex.Shape = ()) -> chex.Array:
        """
        Generate samples from the Many Well distribution
        by loading from pickle file
        """
        with open('./Data/mw_d5_m5_del4_1M.pkl', 'rb') as f:
            mwdata = pickle.load(f)
        mwdata = jnp.array(mwdata)
        row_idx = jax.random.choice(seed, mwdata.shape[0], shape=sample_shape, replace=False)
        mwdata = mwdata[row_idx]
        return mwdata

    def generate_samples(self, key, n_samples):
        """
        Generate multiple samples.
        """
        return self.sample(key, sample_shape=(n_samples,))
    
    def visualise(self, samples: chex.Array = None, axes=None, show=False, prefix='') -> dict:
        """
        Visualize samples from the Double Well distribution.
        Shows the first two dimensions if m >= 2, or first dimension vs first Gaussian dimension if m == 1.
        
        :param samples: Optional array of samples to plot
        :param axes: Optional matplotlib axes for plotting
        :param show: Whether to show the plot
        :param prefix: Prefix for saving the plot
        :return: Dictionary with wandb image
        """
        plt.close()
        fig = plt.figure(figsize=(10, 8))
        if axes is None:
            ax = fig.add_subplot(111)
        else:
            ax = axes
            
        if samples is not None:
            # Plot first two dimensions
            if self.m >= 2:
                ax.scatter(samples[:, 0], samples[:, 1], alpha=0.5, s=10)
                ax.set_xlabel('x₁')
                ax.set_ylabel('x₂')
            else:
                # Plot first double well dimension vs first Gaussian dimension
                ax.scatter(samples[:, 0], samples[:, self.m], alpha=0.5, s=10)
                ax.set_xlabel('x₁ (Double Well)')
                ax.set_ylabel('x_{m+1} (Gaussian)')
        
        # Set plot bounds based on the double well modes
        d_0 = 4.0
        bound = 2.5 * jnp.sqrt(d_0)
        ax.set_xlim(-bound, bound)
        ax.set_ylim(-bound, bound)
        
        plt.title(f"Double Well Samples (m={self.m}, d={self.d})")
        plt.grid(True)
        
        # Remove ticks for cleaner visualization
        plt.xticks([])
        plt.yticks([])
        
        plt.savefig(os.path.join('EnergyFunctions', 'EnergyData', 'Plots', 'vis.png'))
        
        if show:
            plt.show()
        



if __name__ == '__main__':
    # Test configurations
    configs = [
        {"d": 5, "m": 5, "dim_x": 10, 'scaling': 1.0}, 
    ]
    
    for config in configs:
        # Initialize model
        double_well = ManyWellClass(config)
        
        # Generate samples
        key = jax.random.PRNGKey(0)
        samples = double_well.generate_samples(key, 2000)
        
        # Visualize
        double_well.visualise(samples, show=True)
        
        # Print some statistics about the first dimensions
        if config["m"] >= 2:
            print(f"\nStatistics for 2D Double Well (m={config['m']}, d={config['d']}):")
            print(f"Mean of first two dims: ({jnp.mean(samples[:, 0]):.3f}, {jnp.mean(samples[:, 1]):.3f})")
            print(f"Std of first two dims: ({jnp.std(samples[:, 0]):.3f}, {jnp.std(samples[:, 1]):.3f})")
        else:
            print(f"\nStatistics for 1D Double Well (m={config['m']}, d={config['d']}):")
            print(f"Mean of double well dim: {jnp.mean(samples[:, 0]):.3f}")
            print(f"Std of double well dim: {jnp.std(samples[:, 0]):.3f}")
            print(f"Mean of first Gaussian dim: {jnp.mean(samples[:, 1]):.3f}")
            print(f"Std of first Gaussian dim: {jnp.std(samples[:, 1]):.3f}")
    
    
    
