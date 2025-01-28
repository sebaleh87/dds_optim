from .BaseEnergy import EnergyModelClass
import jax
import jax.numpy as jnp
import numpy as np
from functools import partial
import matplotlib.pyplot as plt
import os

class FunnelClass(EnergyModelClass):
    def __init__(self, config):
        """
        Initialize the Funnel distribution (Neal, 2003).
        The distribution is defined as:
        x1 ~ N(0, η²)
        xi ~ N(0, exp(x1)) for i = 2,...,dim
        
        :param config:
            - dim_x: dimension of the distribution (default: 10)
            - eta: scale parameter for x₁ (default: 3.0)
            - sample_bounds: optional tuple of (min, max) for clipping samples
        """
        super().__init__(config)
        self.dim = config.get("dim_x", 10) 
        self.eta = config.get("eta", 3.0)
        self.sample_bounds = config.get("sample_bounds", [-30, 30])
        
        # Constants for log probability computation
        self.log_2pi = jnp.log(2 * jnp.pi)
        self.has_tractable_distribution = True

    @partial(jax.jit, static_argnums=(0,))
    def energy_function(self, x):
        """
        Calculate the energy (negative log probability) of the Funnel distribution.
        
        :param x: Input array of shape (..., dim)
        :return: Energy value (scalar)
        """
        assert x.shape[-1] == self.dim
        x1 = x[..., 0]
        xi = x[..., 1:]

        # Log prob for x1 ~ N(0, eta²)
        log_prob_x1 = -0.5 * self.log_2pi - jnp.log(self.eta) - 0.5 * (x1**2) / (self.eta**2)
        
        # Log prob for xi ~ N(0, exp(x1))
        log_sigma = 0.5 * x1[..., None]
        sigma2 = jnp.exp(x1[..., None])
        log_prob_xi = -0.5 * self.log_2pi - log_sigma - 0.5 * xi**2 / sigma2
        log_prob_xi = jnp.sum(log_prob_xi, axis=-1)

        # Total log probability
        log_prob = log_prob_x1 + log_prob_xi

        return -log_prob
    
    def sample(self, key, sample_shape=()):
        """
        Generate samples from the Funnel distribution.
        
        :param key: JAX random key
        :param sample_shape: Shape of samples to generate (default: single sample)
        :return: Array of samples with shape (*sample_shape, dim)
        """
        key1, key2 = jax.random.split(key)
        
        # Sample x1 ~ N(0, eta²)
        dominant_x = self.eta * jax.random.normal(key1, sample_shape + (1,))
        
        # Sample xi ~ N(0, exp(x1)) for i = 2,...,dim
        std_other = jnp.exp(0.5 * dominant_x)
        x_others = std_other * jax.random.normal(key2, sample_shape + (self.dim - 1,))
        
        # Combine samples
        samples = jnp.concatenate([dominant_x, x_others], axis=-1)
        
        # Apply clipping if sample_bounds are specified
        if self.sample_bounds is not None:
            samples = samples.clip(min=self.sample_bounds[0], max=self.sample_bounds[1])
            
        return samples

    def generate_samples(self, key, n_samples):
        """
        Generate multiple samples from the Funnel distribution.
        
        :param key: JAX random key
        :param n_samples: Number of samples to generate
        :return: Array of samples with shape (n_samples, dim)
        """
        return self.sample(key, sample_shape=(n_samples,))

    def plot_samples(self, samples, title="Funnel Distribution Samples", save_path=None):
        """
        Plot samples from the Funnel distribution and optionally save to file.
        
        :param samples: Array of samples with shape (n_samples, dim)
        :param title: Plot title
        :param save_path: Path to save the plot. If None, will show the plot instead
        """
        if samples.shape[1] < 2:
            raise ValueError("Need at least 2 dimensions to plot")
            
        plt.figure(figsize=(10, 8))
        plt.scatter(samples[:, 0], samples[:, 1], alpha=0.5)
        plt.xlabel('x₁')
        plt.ylabel('x₂')
        plt.title(title)
        plt.grid(True)
        
        if save_path is not None:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()
        return plt.gcf()

    
if __name__ == "__main__":
    # Force CPU
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    
    # Initialize the funnel distribution
    config = {
        "dim_x": 2,  # Using 2D for visualization
        "eta": 3.0,
        "scaling": 1.0
    }
    funnel = FunnelClass(config)
    
    # Generate samples
    key = jax.random.PRNGKey(42)
    n_samples = 5000
    samples = funnel.generate_samples(key, n_samples)
    
    # Create plot directory if needed
    current_dir = os.path.dirname(os.path.abspath(__file__))
    plot_path = os.path.join(current_dir, 'EnergyData', 'Plots', 'funnel_samples.png')
    
    # Use the plotting method to save the plot
    funnel.plot_samples(samples, 
                       title="Funnel Distribution (5000 samples)",
                       save_path=plot_path)
    
    # Print some statistics
    print(f"Mean x₁: {jnp.mean(samples[:, 0]):.3f} (should be close to 0)")
    print(f"Std x₁: {jnp.std(samples[:, 0]):.3f} (should be close to {config['eta']})")
    print(f"Mean x₂: {jnp.mean(samples[:, 1]):.3f} (should be close to 0)")
    