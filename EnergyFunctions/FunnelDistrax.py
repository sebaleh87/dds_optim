from .BaseEnergy import EnergyModelClass
import jax
import jax.numpy as jnp
import distrax
import numpy as np
from functools import partial
import matplotlib.pyplot as plt
import os

class FunnelDistraxClass(EnergyModelClass):
    def __init__(self, config):
        """
        Initialize the Funnel distribution using distrax (Neal, 2003).
        The distribution is defined as:
        x1 ~ N(0, η²)
        xi ~ N(0, exp(x1)) for i = 2,...,dim
        
        :param config: Configuration dictionary
        """
        super().__init__(config)
        self.dim = config.get("dim_x", 10)
        self.eta = config.get("eta", 3.0)
        
        # Initialize the dominant distribution (x1)
        self.dist_dominant = distrax.Normal(loc=jnp.array(0.0), scale=jnp.array(self.eta))
        
        # Constants for other dimensions
        self.mean_other = jnp.zeros(self.dim - 1, dtype=float)
        self.cov_eye = jnp.eye(self.dim - 1).reshape((1, self.dim - 1, self.dim - 1))
        self.has_tractable_distribution = True

    def _dist_other(self, dominant_x):
        """
        Create the conditional distribution for other dimensions given x1.
        
        :param dominant_x: The value of x1
        :return: MultivariateNormalFullCovariance distribution
        """
        variance_other = jnp.exp(dominant_x)
        cov_other = variance_other.reshape(-1, 1, 1) * self.cov_eye
        return distrax.MultivariateNormalFullCovariance(self.mean_other, cov_other)

    @partial(jax.jit, static_argnums=(0,))
    def energy_function(self, x):
        """
        Calculate the energy (negative log probability) of the Funnel distribution.
        
        :param x: Input array of shape (..., dim)
        :return: Energy value (scalar)
        """
        assert x.shape[-1] == self.dim
        
        # Handle batched and unbatched inputs
        batched = x.ndim == 2
        if not batched:
            x = x[None,]
            
        dominant_x = x[..., 0]
        log_density_dominant = self.dist_dominant.log_prob(dominant_x)
        
        # Calculate log prob for other dimensions
        log_sigma = 0.5 * x[..., 0:1]
        sigma2 = jnp.exp(x[..., 0:1])
        neglog_density_other = 0.5 * jnp.log(2 * jnp.pi) + log_sigma + 0.5 * x[..., 1:] ** 2 / sigma2
        log_density_other = jnp.sum(-neglog_density_other, axis=-1)
        
        log_prob = log_density_dominant + log_density_other
        
        if not batched:
            log_prob = jnp.squeeze(log_prob, axis=0)
            
        return -log_prob

    def sample(self, key, sample_shape=()):
        """
        Generate samples from the Funnel distribution.
        
        :param key: JAX random key
        :param sample_shape: Shape of samples to generate
        :return: Array of samples
        """
        key1, key2 = jax.random.split(key)
        
        # Sample x1 from dominant distribution
        dominant_x = self.dist_dominant.sample(seed=key1, sample_shape=sample_shape)
        dominant_x = dominant_x[..., None]  # Add extra dimension to match x_others
        
        # Sample other dimensions conditionally
        x_others = self._dist_other(dominant_x).sample(seed=key2)
        
        return jnp.hstack([dominant_x, x_others])

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
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()
        return plt.gcf()

if __name__ == "__main__":
    # Test the implementation
    config = {
        "dim_x": 2,  # Using 2D for visualization
        "eta": 3.0,
        "scaling": 1.0
    }
    funnel = FunnelDistraxClass(config)
    
    # Generate samples
    key = jax.random.PRNGKey(42)
    n_samples = 5000
    samples = funnel.generate_samples(key, n_samples)
    
    # Create plot directory if needed
    current_dir = os.path.dirname(os.path.abspath(__file__))
    plot_path = os.path.join(current_dir, 'EnergyData', 'Plots', 'funnel_distrax_samples.png')
    
    # Plot and save samples
    funnel.plot_samples(samples, 
                       title="Funnel Distribution (distrax, 5000 samples)",
                       save_path=plot_path)
    
    # Print statistics
    print(f"Mean x₁: {jnp.mean(samples[:, 0]):.3f} (should be close to 0)")
    print(f"Std x₁: {jnp.std(samples[:, 0]):.3f} (should be close to {config['eta']})")
    print(f"Mean x₂: {jnp.mean(samples[:, 1]):.3f} (should be close to 0)") 