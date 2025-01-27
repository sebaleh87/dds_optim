from .BaseEnergy import EnergyModelClass
import jax
import jax.numpy as jnp
import distrax
from functools import partial
import matplotlib.pyplot as plt
import os
import wandb

class FunnelDistraxClass(EnergyModelClass):
    def __init__(self, config):
        """
        Initialize the Funnel distribution (Neal, 2003).
        The distribution is defined as:
        x₁ ~ N(0, η²)
        xᵢ ~ N(0, exp(x₁)) for i = 2,...,dim
        
        :param config: Configuration dictionary containing:
            - dim_x: dimension of the distribution (default: 2)
            - eta: scale parameter for x₁ (default: 3.0)
            - sample_bounds: optional tuple of (min, max) for clipping samples
        """
        super().__init__(config)
        self.dim = config.get("dim_x", 2)
        self.eta = config.get("eta", 3.0)
        self.sample_bounds = config.get("sample_bounds", [-30, 30])
        
        # Setup distributions
        self.dist_dominant = distrax.Normal(jnp.array([0.0]), jnp.array([self.eta]))
        self.mean_other = jnp.zeros(self.dim - 1, dtype=float)
        self.cov_eye = jnp.eye(self.dim - 1).reshape((1, self.dim - 1, self.dim - 1))
        
        self.has_tractable_distribution = True
        
    def _dist_other(self, dominant_x):
        """Helper function to create the conditional distribution for other dimensions"""
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
        return -self.log_prob(x)
    
    def log_prob(self, x):
        """
        Calculate the log probability of the Funnel distribution.
        
        :param x: Input array of shape (..., dim)
        :return: Log probability value
        """
        batched = x.ndim == 2
        if not batched:
            x = x[None,]
            
        dominant_x = x[:, 0]
        log_density_dominant = self.dist_dominant.log_prob(dominant_x)
        
        log_sigma = 0.5 * x[:, 0:1]
        sigma2 = jnp.exp(x[:, 0:1])
        neglog_density_other = 0.5 * jnp.log(2 * jnp.pi) + log_sigma + 0.5 * x[:, 1:] ** 2 / sigma2
        log_density_other = jnp.sum(-neglog_density_other, axis=-1)
        
        log_prob = log_density_dominant + log_density_other
        if not batched:
            log_prob = jnp.squeeze(log_prob, axis=0)
        return log_prob

    def sample(self, key, sample_shape=()):
        """
        Generate samples from the Funnel distribution.
        
        :param key: JAX random key
        :param sample_shape: Shape of samples to generate
        :return: Array of samples
        """
        key1, key2 = jax.random.split(key)
        dominant_x = self.dist_dominant.sample(seed=key1, sample_shape=sample_shape)
        x_others = self._dist_other(dominant_x).sample(seed=key2)
        
        samples = jnp.hstack([dominant_x, x_others])
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
        if self.dim != 2:
            raise ValueError("Plotting is only supported for 2D distributions")
            
        plt.figure(figsize=(10, 8))
        
        # Create grid for density plot
        x, y = jnp.meshgrid(
            jnp.linspace(-10, 5, 100),
            jnp.linspace(-5, 5, 100)
        )
        grid = jnp.c_[x.ravel(), y.ravel()]
        
        # Compute density
        pdf_values = jax.vmap(jnp.exp)(self.log_prob(grid))
        density = jnp.reshape(pdf_values, x.shape)
        
        # Plot density and samples
        plt.contourf(x, y, density, levels=20, cmap='viridis')
        if samples is not None:
            idx = jax.random.choice(jax.random.PRNGKey(0), samples.shape[0], (300,))
            plt.scatter(samples[idx, 0], samples[idx, 1], c='r', alpha=0.5, marker='x')
        
        plt.xticks([])
        plt.yticks([])
        
        if save_path is not None:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()
            
        return {"figures/vis": [wandb.Image(plt.gcf())]}

if __name__ == "__main__":
    # Initialize the funnel distribution
    config = {
        "dim_x": 2,
        "eta": 3.0,
        "sample_bounds": None
    }
    funnel = FunnelDistraxClass(config)
    
    # Generate samples
    key = jax.random.PRNGKey(42)
    n_samples = 5000
    samples = funnel.generate_samples(key, n_samples)
    
    # Create plot directory if needed
    current_dir = os.path.dirname(os.path.abspath(__file__))
    plot_path = os.path.join(current_dir, 'EnergyData', 'Plots', 'funnel_samples.png')
    
    # Plot samples
    funnel.plot_samples(samples, 
                       title="Funnel Distribution (5000 samples)",
                       save_path=plot_path)
    
    # Print some statistics
    print(f"Mean x₁: {jnp.mean(samples[:, 0]):.3f} (should be close to 0)")
    print(f"Std x₁: {jnp.std(samples[:, 0]):.3f} (should be close to {config['eta']})")
    print(f"Mean x₂: {jnp.mean(samples[:, 1]):.3f} (should be close to 0)")