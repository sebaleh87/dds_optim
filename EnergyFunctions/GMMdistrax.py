from .BaseEnergy import EnergyModelClass
import jax
import jax.numpy as jnp
import numpyro.distributions as dist
from functools import partial
import matplotlib.pyplot as plt
import os
import numpy as np

class GMMDistraxClass(EnergyModelClass):
    def __init__(self, config):
        """
        Initialize the Student-t Mixture distribution.
        The distribution is a mixture of multivariate Student-t distributions.
        
        :param config: Configuration dictionary containing:
            - dim_x: dimension of the distribution
            - num_components: number of mixture components
            - df: degrees of freedom for Student-t
        """
        super().__init__(config)
        self.dim = config.get("dim_x", 2)
        self.num_components = config.get("num_components", 40)
        self.loc_scaling = config.get("loc_scaling", 40.)
        self.variance = config.get("variances", 1.)
        
        # Initialize mixture parameters
        key = jax.random.PRNGKey(config.get("seed", 0))

        mean = jax.random.uniform(shape=(self.num_components, self.dim), key=key, minval=-1.0, maxval=1.0) * self.loc_scaling
        scale = jnp.ones(shape=(self.num_components, self.dim)) * self.variance
        self.means = mean
        self.scales = scale
        
        # Setup the mixture distribution
        component_dist = dist.Independent(
            dist.Normal(loc=self.means, scale=self.scales), 
            1
        )
        mixture_weights = dist.Categorical(
            logits=jnp.ones(self.num_components) / self.num_components
        )
        self.mixture_distribution = dist.MixtureSameFamily(
            mixture_weights, 
            component_dist
        )
        
        self.has_tractable_distribution = True

        self.variances = self.variance
        self.x_min = np.min(self.means) + self.shift - 10 * np.max(self.variances)
        self.x_max = np.max(self.means) + self.shift + 10 * np.max(self.variances)
        self.y_min = np.min(self.means) + self.shift - 10 * np.max(self.variances)
        self.y_max = np.max(self.means) + self.shift + 10 * np.max(self.variances)  
        
        self.levels = 80

    @partial(jax.jit, static_argnums=(0,))
    def energy_function(self, x):
        """
        Calculate the energy (negative log probability) of the Student-t mixture.
        
        :param x: Input array of shape (..., dim)
        :return: Energy value (scalar)
        """
        return -self.mixture_distribution.log_prob(x)
    
    def sample(self, key, sample_shape=()):
        """
        Generate samples from the Student-t mixture distribution.
        
        :param key: JAX random key
        :param sample_shape: Shape of samples to generate
        :return: Array of samples
        """
        return self.mixture_distribution.sample(key=key, sample_shape=sample_shape)

    def generate_samples(self, key, n_samples):
        """
        Generate multiple samples from the Student-t mixture distribution.
        
        :param key: JAX random key
        :param n_samples: Number of samples to generate
        :return: Array of samples with shape (n_samples, dim)
        """
        return self.sample(key, sample_shape=(n_samples,))

    def compute_emc(self, samples):
        """
        Compute the Energy-based Monte Carlo (EMC) score for a given set of samples.
        
        :param samples: Array of samples with shape (n_samples, dim)
        :return: EMC score
        """

        # Expand samples to compute component-wise log probabilities
        expanded = jnp.expand_dims(samples, axis=-2)  # Shape: (n_samples, 1, dim)
        
        # Get the component distribution from the mixture
        component_dist = self.mixture_distribution.component_distribution
        
        # Compute log probability for each sample under each component
        component_log_probs = component_dist.log_prob(expanded)  # Shape: (n_samples, num_components)
        
        # Find the most likely component for each sample
        idx = jnp.argmax(component_log_probs, axis=1)  # Shape: (n_samples,)
        
        # Count occurrences of each component
        unique_elements, counts = jnp.unique(idx, return_counts=True)
        
        # Calculate mode distribution (probability of each mode)
        mode_dist = counts / samples.shape[0]  # Shape: (num_unique_components,)
        
        # Calculate entropy with log base equal to number of components
        entropy = -jnp.sum(mode_dist * (jnp.log(mode_dist) / jnp.log(self.num_components)))
        
        return entropy


    def plot_samples(self, samples, title="GMM", save_path=None):
        """
        Plot samples from the Student-t mixture distribution and optionally save to file.
        
        :param samples: Array of samples with shape (n_samples, dim)
        :param title: Plot title
        :param save_path: Path to save the plot. If None, will show the plot instead
        """
        if self.dim != 2:
            raise ValueError("Plotting is only supported for 2D distributions")
            
        plt.figure(figsize=(10, 8))
        
        # Create grid for density plot
        x, y = jnp.meshgrid(
            jnp.linspace(-15, 15, 100),
            jnp.linspace(-15, 15, 100)
        )
        grid = jnp.stack([x.ravel(), y.ravel()], axis=1)
        
        # Compute density
        log_probs = jax.vmap(lambda x: -self.energy_function(x))(grid)
        density = jnp.exp(log_probs).reshape(x.shape)
        
        # Plot density and samples
        plt.contourf(x, y, density, levels=50)
        if samples is not None:
            plt.scatter(samples[:300, 0], samples[:300, 1], 
                       c='r', alpha=0.5, marker='x')
        
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
    config = {
        "dim_x": 2,
        "num_components": 5,
        "df": 2.0,
        'scaling': 1.0
    }
    gmm = GMMDistraxClass(config)
    
    # Generate samples
    key = jax.random.PRNGKey(42)
    n_samples = 5000
    samples = gmm.generate_samples(key, n_samples)
    
    emc = gmm.compute_emc(samples)
    print(emc)
