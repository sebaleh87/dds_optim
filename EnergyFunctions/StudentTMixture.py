from .BaseEnergy import EnergyModelClass
import jax
import jax.numpy as jnp
import numpyro.distributions as dist
from functools import partial
import matplotlib.pyplot as plt
import os

class StudentTMixtureClass(EnergyModelClass):
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
        self.num_components = config.get("num_components", 5)
        self.df = config.get("df", 2.0)
        
        # Initialize mixture parameters
        key = jax.random.PRNGKey(0)
        self.locs = jax.random.uniform(
            key, 
            minval=-10.0, 
            maxval=10.0, 
            shape=(self.num_components, self.dim)
        )
        self.dofs = jnp.ones((self.num_components, self.dim)) * self.df
        self.scales = jnp.ones((self.num_components, self.dim))
        
        # Setup the mixture distribution
        component_dist = dist.Independent(
            dist.StudentT(df=self.dofs, loc=self.locs, scale=self.scales), 
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

    def plot_samples(self, samples, title="Student-t Mixture Samples", save_path=None):
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
    # Initialize the Student-t mixture distribution
    config = {
        "dim_x": 2,
        "num_components": 5,
        "df": 2.0
    }
    stm = StudentTMixtureClass(config)
    
    # Generate samples
    key = jax.random.PRNGKey(42)
    n_samples = 5000
    samples = stm.generate_samples(key, n_samples)
    
    # Create plot directory if needed
    current_dir = os.path.dirname(os.path.abspath(__file__))
    plot_path = os.path.join(current_dir, 'EnergyData', 'Plots', 'student_t_mixture_samples.png')
    
    # Plot samples
    stm.plot_samples(samples, 
                    title="Student-t Mixture Distribution (5000 samples)",
                    save_path=plot_path)
    
    # Print some statistics
    print(f"Mean of samples: {jnp.mean(samples, axis=0)}")
    print(f"Std of samples: {jnp.std(samples, axis=0)}")