
from .BaseEnergy import EnergyModelClass
import jax
import jax.numpy as jnp
from functools import partial

class GaussianMixtureClass(EnergyModelClass):
    def __init__(self, config):
        """
        Initialize the Gaussian Mixture Model.
        
        :param means: List of means for each Gaussian component.
        :param variances: List of variances for each Gaussian component.
        :param weights: List of weights for each Gaussian component.
        """
        self.means = jnp.array(config["means"])
        self.variances = jnp.array(config["variances"])
        self.weights = jnp.array(config["weights"])
        super().__init__(config)

    @partial(jax.jit, static_argnums=(0,))
    def calc_energy(self, x):
        """
        Calculate the energy of the Gaussian Mixture Model.
        
        :param x: Input array.
        :return: Energy value.
        """
        def gaussian(x, mean, variance):
            return -jnp.exp(-0.5 * ((x - mean) ** 2) / variance) / jnp.sqrt(2 * jnp.pi * variance)
        
        gaussians = jnp.array([w * gaussian(x, m, v) for m, v, w in zip(self.means, self.variances, self.weights)])
        return jnp.sum(gaussians, axis=0)
    
    
