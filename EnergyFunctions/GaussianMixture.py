from .BaseEnergy import EnergyModelClass
import jax
import jax.numpy as jnp
import numpy as np
from functools import partial
from jax.scipy.special import logsumexp


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

        self.x_min = self.config['scaling'] * (np.min(self.means) + self.shift - 2 * np.max(self.variances))
        self.x_max = self.config['scaling'] * (np.max(self.means) + self.shift + 2 * np.max(self.variances))
        self.y_min = self.config['scaling'] * (np.min(self.means) + self.shift - 2 * np.max(self.variances))
        self.y_max = self.config['scaling'] * (np.max(self.means) + self.shift + 2 * np.max(self.variances))    
 
        self.levels = 50

    # @partial(jax.jit, static_argnums=(0,))
    # def energy_function(self, x):
    #     """
    #     Calculate the energy of the Gaussian Mixture Model.
        
    #     :param x: Input array.
    #     :return: Energy value.
    #     """
    #     def gaussian(x, mean, variance):
    #         return jnp.exp(jnp.sum(-0.5 * ((x[None, ...] - mean) ** 2 / variance) - jnp.log(jnp.sqrt(2 * jnp.pi * variance)), axis = -1))
        
    #     # print([ gaussian(x, m, v).shape for m, v, w in zip(self.means, self.variances, self.weights)])
    #     # gaussians = jnp.array([w * gaussian(x, m, v) for m, v, w in zip(self.means, self.variances, self.weights)])
    #     # print(gaussians.shape, "gaussians")
    #     return -jnp.log(jnp.mean(gaussian(x, self.means, self.variances), axis=0) + 10**-10)
    
    @partial(jax.jit, static_argnums=(0,))
    def energy_function(self, x):
        """
        Calculate the energy of the Gaussian Mixture Model using logsumexp.
        
        :param x: Input array.
        :return: Energy value.
        """
        def gaussian_log_prob(x, mean, variance):
            return jnp.sum(-0.5 * ((x[None, ...] - mean) ** 2 / variance) - 0.5 * jnp.log(2 * jnp.pi * variance), axis=-1)

        # Compute log probabilities and apply logsumexp to get log of mean of exponentials
        log_probs = gaussian_log_prob(x, self.means, self.variances)
        return -logsumexp(log_probs, axis=0) + jnp.log(log_probs.shape[0])

    
    