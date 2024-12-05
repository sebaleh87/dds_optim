
from .BaseEnergy import EnergyModelClass
import jax
import jax.numpy as jnp
import numpy as np
from functools import partial
from jax.scipy.special import logsumexp


class LeonardJonesClass(EnergyModelClass):
    def __init__(self, config):
        """
        Initialize the Gaussian Mixture Model.
        
        :param means: List of means for each Gaussian component.
        :param variances: List of variances for each Gaussian component.
        :param weights: List of weights for each Gaussian component.
        """

        super().__init__(config)
        self.r = 1
        self.tau = 1.
        self.eps = 1.
        self.c = 0.5

        self.d = 3
        self.n = config["N"]
        self.dim_x = self.d*self.n
        self.invariance = True
        self.n_particles = self.n
        self.particle_dim = self.d
    
    @partial(jax.jit, static_argnums=(0,))
    def energy_function(self, x):
        """
        Calculate the energy of the Gaussian Mixture Model using logsumexp.
        
        :param x: Input array.
        :return: Energy value.
        """
        x = x.reshape(-1, self.d)

        eps = 9*10**-3
        d_ij_squared = jnp.sum((x[:, None, :] - x[None, :, :]) ** 2, axis=-1)
        d_ij_squared = jnp.where(d_ij_squared < eps, d_ij_squared+eps, d_ij_squared)
        mask = jnp.eye(d_ij_squared.shape[0])

        R_ij = (((self.r**2/d_ij_squared)**3 - 2)*(self.r**2/d_ij_squared)**3)
        R_ij = jnp.where(mask, 0, R_ij)

        Energy_LJ = self.eps/(2*self.tau) *jnp.mean(R_ij)
        x_COM = jnp.mean(x, axis=0, keepdims=True)
        Energy_COM = 0.5*jnp.mean(jnp.sum((x - x_COM)**2, axis=-1))
        # print("Energies")
        # print(Energy_LJ, self.c*Energy_COM)
        # print("distances")
        # print(jnp.min(d_ij), jnp.max(d_ij), jnp.mean(d_ij))
        # print("xs")
        # print(jnp.min(x), jnp.max(x), jnp.mean(x))
        return Energy_LJ + self.c*Energy_COM

    
    