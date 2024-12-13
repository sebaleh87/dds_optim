
from .BaseEnergy import EnergyModelClass
import jax
import jax.numpy as jnp
import numpy as np
from functools import partial
from jax.scipy.special import logsumexp


class DoubleWellClass(EnergyModelClass):
    def __init__(self, config):
        """
        Initialize the Gaussian Mixture Model.
        
        :param means: List of means for each Gaussian component.
        :param variances: List of variances for each Gaussian component.
        :param weights: List of weights for each Gaussian component.
        """

        super().__init__(config)
        if(self.config["name"] == "DoubleWell_iter"):
            self.a = 0
            self.b = -4
            self.c = 0.9
            self.tau = 1.


            self.d = 2
            self.n = config["N"]
            self.dim_x = self.d*self.n
            self.chosen_energy_function = self.energy_function_iter
            self.invariance = True
            self.n_particles = self.n
            self.particle_dim = self.d
        elif(self.config["name"] == "DoubleWell_Richter"):
            self.d = self.config["d"]
            self.m = self.config["m"]
            self.b = 1
            self.c = 0.5
            self.dim_x = self.d + self.m
            self.chosen_energy_function = self.energy_function_richter
            self.invariance = False
        else:
            raise ValueError("Energy Config not found")

    @partial(jax.jit, static_argnums=(0,))
    def energy_function(self, x):    
        return self.chosen_energy_function(x)
    
    @partial(jax.jit, static_argnums=(0,))
    def energy_function_richter(self, x):
        """
        Calculate the energy of the Gaussian Mixture Model using logsumexp.
        
        :param x: Input array.
        :return: Energy value.
        """
        d_0 = 4.0

        energy = self.b *jnp.sum((x[:self.m]**2 - d_0))**2 + self.c*jnp.sum(x[self.m:]**2)

        return energy
    
    @partial(jax.jit, static_argnums=(0,))
    def energy_function_iter(self, x):
        """
        Calculate the energy of the Gaussian Mixture Model using logsumexp.
        
        :param x: Input array.
        :return: Energy value.
        """
        d_0 = 4.0
        x = x.reshape(-1, self.d)
        d_ij = jnp.sqrt(jnp.sum((x[:, None, :] - x[None, :, :]) ** 2 , axis=-1) + 10**-8)
        mask = jnp.eye(d_ij.shape[0])

        energy_per_particle = self.a* (d_ij -d_0) + self.b *(d_ij - d_0)**2 + self.c*(d_ij - d_0)**4
        energy_per_particle = jnp.where(mask, 0, energy_per_particle)

        energy = 1/(2*self.tau) * jnp.sum(energy_per_particle)
        # energy = jnp.nan_to_num(energy, 10**4)
        # energy = jnp.where(energy > 10**4, 10**4, energy)
        return energy

    def plot_interatomic_distances(self, x, panel = ""):
        """
        Plot the interatomic distances between particles in the input array.
        
        :param x: Input array.
        """
        x = x.reshape(-1, self.n_particles, self.particle_dim)
        d_ij = np.sqrt(np.sum((x[:, None, :, :] - x[:, :, None, :]) ** 2 , axis=-1))
        plt.hist(d_ij.flatten(), bins=100, density=True)
        plt.xlabel("Interatomic distance")
        plt.ylabel("Density")
        
        wandb.log({f"{panel}/Energy_Landscape": wfig})
        plt.close()
