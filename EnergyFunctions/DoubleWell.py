from matplotlib import pyplot as plt
import wandb
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
        self.d = self.config["d"]
        self.m = self.config["m"]
        self.b = 1
        self.c = 0.5
        self.dim_x = self.d + self.m
        self.chosen_energy_function = self.energy_function_richter
        self.invariance = False


    @partial(jax.jit, static_argnums=(0,))
    def energy_function(self, x):    
        """
        Calculate the energy of the Gaussian Mixture Model using logsumexp.
        
        :param x: Input array.
        :return: Energy value.
        """
        d_0 = 4.0

        energy = self.b *jnp.sum((x[:self.m]**2 - d_0))**2 + self.c*jnp.sum(x[self.m:]**2)

        return energy
    
