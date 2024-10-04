from .BaseEnergy import EnergyModelClass
import jax
import jax.numpy as jnp
from functools import partial

class MexicanHatClass(EnergyModelClass):
    def __init__(self, config, a=1.0):
        """
        Initialize the Mexican Hat Potential.
        
        :param a: Position of the minima.
        """
        self.a = a
        super().__init__(config)

    @partial(jax.jit, static_argnums=(0,))
    def calc_energy(self, x):
        """
        Calculate the energy of the Mexican Hat Potential.
        
        :param x: Input array.
        :return: Energy value.
        """
        return (x ** 2 - self.a ** 2) ** 2