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
        pos = 3
        f = 1.3
        self.a1 = jnp.array([[pos,pos]])
        self.a2 = jnp.array([[-pos,-pos]])
        self.a3 = jnp.array([[pos,-pos]])
        self.a4 = jnp.array([[-pos*f,pos]])
        self.A_list = [self.a1, self.a2, self.a3, self.a4]
        self.norm = 200
        super().__init__(config)

    @partial(jax.jit, static_argnums=(0,))
    def energy_function(self, x):
        """
        Calculate the energy of the Mexican Hat Potential.
        
        :param x: Input array.
        :return: Energy value.
        """
        value = 1.
        for a in self.A_list:
            value *= jnp.sum((x - a) ** 2, axis = -1)

        return jnp.sqrt(value)/self.norm