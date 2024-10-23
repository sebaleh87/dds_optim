from .BaseEnergy import EnergyModelClass
import jax
import jax.numpy as jnp
from functools import partial

class RastriginClass(EnergyModelClass):
    def __init__(self, config ):
        super().__init__(config)
        self.norm = 80
        self.shift = config["shift"]

    @partial(jax.jit, static_argnums=(0,))
    def energy_function(self, x):
        """
        Rastrigin function is given by:
        f(x) = A * n + sum([x_i^2 - A * cos(2*pi*x_i)]) for all dimensions
        Where A = 10, and n is the dimension of x.
        """
        A = 10
        x = x - self.shift
        energy_value = A * len(x) + jnp.sum(x**2 - A * jnp.cos(2 * jnp.pi * x))
        return energy_value#/self.norm
    

