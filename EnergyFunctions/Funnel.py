from .BaseEnergy import EnergyModelClass
import jax
import jax.numpy as jnp
import numpy as np
from functools import partial

class FunnelClass(EnergyModelClass):
    def __init__(self, config):
        """
        Initialize the Funnel distribution (Neal, 2003).
        The distribution is defined as:
        x1 ~ N(0, η²)
        xi ~ N(0, exp(x1)) for i = 2,...,dim
        
        :param config:
        """
        super().__init__(config)
        self.dim = config.get("dim_x", 10) 
        self.eta = config.get("eta", 3.0) 


    @partial(jax.jit, static_argnums=(0,))
    def energy_function(self, x):
        """
        Calculate the energy of the Funnel distribution.
        
        :param x: Input array of shape (..., dim)
        :return: Energy value (scalar)
        """
        assert x.shape[-1] == self.dim
        x1 = x[..., 0]
        xi = x[..., 1:]

        energy_x1 = 0.5 * (x1**2) / (self.eta**2)
        
        energy_xi = 0.5 * jnp.sum(xi**2 * jnp.exp(-energy_x1)[..., None], axis=-1) + \
                   0.5 * (self.dim - 1) * x1
        

        #energy = jnp.clip(energy_x1 + energy_xi, a_min=-1e6, a_max=1e6)
        energy = energy_x1 + energy_xi
        return energy

    def test_energy(self, n_samples=5):
        """
        Test method to print actual energy values for random inputs
        """
        # Create some test points
        rng = np.random.default_rng(42)
        test_points = rng.normal(0, 1, (n_samples, self.dim))
        
        
        x = jnp.array(test_points)
        
        
        x1 = x[..., 0]
        xi = x[..., 1:]
        
        energy_x1 = 0.5 * (x1**2) / (self.eta**2)
        energy_xi = 0.5 * jnp.sum(xi**2 * jnp.exp(-x1)[..., None], axis=-1) + \
                   0.5 * (self.dim - 1) * x1
        
        total_energy = energy_x1 + energy_xi
        
        print("\nFunnel Energy Test:")
        print(f"Input shape: {x.shape}")
        for i in range(n_samples):
            print(f"\nSample {i+1}:")
            print(f"x1: {x1[i]:.3f}")
            print(f"energy_x1: {energy_x1[i]:.3f}")
            print(f"energy_xi: {energy_xi[i]:.3f}")
            print(f"total_energy: {total_energy[i]:.3f}")
        print(f"total_energy: {total_energy}")
        print(f"energy: {self.energy_function(x)}")

if __name__ == "__main__":
    funnel = FunnelClass({"dim_x": 10, "eta": 3.0, "scaling": 1})
    funnel.test_energy()
    