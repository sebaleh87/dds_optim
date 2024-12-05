
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
        d_ij = jnp.sqrt(jnp.sum((x[:, None, :] - x[None, :, :]) ** 2, axis=-1))
        mask = jnp.eye(d_ij.shape[0])

        energy_per_particle = self.a* (d_ij -d_0) + self.b *(d_ij - d_0)**2 + self.c*(d_ij - d_0)**4
        energy_per_particle = jnp.where(mask, 0, energy_per_particle)

        energy = 1/(2*self.tau) * jnp.sum(energy_per_particle)
        # energy = jnp.nan_to_num(energy, 10**4)
        # energy = jnp.where(energy > 10**4, 10**4, energy)
        return energy

    # @partial(jax.jit, static_argnums=(0,))
    # def energy_function(self, x):

    #     return custom_energy_grad(x)


    # Register the custom VJP with JAX

# @jax.custom_vjp
# def custom_energy_grad(x):
#     return compute_energy(x)

# def compute_energy(x):
#     d = 2
#     a = 0.
#     b = -4.
#     c = 0.9
#     tau = 1.
#     d_0 = 4.0
#     x = x.reshape(-1, d)
#     d_ij = jnp.sqrt(jnp.sum((x[:, None, :] - x[None, :, :]) ** 2, axis=-1))
#     mask = jnp.eye(d_ij.shape[0])

#     energy_per_particle = a * (d_ij - d_0) + b * (d_ij - d_0) ** 2 + c * (d_ij - d_0) ** 4
#     energy_per_particle = jnp.where(mask, 0, energy_per_particle)

#     energy = 1 / (2 * tau) * jnp.sum(energy_per_particle)
#     print("energy", jnp.min(energy), jnp.max(energy))
#     return energy
    
# def energy_function_bwd(x, grad_energy):
#     # `aux_data` is `x` from the forward pass, `grad_energy` is the gradient from the loss

#     # Compute the gradients of the energy function w.r.t. x
#     grad_x = jax.grad(lambda xin: compute_energy(xin))(x )
    
#     # Apply gradient clipping (e.g., clip by norm)
#     clip_value = 10**3  # Adjust this value as needed
#     grad_x_clipped = jnp.clip(grad_x, -clip_value, clip_value)
#     grad_x_clipped = jnp.where(jnp.isnan(grad_x), 0, grad_x_clipped)

#     print("gradient", jnp.min(grad_x_clipped), jnp.max(grad_x_clipped))
#     print("grad_energy", grad_energy.shape, grad_x_clipped.shape)
#     # Return the clipped gradients multiplied by the incoming gradient
#     return (grad_energy * grad_x_clipped,)    

# custom_energy_grad.defvjp(compute_energy, energy_function_bwd)