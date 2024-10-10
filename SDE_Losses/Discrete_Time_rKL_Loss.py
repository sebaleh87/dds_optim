from .Base_SDE_Loss import Base_SDE_Loss_Class
import jax
from jax import numpy as jnp
from functools import partial

class Discrete_Time_rKL_Loss_Class(Base_SDE_Loss_Class):

    def __init__(self, SDE_config, Optimizer_Config, EnergyClass, model):
        super().__init__(SDE_config, Optimizer_Config, EnergyClass, model)

    #@partial(jax.jit, static_argnums=(0,))  
    def compute_loss(self, params, key, n_integration_steps = 100, n_states = 10, temp = 1.0, x_dim = 2):
        SDE_tracer, key = self.SDE_type.simulate_reverse_sde_scan(self.model , params, key, n_integration_steps = n_integration_steps, n_states = n_states, x_dim = x_dim)
        noise_loss_values = SDE_tracer["noise_loss_value"]
        entropy_loss_values = SDE_tracer["entropy_loss_value"]
        entropy_loss = jnp.sum(entropy_loss_values, axis = -1)
        noise_loss = jnp.sum(noise_loss_values, axis = -1) 


        x_last = SDE_tracer["x_final"]

        Energy = jax.vmap(self.EnergyClass.calc_energy, in_axes = (0,))(x_last)
        mean_Energy = jnp.mean(Energy)

        loss = temp*noise_loss+ temp*entropy_loss + mean_Energy
        return loss, {"mean_energy": mean_Energy, "noise_loss": noise_loss, "entropy_loss": entropy_loss, "key": key, "X_0": x_last}