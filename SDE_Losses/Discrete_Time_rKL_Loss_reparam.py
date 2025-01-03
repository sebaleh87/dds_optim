from .Base_SDE_Loss import Base_SDE_Loss_Class
import jax
from jax import numpy as jnp
from functools import partial

class Discrete_Time_rKL_Loss_Class_reparam(Base_SDE_Loss_Class):

    def __init__(self, SDE_config, Optimizer_Config, EnergyClass, Network_Config, model):
        self.temp_mode = SDE_config["SDE_Type_Config"]["temp_mode"]
        super().__init__(SDE_config, Optimizer_Config, EnergyClass, Network_Config, model)

    @partial(jax.jit, static_argnums=(0,), static_argnames=("n_integration_steps", "n_states", "x_dim"))  
    def compute_loss(self, params, key, n_integration_steps = 100, n_states = 10, temp = 1.0, x_dim = 2):
        SDE_tracer, key = self.SDE_type.simulate_reverse_sde_scan(self.model , params, key, n_integration_steps = n_integration_steps, n_states = n_states, x_dim = x_dim)
        noise_loss_values = SDE_tracer["noise_loss_value"]
        entropy_loss_values = SDE_tracer["entropy_loss_value"]
        entropy_loss = jnp.sum(entropy_loss_values, axis = 0)
        noise_loss = jnp.sum(noise_loss_values, axis = 0) 


        x_last = SDE_tracer["x_final"]

        Energy = self.vmap_calc_Energy(x_last)
        mean_Energy = jnp.mean(Energy)

        if(self.temp_mode):
            loss = temp*entropy_loss + noise_loss + mean_Energy
        else:
            loss = temp*entropy_loss + temp*noise_loss + mean_Energy
        return loss, {"mean_energy": mean_Energy, "best_Energy": jnp.min(Energy), "noise_loss": noise_loss, "entropy_loss": entropy_loss, "key": key, "X_0": x_last}