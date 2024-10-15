from .Base_SDE_Loss import Base_SDE_Loss_Class
import jax
from jax import numpy as jnp
from functools import partial

class Discrete_Time_rKL_Loss_Class_log_deriv(Base_SDE_Loss_Class):

    def __init__(self, SDE_config, Optimizer_Config, EnergyClass, model):
        self.temp_mode = SDE_config["SDE_Type_Config"]["temp_mode"]
        super().__init__(SDE_config, Optimizer_Config, EnergyClass, model)
        self.SDE_type.stop_gradient = True

    @partial(jax.jit, static_argnums=(0,), static_argnames=("n_integration_steps", "n_states", "x_dim"))  
    def compute_loss(self, params, key, n_integration_steps = 100, n_states = 10, temp = 1.0, x_dim = 2):
        SDE_tracer, key = self.SDE_type.simulate_reverse_sde_scan(self.model , params, key, n_integration_steps = n_integration_steps, n_states = n_states, x_dim = x_dim)
        log_probs = SDE_tracer["log_probs"]
        noise_loss_values = SDE_tracer["noise_loss_arr"]
        entropy_loss_values = SDE_tracer["entropy_loss_arr"]

        x_last = SDE_tracer["x_final"]
        Energy = self.vmap_calc_Energy(x_last)
        mean_Energy = jnp.mean(Energy)

        ### TODO stop gradient of X!
        cum_sum_log_probs = jax.lax.cumsum(log_probs, axis = 0)

        cum_sum_log_probs_for_entropy = jnp.concatenate([jnp.ones_like(cum_sum_log_probs[0])[None, ...],cum_sum_log_probs[:-1]], axis = 0)
        Entropy_term_reparam = jnp.mean(jnp.sum(entropy_loss_values, axis = 0))


        Entropy_term_reward = jax.lax.stop_gradient(entropy_loss_values - jnp.mean(entropy_loss_values, axis = 1, keepdims=True))
        Entropy_term_reinforce = jnp.mean(jnp.sum(Entropy_term_reward*cum_sum_log_probs_for_entropy, axis = 0))

        noise_loss_reward = jax.lax.stop_gradient(noise_loss_values - jnp.mean(noise_loss_values, axis = 1, keepdims=True))
        noise_loss_term = jnp.mean(jnp.sum(noise_loss_reward* cum_sum_log_probs, axis = 0))

        Energy_reward = jax.lax.stop_gradient(Energy - jnp.mean(Energy))
        Energy_term =  jnp.mean(Energy_reward * cum_sum_log_probs[-1])

        entropy_loss = Entropy_term_reparam + Entropy_term_reinforce
        noise_loss = noise_loss_term

        if(self.temp_mode):
            loss = temp*entropy_loss + noise_loss + Energy_term
        else:
            loss = temp*entropy_loss + temp*noise_loss + Energy_term

        log_noise_Loss = jnp.mean(jnp.sum(noise_loss_values, axis = 0))
        return loss, {"mean_energy": mean_Energy, "best_Energy": jnp.min(Energy), "noise_loss": log_noise_Loss, "entropy_loss": Entropy_term_reparam, "key": key, "X_0": x_last}