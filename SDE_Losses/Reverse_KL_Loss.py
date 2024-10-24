from .Base_SDE_Loss import Base_SDE_Loss_Class
import jax
from jax import numpy as jnp
from functools import partial


class Reverse_KL_Loss_Class(Base_SDE_Loss_Class):

    def __init__(self, SDE_config, Optimizer_Config, EnergyClass, Network_Config, model):
        super().__init__(SDE_config, Optimizer_Config, EnergyClass, Network_Config, model)
        self.vmap_diff_factor = jax.vmap(self.SDE_type.get_diffusion, in_axes=(None, None, 0))
        self.vmap_drift_divergence = jax.vmap(self.SDE_type.beta, in_axes = (None, 0))

    @partial(jax.jit, static_argnums=(0,), static_argnames=("n_integration_steps", "n_states", "x_dim"))  
    def compute_loss(self, params, Energy_params, SDE_params, key, n_integration_steps = 100, n_states = 10, temp = 1.0, x_dim = 2):
        SDE_tracer, key = self.SDE_type.simulate_reverse_sde_scan(self.model , params, Energy_params, SDE_params, key, n_integration_steps = n_integration_steps, n_states = n_states, x_dim = x_dim)
        score = jnp.array(SDE_tracer["scores"])
        ts = jnp.array(SDE_tracer["ts"])
        dt = 1./n_integration_steps

        x_prior = SDE_tracer["x_prior"]
        x_last = SDE_tracer["x_final"]

        log_prior = self.SDE_type.get_log_prior(x_prior)
        mean_log_prior = jnp.mean(log_prior)

        Energy, key = self.EnergyClass.vmap_calc_energy(x_last, Energy_params, key)
        mean_Energy = jnp.mean(Energy)
        diff_factor = self.vmap_diff_factor(SDE_params, None, ts)
        drift_divergence = self.vmap_drift_divergence( SDE_params, ts)[:,None, :]
        #print("shapes", score.shape, diff_factor.shape, drift_divergence.shape)
        R_diff = jnp.mean(jnp.sum(dt*(1/2*jnp.sum( ( diff_factor*score)**2, axis = -1) - jnp.sum(drift_divergence, axis = -1))  , axis = 0))
        loss = temp*R_diff + temp*mean_log_prior + mean_Energy
        return loss, {"mean_energy": mean_Energy, "R_diff": R_diff, "likelihood_ratio": jnp.mean(loss), "key": key, "X_0": x_last, "beta_min": jnp.exp(SDE_params["log_beta_min"]), "beta_delta": jnp.exp(SDE_params["log_beta_delta"])}