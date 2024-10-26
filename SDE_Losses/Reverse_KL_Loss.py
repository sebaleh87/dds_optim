from .Base_SDE_Loss import Base_SDE_Loss_Class
import jax
from jax import numpy as jnp
from functools import partial


class Reverse_KL_Loss_Class(Base_SDE_Loss_Class):

    def __init__(self, SDE_config, Optimizer_Config, EnergyClass, Network_Config, model):
        super().__init__(SDE_config, Optimizer_Config, EnergyClass, Network_Config, model)
        self.vmap_diff_factor = jax.vmap(self.SDE_type.get_diffusion, in_axes=(None, None, 0))
        self.vmap_drift_divergence = jax.vmap(self.SDE_type.beta, in_axes = (None, 0))
        self.vmap_get_log_prior = jax.vmap(self.SDE_type.get_log_prior, in_axes = (None, 0))

    @partial(jax.jit, static_argnums=(0,), static_argnames=("n_integration_steps", "n_states", "x_dim"))  
    def compute_loss(self, params, Energy_params, SDE_params, key, n_integration_steps = 100, n_states = 10, temp = 1.0, x_dim = 2):
        SDE_tracer, key = self.SDE_type.simulate_reverse_sde_scan(self.model , params, Energy_params, SDE_params, key, n_integration_steps = n_integration_steps, n_states = n_states, x_dim = x_dim)
        score = SDE_tracer["scores"]
        ts = SDE_tracer["ts"]
        dts = SDE_tracer["dts"][...,None]

        x_prior = SDE_tracer["x_prior"]
        x_last = SDE_tracer["x_final"]

        log_prior = self.vmap_get_log_prior(SDE_params, x_prior)
        #print("log_prior", log_prior.shape, x_prior.shape)
        mean_log_prior = jnp.mean(jnp.sum(log_prior, axis = -1))

        Energy, key = self.EnergyClass.vmap_calc_energy(x_last, Energy_params, key)
        mean_Energy = jnp.mean(Energy)
        diff_factor = self.vmap_diff_factor(SDE_params, None, ts)
        drift_divergence = self.vmap_drift_divergence( SDE_params, ts)[:,None, :]
        #print("shapes", score.shape, diff_factor.shape, drift_divergence.shape)
        f = (1/2*jnp.sum( ( diff_factor*score)**2, axis = -1) - jnp.sum(drift_divergence, axis = -1))

        R_diff = jnp.mean(jnp.sum(dts*f  , axis = 0))
        loss = temp*R_diff + temp*mean_log_prior + mean_Energy
        Entropy = -(R_diff + mean_log_prior)
        Free_Energy = R_diff + mean_log_prior + mean_Energy
        return loss, {"mean_energy": mean_Energy, "Free_Energy_at_T=1": Free_Energy, "Entropy": Entropy, "R_diff": R_diff, "likelihood_ratio": jnp.mean(loss), 
                      "key": key, "X_0": x_last, "mean_X_prior": jnp.mean(x_prior), "std_X_prior": jnp.mean(jnp.std(x_prior, axis = 0)), 
                       "sigma": jnp.exp(SDE_params["log_sigma"]),
                      "beta_min": jnp.exp(SDE_params["log_beta_min"]), "beta_delta": jnp.exp(SDE_params["log_beta_delta"]), "mean": SDE_params["mean"]}