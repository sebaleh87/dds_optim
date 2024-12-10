from .Base_SDE_Loss import Base_SDE_Loss_Class
import jax
from jax import numpy as jnp
from functools import partial


class Reverse_KL_Loss_Class(Base_SDE_Loss_Class):

    def __init__(self, SDE_config, Optimizer_Config, EnergyClass, Network_Config, model):
        super().__init__(SDE_config, Optimizer_Config, EnergyClass, Network_Config, model)
        self.vmap_diff_factor = jax.vmap(self.SDE_type.get_diffusion, in_axes=(None, None, 0))
        self.vmap_drift_divergence = jax.vmap(self.SDE_type.get_div_drift, in_axes = (None, 0))
        self.vmap_get_log_prior = jax.vmap(self.SDE_type.get_log_prior, in_axes = (None, 0))

    @partial(jax.jit, static_argnums=(0,))  
    def evaluate_loss(self, Energy_params, SDE_params, SDE_tracer, key, temp = 1.0):
        score = SDE_tracer["scores"]
        ts = SDE_tracer["ts"]
        dW = SDE_tracer["dW"]
        dts = SDE_tracer["dts"][...,None]

        x_prior = SDE_tracer["x_prior"]
        x_last = SDE_tracer["x_final"]

        log_prior = self.vmap_get_log_prior(SDE_params, x_prior)
        #print("log_prior", log_prior.shape, x_prior.shape)
        mean_log_prior = jnp.mean(log_prior)

        Energy, key = self.EnergyClass.vmap_calc_energy(x_last, Energy_params, key)
        mean_Energy = jnp.mean(Energy)
        diff_factor = self.vmap_diff_factor(SDE_params, None, ts)
        #print("adas", self.vmap_drift_divergence( SDE_params, ts).shape)
        drift_divergence = self.vmap_drift_divergence( SDE_params, ts)[:,None, :]
        #print("shapes", score.shape, diff_factor.shape, drift_divergence.shape)
        U = diff_factor*score
        f = (1/2*jnp.sum( ( U)**2, axis = -1) - jnp.sum(drift_divergence, axis = -1))
        
        S = jnp.sum(jnp.sum(U * dW, axis = -1), axis = 0)
        R_diff = jnp.sum(dts*f  , axis = 0)
        mean_R_diff = jnp.mean(R_diff)

        loss = temp*mean_R_diff + temp*mean_log_prior + mean_Energy
        Entropy = -(mean_R_diff + mean_log_prior)

        #print("RKL LOss", mean_R_diff, mean_log_prior, beta*mean_Energy)

        res_dict = self.compute_partition_sum(R_diff, S, log_prior, Energy)
        log_Z = res_dict["log_Z"]
        Free_Energy, n_eff, NLL = res_dict["Free_Energy"], res_dict["n_eff"], res_dict["NLL"]

        return loss, {"mean_energy": mean_Energy, "Free_Energy_at_T=1": Free_Energy, "Entropy": Entropy, "R_diff": R_diff, "likelihood_ratio": jnp.mean(loss), 
                      "key": key, "X_0": x_last, "mean_X_prior": jnp.mean(x_prior), "std_X_prior": jnp.mean(jnp.std(x_prior, axis = 0)), 
                       "sigma": jnp.exp(SDE_params["log_sigma"]),
                      "beta_min": jnp.exp(SDE_params["log_beta_min"]), "beta_delta": jnp.exp(SDE_params["log_beta_delta"]), "mean": SDE_params["mean"]
                      , "log_Z_at_T=1": log_Z, "n_eff": n_eff, "NLL": NLL}


