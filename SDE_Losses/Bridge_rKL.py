from .Base_SDE_Loss import Base_SDE_Loss_Class
import jax
from jax import numpy as jnp
from functools import partial

class Bridge_rKL_Loss_Class(Base_SDE_Loss_Class):

    def __init__(self, SDE_config, Optimizer_Config,  EnergyClass, Network_Config, model):
        super().__init__(SDE_config, Optimizer_Config, EnergyClass, Network_Config, model)
        self.update_params = self.update_net_params_only

    @partial(jax.jit, static_argnums=(0,))  
    def evaluate_loss(self, params, Energy_params, SDE_params, SDE_tracer, key, temp = 1.0):
        forward_diff_log_probs = SDE_tracer["forward_diff_log_probs"]
        reverse_log_probs = SDE_tracer["reverse_log_probs"]
        entropy_loss = jnp.sum(reverse_log_probs, axis = 0)
        noise_loss = -jnp.sum(forward_diff_log_probs, axis = 0) 

        x_prior = SDE_tracer["x_prior"]
        x_last = SDE_tracer["x_final"]

        log_prior = self.vmap_get_log_prior(SDE_params, x_prior)

        Energy, key = self.EnergyClass.vmap_calc_energy(x_last, Energy_params, key)
        mean_Energy = jnp.mean(Energy)

        res_dict = self.compute_partition_sum(entropy_loss, noise_loss, log_prior, Energy)
        log_Z = res_dict["log_Z"]
        Free_Energy, n_eff, NLL = res_dict["Free_Energy"], res_dict["n_eff"], res_dict["NLL"]


        if(self.optim_mode == "optim"):
            loss = jnp.mean(temp*entropy_loss + temp*noise_loss + Energy + temp*log_prior)
        elif(self.optim_mode == "equilibrium"):
            loss = jnp.mean(entropy_loss + noise_loss + Energy/temp + log_prior)
        return loss, {"loss": loss, "Free_Energy_at_T=1": Free_Energy, "log_Z_at_T=1": log_Z, "n_eff": n_eff, "mean_energy": mean_Energy, 
                      "best_Energy": jnp.min(Energy), "noise_loss": noise_loss, "entropy_loss": entropy_loss, "key": key, "X_0": x_last, 
                      "sigma": jnp.exp(SDE_params["log_sigma"]),"beta_min": jnp.exp(SDE_params["log_beta_min"]),
                        "beta_delta": jnp.exp(SDE_params["log_beta_delta"]), "mean": SDE_params["mean"]
                        }
    


    def compute_partition_sum(self, R_diff, S, log_prior, Energy):
        Z_estim = R_diff + S + log_prior + Energy
        log_Z = jnp.mean(-Z_estim)
        Free_Energy = -log_Z
        log_weights = -Z_estim
        normed_weights = jax.nn.softmax(log_weights, axis = -1)

        n_eff = 1/(jnp.sum(normed_weights**2)*Z_estim.shape[0])

        NLL = -jnp.mean(R_diff + S + log_prior) 
        res_dict = {"Free_Energy": Free_Energy, "normed_weights": normed_weights, "log_Z": log_Z, "n_eff": n_eff, "NLL": NLL}
        return res_dict