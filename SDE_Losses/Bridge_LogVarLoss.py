from .Base_SDE_Loss import Base_SDE_Loss_Class
import jax
from jax import numpy as jnp
from functools import partial

class Bridge_LogVarLoss_Class(Base_SDE_Loss_Class):

    def __init__(self, SDE_config, Optimizer_Config,  EnergyClass, Network_Config, model):
        super().__init__(SDE_config, Optimizer_Config, EnergyClass, Network_Config, model)
        self.SDE_type.stop_gradient = True
        #self.update_params = self.update_net_params_only

    @partial(jax.jit, static_argnums=(0,))  
    def evaluate_loss(self, params, Energy_params, SDE_params, SDE_tracer, key, temp = 1.0):
        forward_diff_log_probs = SDE_tracer["forward_diff_log_probs"]
        reverse_log_probs = SDE_tracer["reverse_log_probs"]
        entropy_loss = jnp.sum(reverse_log_probs, axis = 0)
        noise_loss = -jnp.sum(forward_diff_log_probs, axis = 0) 
        entropy_minus_noise = jnp.sum(reverse_log_probs - forward_diff_log_probs, axis = 0)

        x_prior = SDE_tracer["x_prior"]
        x_last = SDE_tracer["x_final"]

        log_prior = self.vmap_get_log_prior(SDE_params, x_prior)

        Energy, key = self.EnergyClass.vmap_calc_energy(x_last, Energy_params, key)
        mean_Energy = jnp.mean(Energy)


        if(self.optim_mode == "optim"):
            obs = temp*entropy_minus_noise + Energy + temp*log_prior
        elif(self.optim_mode == "equilibrium"):
            obs = entropy_minus_noise + Energy/temp + log_prior

        log_var_loss = jnp.mean((obs)**2) - jnp.mean(obs)**2#jnp.var(obs)

        res_dict = self.compute_partition_sum(entropy_minus_noise, jnp.zeros_like(entropy_minus_noise), log_prior, Energy)
        log_Z = res_dict["log_Z"]
        Free_Energy, n_eff, NLL = res_dict["Free_Energy"], res_dict["n_eff"], res_dict["NLL"]

        Entropy = jnp.mean(entropy_loss) #+ jnp.mean(log_prior)

        return log_var_loss, {"loss": log_var_loss, "losses/log_var": log_var_loss, "Entropy": Entropy, "Free_Energy_at_T=1": Free_Energy, "log_Z_at_T=1": log_Z, "n_eff": n_eff, "mean_energy": mean_Energy, 
                      "best_Energy": jnp.min(Energy), "noise_loss": noise_loss, "entropy_loss": entropy_loss, "key": key, "X_0": x_last, 
                      "sigma": jnp.exp(SDE_params["log_sigma"]),"beta_min": jnp.exp(SDE_params["log_beta_min"]),
                        "beta_delta": jnp.exp(SDE_params["log_beta_delta"]), "mean": SDE_params["mean"], "sigma_prior": jnp.exp(SDE_params["log_sigma_prior"])
                        }
    
