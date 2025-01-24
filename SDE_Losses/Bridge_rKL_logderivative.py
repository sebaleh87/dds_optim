from .Base_SDE_Loss import Base_SDE_Loss_Class
import jax
from jax import numpy as jnp
from functools import partial

#see Sequential Controlled Langevin Diffusions (16)

class Bridge_rKL_logderiv_Loss_Class(Base_SDE_Loss_Class):

    def __init__(self, SDE_config, Optimizer_Config,  EnergyClass, Network_Config, model):
        super().__init__(SDE_config, Optimizer_Config, EnergyClass, Network_Config, model)
        self.SDE_type.stop_gradient = True
        #self.update_params = self.update_net_params_only

    @partial(jax.jit, static_argnums=(0,))  
    def evaluate_loss(self, params, Energy_params, SDE_params, SDE_tracer, key, temp = 1.0):
        ### TODO check why mean is not learned!
        ts = SDE_tracer["ts"]
        forward_diff_log_probs = SDE_tracer["forward_diff_log_probs"]
        reverse_log_probs = SDE_tracer["reverse_log_probs"]
        log_prob_prior_scaled = SDE_tracer["log_prob_prior_scaled"]

        entropy_minus_noise = jnp.sum(reverse_log_probs - forward_diff_log_probs, axis = 0)

        x_prior = SDE_tracer["x_prior"]
        x_last = SDE_tracer["x_final"]

        log_prior = self.vmap_get_log_prior(SDE_params, x_prior)

        Energy, key = self.EnergyClass.vmap_calc_energy(x_last, Energy_params, key)
        mean_Energy = jnp.mean(Energy)

        entropy_loss = jnp.mean(jnp.sum(reverse_log_probs, axis = 0) )
        noise_loss = jnp.mean(-jnp.sum(forward_diff_log_probs, axis = 0))


        if self.SDE_type.config['use_off_policy']:  
            log_prob_prior_scaled = SDE_tracer["log_prob_prior_scaled"]
            scale_log_prob = SDE_tracer["scale_log_prob"]
            log_prob_noise = SDE_tracer["log_prob_noise"]
            log_weights = reverse_log_probs - log_prob_noise + log_prior - log_prob_prior_scaled #- scale_log_prob
            off_policy_weights = jax.lax.stop_gradient(jnp.exp(log_weights - jnp.max(log_weights)))
            loss = self.compute_rKL_log_deriv(SDE_params, log_prior, reverse_log_probs, forward_diff_log_probs, entropy_minus_noise,Energy, temp, ts, off_policy_weights)
        else:
            off_policy_weights = 1.
            loss = self.compute_rKL_log_deriv(SDE_params, log_prior, reverse_log_probs, forward_diff_log_probs, entropy_minus_noise,Energy, temp, ts)

        log_dict = {"loss": loss, "mean_energy": mean_Energy, 
                      "best_Energy": jnp.min(Energy), "noise_loss": noise_loss, "entropy_loss": entropy_loss, "key": key, "X_0": x_last, 
                      "sigma": jnp.exp(SDE_params["log_sigma"]),"beta_min": jnp.exp(SDE_params["log_beta_min"]),
                        "beta_delta": jnp.exp(SDE_params["log_beta_delta"]), "mean": SDE_params["mean"], "sigma_prior": jnp.exp(SDE_params["log_sigma_prior"])
                        }

        log_dict = self.compute_partition_sum(entropy_minus_noise, jnp.zeros_like(entropy_minus_noise), log_prior, Energy, log_dict, off_policy_weights = off_policy_weights)

        return loss, log_dict

    def compute_rKL_log_deriv(self, SDE_params, log_prior, reverse_log_probs, forward_diff_log_probs, entropy_minus_noise,Energy, temp, ts, off_policy_weights = 1.):

        if(self.optim_mode == "optim"):
            sum_reverse_log_probs = jnp.sum(reverse_log_probs, axis = 0) + log_prior
            radon_dykodin_derivative = T*log_prior + T*entropy_minus_noise + Energy

            reward = jax.lax.stop_gradient(radon_dykodin_derivative - jnp.mean(radon_dykodin_derivative, keepdims=True, axis = 0))
            loss = jnp.mean(off_policy_weights * reward * sum_reverse_log_probs) + jnp.mean(off_policy_weights * radon_dykodin_derivative)
        elif(self.optim_mode == "equilibrium"):
            sum_reverse_log_probs = jnp.sum(reverse_log_probs, axis = 0) + log_prior
            radon_dykodin_derivative = log_prior + entropy_minus_noise + Energy/temp
            radon_nykodin_wo_reverse = -jnp.sum(forward_diff_log_probs, axis = 0) + Energy/temp

            #print("log_prior", log_prior.shape, sum_reverse_log_probs.shape, radon_dykodin_derivative.shape)
            reward = jax.lax.stop_gradient(radon_dykodin_derivative - jnp.mean(radon_dykodin_derivative, keepdims=True, axis = 0))
            loss = jnp.mean(off_policy_weights * reward * sum_reverse_log_probs) + jnp.mean(off_policy_weights * radon_nykodin_wo_reverse)

        return loss
    
