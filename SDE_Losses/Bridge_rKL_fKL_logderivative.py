from .Base_SDE_Loss import Base_SDE_Loss_Class
import jax
from jax import numpy as jnp
from functools import partial
from jax import nn
from .loss_utils import compute_rKL_log_deriv, compute_fKL_log_deriv
# Note: Functions from Bridge_fKL_logderivative.py and Bridge_rKL_logderivative.py 
# have been moved to loss_utils.py and are imported here

### TODO implement add Bridge_rKL_rKL_logderiv_Loss_Class to SDE_Loss_registry in __init__.py and also to argparse in main.py

### try out on GMM-2D and if it works try out sweeps in Configs/Sweeps/GMM/

class Bridge_rKL_fKL_logderiv_Loss_Class(Base_SDE_Loss_Class):

    def __init__(self, SDE_config, Optimizer_Config,  EnergyClass, Network_Config, model):
        super().__init__(SDE_config, Optimizer_Config, EnergyClass, Network_Config, model)
        self.SDE_type.stop_gradient = True
        self.quantile = SDE_config["quantile"]
        self.weight_temperature = SDE_config["weight_temperature"]
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
            raise ValueError("for now implement version without off policy")
            log_prob_prior_scaled = SDE_tracer["log_prob_prior_scaled"]
            log_prob_noise = SDE_tracer["log_prob_noise"]
            log_prob_on_policy = SDE_tracer["log_prob_on_policy"]

            reverse_log_probs_for_weights = jnp.concatenate([log_prob_prior_scaled[None, :], log_prob_noise], axis = 0)
            forward_log_probs_for_weights = jnp.concatenate([forward_diff_log_probs, -Energy[None, :]/temp, ], axis = 0)

            log_weights = jnp.sum(forward_log_probs_for_weights - reverse_log_probs_for_weights, axis = 0)

            off_policy_weights_normed = jax.lax.stop_gradient(jax.nn.softmax(log_weights, axis = -1))
            off_policy_weights_normed_times_n_samples = off_policy_weights_normed* log_prob_on_policy.shape[-1] ### multiply wiht numer of states so that mean instead of sum can be used later on
            loss, L1, L2 = self.compute_fKL_log_deriv(SDE_params, log_prior, reverse_log_probs, forward_diff_log_probs, entropy_minus_noise,Energy, temp, 
                        off_policy_weights_normed_times_n_samples, use_off_policy = True)
        else:
            off_policy_weights_normed_times_n_samples = 1.
            
            # Calculate both rKL and fKL losses using the imported utility functions
            fKL_loss, fKL_L1, fKL_L2 = compute_fKL_log_deriv(
                self.optim_mode, log_prior, reverse_log_probs, forward_diff_log_probs, 
                entropy_minus_noise, Energy, temp
            )
            
            rKL_loss, rKL_unbiased, rKL_centered = compute_rKL_log_deriv(
                self.optim_mode, log_prior, reverse_log_probs, forward_diff_log_probs, 
                entropy_minus_noise, Energy, temp
            )
            
            # Combine the losses
            loss = (fKL_loss + rKL_loss)
            L1 = (fKL_L1 + rKL_unbiased)
            L2 = (fKL_L2 + rKL_centered)

        log_dict = {"loss": loss, "mean_energy": mean_Energy, "losses/L1": L1, "losses/L2": L2,
                      "best_Energy": jnp.min(Energy), "noise_loss": noise_loss, "entropy_loss": entropy_loss, "key": key, "X_0": x_last, 
                      "sigma": jnp.exp(SDE_params["log_sigma"]),"beta_min": jnp.exp(SDE_params["log_beta_min"]),
                        "beta_delta": jnp.exp(SDE_params["log_beta_delta"]), "mean": SDE_params["mean"], "sigma_prior": jnp.exp(SDE_params["log_sigma_prior"]),
                        "fKL_loss": fKL_loss, "rKL_loss": rKL_loss
                        }

        log_dict = self.compute_partition_sum(entropy_minus_noise, jnp.zeros_like(entropy_minus_noise), log_prior, Energy, log_dict, off_policy_weights = off_policy_weights_normed_times_n_samples)

        return loss, log_dict
