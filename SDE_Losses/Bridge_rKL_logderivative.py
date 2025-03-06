from .Base_SDE_Loss import Base_SDE_Loss_Class
import jax
from jax import numpy as jnp
from functools import partial
from jax import nn

#see Sequential Controlled Langevin Diffusions (16)

class Bridge_rKL_logderiv_Loss_Class(Base_SDE_Loss_Class):

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
            log_prob_prior_scaled = SDE_tracer["log_prob_prior_scaled"]
            log_prob_noise = SDE_tracer["log_prob_noise"]
            log_prob_on_policy = SDE_tracer["log_prob_on_policy"]

            delta_log_prior = log_prior - log_prob_prior_scaled 
            delta_log_prob = log_prob_on_policy - log_prob_noise
            delta_log_weights = jnp.concatenate([delta_log_prior[None, :], delta_log_prob], axis = 0)
            if(self.quantile != 0):
                quantile = self.quantile
                log_max_quantile = jnp.quantile(delta_log_weights, quantile, axis = -1)
                log_weights_max_quantile = log_max_quantile
                delta_log_weights = jnp.maximum(delta_log_weights, log_weights_max_quantile[:, None])

            log_weights = jnp.sum(self.weight_temperature* delta_log_weights, axis = 0)



            # log_weights = jnp.nan_to_num(log_weights, nan=0.0, posinf=1e10, neginf=-1e10)
            # Energy = jnp.nan_to_num(Energy, nan=1e10, posinf=1e10)
            off_policy_weights_normed = jax.lax.stop_gradient(jax.nn.softmax(log_weights, axis = -1))
            off_policy_weights_normed_times_n_samples = off_policy_weights_normed* log_prob_on_policy.shape[-1] ### multiply wiht numer of states so that mean instead of sum can be used later on
            loss, unbiased_loss, centered_loss = self.compute_rKL_log_deriv(SDE_params, log_prior, reverse_log_probs, forward_diff_log_probs, entropy_minus_noise,Energy, temp, 
                        off_policy_weights_normed_times_n_samples, use_off_policy = True)
        else:
            off_policy_weights_normed_times_n_samples = 1.
            loss, unbiased_loss, centered_loss = self.compute_rKL_log_deriv(SDE_params, log_prior, reverse_log_probs, forward_diff_log_probs, entropy_minus_noise,Energy, temp)

        log_dict = {"loss": loss, "mean_energy": mean_Energy, "losses/unbiased_loss": unbiased_loss, "losses/centered_loss": centered_loss,
                      "best_Energy": jnp.min(Energy), "noise_loss": noise_loss, "entropy_loss": entropy_loss, "key": key, "X_0": x_last, 
                      "sigma": jnp.exp(SDE_params["log_sigma"]),"beta_min": jnp.exp(SDE_params["log_beta_min"]),
                        "beta_delta": jnp.exp(SDE_params["log_beta_delta"]), "mean": SDE_params["mean"], "sigma_prior": jnp.exp(SDE_params["log_sigma_prior"])
                        }

        log_dict = self.compute_partition_sum(entropy_minus_noise, jnp.zeros_like(entropy_minus_noise), log_prior, Energy, log_dict, off_policy_weights = off_policy_weights_normed_times_n_samples)

        return loss, log_dict

    def compute_rKL_log_deriv(self, SDE_params, log_prior, reverse_log_probs, forward_diff_log_probs, entropy_minus_noise,Energy, temp, off_policy_weights = 1., use_off_policy = False):

        if(self.optim_mode == "optim"):
            sum_reverse_log_probs = jnp.sum(reverse_log_probs, axis = 0) + log_prior
            radon_dykodin_derivative = temp*log_prior + temp*entropy_minus_noise + Energy
            radon_nykodin_wo_reverse = -temp*jnp.sum(forward_diff_log_probs, axis = 0) + Energy

        elif(self.optim_mode == "equilibrium"):
            sum_reverse_log_probs = jnp.sum(reverse_log_probs, axis = 0) + log_prior
            radon_dykodin_derivative = log_prior + entropy_minus_noise + Energy/temp
            radon_nykodin_wo_reverse = -jnp.sum(forward_diff_log_probs, axis = 0) + Energy/temp

        #print("log_prior", log_prior.shape, sum_reverse_log_probs.shape, radon_dykodin_derivative.shape)
        #unbiased_mean = jax.lax.stop_gradient(jnp.mean(off_policy_weights*radon_dykodin_derivative, keepdims=True, axis = 0))
        if(use_off_policy):
            if(False):
                unbiased_mean = jax.lax.stop_gradient(jnp.mean(off_policy_weights*radon_dykodin_derivative, keepdims=True, axis = 0))
                reward = jax.lax.stop_gradient((radon_dykodin_derivative))
                L1 = jnp.mean((off_policy_weights* reward - unbiased_mean) * sum_reverse_log_probs)
                loss = L1 + jnp.mean(off_policy_weights * radon_nykodin_wo_reverse)
                unbiased_loss = jnp.mean((off_policy_weights* reward) * sum_reverse_log_probs) + jnp.mean(off_policy_weights * radon_nykodin_wo_reverse)
                centered_loss = L1
            else:
                biased_mean = jax.lax.stop_gradient(jnp.mean(radon_dykodin_derivative, keepdims=True, axis = 0))
                reward = jax.lax.stop_gradient((radon_dykodin_derivative - biased_mean))
                L1 = jnp.mean((off_policy_weights* reward ) * sum_reverse_log_probs)
                loss = L1 + jnp.mean(off_policy_weights * radon_nykodin_wo_reverse)

                unbiased_loss = jnp.mean((off_policy_weights* reward) * sum_reverse_log_probs) + jnp.mean(off_policy_weights * radon_nykodin_wo_reverse)
                centered_loss = L1

        else:
            unbiased_mean = jax.lax.stop_gradient(jnp.mean(radon_dykodin_derivative, keepdims=True, axis = 0))
            reward = jax.lax.stop_gradient((radon_dykodin_derivative-unbiased_mean))
            L1 = jnp.mean(reward * sum_reverse_log_probs)
            loss = L1+ jnp.mean( radon_nykodin_wo_reverse)

            unbiased_loss = jnp.mean(jax.lax.stop_gradient((radon_dykodin_derivative)) * sum_reverse_log_probs) + jnp.mean( radon_nykodin_wo_reverse)
            centered_loss = L1

        return loss, unbiased_loss, centered_loss

    
