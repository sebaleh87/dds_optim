from .Base_SDE_Loss import Base_SDE_Loss_Class
import jax
from jax import numpy as jnp
from functools import partial
from jax import nn
from .loss_utils import compute_rKL_log_deriv, compute_fKL_log_deriv, compute_fKL_reparam

#see Sequential Controlled Langevin Diffusions (16)

class Bridge_fKL_reparam_Loss_Class(Base_SDE_Loss_Class):

    def __init__(self, SDE_config, Optimizer_Config,  EnergyClass, Network_Config, model):
        super().__init__(SDE_config, Optimizer_Config, EnergyClass, Network_Config, model)
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

            reverse_log_probs_for_weights = jnp.concatenate([log_prob_prior_scaled[None, :], log_prob_noise], axis = 0)
            forward_log_probs_for_weights = jnp.concatenate([forward_diff_log_probs, -Energy[None, :]/temp, ], axis = 0)

            log_weights = jnp.sum(forward_log_probs_for_weights - reverse_log_probs_for_weights, axis = 0)

            off_policy_weights_normed = jax.lax.stop_gradient(jax.nn.softmax(log_weights, axis = -1))
            off_policy_weights_normed_times_n_samples = off_policy_weights_normed* log_prob_on_policy.shape[-1] ### multiply wiht numer of states so that mean instead of sum can be used later on
            loss, L1, L2 = compute_fKL_reparam(self.optim_mode, SDE_params, log_prior, reverse_log_probs, forward_diff_log_probs, entropy_minus_noise, Energy, temp, 
                        off_policy_weights_normed_times_n_samples, use_off_policy = True)
        else:
            off_policy_weights_normed_times_n_samples = 1.
            loss, L1, L2 = compute_fKL_reparam(self.optim_mode, SDE_params, log_prior, reverse_log_probs, forward_diff_log_probs, entropy_minus_noise,Energy, temp)

        log_dict = {"loss": loss, "mean_energy": mean_Energy, "losses/L1": L1, "losses/L2": L2,
                      "best_Energy": jnp.min(Energy), "noise_loss": noise_loss, "entropy_loss": entropy_loss, "key": key, "X_0": x_last, 
                      "sigma": jnp.exp(SDE_params["log_sigma"]),"beta_min": jnp.exp(SDE_params["log_beta_min"]),
                        "beta_delta": jnp.exp(SDE_params["log_beta_delta"]), "mean": SDE_params["mean"], "sigma_prior": jnp.exp(SDE_params["log_sigma_prior"])
                        }

        log_dict = self.compute_partition_sum(entropy_minus_noise, jnp.zeros_like(entropy_minus_noise), log_prior, Energy, log_dict, off_policy_weights = off_policy_weights_normed_times_n_samples)

        return loss, log_dict

    def compute_rKL_LD(self, SDE_tracer, params, Energy_params, SDE_params):
        xs_stopped = jax.lax.stop_gradient(SDE_tracer["xs"])

        ### TODO make forward pass

        pass


#simpliefied Kl loss terms 
    
# reverse_log_probs_over_t = reverse_log_probs
# reverse_log_probs_over_t = reverse_log_probs_over_t.at[0].set(reverse_log_probs_over_t[0] + log_prior)
# forward_log_probs_over_t = jnp.concatenate([forward_diff_log_probs, -Energy[None, :]/temp, ], axis = 0)

# unnormed_log_weights = jax.lax.cumsum(forward_diff_log_probs - reverse_log_probs, axis = 0) - log_prior[None, :]
# importance_weights = jax.lax.stop_gradient(jax.nn.softmax(unnormed_log_weights, axis = -1))*unnormed_log_weights.shape[-1]

# radon_nycodin_per_step = forward_diff_log_probs - reverse_log_probs
# radon_nycodin_per_step = radon_nycodin_per_step.at[0].set(radon_nycodin_per_step[0] - log_prior)
# forward_kl_cum_sum = jax.lax.cumsum(forward_log_probs_over_t[0:-1], axis = 0)

# unbiased_mean = jax.lax.stop_gradient(jnp.mean(radon_nycodin_per_step, keepdims=True, axis = -1))
# reward = jax.lax.stop_gradient((radon_nycodin_per_step-unbiased_mean))
# L1 = jnp.mean(jnp.sum(importance_weights*reward * forward_kl_cum_sum, axis = 0))
# L2 = jnp.mean(jnp.sum(-importance_weights*reverse_log_probs_over_t, axis = 0))
# loss = L1 + L2

# L1_log = jnp.mean(jnp.sum(importance_weights*(reward + unbiased_mean) * forward_kl_cum_sum, axis = 0))
# L2_log = L2
# print((importance_weights*reward * forward_kl_cum_sum).shape, importance_weights.shape, reward.shape, forward_kl_cum_sum.shape, radon_nycodin_per_step.shape)