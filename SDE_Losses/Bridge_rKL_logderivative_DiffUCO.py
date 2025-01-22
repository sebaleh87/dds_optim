from .Base_SDE_Loss import Base_SDE_Loss_Class
import jax
from jax import numpy as jnp
from functools import partial

#see Sequential Controlled Langevin Diffusions (16)

class Bridge_rKL_logderiv_DiffUCO_Loss_Class(Base_SDE_Loss_Class):

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

        entropy_minus_noise = jnp.sum(reverse_log_probs - forward_diff_log_probs, axis = 0)

        x_prior = SDE_tracer["x_prior"]
        x_last = SDE_tracer["x_final"]

        log_prior = self.vmap_get_log_prior(SDE_params, x_prior)

        Energy, key = self.EnergyClass.vmap_calc_energy(x_last, Energy_params, key)
        mean_Energy = jnp.mean(Energy)

        entropy_loss = jnp.mean(jnp.sum(reverse_log_probs, axis = 0) )
        noise_loss = jnp.mean(-jnp.sum(forward_diff_log_probs, axis = 0))


        loss = self.compute_rKL_log_deriv_DiffUCO(SDE_params, log_prior, reverse_log_probs, forward_diff_log_probs, entropy_minus_noise,Energy, temp, ts)

        log_dict = {"loss": loss, "mean_energy": mean_Energy, 
                      "best_Energy": jnp.min(Energy), "noise_loss": noise_loss, "entropy_loss": entropy_loss, "key": key, "X_0": x_last, 
                      "sigma": jnp.exp(SDE_params["log_sigma"]),"beta_min": jnp.exp(SDE_params["log_beta_min"]),
                        "beta_delta": jnp.exp(SDE_params["log_beta_delta"]), "mean": SDE_params["mean"], "sigma_prior": jnp.exp(SDE_params["log_sigma_prior"])
                        }

        log_dict = self.compute_partition_sum(entropy_minus_noise, jnp.zeros_like(entropy_minus_noise), log_prior, Energy, log_dict)

        return loss, log_dict

    def compute_rKL_log_deriv_DiffUCO(self, SDE_params, log_prior, reverse_log_probs, forward_diff_log_probs, entropy_minus_noise,Energy, temp, ts):
        ### TODO compute exact log_prior entropy
        neg_Entropy_prior = - self.SDE_type.get_entropy_prior(SDE_params)
        Entropy_prior_loss = neg_Entropy_prior

        ### TODO compute exact reverse_log_prob_entropy reinforce loss and exact loss
        cumsum_reverse_log_probs = jax.lax.cumsum(reverse_log_probs, axis = 0) + log_prior[None, :]
        #cum_sum_log_probs_shifted = jax.lax.cumsum(jnp.concatenate([log_prior[None, ...], reverse_log_probs[:-1]], axis = 0))

        neg_Entropy_diff_step = - jax.vmap(self.SDE_type.get_entropy_diff_step, in_axes = (None, 0))(SDE_params, ts)
        #entropy_baseline = jax.lax.stop_gradient(Entropy_diff_step - jnp.mean(Entropy_diff_step, axis = -1, keepdims=True))
        #Entropy_reinforce_loss_per_diff_step = jnp.sum(entropy_baseline *cum_sum_log_probs_shifted, axis = 0)
        Entropy_reinforce_loss = 0.#jnp.mean(Entropy_reinforce_loss_per_diff_step)

        Entropy_loss_exact = jnp.mean(jnp.sum(neg_Entropy_diff_step, axis = 0))

        ### TODO compute cross entropy terms and integrate out the future
        forward_diff_log_probs_baseline = jax.lax.stop_gradient(forward_diff_log_probs - jnp.mean(forward_diff_log_probs, axis = -1, keepdims=True))
        cross_entropy_reinforce_loss = -jnp.mean(jnp.sum(forward_diff_log_probs_baseline*cumsum_reverse_log_probs, axis = 0))
        #cross_entropy_reinforce_loss_2 = -jnp.mean(forward_diff_log_probs_baseline[:-1]*cum_sum_log_probs_shifted[:-1])
        cross_entropy_loss_second = -jnp.mean(jnp.sum(forward_diff_log_probs, axis = 0))


        ### TODO compute reinforce loss of energy
        sum_reverse_log_probs = jnp.sum(reverse_log_probs, axis = 0) + log_prior
        energy_reduced = jax.lax.stop_gradient((Energy - jnp.mean(Energy, keepdims=True, axis = 0))/temp)
        Energy_loss = jnp.mean(energy_reduced*sum_reverse_log_probs) 

        loss = Entropy_prior_loss + Energy_loss + Entropy_loss_exact+ Entropy_reinforce_loss + cross_entropy_loss_second  + cross_entropy_reinforce_loss
        return loss
    