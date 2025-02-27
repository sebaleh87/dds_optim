from .Base_SDE_Loss import Base_SDE_Loss_Class
import jax
from jax import numpy as jnp
from functools import partial
from jax import nn

#see Sequential Controlled Langevin Diffusions (16)

class Bridge_fKL_subtraj_Loss_Class(Base_SDE_Loss_Class):

    def __init__(self, SDE_config, Optimizer_Config,  EnergyClass, Network_Config, model):
        super().__init__(SDE_config, Optimizer_Config, EnergyClass, Network_Config, model)
        self.SDE_type.stop_gradient = True
        #self.update_params = self.update_net_params_only

    @partial(jax.jit, static_argnums=(0,))  
    def evaluate_loss(self, params, Energy_params, SDE_params, SDE_tracer, key, temp = 1.0):
        ### TODO check why mean is not learned!
        ts = SDE_tracer["ts"]
        interpol_log_probs = SDE_tracer["interpol_log_probs"][...,0]
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

        ### TODO compute off policy weights
        ### TODO compute DB-fKL loss D_KL_t, t-1
        print(interpol_log_probs.shape , forward_diff_log_probs.shape, log_prior.shape, jax.lax.cumsum(forward_diff_log_probs - reverse_log_probs, axis = 0).shape)
        unnormed_log_weights = jax.lax.cumsum(forward_diff_log_probs - reverse_log_probs, axis = 0) + interpol_log_probs[1:] - log_prior[None, ...]
        importance_weights = jax.lax.stop_gradient(jax.nn.softmax(unnormed_log_weights, axis = -1))*unnormed_log_weights.shape[-1] ### multiply wiht numer of states so that mean instead of sum can be used later on

        reverse_step = reverse_log_probs + interpol_log_probs[:-1]
        forward_step = forward_diff_log_probs + interpol_log_probs[1:]
        radon_nycodin_deriv = forward_step - reverse_step
        radon_nycodin_deriv_baseline = jax.lax.stop_gradient(jnp.mean(radon_nycodin_deriv, axis = -1, keepdims = True))
        Reward = jax.lax.stop_gradient(radon_nycodin_deriv - radon_nycodin_deriv_baseline)
        D_KL_per_t = jnp.mean((importance_weights*Reward)*forward_step, axis = -1) - jnp.mean(importance_weights*reverse_step, axis = -1)
        print("shapes", D_KL_per_t.shape, reverse_step.shape, forward_step.shape, importance_weights.shape)
        loss = jnp.sum(D_KL_per_t)

        log_dict = {"loss": loss, "mean_energy": mean_Energy, 
                      "best_Energy": jnp.min(Energy), "noise_loss": noise_loss, "entropy_loss": entropy_loss, "key": key, "X_0": x_last, 
                      "sigma": jnp.exp(SDE_params["log_sigma"]),"beta_min": jnp.exp(SDE_params["log_beta_min"]),
                        "beta_delta": jnp.exp(SDE_params["log_beta_delta"]), "mean": SDE_params["mean"], "sigma_prior": jnp.exp(SDE_params["log_sigma_prior"])
                        }

        log_dict = self.compute_partition_sum(entropy_minus_noise, jnp.zeros_like(entropy_minus_noise), log_prior, Energy, log_dict)

        return loss, log_dict

    def compute_fKL_DB(self, SDE_params, log_prior, reverse_log_probs, forward_diff_log_probs, entropy_minus_noise,Energy, temp, off_policy_weights = 1., use_off_policy = False):

        if(self.optim_mode == "optim"):
            sum_reverse_log_probs = jnp.sum(reverse_log_probs, axis = 0) + log_prior
            radon_dykodin_derivative = temp*log_prior + temp*entropy_minus_noise + Energy
            radon_nykodin_wo_reverse = -jnp.sum(forward_diff_log_probs, axis = 0) + Energy/temp

        elif(self.optim_mode == "equilibrium"):
            sum_reverse_log_probs = jnp.sum(reverse_log_probs, axis = 0) + log_prior
            radon_dykodin_derivative = log_prior + entropy_minus_noise + Energy/temp
            radon_nykodin_wo_reverse = -jnp.sum(forward_diff_log_probs, axis = 0) + Energy/temp

        unbiased_mean = jax.lax.stop_gradient(jnp.mean(radon_dykodin_derivative, keepdims=True, axis = 0))
        reward = jax.lax.stop_gradient((radon_dykodin_derivative-unbiased_mean))
        #loss = jnp.mean((off_policy_weights* reward - unbiased_mean) * sum_reverse_log_probs) + jnp.mean(off_policy_weights * radon_nykodin_wo_reverse)
        loss = jnp.mean(reward * sum_reverse_log_probs) + jnp.mean( radon_nykodin_wo_reverse)

        return loss
    
