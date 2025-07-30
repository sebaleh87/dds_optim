import jax
from jax import numpy as jnp

def compute_rKL_log_deriv(optim_mode, log_prior, reverse_log_probs, forward_diff_log_probs, entropy_minus_noise, Energy, temp, off_policy_weights=1., use_off_policy=False):
    """
    Compute the reverse KL divergence loss using log derivatives.
    """
    if optim_mode == "optim":
        sum_reverse_log_probs = jnp.sum(reverse_log_probs, axis=0) + log_prior
        radon_dykodin_derivative = temp*log_prior + temp*entropy_minus_noise + Energy
        radon_nykodin_wo_reverse = -temp*jnp.sum(forward_diff_log_probs, axis=0) + Energy

    elif optim_mode == "equilibrium":
        sum_reverse_log_probs = jnp.sum(reverse_log_probs, axis=0) + log_prior
        radon_dykodin_derivative = log_prior + entropy_minus_noise + Energy/temp
        radon_nykodin_wo_reverse = -jnp.sum(forward_diff_log_probs, axis=0) + Energy/temp

    if use_off_policy:
        biased_mean = jax.lax.stop_gradient(jnp.mean(radon_dykodin_derivative, keepdims=True, axis=0))
        reward = jax.lax.stop_gradient((radon_dykodin_derivative - biased_mean))
        L1 = jnp.mean((off_policy_weights * reward) * sum_reverse_log_probs)
        loss = L1 + jnp.mean(off_policy_weights * radon_nykodin_wo_reverse)

        unbiased_loss = jnp.mean((off_policy_weights * reward) * sum_reverse_log_probs) + jnp.mean(off_policy_weights * radon_nykodin_wo_reverse)
        centered_loss = L1
    else:
        unbiased_mean = jax.lax.stop_gradient(jnp.mean(radon_dykodin_derivative, keepdims=True, axis=0))
        reward = jax.lax.stop_gradient((radon_dykodin_derivative - unbiased_mean))
        L1 = jnp.mean(reward * sum_reverse_log_probs)
        loss = L1 + jnp.mean(radon_nykodin_wo_reverse)

        unbiased_loss = jnp.mean(jax.lax.stop_gradient((radon_dykodin_derivative)) * sum_reverse_log_probs) + jnp.mean(radon_nykodin_wo_reverse)
        centered_loss = L1

    return loss, unbiased_loss, centered_loss


def compute_fKL_log_deriv(optim_mode, log_prior, reverse_log_probs, forward_diff_log_probs, entropy_minus_noise, Energy, temp, off_policy_weights=1., use_off_policy=False):
    """
    Compute the forward KL divergence loss using log derivatives.
    """
    log_prob_target = -Energy/temp
    
    if optim_mode == "optim":
        sum_forward_log_probs = temp*jnp.sum(forward_diff_log_probs, axis=0) - Energy
        radon_dykodin_derivative = -(temp*log_prior + temp*entropy_minus_noise + Energy)
        radon_nykodin_wo_forward = -(jnp.sum(reverse_log_probs, axis=0) + log_prior)

    elif optim_mode == "equilibrium":
        sum_forward_log_probs = jnp.sum(forward_diff_log_probs, axis=0) + log_prob_target
        radon_dykodin_derivative = -(log_prior + entropy_minus_noise + Energy/temp)
        radon_nykodin_wo_forward = -(jnp.sum(reverse_log_probs, axis=0) + log_prior)

    if use_off_policy:
        biased_mean = jax.lax.stop_gradient(jnp.mean(radon_dykodin_derivative, keepdims=True, axis=0))
        reward = jax.lax.stop_gradient((radon_dykodin_derivative - biased_mean))
        L1 = jnp.mean((off_policy_weights * reward) * sum_forward_log_probs)
        loss = L1 + jnp.mean(off_policy_weights * radon_nykodin_wo_forward)

        L1_log = jnp.mean((off_policy_weights * reward) * sum_forward_log_probs) 
        L2_log = jnp.mean(off_policy_weights * radon_nykodin_wo_forward)
        
        # Note: The original implementation raises an error here with "Not implemented"
        # I've removed that since this needs to be a functional utility
        return loss, L1_log, L2_log
    else:
        # quantile = 0.1
        # log_max_quantile = jnp.quantile(radon_dykodin_derivative, quantile, axis = -1)
        # log_weights_max_quantile = log_max_quantile
        # delta_log_weights = jnp.maximum(radon_dykodin_derivative, log_weights_max_quantile)

        # print(log_prior.shape, delta_log_weights.shape,log_max_quantile.shape, log_weights_max_quantile.shape)

        # THIS is not the implementation of fKL as a regularizer to prevent mode collapse
        importance_weights = 1.#jax.lax.stop_gradient(jax.nn.softmax(radon_dykodin_derivative, axis=-1)) * radon_dykodin_derivative.shape[-1]

        unbiased_mean = jax.lax.stop_gradient(jnp.mean(radon_dykodin_derivative, keepdims=True, axis=-1))
        reward = jax.lax.stop_gradient((radon_dykodin_derivative - unbiased_mean))
        L1 = jnp.mean(importance_weights*reward * sum_forward_log_probs)
        L2 = jnp.mean(importance_weights * radon_nykodin_wo_forward)
        loss = L1 + L2

        L1_log = jnp.mean(jax.lax.stop_gradient(importance_weights * radon_dykodin_derivative) * sum_forward_log_probs) 
        L2_log = jnp.mean(importance_weights * radon_nykodin_wo_forward)

    return loss, L1_log, L2_log

def compute_fKL_reparam(optim_mode, SDE_params, log_prior, reverse_log_probs, forward_diff_log_probs, entropy_minus_noise,Energy, temp, off_policy_weights = 1., use_off_policy = False):
    log_prob_target = - Energy/temp
    if(optim_mode == "optim"):
        sum_forward_log_probs = temp*jnp.sum(forward_diff_log_probs, axis = 0) - Energy
        #sum_forward_log_probs_without_energy = temp*jnp.sum(forward_diff_log_probs, axis = 0)
        radon_dykodin_derivative = -(temp*log_prior + temp*entropy_minus_noise + Energy)
        #radon_dykodin_derivative_no_energy = -(log_prior + entropy_minus_noise)
        radon_nykodin_wo_forward = -(jnp.sum(reverse_log_probs, axis = 0) + log_prior)

    elif(optim_mode == "equilibrium"):
        sum_forward_log_probs = jnp.sum(forward_diff_log_probs, axis = 0) + log_prob_target
        #sum_forward_log_probs_without_energy = jnp.sum(forward_diff_log_probs, axis = 0)
        radon_dykodin_derivative = -(log_prior + entropy_minus_noise + Energy/temp)
        #radon_dykodin_derivative_no_energy = -(log_prior + entropy_minus_noise)
        radon_nykodin_wo_forward = -(jnp.sum(reverse_log_probs, axis = 0) + log_prior)

    if(use_off_policy):

        raise ValueError("Not implemented")
    else:
        importance_weights = jax.nn.softmax(radon_dykodin_derivative, axis = -1)*radon_dykodin_derivative.shape[-1]

        L1 = jnp.mean(importance_weights*radon_dykodin_derivative)
        loss = L1

        L1_log = L1
        L2_log = 0.

    return loss, L1_log, L2_log
