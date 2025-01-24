from .Base_SDE_Loss import Base_SDE_Loss_Class
import jax
from jax import numpy as jnp
from functools import partial


class Reverse_KL_Loss_log_deriv_Class(Base_SDE_Loss_Class):

    def __init__(self, SDE_config, Optimizer_Config, EnergyClass, Network_Config, model):
        super().__init__(SDE_config, Optimizer_Config, EnergyClass, Network_Config, model)
        self.SDE_type.stop_gradient = True
        self.vmap_get_drift = jax.vmap(self.SDE_type.get_drift, in_axes = (None, 0, 0))
        #self.vmap_log_prior_sigma_scale = jax.vmap(self.SDE_type.get_log_prior, in_axes = (None, 0, None))

    @partial(jax.jit, static_argnums=(0,))  
    def evaluate_loss(self, params, Energy_params, SDE_params, SDE_tracer, key, temp = 1.0):
        score = SDE_tracer["scores"]
        ts = SDE_tracer["ts"]
        dW = SDE_tracer["dW"]
        dts = SDE_tracer["dts"][...,None]
        xs = SDE_tracer["xs"]

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

        forward_drift = self.vmap_get_drift(SDE_params, xs, ts)
        ### TODO get forward drift and score
        if self.SDE_type.config['use_off_policy']:  
            noise_scale = SDE_tracer["noise_scale"]
            scale_log_prob = SDE_tracer["scale_log_prob"]
            diff_factor_X_scaled = jax.lax.stop_gradient(diff_factor*noise_scale[None, ...])
            drift_X_scaled = jax.lax.stop_gradient((diff_factor_X_scaled)**2*score - forward_drift)
            log_prior_X_scaled = jax.lax.stop_gradient(self.SDE_type.get_log_prior(SDE_params, x_prior, noise_scale) + scale_log_prob)

            drift_Y = diff_factor**2*score - forward_drift
            drift_X = jax.lax.stop_gradient(drift_Y )  
            log_prior_X = jax.lax.stop_gradient(log_prior)
            log_prior_Y = log_prior
            diff_factor_X = jax.lax.stop_gradient(diff_factor)
            diff_factor_Y = diff_factor

            _, _, p_weights = self.compute_radon_nykodyn_derivative( diff_factor_X_scaled, diff_factor_X, drift_X_scaled, drift_X, dW, dts, log_prior_X_scaled, log_prior_X)
            log_p_ref, _, _ = self.compute_radon_nykodyn_derivative( diff_factor_X, diff_factor_Y, drift_X, drift_Y, dW, dts, log_prior_X, log_prior_Y)

            U = diff_factor*score
            stop_grad_U = jax.lax.stop_gradient(diff_factor_X_scaled*score)
            #f = (1/2*jnp.sum( ( U)**2, axis = -1) + jnp.sum(drift_divergence, axis = -1))
            f = (jnp.sum( U * stop_grad_U - U**2/2, axis = -1) + jnp.sum(drift_divergence, axis = -1))
            f_no_stop = (jnp.sum( U**2/2, axis = -1) + jnp.sum(drift_divergence, axis = -1))
            S_no_stop = 0.
        else:
            drift_Y = diff_factor**2*score - forward_drift
            drift_X = jax.lax.stop_gradient(drift_Y )  
            log_prior_X = jax.lax.stop_gradient(log_prior)
            log_prior_Y = log_prior
            diff_factor_X = jax.lax.stop_gradient(diff_factor)
            diff_factor_Y = diff_factor

            log_p_ref, _, _ = self.compute_radon_nykodyn_derivative( diff_factor_X, diff_factor_Y, drift_X, drift_Y, dW, dts, log_prior_X, log_prior_Y)
            p_weights = 1.

            U = diff_factor*score
            stop_grad_U = jax.lax.stop_gradient(diff_factor_X*score)
            #f = (1/2*jnp.sum( ( U)**2, axis = -1) + jnp.sum(drift_divergence, axis = -1))
            f = (jnp.sum( U * stop_grad_U - U**2/2, axis = -1) + jnp.sum(drift_divergence, axis = -1))
            f_no_stop = f
            S_no_stop = jnp.sum(jnp.sum(U * dW, axis = -1), axis = 0)

        S = jnp.sum(jnp.sum(U * dW, axis = -1), axis = 0)
        R_diff = jnp.sum(dts*f  , axis = 0)
        R_diff_no_stop = jnp.sum(dts*f_no_stop  , axis = 0)
        mean_R_diff = jnp.mean(R_diff)

        if(not self.Network_Config["model_mode"] == "latent"):
            if(self.optim_mode == "optim"):
                loss_value = temp*(R_diff + S+ log_prior) + Energy
                loss_value_no_stop = temp*(R_diff_no_stop + S_no_stop+ log_prior) + Energy
            elif(self.optim_mode == "equilibrium"):
                loss_value = (R_diff + S+ log_prior) + Energy/temp
                loss_value_no_stop = (R_diff_no_stop + S_no_stop + log_prior) + Energy/temp
            else:
                raise ValueError(f"Unknown optim_mode: {self.optim}")
            
            loss_baseline = jax.lax.stop_gradient(loss_value - jnp.mean(loss_value, keepdims=True))
            loss_log_deriv = jnp.mean(p_weights*loss_baseline* log_p_ref)
            loss = loss_log_deriv + jnp.mean(p_weights*loss_value_no_stop)
        else:
            raise ValueError("Not implemented yet")
        
        Entropy = -(mean_R_diff + mean_log_prior)

        #print("RKL LOss", mean_R_diff, mean_log_prior, beta*mean_Energy)
        log_dict = {"mean_energy": mean_Energy, "Entropy": Entropy, "R_diff": R_diff, "likelihood_ratio": jnp.mean(loss), 
                      "key": key, "X_0": x_last, "mean_X_prior": jnp.mean(x_prior), #"std_X_prior": jnp.mean(jnp.std(x_prior, axis = 0)), 
                       "sigma": jnp.exp(SDE_params["log_sigma"]),
                      "beta_min": jnp.exp(SDE_params["log_beta_min"]), "beta_delta": jnp.exp(SDE_params["log_beta_delta"]), "mean": SDE_params["mean"]}
        log_dict = self.compute_partition_sum(R_diff, S, log_prior, Energy, log_dict)

        return loss, log_dict
    
    def compute_radon_nykodyn_derivative(self, diff_factor_X, diff_factor_Y, drift_X, drift_Y, dW, dts, log_prior_X, log_prior_Y):
        # X is the proposal diff path and Y the true diff path
        epsilons = dW/jnp.sqrt(dts[:, None, :])
        term_1 = jnp.sum(jnp.sum(jnp.log(diff_factor_X) - jnp.log(diff_factor_Y) , axis = 0), axis = -1) ## sum over diff steps
        term_2 = -jnp.sum(jnp.sum( 0.5*((diff_factor_X/diff_factor_Y)**2 -1) *epsilons**2, axis = 0), axis = -1)
        term_3 = - jnp.sum(jnp.sum(1/(2* diff_factor_Y**2) * (drift_X - drift_Y)**2 *dts[:, None, :], axis = 0), axis = -1)
        term_4 = -jnp.sum(jnp.sum( diff_factor_X/diff_factor_Y**2 * (drift_X - drift_Y) * dW, axis = 0), axis = -1)
        log_p_ref = term_1 + term_2 + term_3 + term_4 + log_prior_Y - log_prior_X

        p_ref = jnp.exp(log_p_ref)
        p_weights = jax.nn.softmax(log_p_ref, axis = -1)*p_ref.shape[-1] #jax.lax.stop_gradient(p_ref)#jax.nn.softmax(log_p_ref, axis = -1)*p_ref.shape[-1])
        return log_p_ref, p_ref, p_weights


