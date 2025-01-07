import jax
from jax import random
import matplotlib.pyplot as plt
import jax.numpy as jnp
import flax.linen as nn

from .Time_Importance_Sampler.numerical_inverse import NumericalIntSampler
import wandb
from matplotlib import pyplot as plt
from .Base_SDE import Base_SDE_Class


class Bridge_SDE_Class(Base_SDE_Class):
    def __init__(self, SDE_Type_Config, Network_Config, Energy_Class):
        super().__init__(SDE_Type_Config, Network_Config, Energy_Class)
        ### TODO remove learnability of diffusion parameters

    def get_SDE_params(self):
        if(self.invariance):
            ### if beta is learnable this also ahs to be dim(1)
            SDE_params = {"log_beta_delta": jnp.log(self.config["beta_max"] - self.config["beta_min"]), 
            "log_beta_min": jnp.log(self.config["beta_min"]),
            "log_sigma": jnp.log(self.sigma_init), "mean": jnp.zeros((self.dim_x,)),
            "log_sigma_prior": jnp.log(self.sigma_init)}

        else:
            SDE_params = {"log_beta_delta": jnp.log(self.config["beta_max"] - self.config["beta_min"])* jnp.ones((self.dim_x,)), 
                        "log_beta_min": jnp.log(self.config["beta_min"])* jnp.ones((self.dim_x,)),
                        "log_sigma": jnp.log(self.sigma_init)* jnp.ones((self.dim_x,)), "mean": jnp.zeros((self.dim_x,)),
                        "log_sigma_prior": jnp.log(self.sigma_init)* jnp.ones((self.dim_x,)),
                        "beta_interpol_params": jnp.ones((self.n_integration_steps))}
        return SDE_params

    def interpol_func(self, x, t, SDE_params, Energy_params, temp, key):
        clipped_temp = jnp.clip(temp, min = 0.0001)
        step_index = round((1-t)*self.n_integration_steps)
        beta_params = SDE_params["beta_interpol_params"]
        beta_activ = nn.softplus(beta_params)
        where_true = 1*(jnp.arange(0, self.n_integration_steps) < step_index)
        beta_interpol = 1 - jnp.sum(where_true*beta_activ)/ jnp.sum(beta_activ)
        Energy_value, key = jax.lax.stop_gradient(self.Energy_Class.calc_energy(x, Energy_params, key))
        interpol = (beta_interpol)*self.get_log_prior(SDE_params,x)  - (1-beta_interpol)*Energy_value / clipped_temp
        return interpol, key

    def get_log_prior(self, SDE_params, x):
        mean = self.get_mean_prior(SDE_params)
        #print("VP_SDE", x.shape, mean.shape, sigma.shape)
        if(self.invariance):
            overall_sigma = self.return_prior_covar(SDE_params)
            log_pdf_vec =  jax.scipy.stats.norm.logpdf(x, loc=mean, scale=overall_sigma) + 0.5*jnp.log(2 * jnp.pi * overall_sigma)/overall_sigma.shape[0]*self.Energy_Class.particle_dim
            return jnp.sum(log_pdf_vec, axis = -1)
        else:
            prior_sigma = self.return_prior_covar(SDE_params)
            #return jax.random.multivariate_normal(random.PRNGKey(0), mean, jnp.diag(overall_sigma**2), x.shape[0])
            log_pdf_vec = jax.scipy.stats.norm.logpdf(x, loc=mean, scale=prior_sigma) 
            log_pdf = jnp.sum(log_pdf_vec, axis = -1)
            return log_pdf
        
    def get_mean_prior(self, SDE_params):
        if(self.invariance):
            mean = jnp.zeros((self.dim_x,))
        else:
            mean = SDE_params["mean"]*jnp.zeros((self.dim_x,))
        overall_mean = mean 
        return overall_mean

    def get_SDE_sigma(self, SDE_params):
        if(self.invariance):
            sigma = jnp.exp(SDE_params["log_sigma"])*jnp.ones((self.dim_x,))
            return sigma
        else:
            sigma = jnp.exp(SDE_params["log_sigma"])
            return sigma

    def sample_prior(self, SDE_params, key, n_states):
        key, subkey = random.split(key)
        prior_mean = self.get_mean_prior(SDE_params)
        if(self.invariance):
            overall_sigma = self.return_prior_covar(SDE_params)
            x_prior = random.normal(subkey, shape=(n_states, self.dim_x))*overall_sigma[None, :] + prior_mean[None, :]
        else:
            prior_sigma = self.return_prior_covar(SDE_params)
            x_prior = random.normal(subkey, shape=(n_states, self.dim_x))*prior_sigma[None, :] + prior_mean[None, :]
        return x_prior, key    
    
    def return_prior_covar(self, SDE_params):
        if(self.invariance):
            sigma = jnp.exp(SDE_params["log_sigma_prior"])*jnp.ones((self.dim_x,))
            overall_sigma = sigma
            return overall_sigma
        else:
            sigma = jnp.exp(SDE_params["log_sigma_prior"])
            return sigma

    def beta(self, SDE_params, t):
        beta_min, beta_max = self.get_beta_min_and_max(SDE_params)
        return beta_min + (beta_max-beta_min)*jnp.cos(jnp.pi/2*(1-t)) 

    def get_diffusion(self, SDE_params, x, t):
        sigma, _ = self.get_SDE_sigma(SDE_params)
        diffusion = sigma*self.beta(SDE_params, t)
        return diffusion[None, :] 

    def calc_diff_log_prob(self, mean, loc, scale):
        ### TODO adapt this in case of EN invariance
        if(self.invariance):
            raise ValueError("Invariance not implemented for this SDE")
        else:
            log_pdf_vec = jax.scipy.stats.norm.logpdf(mean, loc=loc, scale=scale)

        return jnp.sum(log_pdf_vec, axis = -1)

    def forward_sde(self, SDE_params, x, t, key):
        pass

    def sample_noise(self, SDE_params, x, t, dt, key):
        key, subkey = random.split(key)
        noise = random.normal(subkey, shape=x.shape)
        
        diffusion = self.get_diffusion(SDE_params, x, t)

        dx = diffusion * noise*jnp.sqrt(dt)
        return dx, key
    
    def compute_reverse_drift(self, diffusion, score, grad):
        reverse_drift = diffusion**2*score
        return reverse_drift
    
    def reverse_sde(self, SDE_params, score, grad, x, t, dt, key):
        diffusion = self.get_diffusion(SDE_params, x, t)

        reverse_drift_t_g_s = self.compute_reverse_drift(diffusion, score, grad) #TODO check is this power of two correct? I think yes because U = diffusion*score

        dx, key = self.sample_noise(SDE_params, x, t, dt, key)

        if(self.invariance == True):
            dx = self.subtract_COM(dx)
            #reverse_drift = self.subtract_COM(reverse_drift)

        if(self.stop_gradient):
            x_next = jax.lax.stop_gradient(x + reverse_drift_t_g_s*dt  + dx)
        else:
            x_next = x + reverse_drift_t_g_s  * dt  + dx

        log_prob_t_g_s = self.calc_diff_log_prob(x_next, x + reverse_drift_t_g_s*dt, diffusion*jnp.sqrt(dt))


        ### TODO check at which x drift ref should be evaluated?
        reverse_out_dict = {"x_next": x_next, "t_next": t - dt, "diffusion": diffusion, "reverse_drift": reverse_drift_t_g_s, "reverse_log_prob": log_prob_t_g_s, "dx": dx}
        return reverse_out_dict, key
    
    def simulate_reverse_sde_scan(self, model, params, Energy_params, SDE_params, temp, key, n_states = 100, x_dim = 2, n_integration_steps = 1000):
        dt = 1./n_integration_steps
        t = 1.
        def scan_fn(carry, step):
            x, t, key, carry_dict = carry
            # if(jnp.isnan(x).any()):
            #     print("score", x)
            #     raise ValueError("score is nan")
            hidden_state = carry_dict["hidden_state"]
            score, new_hidden_state, grad, key = self.apply_model(model, x, t, params, Energy_params, SDE_params, hidden_state, temp, key)
            carry_dict["hidden_state"] = new_hidden_state

            reverse_out_dict, key = self.reverse_sde(SDE_params, score, grad, x, t, dt, key)

            SDE_tracker_step = {
            "interpolated_grad": grad,
            "dx": reverse_out_dict["dx"],
            "xs": x,
            "ts": t,
            "diffusions": reverse_out_dict["diffusion"],
            "reverse_log_probs": reverse_out_dict["reverse_log_prob"],
            "reverse_drifts": reverse_out_dict["reverse_drift"],
            "dts": dt,
            "key": key,
            "hidden_state": carry_dict["hidden_state"]
            }

            x = reverse_out_dict["x_next"]
            t = reverse_out_dict["t_next"]
            return (x, t, key, carry_dict), SDE_tracker_step

        x_prior, key = self.sample_prior(SDE_params, key, n_states)
    
        if(self.stop_gradient):
            x_prior = jax.lax.stop_gradient(x_prior)

        if(self.invariance == True):
            x_prior = self.subtract_COM(x_prior)

        init_carry = jnp.zeros((n_states, self.Network_Config["n_hidden"]))
        carry_dict = {"hidden_state": [(init_carry, init_carry)  for i in range(self.Network_Config["n_layers"])]}
        (x_final, t_final, key, carry_dict), SDE_tracker_steps = jax.lax.scan(
            scan_fn,
            (x_prior, t, key, carry_dict),
            jnp.arange(n_integration_steps)
        )

        ### TODO make last forward pass here
        hidden_state = carry_dict["hidden_state"]
        score, new_hidden_state, grad, key = self.apply_model(model, x_final, t_final, params, Energy_params, SDE_params, hidden_state, temp, key)
        diffusion_final = self.get_diffusion(SDE_params, x_final, t_final)
        reverse_drift_final = self.compute_reverse_drift(diffusion_final, score, grad)
        #carry_dict["hidden_state"] = new_hidden_state

        xs = jnp.concatenate([x_prior[None, :], SDE_tracker_steps["xs"]], axis = 0)
        interpol_grads = jnp.concatenate([SDE_tracker_steps["interpolated_grad"], grad[None, :]], axis = 0)
        #hidden_states = jnp.concatenate([ SDE_tracker_steps["hidden_state"], hidden_state[None, :]], axis = 0)
        diffusions = jnp.concatenate([SDE_tracker_steps["diffusions"], diffusion_final[None, :]], axis = 0)
        reverse_drifts = jnp.concatenate([SDE_tracker_steps["reverse_drifts"], reverse_drift_final[None, :]], axis = 0)

        x_prev = xs[:-1]
        x_next = xs[1:]
        diffusion_next = diffusions[1:]
        grads_next = interpol_grads[1:]
        reverse_drifts = reverse_drifts[1:]

        forward_drift = (diffusion_next**2*grads_next - reverse_drifts)
        x_pos_next = x_next + forward_drift*dt

        forward_diff_log_probs = jax.vmap(self.calc_diff_log_prob, in_axes=(0, 0, 0))(x_prev, x_pos_next, diffusion_next*jnp.sqrt(dt))


        SDE_tracker = {
            "dx": SDE_tracker_steps["dx"],
            "xs": SDE_tracker_steps["xs"],
            "ts": SDE_tracker_steps["ts"],
            "forward_diff_log_probs": forward_diff_log_probs,
            "reverse_log_probs": SDE_tracker_steps["reverse_log_probs"],
            "dts": SDE_tracker_steps["dts"],
            "x_final": x_final,
            "x_prior": x_prior,
            #"hidden_states": hidden_states,
            "keys": SDE_tracker_steps["key"],
            "interpolated_grads": interpol_grads

        }

        return SDE_tracker, key
    
    def simulated_SDE_from_x(self, model, params, Energy_params, SDE_params, x, key, n_integration_steps = 1000):
        def scan_fn(carry, step):
            x, t, key, carry_dict = carry
            # if(jnp.isnan(x).any()):
            #     print("score", x)
            #     raise ValueError("score is nan")
            t_arr = t*jnp.ones((x.shape[0], 1)) 
            if(self.use_interpol_gradient):
                if(self.network_has_hidden_state):
                    Energy, grad, key = self.vmap_prior_target_grad_interpolation(x, t, Energy_params, SDE_params, key) 
                    Energy_value = Energy #Energy[...,None]
                    in_dict = {"x": x, "Energy_value": Energy_value, "t": t_arr, "grads": grad, "hidden_state": carry_dict["hidden_state"]}
                    out_dict = model.apply(params, in_dict, train = True)
                    score = out_dict["score"]
                    carry_dict["hidden_state"] = out_dict["hidden_state"]
                else:
                    Energy, grad, key = self.vmap_prior_target_grad_interpolation(x, t, Energy_params, SDE_params, key) 
                    Energy_value = Energy
                    in_dict = {"x": x, "Energy_value": Energy_value,  "t": t_arr, "grads": grad}
                    out_dict = model.apply(params, in_dict, train = True)
                    score = out_dict["score"]
            # if(jnp.isnan(concat_values).any()):
            #     print("concat_values", concat_values)
            #     raise ValueError("concat_values is nan")
                
            else:
                ### TODO x dim should be increased by 1
                grad = jnp.zeros((x.shape[0], self.dim_x))
                in_dict = {"x": x, "t": t_arr, "Energy_value": jnp.zeros((x.shape[0], 1)),  "grads": grad}
                out_dict = model.apply(params, in_dict, train = True)
                score = out_dict["score"]


            dt = self.reversed_dt_values[step]
            reverse_out_dict, key = self.reverse_sde(SDE_params, score, x, t, dt, key)

            SDE_tracker_step = {
            "interpolated_grad": grad,
            "dW": reverse_out_dict["dW"],
            "xs": x,
            "ts": t,
            "scores": score,
            "forward_drift": reverse_out_dict["forward_drift"],
            "reverse_drift": reverse_out_dict["reverse_drift"],
            "drift_ref": reverse_out_dict["drift_ref"],
            "dts": dt
            }

            x = reverse_out_dict["x_next"]
            t = reverse_out_dict["t_next"]
            return (x, t, key, carry_dict), SDE_tracker_step

        if(self.invariance == True):
            x = self.subtract_COM(x)

        t = 1.0
        dt = 1. / n_integration_steps
        ### this function is primarily so that i can test the equivaraicne of the sde
        init_carry = jnp.zeros((x.shape[0], self.Network_Config["n_hidden"]))
        carry_dict = {"hidden_state": [(init_carry, init_carry)  for i in range(self.Network_Config["n_layers"])]}
        print("key",key)
        (x_final, t_final, key, carry_dict), SDE_tracker_steps = jax.lax.scan(
            scan_fn,
            (x, t, key, carry_dict),
            jnp.arange(n_integration_steps)
        )

        return x_final, SDE_tracker_steps
    


