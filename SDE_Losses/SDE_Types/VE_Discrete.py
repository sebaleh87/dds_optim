import jax
from jax import random
import matplotlib.pyplot as plt
import jax.numpy as jnp
import flax.linen as nn

from .Time_Importance_Sampler.numerical_inverse import NumericalIntSampler
import wandb
from matplotlib import pyplot as plt
from .Base_SDE import Base_SDE_Class, inverse_softplus

### Bridge as in SEQUENTIAL CONTROLLED LANGEVIN DIFFUSIONS
class VE_Discrete_Class(Base_SDE_Class):
    def __init__(self, SDE_Type_Config, Network_Config, Energy_Class):
        super().__init__(SDE_Type_Config, Network_Config, Energy_Class)
        self.vmap_get_log_prior = jax.vmap(self.get_log_prior, in_axes=(None, 0, 0))
        self.laplace_width = self.config["laplace_width"]
        self.mixture_prob = self.config["mixture_probs"]
        ### TODO remove learnability of diffusion parameters

    def get_SDE_params(self):
        if(self.invariance):
            ### if beta is learnable this also ahs to be dim(1)
            SDE_params = {"log_beta_delta": jnp.log((self.config["beta_max"] - self.config["beta_min"])), 
            "log_beta_min": jnp.log(self.config["beta_min"]),
            "log_sigma": jnp.log(1.), "mean": jnp.zeros((self.dim_x,)),
            "log_sigma_prior": jnp.log(self.sigma_init)}

        else:
            # rand_weights = jax.random.normal(random.PRNGKey(0), shape=(self.n_integration_steps,))
            # rand_weights_repulse = jax.random.normal(random.PRNGKey(0), shape=(self.n_integration_steps,))
            SDE_params = {"log_beta_delta": jnp.log((self.config["beta_max"] - self.config["beta_min"] + 10**-3))* jnp.ones((self.dim_x,)), 
                        "log_beta_min": jnp.log(self.config["beta_min"])* jnp.ones((self.dim_x,)),
                        "log_sigma": jnp.log(1.)* jnp.ones((self.dim_x,)), "mean": jnp.zeros((self.dim_x,)),
                        "log_sigma_prior": jnp.log(self.sigma_init)* jnp.ones((self.dim_x,))}
            if(self.config["beta_schedule"] == "learned"):
                SDE_params["log_beta_over_time"] = jnp.log(self.config["beta_max"])*jnp.ones((self.n_integration_steps, self.dim_x))
                # del SDE_params["log_beta_delta"]
                # del SDE_params["log_beta_min"]
            elif(self.config["beta_schedule"] == "neural"):
                #self.inverse_beta_init = inverse_softplus((self.config["beta_max"] - self.config["beta_min"] + 10**-3))
                self.inverse_beta_init = jnp.log((self.config["beta_max"]))


        return SDE_params

    def get_log_prior(self, SDE_params, x, sigma_scale_factor = 1.):
        mean = self.get_mean_prior(SDE_params)
        #print("VP_SDE", x.shape, mean.shape, sigma.shape)
        if(self.invariance):
            prior_sigma = self.return_prior_covar(SDE_params, sigma_scale_factor=sigma_scale_factor)
            log_pdf_vec =  jax.scipy.stats.norm.logpdf(x, loc=mean, scale=prior_sigma) + 0.5*jnp.log(2 * jnp.pi * prior_sigma)/prior_sigma.shape[0]*self.Energy_Class.particle_dim
            return jnp.sum(log_pdf_vec, axis = -1)
        else:
            prior_sigma = self.return_prior_covar(SDE_params, sigma_scale_factor=sigma_scale_factor)
            #return jax.random.multivariate_normal(random.PRNGKey(0), mean, jnp.diag(overall_sigma**2), x.shape[0])
            log_pdf_vec = jax.scipy.stats.norm.logpdf(x, loc=mean, scale=prior_sigma) 
            log_pdf = jnp.sum(log_pdf_vec, axis = -1)
            return log_pdf

    def prior_target_grad_interpolation(self, x, counter, Energy_params, SDE_params, temp, key):
        #x = jax.lax.stop_gradient(x) ### TODO for bridges in rKL w repara this should not be stopped
        #interpol = lambda x: self.Energy_Class.calc_energy(x, Energy_params, key)
        (log_prob, key), (grad)  = jax.value_and_grad(self.interpol_func, has_aux=True)( x, counter[0], SDE_params, Energy_params, temp, key)

        #grad = jnp.clip(grad, -10**2, 10**2)
        return jnp.expand_dims(log_prob, axis = -1), grad

    def get_entropy_prior(self, SDE_params):
        if(self.invariance):
            raise ValueError("not implemented")
        else:
            prior_sigma = self.return_prior_covar(SDE_params)
            entropy = 0.5 * jnp.sum(jnp.log(2 * jnp.pi) + 2* jnp.log(prior_sigma) + 1, axis = -1)
            return entropy
        
    def get_entropy_diff_step(self, SDE_params, t):
        if(self.invariance):
            raise ValueError("not implemented")
        else:
            diff_sigma = self.get_diffusion(SDE_params, None, t)
            entropy = 0.5 * jnp.sum(jnp.log(2 * jnp.pi) + 2* jnp.log(diff_sigma) + 1, axis = -1)
            return entropy
        
    def get_mean_prior(self, SDE_params):
        if(self.invariance):
            mean = jnp.zeros((self.dim_x,))
        else:
            mean = SDE_params["mean"]
        overall_mean = mean 
        return overall_mean

    def get_SDE_sigma(self, SDE_params):
        if(self.invariance):
            sigma = jnp.exp(SDE_params["log_sigma"])*jnp.ones((self.dim_x,))
        else:
            sigma = jnp.exp(SDE_params["log_sigma"])

        if(self.config["beta_schedule"]== "learned"):
            sigma = jax.lax.stop_gradient(sigma)
        return sigma, None

    def sample_prior(self, SDE_params, key, n_states, sigma_scale_factor = 1.):
        key, subkey = random.split(key)
        prior_mean = self.get_mean_prior(SDE_params)[None, :]
        if(self.invariance):
            overall_sigma = self.return_prior_covar(SDE_params, sigma_scale_factor = sigma_scale_factor)
            x_prior = random.normal(subkey, shape=(n_states, self.dim_x))*overall_sigma + prior_mean
        else:
            prior_sigma = self.return_prior_covar(SDE_params, sigma_scale_factor = sigma_scale_factor)
            x_prior = random.normal(subkey, shape=(n_states, self.dim_x))*prior_sigma + prior_mean
        return x_prior, key    
    
    def sample_prior_mixture_with_log_probs(self, SDE_params, key, n_states, sigma_scale_factor = 1.):
        prior_mean = self.get_mean_prior(SDE_params)[None, :]
        if(self.invariance):
            raise ValueError("not implemented")
        else:
            prior_sigma = self.return_prior_covar(SDE_params, sigma_scale_factor = sigma_scale_factor)
            decay_value = sigma_scale_factor - 1.
            curr_mixture_prob = self.mixture_prob*decay_value
            entropy_factor = self.laplace_width

            mixed_noise, off_policy_log_prob_noise, key = self.sample_from_mixture( curr_mixture_prob, entropy_factor, (n_states, self.dim_x) ,key)
            
            gauss_log_prob_noise = jnp.sum(jax.scipy.stats.norm.logpdf(mixed_noise, loc=0, scale=1), axis = -1)
            log_prob_noise_mixed = self.calc_mixture_probs( curr_mixture_prob, off_policy_log_prob_noise, gauss_log_prob_noise)
            x_prior = mixed_noise*prior_sigma + prior_mean
        return x_prior, log_prob_noise_mixed, key  
    
    def return_prior_covar(self, SDE_params, sigma_scale_factor = 1.):
        if(self.invariance):
            sigma = jnp.exp(SDE_params["log_sigma_prior"])*jnp.ones((self.dim_x,))
            overall_sigma = sigma*sigma_scale_factor
            return overall_sigma
        else:
            sigma = jnp.exp(SDE_params["log_sigma_prior"])
            return sigma*sigma_scale_factor

    def get_beta_min_and_max(self, SDE_params):
        if(self.invariance):
            beta_min = jnp.exp(SDE_params["log_beta_min"])*jnp.ones((self.dim_x,))
            beta_delta = jnp.exp(SDE_params["log_beta_delta"])*jnp.ones((self.dim_x,))
            beta_max = beta_min + beta_delta
            return beta_min, beta_max
        else:
            beta_delta = jnp.exp(SDE_params["log_beta_delta"])
            beta_min = jnp.exp(SDE_params["log_beta_min"])
            beta_max = beta_min + beta_delta
            return beta_min, beta_max

    def beta(self, SDE_params, t):
        beta_min, beta_max = self.get_beta_min_and_max(SDE_params)
        return (beta_min*(beta_max/beta_min)**(t/self.n_integration_steps))**2*(jnp.log(beta_max) - jnp.log(beta_min))


    def get_diffusion(self, SDE_params, x, t):
        sigma, _ = self.get_SDE_sigma(SDE_params)
        if(self.config["beta_schedule"] == "neural"):
            return jax.lax.stop_gradient(sigma[None, ...])*self.beta(SDE_params, t)
        else:
            diffusion = sigma*self.beta(SDE_params, t)
            return diffusion[None, :] 

    def calc_diff_log_prob(self, mean, loc, scale):
        if(self.invariance):
            log_pdf_vec = jax.scipy.stats.norm.logpdf(mean, loc=loc, scale=scale) + 0.5*jnp.log(2 * jnp.pi * scale)/scale.shape[0]*self.Energy_Class.particle_dim
        else:
            log_pdf_vec = jax.scipy.stats.norm.logpdf(mean, loc=loc, scale=scale)

        return jnp.sum(log_pdf_vec, axis = -1)

    def forward_sde(self, SDE_params, x, t, key):
        pass

    def sample_from_mixture(self, off_policy_probs, entropy_factor, shape ,key):
        key, subkey = random.split(key)
        ps = random.uniform(subkey, shape = (shape[0], 1))

        if(self.config["off_policy_mode"] == "gaussian"):
            key, subkey = random.split(key)
            laplace_scale = entropy_factor
            laplace_noise = random.normal(subkey, shape=shape)*laplace_scale
        elif(self.config["off_policy_mode"] == "laplace"):
            key, subkey = random.split(key)
            laplace_scale = jnp.sqrt(jnp.pi/(2*jnp.exp(1.)))*entropy_factor
            laplace_noise = random.laplace(subkey, shape=shape)*laplace_scale
        else:
            raise ValueError("off policy mode not implemented")

        gauss_noise = random.normal(subkey, shape=shape)
        mixed_noise = jnp.where(ps < off_policy_probs, laplace_noise, gauss_noise)

        if(self.config["off_policy_mode"] == "gaussian"):
            off_policy_log_prob_noise = jnp.sum(jax.scipy.stats.norm.logpdf(mixed_noise, loc=0, scale=laplace_scale), axis=-1)
        elif(self.config["off_policy_mode"] == "laplace"):
            off_policy_log_prob_noise = jnp.sum(jax.scipy.stats.laplace.logpdf(mixed_noise, loc=0, scale=laplace_scale), axis=-1)
        else:
            raise ValueError("off policy mode not implemented")
        return mixed_noise, off_policy_log_prob_noise, key

    def calc_mixture_probs(self, off_policy_prob, off_policy_log_prob, on_policy_log_prob):
        log_prob_noise_mixed = jax.scipy.special.logsumexp(
                jnp.stack([off_policy_log_prob + jnp.log(off_policy_prob), on_policy_log_prob + jnp.log1p(-off_policy_prob)], axis=-1),
                axis=-1
            )
        return log_prob_noise_mixed

    def sample_noise(self, SDE_params, x, t, dt, key, sigma_scale_factor = 1.):
        key, subkey = random.split(key)

        if(self.config["use_off_policy"] and (self.config["off_policy_mode"] == "laplace" or self.config["off_policy_mode"] == "gaussian")):
            decay_value = sigma_scale_factor - 1.
            curr_mixture_prob = self.mixture_prob*decay_value
            entropy_factor = self.laplace_width
            vec_ps = jnp.ones((x.shape[0], ))*(curr_mixture_prob )

            mixed_noise, off_policy_log_prob_noise, key = self.sample_from_mixture( curr_mixture_prob, entropy_factor, x.shape ,key)

            diffusion = self.get_diffusion(SDE_params, x, t)
            dx = jnp.sqrt(dt)*diffusion * mixed_noise
            
            gauss_log_prob_noise = jnp.sum(jax.scipy.stats.norm.logpdf(mixed_noise, loc=0, scale=1), axis = -1)
            log_prob_noise_mixed = self.calc_mixture_probs( vec_ps, off_policy_log_prob_noise, gauss_log_prob_noise)
            log_prob_noise = jnp.where(vec_ps == 0, gauss_log_prob_noise, log_prob_noise_mixed)
            log_prob_on_policy = gauss_log_prob_noise
            off_policy_log_weights = jax.scipy.special.logsumexp(
                jnp.stack([off_policy_log_prob_noise - gauss_log_prob_noise + jnp.log(vec_ps), jnp.log1p(-vec_ps)], axis=-1),
                axis=-1
            )
        else:
            noise = random.normal(subkey, shape=x.shape)
            log_prob_noise = jnp.sum(jax.scipy.stats.norm.logpdf(noise, loc=0, scale=1), axis = -1)
            diffusion = self.get_diffusion(SDE_params, x, t)*sigma_scale_factor
            dx = jnp.sqrt(dt)*diffusion * noise
            log_prob_on_policy  = log_prob_noise
            off_policy_log_weights = jnp.ones_like(log_prob_noise)
        return dx, log_prob_noise, log_prob_on_policy, off_policy_log_weights, key
    
    def compute_reverse_drift(self, diffusion, score, grad):
        reverse_drift = diffusion**2*score
        return reverse_drift
    
    def reverse_sde(self, SDE_params, score, grad, x, t, dt, key, sigma_scale_factor = 1.):
        diffusion = self.get_diffusion(SDE_params, x, t)
        if(self.config["off_policy_mode"] == "scale_drift"):
            diffusion_for_drift = diffusion*sigma_scale_factor
        else:
            diffusion_for_drift = diffusion

        reverse_drift_t_g_s = self.compute_reverse_drift(diffusion_for_drift, score, grad) #TODO check is this power of two correct? I think yes because U = diffusion*score
        x_drift_update = reverse_drift_t_g_s

        dx, log_prob_noise, log_prob_on_policy, off_policy_log_weights, key = self.sample_noise(SDE_params, x, t, dt, key, sigma_scale_factor = sigma_scale_factor)

        if(self.invariance == True):
            dx = self.subtract_COM(dx)

        if(self.stop_gradient):
            x_next = jax.lax.stop_gradient(x + x_drift_update*dt  + dx)
        else:
            x_next = x + x_drift_update * dt  + dx


        ### TODO check at which x drift ref should be evaluated?
        reverse_out_dict = {"x_next": x_next, "t_next": t - dt, "diffusion": diffusion, "log_prob_on_policy": log_prob_on_policy, "off_policy_log_weights": off_policy_log_weights,
                            "reverse_drift": reverse_drift_t_g_s, "dx": dx, "log_prob_noise": log_prob_noise} #"reverse_log_prob": log_prob_t_g_s
        return reverse_out_dict, key


    def simulate_reverse_sde_scan(self, model, params, Interpol_params, SDE_params, temp, key, n_states = 100, sample_mode = "train", n_integration_steps = 1000):
        ### since we use discrete time models dt is 1 and t = n_integration_steps (this is different from when we use SDEs formulation)
        for interpol_key in Interpol_params.keys():
            SDE_params[interpol_key] = Interpol_params[interpol_key]
        dt = 1.
        t = n_integration_steps
        counter = 0
        sigma_scale, scale_log_prob, temp, key = self.get_sigma_noise(n_states, key, sample_mode, temp)

        def scan_fn(carry, step):
            x, t, key, carry_dict = carry
            counter = step
            hidden_state = carry_dict["hidden_state"]

            score, new_hidden_state, grad, SDE_params_extended, interpol_log_prob, key = self.apply_model(model, x, t/self.n_integration_steps, counter, params, Interpol_params, SDE_params, hidden_state, temp, key)
            carry_dict["hidden_state"] = new_hidden_state

            reverse_out_dict, key = self.reverse_sde(SDE_params_extended, score, grad, x, t, dt, key, sigma_scale_factor= sigma_scale)

            SDE_tracker_step = {
            "interpolated_grad": grad,
            "dx": reverse_out_dict["dx"],
            "xs": x,
            "ts": jnp.array(t, dtype = jnp.float32),
            "diffusions": reverse_out_dict["diffusion"],
            "reverse_drifts": reverse_out_dict["reverse_drift"],
            "dts": jnp.array(dt, dtype = jnp.float32),
            "key": key,
            #"hidden_state": carry_dict["hidden_state"],
            "log_prob_noise": reverse_out_dict["log_prob_noise"],
            "interpol_log_probs": interpol_log_prob,
            "log_prob_on_policy": reverse_out_dict["log_prob_on_policy"],
            "off_policy_log_weights": reverse_out_dict["off_policy_log_weights"]
            }

            x = reverse_out_dict["x_next"]
            t = reverse_out_dict["t_next"]
            return (x, t, key, carry_dict), SDE_tracker_step

        if(self.config["use_off_policy"] and (self.config["off_policy_mode"] == "laplace" or self.config["off_policy_mode"] == "gaussian")):
            x_prior, log_prob_prior_scaled, key = self.sample_prior_mixture_with_log_probs(SDE_params, key, n_states, sigma_scale_factor = sigma_scale)
        else:
            x_prior, key = self.sample_prior(SDE_params, key, n_states, sigma_scale_factor = sigma_scale)
            log_prob_prior_scaled = self.vmap_get_log_prior(SDE_params, x_prior, sigma_scale)
    
        if(self.stop_gradient):
            x_prior = jax.lax.stop_gradient(x_prior)

        if(self.invariance == True):
            x_prior = self.subtract_COM(x_prior)

        init_carry = jnp.zeros((n_states, self.Network_Config["n_hidden"]), dtype = jnp.float32)
        carry_dict = {"hidden_state": [(init_carry, init_carry)  for i in range(self.Network_Config["n_layers"])]}
        (x_final, t_final, key, carry_dict), SDE_tracker_steps = jax.lax.scan(
            scan_fn,
            (x_prior, t, key, carry_dict),
            jnp.arange(n_integration_steps)
        )


        ### TODO make last forward pass here
        hidden_state = carry_dict["hidden_state"]
        score, new_hidden_state, grad, SDE_params_extended, interpol_log_prob, key = self.apply_model(model, x_final, t_final/self.n_integration_steps, counter, params, Interpol_params, SDE_params, hidden_state, temp, key)
        diffusion_final = self.get_diffusion(SDE_params_extended, x_final, t_final)
        reverse_drift_final = self.compute_reverse_drift(diffusion_final, score, grad)
        #carry_dict["hidden_state"] = new_hidden_state

        xs = jnp.concatenate([SDE_tracker_steps["xs"], x_final[None, :]], axis = 0)
        interpol_grads = jnp.concatenate([SDE_tracker_steps["interpolated_grad"], grad[None, :]], axis = 0)
        #hidden_states = jnp.concatenate([ SDE_tracker_steps["hidden_state"], hidden_state[None, :]], axis = 0)
        diffusions = jnp.concatenate([SDE_tracker_steps["diffusions"], diffusion_final[None, :]], axis = 0)
        reverse_drifts = jnp.concatenate([SDE_tracker_steps["reverse_drifts"], reverse_drift_final[None, :]], axis = 0)
        interpol_log_probs = jnp.concatenate([SDE_tracker_steps["interpol_log_probs"], interpol_log_prob[None, :]], axis = 0)

        x_prev = xs[:-1]
        x_next = xs[1:]
        diffusion_prev = diffusions[0:-1]
        diffusion_next = diffusions[1:]
        grads_next = interpol_grads[1:]
        reverse_drifts_next = reverse_drifts[1:]
        reverse_drifts_prev = reverse_drifts[0:-1]

        x_pos_next = x_next


        ## TODO compute forward log probs here
        reverse_diff_log_probs = jax.vmap(self.calc_diff_log_prob, in_axes=(0, 0, 0))(x_next, x_prev + reverse_drifts_prev*dt, diffusion_prev*jnp.sqrt(dt))
        forward_diff_log_probs = jax.vmap(self.calc_diff_log_prob, in_axes=(0, 0, 0))(x_prev, x_pos_next, diffusion_next*jnp.sqrt(dt))
        log_prob_noise = SDE_tracker_steps["log_prob_noise"]

        SDE_tracker = {
            "log_prob_prior_scaled": log_prob_prior_scaled,
            "log_prob_noise": log_prob_noise,
            "scale_log_prob": scale_log_prob,
            "noise_scale": sigma_scale,
            "dx": SDE_tracker_steps["dx"],
            "xs": SDE_tracker_steps["xs"],
            "ts": SDE_tracker_steps["ts"],
            "forward_diff_log_probs": forward_diff_log_probs,
            "reverse_log_probs": reverse_diff_log_probs,
            "dts": SDE_tracker_steps["dts"],
            "x_final": x_final,
            "x_prior": x_prior,
            "keys": SDE_tracker_steps["key"],
            "interpolated_grads": interpol_grads,
            "interpol_log_probs": interpol_log_probs,
            "log_prob_on_policy": SDE_tracker_steps["log_prob_on_policy"],
            "off_policy_log_weights": SDE_tracker_steps["off_policy_log_weights"]

        }

        return SDE_tracker, key



