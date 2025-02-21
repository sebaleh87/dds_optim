import jax
from jax import random
import matplotlib.pyplot as plt
import jax.numpy as jnp
import flax.linen as nn

from .Time_Importance_Sampler.numerical_inverse import NumericalIntSampler
import wandb
from matplotlib import pyplot as plt
class Base_SDE_Class:

    def __init__(self, config, Network_Config, Energy_Class) -> None:
        self.config = config
        self.stop_gradient = False
        self.Energy_Class = Energy_Class
        self.use_interpol_gradient = config["use_interpol_gradient"]
        self.n_integration_steps = config["n_integration_steps"]
        self.Network_Config = Network_Config

        if(self.Network_Config["model_mode"] == "latent"):
            self.dim_x = self.Network_Config["latent_dim"]
        else:
            self.dim_x = self.Network_Config["x_dim"]

        if("LSTM" in Network_Config["name"]):
            self.network_has_hidden_state = True
        else:
            self.network_has_hidden_state = False

        if(self.config["SDE_weightening"] != "normal"):
            self.NumericalIntSampler_Class = NumericalIntSampler(self.weightening, self.den_weighting, n_integration_steps = self.config["n_integration_steps"])
            t_values, dt_values = self.NumericalIntSampler_Class.get_dt_values()

            # plt.figure()
            # plt.plot(t_values, dt_values)
            # plt.plot(t_values, jax.lax.cumsum(dt_values))
            # wandb.log({"figures/dt_values": wandb.Image(plt)})
            # plt.close()
            self.reversed_dt_values = jnp.flip(dt_values)
        else:
            self.reversed_dt_values = jnp.ones((self.config["n_integration_steps"],))*1./self.config["n_integration_steps"]


        self.invariance = self.Energy_Class.invariance
        
        self.sigma_init = config["sigma_init"]
        self.learn_covar = config["learn_covar"]
        self.repulsion_strength = config["repulsion_strength"]
        self.sigma_scale_factor = config["sigma_scale_factor"]
        self.learn_interpolation_params = config["learn_interpolation_params"]
        # self.noise_scale_factor = config["noise_scale_factor"]

    def weightening(self, t):
        SDE_params = self.get_SDE_params()
        weight = jnp.mean((1-jnp.exp(- 2*jax.vmap(self.beta_int, in_axes=(None, 0))(SDE_params, t))), axis = -1)
        return weight
    
    def den_weighting(self, t):
        SDE_params = self.get_SDE_params()
        den_weight =  jnp.mean(2*jax.vmap(self.beta, in_axes=(None, 0))(SDE_params, t), axis = -1)
        return den_weight
    
    def get_div_drift(self, SDE_params, t):
        raise NotImplementedError("get_diffusion method not implemented")

    def get_Interpol_params(self):
        InterpoL_params = {"beta_interpol_params": jnp.ones((self.n_integration_steps)),
                            "repulsion_interpol_params": jnp.ones((self.n_integration_steps))}
        return InterpoL_params

    def get_SDE_params(self):
        raise NotImplementedError("get_diffusion method not implemented")


    def get_SDE_mean(self, SDE_params):
        raise NotImplementedError("get_diffusion method not implemented")

    def get_SDE_sigma(self, SDE_params):
        raise NotImplementedError("get_diffusion method not implemented")


    def compute_p_xt_g_x0_statistics(self, x0, xt, t):
        raise NotImplementedError("get_diffusion method not implemented")

    def get_log_prior(self, x):
        raise NotImplementedError("get_diffusion method not implemented")
    
    def sample_prior(self, SDE_params, key, n_states, sigma_scale_factor = 1.):
        key, subkey = random.split(key)
        prior_mean = self.get_mean_prior(SDE_params)
        if(self.invariance):
            overall_sigma = self.return_prior_covar(SDE_params, sigma_scale_factor = sigma_scale_factor)
            x_prior = random.normal(subkey, shape=(n_states, self.dim_x))*overall_sigma + prior_mean
        elif(not self.learn_covar):
            prior_sigma = self.return_prior_covar(SDE_params, sigma_scale_factor = sigma_scale_factor)
            x_prior = random.normal(subkey, shape=(n_states, self.dim_x))*prior_sigma + prior_mean
        else:
            overall_covar = self.return_prior_covar(SDE_params, sigma_scale_factor = sigma_scale_factor)
            x_prior = jax.random.multivariate_normal(subkey, prior_mean, overall_covar, (n_states,))
        return x_prior, key
    
    
    def vmap_prior_target_grad_interpolation(self, x, counter, Energy_params, SDE_params, temp, key, clip_overall_score = 10**3):
        key, subkey = random.split(key)
        batched_subkey = random.split(subkey, x.shape[0])
        vmap_log_prob, vmap_grad = jax.vmap(self.prior_target_grad_interpolation, in_axes=(0, 0, None, None, None, 0))(x, counter, Energy_params, SDE_params, temp, batched_subkey)
        #print("vmap_grad", jnp.mean(jax.lax.stop_gradient(vmap_grad)))
        #vmap_grad = jnp.where(jnp.isfinite(vmap_grad), vmap_grad, 0)
        vmap_grad = jnp.where(jnp.isnan(vmap_grad), 0, vmap_grad)
        # #vmap_grad = jnp.clip(vmap_grad, -10**4, 10**4)
        # grad_norm = jnp.linalg.norm(vmap_grad, axis = -1, keepdims = True)
        # vmap_grad = jnp.where(grad_norm > clip_overall_score, clip_overall_score*vmap_grad/grad_norm, vmap_grad)

        #vmap_div_energy, vmap_grad_div = self.get_diversity_log_prob_grad(x,counter,SDE_params) 
        vmap_log_prob = vmap_log_prob #+ vmap_div_energy
        vmap_grad = vmap_grad #+ vmap_grad_div

        return vmap_log_prob, vmap_grad, key
    
    def prior_target_grad_interpolation(self, x, counter, Energy_params, SDE_params, temp, key):
        x_stopped = jax.lax.stop_gradient(x) ### TODO for bridges in rKL w repara this should not be stopped
        #interpol = lambda x: self.Energy_Class.calc_energy(x, Energy_params, key)
        (log_prob, key), (grad)  = jax.value_and_grad(self.interpol_func, has_aux=True)( x_stopped, counter[0], SDE_params, Energy_params, temp, key)
        #grad = jnp.clip(grad, -10**2, 10**2)
        return jnp.expand_dims(log_prob, axis = -1), grad

    def interpol_func(self, x, counter, SDE_params, Energy_params, temp, key):
        clipped_temp = jnp.clip(temp, min = 0.0001)
        
        if(self.learn_interpolation_params):
            interpolation_params = SDE_params
        else:
            interpolation_params = jax.lax.stop_gradient(SDE_params)

        beta_interpol = self.compute_energy_interpolation_time(interpolation_params, counter, SDE_param_key = "beta_interpol_params")
        Energy_value, key = self.Energy_Class.calc_energy(x, Energy_params, key)
        log_prior_params = SDE_params
        log_prior = self.get_log_prior(log_prior_params,x)  ### only stop gradient for log prior but not for beta_interpol or x
        interpol = (beta_interpol)*log_prior  - (1-beta_interpol)*Energy_value / clipped_temp
        return interpol, key
    
    def get_diversity_log_prob_grad(self, x_batch, counter, SDE_params):
        x_batch = jax.lax.stop_gradient(x_batch)
        (div_Energy, _), (batched_x_grad) = jax.value_and_grad(self.diversity_log_prob, has_aux=True)(x_batch, counter, SDE_params)
        return div_Energy, batched_x_grad
    
    def diversity_log_prob(self, x_batch, counter ,SDE_params, eps = 10**-8):
        div_beta_interpol = self.compute_energy_interpolation_time(SDE_params, counter[0], SDE_param_key = "repulsion_interpol_params")
        distances = jnp.sqrt(jnp.sum((x_batch[:, None, :] - x_batch[None, :, :])**2, axis = -1) + eps)
        mask = jnp.eye(distances.shape[0])
        max_value = jax.lax.stop_gradient(jnp.max(distances))
        masked_distances = jnp.where(mask == 1, max_value+1., distances)
        div_Energy = - jnp.sum(jnp.min(masked_distances, axis = -1)**0.5)**2

        return - self.repulsion_strength*div_Energy*(div_beta_interpol)*(1-div_beta_interpol), (div_beta_interpol)*(1-div_beta_interpol)
    
    def compute_energy_interpolation_time(self, SDE_params, counter, SDE_param_key = "beta_interpol_params"):
        step_index = self.n_integration_steps-counter
        beta_params = SDE_params[SDE_param_key]
        beta_activ = nn.softplus(beta_params)
        where_true = 1*(jnp.arange(0, self.n_integration_steps) < step_index)
        beta_interpol = jnp.sum(where_true*beta_activ)/ jnp.sum(beta_activ)
        return beta_interpol

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

    def return_sigma_scale_factor(self, scale_strength, shape, key):
        key, subkey = random.split(key)
        #TODO the following distribution produces heavy outliers! Fat tail distribution
        if self.config['use_off_policy']:
            sigma_scale_factor = 1 + jax.random.exponential(subkey, shape) * scale_strength
            log_prob = jnp.zeros((shape[0],))# jnp.sum(jax.scipy.stats.expon.logpdf(sigma_scale_factor - 1, scale=1/scale_strength), axis = -1)

        else:
            sigma_scale_factor = 1.*jnp.ones((shape[0],))
            log_prob = jnp.zeros((shape[0],))
        return sigma_scale_factor, log_prob, key
    

    def get_diffusion(self, SDE_params, x, t):
        """
        Method to get the diffusion term of the SDE.
        
        Parameters:
        t (float): Time variable.
        x (float): State variable.
        
        Returns:
        float: Diffusion term.
        """
        raise NotImplementedError("get_diffusion method not implemented")

    def get_drift(self, SDE_params, x, t):
        """
        Method to get the drift term of the SDE.
        
        Parameters:
        t (float): Time variable.
        x (float): State variable.
        
        Returns:
        float: Drift term.
        """
        raise NotImplementedError("get_drift method not implemented")

    def forward_sde(self, x, t, dt, key):
        """
        Method to simulate the reverse SDE.
        
        Parameters:
        xT (float): Final state.
        t0 (float): Initial time.
        t1 (float): Final time.
        dt (float): Time step.
        
        Returns:
        list: Simulated path of the state variable.
        """
        raise NotImplementedError("simulate_reverse_sde method not implemented")

    def simulate_forward_sde(self, x0, t, key, n_integration_steps = 1000):
        x = x0
        t = 0.0
        dt = 1./n_integration_steps

        SDE_tracker = {"xs": [], "ts": []}
        for step in range(n_integration_steps):
            x, t, key = self.forward_sde(x, t, dt, key)

            SDE_tracker["xs"].append(x)
            SDE_tracker["ts"].append(t) 

        return SDE_tracker, key
    
    def subtract_COM(self, x):
        resh_x = x.reshape((x.shape[0], self.Energy_Class.n_particles, self.Energy_Class.particle_dim))
        shifted_x = resh_x - jnp.mean(resh_x, axis = 1, keepdims=True)
        x_cernered = shifted_x.reshape(x.shape)
        return x_cernered

    def get_scaled_diffusion(self, SDE_params, x, t, sigma_scale_factor):
        diffusion = self.get_diffusion(SDE_params, x, t) * sigma_scale_factor
        return diffusion
    
    def reverse_sde(self, SDE_params, score, x, t, dt, sigma_scale_factor, key):
        forward_drift = self.get_drift(SDE_params, x, t)
        diffusion = self.get_scaled_diffusion(SDE_params, x, t, sigma_scale_factor)

        reverse_drift = diffusion**2*score - forward_drift #TODO check is this power of two correct? I think yes because U = diffusion*score

        key, subkey = random.split(key)
        noise = random.normal(subkey, shape=x.shape)
        dW = jnp.sqrt(dt) * noise

        if(self.invariance == True):
            dW = self.subtract_COM(dW)
            #reverse_drift = self.subtract_COM(reverse_drift)

        if(self.stop_gradient):
            x_next = jax.lax.stop_gradient(x + reverse_drift  * dt  + diffusion * dW)
        else:
            x_next = x + reverse_drift  * dt  + diffusion * dW

        ### TODO check at which x drift ref should be evaluated?
        reverse_out_dict = {"x_next": x_next, "t_next": t - dt, "drift_ref": x, "forward_drift": forward_drift, "reverse_drift": reverse_drift, "dW": dW}
        return reverse_out_dict, key

    
    def apply_model(self, model, x, t, counter, params, Energy_params, SDE_params, hidden_state, temp, key):
        t_arr = t*jnp.ones((x.shape[0], 1)) 
        counter_arr = counter*jnp.ones((x.shape[0], 1)) 
        new_hidden_state = hidden_state
        if(self.use_interpol_gradient):
            if(self.network_has_hidden_state):
                log_prob, grad, key = self.vmap_prior_target_grad_interpolation(x, counter_arr, Energy_params, SDE_params, temp, key) 
                log_prob_value = log_prob #Energy[...,None]
                in_dict = {"x": x, "Energy_value": Energy_value, "t": t_arr, "grads": grad, "hidden_state": hidden_state}
                out_dict = model.apply(params, in_dict, train = True)
                score = out_dict["score"]
                new_hidden_state = out_dict["hidden_state"]
            else:
                log_prob, grad, key = self.vmap_prior_target_grad_interpolation(x, counter_arr, Energy_params, SDE_params, temp, key) 
                log_prob_value = log_prob
                in_dict = {"x": x, "grads": grad,  "t": t_arr}
                out_dict = model.apply(params, in_dict, train = True)
                score = out_dict["score"]
        # if(jnp.isnan(concat_values).any()):
        #     print("concat_values", concat_values)
        #     raise ValueError("concat_values is nan")
            
        else:
            grad = jnp.zeros((x.shape[0], self.dim_x))
            in_dict = {"x": x, "t": t_arr, "Energy_value": jnp.zeros((x.shape[0], 1)),  "grads": grad}
            out_dict = model.apply(params, in_dict, train = True)
            score = out_dict["score"]

        if(self.config["beta_schedule"] == "neural"):
            SDE_params["log_beta_x_t"] = out_dict["log_beta_x_t"]

        return score, new_hidden_state, grad, SDE_params, log_prob, key
    
    def get_sigma_noise(self,  n_states, key, sample_mode, temp):
        ### if self.config['use_off_policy'] true temp is not treated as a temperature but as an annealed scaling for self.sigma_scale_factor, assumes temp >= 1.
        shape= [n_states, self.dim_x]
        if self.config['use_off_policy']:    
            if(self.config["off_policy_mode"] == "laplace" or self.config["off_policy_mode"] == "gaussian"):
                if(sample_mode == "train"):
                    sigma_scale = temp
                    scale_log_prob = jnp.zeros((n_states,))                   
                elif(sample_mode == "val"):
                    sigma_scale = temp
                    scale_log_prob = jnp.zeros((n_states,))
                    # sigma_scale = (self.sigma_scale_factor**2 + 1)*jnp.ones(shape)    #this is the mode, not the expectation value
                    # scale_log_prob = jnp.zeros((n_states,))
                else:
                    sigma_scale = 1.
                    scale_log_prob = jnp.zeros((n_states,))

            else:
                annealed_scale = temp - 1. 
                if(sample_mode == "train"):
                    sigma_scale, scale_log_prob, key = self.return_sigma_scale_factor(self.sigma_scale_factor*annealed_scale, shape, key)
                elif(sample_mode == "val"):
                    sigma_scale, scale_log_prob, key = self.return_sigma_scale_factor(self.sigma_scale_factor*annealed_scale, shape, key)
                    # sigma_scale = (self.sigma_scale_factor**2 + 1)*jnp.ones(shape)    #this is the mode, not the expectation value
                    # scale_log_prob = jnp.zeros((n_states,))
                else:
                    sigma_scale = 1.*jnp.ones(shape)
                    scale_log_prob = jnp.zeros((n_states,))
            

            temp = 1.
        else:
            temp = temp
            sigma_scale = 1.*jnp.ones(shape)
            scale_log_prob = jnp.zeros((n_states,))

        return sigma_scale, scale_log_prob, temp, key
    
    def simulate_reverse_sde_scan(self, model, params, Interpol_params, SDE_params, temp, key, n_states = 100, sample_mode = "train", n_integration_steps = 1000):
        
        for interpol_key in Interpol_params.keys():
            SDE_params[interpol_key] = Interpol_params[interpol_key]

        sigma_scale, scale_log_prob, temp, key = self.get_sigma_noise(n_states, key, sample_mode, temp)

        def scan_fn(carry, step):
            x, t, counter, key, carry_dict = carry
            hidden_state = carry_dict["hidden_state"]
            score, new_hidden_state, grad, SDE_params_extended, interpol_log_prob, key = self.apply_model(model, x, t, counter, params, Interpol_params, SDE_params, hidden_state, temp, key)
            carry_dict["hidden_state"] = new_hidden_state

            dt = self.reversed_dt_values[step]
            
            reverse_out_dict, key = self.reverse_sde(SDE_params_extended, score, x, t, dt, sigma_scale, key)

            SDE_tracker_step = {
            "interpolated_grad": grad,
            "dW": reverse_out_dict["dW"],
            "xs": x,
            "ts": t,
            "scores": score,
            "forward_drift": reverse_out_dict["forward_drift"],
            "reverse_drift": reverse_out_dict["reverse_drift"],
            "drift_ref": reverse_out_dict["drift_ref"],
            "dts": dt,
            "key": key,
            "hidden_state": carry_dict["hidden_state"],
            "interpol_log_probs": interpol_log_prob
            }

            x = reverse_out_dict["x_next"]
            t = reverse_out_dict["t_next"]
            return (x, t, counter + 1, key, carry_dict), SDE_tracker_step
    

        x_prior, key = self.sample_prior(SDE_params, key, n_states, sigma_scale_factor = sigma_scale)

        if(self.stop_gradient):
            x_prior = jax.lax.stop_gradient(x_prior)

        if(self.invariance == True):
            x_prior = self.subtract_COM(x_prior)

        # print("x_prior", x_prior.shape, mean.shape, sigma.shape)
        # print(jnp.mean(x_prior), jnp.mean(mean))
        t = 1.0
        dt = 1. / n_integration_steps
        counter = 0

        #print("no scan", model.apply(params, x0[0:10], t*jnp.ones((10, 1))))
        init_carry = jnp.zeros((n_states, self.Network_Config["n_hidden"]))
        carry_dict = {"hidden_state": [(init_carry, init_carry)  for i in range(self.Network_Config["n_layers"])]}
        ### scan because jit would take too long
        (x_final, t_final, counter, key, carry_dict), SDE_tracker_steps = jax.lax.scan(
            scan_fn,
            (x_prior, t, counter, key, carry_dict),
            jnp.arange(n_integration_steps)
        )

        SDE_tracker = {
            "scale_log_prob": scale_log_prob,
            "noise_scale": sigma_scale,
            "dW": SDE_tracker_steps["dW"],
            "xs": SDE_tracker_steps["xs"],
            "ts": SDE_tracker_steps["ts"],
            "scores": SDE_tracker_steps["scores"],
            "forward_drift": SDE_tracker_steps["forward_drift"],
            "reverse_drift": SDE_tracker_steps["reverse_drift"],
            "drift_ref": SDE_tracker_steps["drift_ref"],
            "dts": SDE_tracker_steps["dts"],
            "x_final": x_final,
            "x_prior": x_prior,
            "hidden_states": SDE_tracker_steps["hidden_state"],
            "keys": SDE_tracker_steps["key"],
            "interpolated_grads": SDE_tracker_steps["interpolated_grad"],
            "interpol_log_probs": SDE_tracker_steps["interpol_log_probs"]

        }


        if(self.Network_Config["model_mode"] == "latent"):
            # compute decoder and encoder probability
            ### TODO make sure that the process before is done in latent dim
            z_final = x_final
            decode_in_dict = {"z": z_final}
            decode_out_dict = model.apply(params, decode_in_dict, train = True, forw_mode = "decode")
            mean_decode = decode_out_dict["mean_x"]
            log_var_decode = decode_out_dict["log_var_x"]
            ### todo sample from decoder
            key, subkey = jax.random.split(key)
            noise = jax.random.normal(subkey, shape = mean_decode.shape)
            x_final = noise * jnp.exp(0.5*log_var_decode) + mean_decode

            if(self.stop_gradient):
                x_final = jax.lax.stop_gradient(x_final)

            log_p_decode = jnp.sum(jax.scipy.stats.norm.logpdf(x_final, loc = mean_decode, scale = jnp.exp(0.5*log_var_decode)), axis = -1)

            in_dict = {"x": x_final}
            encode_out_dict = model.apply(params, in_dict, train = True, forw_mode = "encode")
            mean_decode_z = encode_out_dict["mean_z"]
            log_var_decode_z = encode_out_dict["log_var_z"]

            ### TODO evaluate p_encode(z|x)
            log_p_encode = jnp.sum(jax.scipy.stats.norm.logpdf(z_final, loc = mean_decode_z, scale = jnp.exp(0.5*log_var_decode_z)), axis = -1)
            
            xs_updated = jnp.concatenate([SDE_tracker_steps["xs"], x_final[None, ...]], axis = 0)
            latent_SDE_dict = {"log_p_decode": log_p_decode, "log_p_encode": log_p_encode, "x_final": x_final, "xs": xs_updated}

            for dict_key in latent_SDE_dict.keys():
                SDE_tracker[dict_key] = latent_SDE_dict[dict_key]


        return SDE_tracker, key
    

def inverse_softplus(x):
    return jnp.log(jnp.exp(x) - 1)
    


