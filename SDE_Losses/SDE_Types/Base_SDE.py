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
        self.dim_x = self.Energy_Class.dim_x
        self.use_interpol_gradient = config["use_interpol_gradient"]
        self.Network_Config = Network_Config
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

    def weightening(self, t):
        SDE_params = self.get_SDE_params()
        weight = jnp.mean((1-jnp.exp(- 2*jax.vmap(self.beta_int, in_axes=(None, 0))(SDE_params, t))), axis = -1)
        return weight
    
    def den_weighting(self, t):
        SDE_params = self.get_SDE_params()
        den_weight =  jnp.mean(2*jax.vmap(self.beta, in_axes=(None, 0))(SDE_params, t), axis = -1)
        return den_weight

    def get_SDE_params(self):

        SDE_params = {"log_beta_delta": jnp.log(self.config["beta_max"])* jnp.ones((self.dim_x,)), 
                      "log_beta_min": jnp.log(self.config["beta_min"])* jnp.ones((self.dim_x,)),
                      "log_sigma": jnp.log(1)* jnp.ones((self.dim_x,)), "mean": jnp.zeros((self.dim_x,))}
        return SDE_params

    def compute_p_xt_g_x0_statistics(self, x0, xt, t):
        raise NotImplementedError("get_diffusion method not implemented")

    def get_log_prior(self, x):
        raise NotImplementedError("get_diffusion method not implemented")
    
    def vmap_prior_target_grad_interpolation(self, x, t, Energy_params, SDE_params, key):
        key, subkey = random.split(key)
        batched_subkey = random.split(subkey, x.shape[0])
        vmap_energy, vmap_grad = jax.vmap(self.prior_target_grad_interpolation, in_axes=(0, None, None, None, 0))(x,t, Energy_params, SDE_params, batched_subkey)
        #print("vmap_grad", jnp.mean(jax.lax.stop_gradient(vmap_grad)))
        vmap_grad = jnp.where(jnp.isfinite(vmap_grad), vmap_grad, 0)
        return vmap_energy, vmap_grad, key
    
    def prior_target_grad_interpolation(self, x, t, Energy_params, SDE_params, key):
        
        #interpol = lambda x: self.Energy_Class.calc_energy(x, Energy_params, key)
        (Energy, key), (grad)  = jax.value_and_grad(self.interpol_func, has_aux=True)( x,t, SDE_params, Energy_params, key)
        #grad = jnp.clip(grad, -10**2, 10**2)
        return jnp.expand_dims(Energy, axis = -1), grad

    def get_diffusion(self, t, x, log_sigma):
        """
        Method to get the diffusion term of the SDE.
        
        Parameters:
        t (float): Time variable.
        x (float): State variable.
        
        Returns:
        float: Diffusion term.
        """
        raise NotImplementedError("get_diffusion method not implemented")

    def get_drift(self, t, x):
        """
        Method to get the drift term of the SDE.
        
        Parameters:
        t (float): Time variable.
        x (float): State variable.
        
        Returns:
        float: Drift term.
        """
        raise NotImplementedError("get_drift method not implemented")

    def reverse_sde(self, SDE_params, score, x, t, dt, key, mode = "DIS"):
        """
        Method to simulate the forward SDE.
        
        Parameters:
        x0 (float): Initial state.
        t0 (float): Initial time.
        t1 (float): Final time.
        dt (float): Time step.
        
        Returns:
        list: Simulated path of the state variable.
        """
        raise NotImplementedError("simulate_forward_SDE method not implemented")

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
    
    def reverse_sde(self, SDE_params, score, x, t, dt, key):
        forward_drift = self.get_drift(SDE_params, x, t)
        diffusion = self.get_diffusion(SDE_params, x, t)

        reverse_drift = diffusion**2*score - forward_drift #TODO check is this power of two correct? I think yes because U = diffusion*score

        key, subkey = random.split(key)
        noise = random.normal(subkey, shape=x.shape)
        dW = jnp.sqrt(dt) * noise
        dW_sampled = dW

        if(self.invariance == True):
            dW = self.subtract_COM(dW)
            #reverse_drift = self.subtract_COM(reverse_drift)

        if(self.stop_gradient):
            x_next = jax.lax.stop_gradient(x + reverse_drift  * dt  + diffusion * dW)
        else:
            x_next = x + reverse_drift  * dt  + diffusion * dW

        ### TODO check at which x drift ref should be evaluated?
        reverse_out_dict = {"x_next": x_next, "t_next": t - dt, "drift_ref": x, "forward_drift": forward_drift, "reverse_drift": reverse_drift, "dW": dW_sampled}
        return reverse_out_dict, key

    
    def simulate_reverse_sde_scan(self, model, params, Energy_params, SDE_params, key, n_states = 100, x_dim = 2, n_integration_steps = 1000):
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
                grad = jnp.zeros((x.shape[0], x_dim))
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

        key, subkey = random.split(key)
        sigma = jnp.exp(SDE_params["log_sigma"])
        mean = SDE_params["mean"]
        x_prior = random.normal(subkey, shape=(n_states, x_dim))*sigma[None, :] + mean[None, :]

        if(self.stop_gradient):
            x_prior = jax.lax.stop_gradient(x_prior)
        x_prior_sampled = x_prior

        if(self.invariance == True):
            x_prior = self.subtract_COM(x_prior)

        # print("x_prior", x_prior.shape, mean.shape, sigma.shape)
        # print(jnp.mean(x_prior), jnp.mean(mean))
        t = 1.0
        dt = 1. / n_integration_steps

        #print("no scan", model.apply(params, x0[0:10], t*jnp.ones((10, 1))))
        init_carry = jnp.zeros((n_states, self.Network_Config["n_hidden"]))
        carry_dict = {"hidden_state": [(init_carry, init_carry)  for i in range(self.Network_Config["n_layers"])]}
        (x_final, t_final, key, carry_dict), SDE_tracker_steps = jax.lax.scan(
            scan_fn,
            (x_prior, t, key, carry_dict),
            jnp.arange(n_integration_steps)
        )

        SDE_tracker = {
            "dW": SDE_tracker_steps["dW"],
            "xs": SDE_tracker_steps["xs"],
            "ts": SDE_tracker_steps["ts"],
            "scores": SDE_tracker_steps["scores"],
            "forward_drift": SDE_tracker_steps["forward_drift"],
            "reverse_drift": SDE_tracker_steps["reverse_drift"],
            "drift_ref": SDE_tracker_steps["drift_ref"],
            "dts": SDE_tracker_steps["dts"],
            "x_final": x_final,
            "x_prior": x_prior_sampled
        }

        return SDE_tracker, key

