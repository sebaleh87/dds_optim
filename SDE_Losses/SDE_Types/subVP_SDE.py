import jax
from jax import random
import matplotlib.pyplot as plt
import jax.numpy as jnp
import wandb
import numpy as np
from .Base_SDE import Base_SDE_Class

class subVP_SDE_Class(Base_SDE_Class):
    def __init__(self, SDE_Type_Config, Network_Config, Energy_Class):
        self.beta_min = SDE_Type_Config["beta_min"]
        self.beta_max = SDE_Type_Config["beta_max"]
        self.sigma_sde = 1.
        self.config = SDE_Type_Config
        super().__init__(SDE_Type_Config, Network_Config, Energy_Class)
        ### THIS code assumes that sigma of reference distribution is 1

    def get_log_prior(self, SDE_params, x):
        sigma = jnp.exp(SDE_params["log_sigma"])
        mean = SDE_params["mean"]
        return jax.scipy.stats.norm.logpdf(x, loc=mean, scale=sigma)

    def compute_p_xt_g_x0_statistics(self, x0, xt, t):
        mean_xt = x0 * jnp.exp(-self.beta_int(t)) 
        std_xt = jnp.sqrt(1.-1.*jnp.exp(-2*self.beta_int(t)))
        statistics_dict = {"mean": mean_xt, "std": std_xt}
        return statistics_dict
    
    def beta_int(self, SDE_params, t):
        beta_delta = jnp.exp(SDE_params["log_beta_delta"])
        beta_min = jnp.exp(SDE_params["log_beta_min"])
        beta_max = beta_min + beta_delta
        beta_int_value = 1/2*t**2*(beta_max-beta_min) + t*beta_min
        return beta_int_value

    def beta(self, SDE_params, t):
        beta_delta = jnp.exp(SDE_params["log_beta_delta"])
        beta_min = jnp.exp(SDE_params["log_beta_min"])
        beta_max = beta_min + beta_delta
        return (beta_min + t * (beta_max - beta_min))

    def forward_sde(self, SDE_params, x, t, dt, key):
        drift = self.get_drift(SDE_params, x, t)
        diffusion = self.get_diffusion(SDE_params, x, t)

        key, subkey = random.split(key)
        noise = random.normal(subkey, shape=x.shape)
        x_next = x + drift * dt + diffusion * jnp.sqrt(dt) * noise
        return x_next, t + dt, key

    ### THIs implements drift and diffusion as in vargas papers
    def get_drift(self, SDE_params, x, t):
        return - self.beta(SDE_params, t) * x
    
    def get_diffusion(self, SDE_params, x, t):
        sigma = jnp.exp(SDE_params["log_sigma"])
        diffusion = sigma*jnp.sqrt(2*self.beta(SDE_params, t)*(1-jnp.exp(- 2*self.beta_int(SDE_params, t))))
        return diffusion[None, :] 
    
    def reverse_sde(self, SDE_params, score, x, t, dt, key):
        ### TODO implement hacks
        ### TODO also use gradet of target sto parameterize the score?
        # initialize to optial controls at t= 0 and t = 1
        beta_t = self.beta(SDE_params, t)[None, :] 
        forward_drift = self.get_drift(SDE_params, x, t)
        diffusion = self.get_diffusion(SDE_params, x, t)

        reverse_drift = diffusion**2*score - forward_drift #(forward_drift - beta_t * score)


        key, subkey = random.split(key)
        noise = random.normal(subkey, shape=x.shape)
        dW = jnp.sqrt(dt) * noise

        if(self.stop_gradient):
            x_next = jax.lax.stop_gradient(x + reverse_drift  * dt  + diffusion * dW)
        else:
            x_next = x + reverse_drift  * dt  + diffusion * dW

        ### TODO check at which x drift ref should be evaluated?
        reverse_out_dict = {"x_next": x_next, "t_next": t - dt, "drift_ref": x, "beta_t": beta_t, "forward_drift": forward_drift, "reverse_drift": reverse_drift, "dW": dW}
        return reverse_out_dict, key

    def sample(self, shape, key):
        return random.normal(key, shape)
    
    
    def init_p_ref(self,x_dim):
        sigma_ref = 1.*jnp.ones((x_dim,))
        mean_ref = 0.*jnp.ones((x_dim,)) 
        return {"sigma_ref_param": sigma_ref, "mean_ref": mean_ref}
    


