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

    def get_log_prior(self, x, log_sigma = 1.):
        sigma = 1.
        return jax.scipy.stats.norm.logpdf(x, loc=0, scale=sigma)

    def compute_p_xt_g_x0_statistics(self, x0, xt, t):
        mean_xt = x0 * jnp.exp(-self.beta_int(t)) 
        std_xt = jnp.sqrt(1.-1.*jnp.exp(-2*self.beta_int(t)))
        statistics_dict = {"mean": mean_xt, "std": std_xt}
        return statistics_dict
    
    def beta_int(self, t):
        beta_int_value = 1/4*t**2*(self.beta_max-self.beta_min) + 0.5*t*self.beta_min
        return beta_int_value

    def beta(self, t):
        return 0.5*(self.beta_min + t * (self.beta_max - self.beta_min))

    def forward_sde(self, x, t, dt, key):
        drift = self.get_drift(x, t)
        diffusion = self.get_diffusion(x, t)

        key, subkey = random.split(key)
        noise = random.normal(subkey, shape=x.shape)
        x_next = x + drift * dt + diffusion * jnp.sqrt(dt) * noise
        return x_next, t + dt, key

    ### THIs implements drift and diffusion as in vargas papers
    def get_drift(self, x, t):
        return - self.beta(t) * x
    
    def get_diffusion(self, x, t, log_sigma):
        sigma = jnp.exp(log_sigma)
        diffusion = sigma*jnp.sqrt(2*self.beta(t)*(1-jnp.exp(- 4*self.beta_int(t))))
        return diffusion
    
    def reverse_sde(self, score, log_sigma, x, t, dt, key):
        ### TODO implement hacks
        ### TODO also use gradet of target sto parameterize the score?
        # initialize to optial controls at t= 0 and t = 1
        beta_t = self.beta(t)
        forward_drift = self.get_drift(x, t)
        diffusion = self.get_diffusion(x, t, log_sigma)

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
    


