import jax
from jax import random
import matplotlib.pyplot as plt
import jax.numpy as jnp
import wandb
import numpy as np
from .Base_SDE import Base_SDE_Class

class VE_SDE_Class(Base_SDE_Class):
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
        #print("VP_SDE", x.shape, mean.shape, sigma.shape)
        
        return jax.scipy.stats.norm.logpdf(x, loc=mean, scale=sigma) 
    
    def beta(self, SDE_params, t):
        beta_delta = jnp.exp(SDE_params["log_beta_delta"])
        beta_min = jnp.exp(SDE_params["log_beta_min"])
        beta_max = beta_min + beta_delta
        return beta_min*(beta_max/beta_min)**t ### TODO chekc this factor 0.5


    def interpol_func(self, x, t, SDE_params, Energy_params, key):
        Energy_value, key = self.Energy_Class.calc_energy(x, Energy_params, key)
        interpol = Energy_value
        return interpol, key

    ### THIs implements drift and diffusion as in vargas papers
    def get_drift(self, SDE_params,x, t):
        return jnp.zeros_like(x)
    
    def get_div_drift(self, SDE_params, t):
        return jnp.zeros_like(SDE_params["mean"])
    
    def get_diffusion(self, SDE_params, x, t):
        sigma = jnp.exp(SDE_params["log_sigma"])
        diffusion = sigma*jnp.sqrt(2*self.beta(SDE_params, t))
        return diffusion[None, :] 
    
    def sample(self, shape, key):
        return random.normal(key, shape)
    
    
    def init_p_ref(self,x_dim):
        sigma_ref = 1.*jnp.ones((x_dim,))
        mean_ref = 0.*jnp.ones((x_dim,)) 
        return {"sigma_ref_param": sigma_ref, "mean_ref": mean_ref}
    


