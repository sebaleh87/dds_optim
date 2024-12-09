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

    def get_SDE_params(self):
        if(self.invariance):
            SDE_params = {"log_beta_delta": jnp.log(self.config["beta_max"])* jnp.ones((self.dim_x,)), 
            "log_beta_min": jnp.log(self.config["beta_min"])* jnp.ones((self.dim_x,)),
            "log_sigma": jnp.log(1), "mean": jnp.zeros((self.dim_x,)), 
            "log_sigma_t": jnp.log(1)}

        else:
            SDE_params = {"log_beta_delta": jnp.log(self.config["beta_max"])* jnp.ones((self.dim_x,)), 
                        "log_beta_min": jnp.log(self.config["beta_min"])* jnp.ones((self.dim_x,)),
                        "log_sigma": jnp.log(1)* jnp.ones((self.dim_x,)), "mean": jnp.zeros((self.dim_x,)), 
                        "log_sigma_t": jnp.log(1)* jnp.ones((self.dim_x,))}
        return SDE_params

    def get_SDE_mean(self, SDE_params):
        if(self.invariance):
            mean = jnp.zeros((self.dim_x,))
        else:
            mean = SDE_params["mean"]
        return mean

    def get_SDE_sigma(self, SDE_params):
        if(self.invariance):
            sigma = jnp.exp(SDE_params["log_sigma"])*jnp.ones((self.dim_x,))
            sigma_t = jnp.exp(SDE_params["log_sigma_t"])*jnp.ones((self.dim_x,))
        else:
            sigma = jnp.exp(SDE_params["log_sigma"])
            sigma_t = jnp.exp(SDE_params["log_sigma_t"])
        return sigma, sigma_t

    def get_log_prior(self, SDE_params, x):
        sigma, sigma_t = self.get_SDE_sigma(SDE_params)
        overall_sigma = jnp.sqrt(sigma**2*self.beta_int(SDE_params, 1) + sigma_t**2)
        mean = self.get_SDE_mean(SDE_params)
        #print("VP_SDE", x.shape, mean.shape, sigma.shape)
        if(self.invariance):
            return jax.scipy.stats.norm.logpdf(x, loc=mean, scale=overall_sigma) + 0.5*jnp.log(2 * jnp.pi * overall_sigma[0])*self.Energy_Class.particle_dim
        else:
            return jax.scipy.stats.norm.logpdf(x, loc=mean, scale=overall_sigma) 
    
    def sample_prior(self, SDE_params, key, n_states):
        key, subkey = random.split(key)
        sigma, sigma_t = self.get_SDE_sigma(SDE_params)
        overall_sigma = jnp.sqrt(sigma**2*self.beta_int(SDE_params, 1) + sigma_t**2)
        mean = self.get_SDE_mean(SDE_params)
        x_prior = random.normal(subkey, shape=(n_states, self.dim_x))*overall_sigma[None, :] + mean[None, :]
        return x_prior, key

    def beta_int(self, SDE_params, t):
        beta_delta = jnp.exp(SDE_params["log_beta_delta"])
        beta_min = jnp.exp(SDE_params["log_beta_min"])
        beta_max = beta_min + beta_delta
        return beta_min*((beta_max/beta_min)**t-1)/(jnp.log(beta_max)- jnp.log(beta_min)) ### TODO chekc this factor 0.5

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
        mean = self.get_SDE_mean(SDE_params)
        return jnp.zeros_like(mean)
    
    def get_diffusion(self, SDE_params, x, t):
        sigma, sigma_t = self.get_SDE_sigma(SDE_params)
        diffusion = sigma*jnp.sqrt(2*self.beta(SDE_params, t))
        return diffusion[None, :] 
    
    def sample(self, shape, key):
        return random.normal(key, shape)
    
    


