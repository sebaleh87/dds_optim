import jax
from jax import random
import matplotlib.pyplot as plt
import jax.numpy as jnp
import wandb
import numpy as np
from .Base_SDE import Base_SDE_Class

class VP_SDE_Class(Base_SDE_Class):
    def __init__(self, SDE_Type_Config, Network_Config, Energy_Class):
        self.beta_min = SDE_Type_Config["beta_min"]
        self.beta_max = SDE_Type_Config["beta_max"]
        self.sigma_sde = 1.
        self.config = SDE_Type_Config
        super().__init__(SDE_Type_Config, Network_Config, Energy_Class)
        ### THIS code assumes that sigma of reference distribution is 1

    def get_log_prior(self, SDE_params, x):
        sigma = self.get_SDE_sigma(SDE_params)
        mean = self.get_SDE_mean(SDE_params)
        #print("VP_SDE", x.shape, mean.shape, sigma.shape)
        if(self.invariance):
            return jax.scipy.stats.norm.logpdf(x, loc=mean, scale=sigma) + 0.5*jnp.log(2 * jnp.pi * sigma[0])*self.Energy_Class.particle_dim
        else:
            return jax.scipy.stats.norm.logpdf(x, loc=mean, scale=sigma) 

    def sample_prior(self, SDE_params, key, n_states):
        key, subkey = random.split(key)
        sigma = self.get_SDE_sigma(SDE_params)
        mean = self.get_SDE_mean(SDE_params)
        x_prior = random.normal(subkey, shape=(n_states, self.dim_x))*sigma[None, :] + mean[None, :]
        return x_prior, key

    def get_SDE_params(self):
        SDE_params = {"log_beta_delta": jnp.log(self.config["beta_max"])* jnp.ones((self.dim_x,)), 
                      "log_beta_min": jnp.log(self.config["beta_min"])* jnp.ones((self.dim_x,)),
                      "log_sigma": jnp.log(1)* jnp.ones((self.dim_x,)), "mean": jnp.zeros((self.dim_x,))}
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
        else:
            sigma = SDE_params["sigma"]
        return sigma

    def compute_p_xt_g_x0_statistics(self, x0, xt, t):
        mean_xt = x0 * jnp.exp(-self.beta_int(t)) 
        std_xt = jnp.sqrt(1.-1.*jnp.exp(-2*self.beta_int(t)))
        statistics_dict = {"mean": mean_xt, "std": std_xt}
        return statistics_dict
    
    def beta_int(self, SDE_params, t):
        ### TODO check if t is correct here!
        beta_delta = jnp.exp(SDE_params["log_beta_delta"])
        beta_min = jnp.exp(SDE_params["log_beta_min"])
        beta_max = beta_min + beta_delta
        beta_int_value = 0.5*t**2*(beta_max-beta_min) + t*beta_min
        return beta_int_value

    def beta(self, SDE_params, t):
        beta_delta = jnp.exp(SDE_params["log_beta_delta"])
        beta_min = jnp.exp(SDE_params["log_beta_min"])
        beta_max = beta_min + beta_delta
        return (beta_min + t * (beta_max - beta_min)) ### TODO chekc this factor 0.5

    def forward_sde(self, x, t, dt, key):
        drift = self.get_drift(x, t)
        diffusion = self.get_diffusion(x, t)

        key, subkey = random.split(key)
        noise = random.normal(subkey, shape=x.shape)
        x_next = x + drift * dt + diffusion * jnp.sqrt(dt) * noise
        return x_next, t + dt, key

    def interpol_func(self, x, t, SDE_params, Energy_params, key):
        Energy_value, key = self.Energy_Class.calc_energy(x, Energy_params, key)
        interpol = (t)*jnp.sum(self.get_log_prior(SDE_params,x), axis = -1) + (1-t)*Energy_value
        return interpol, key

    ### THIs implements drift and diffusion as in vargas papers
    def get_drift(self, SDE_params, x, t):
        return - self.beta(SDE_params, t)[None, :] * (x-SDE_params["mean"][None, :])
    
    def get_div_drift(self, SDE_params, t):
        return - self.beta(SDE_params, t)
    
    def get_diffusion(self, SDE_params, x, t):
        sigma = self.get_SDE_sigma(SDE_params)
        diffusion = sigma*jnp.sqrt(2*self.beta(SDE_params, t))
        return diffusion[None, :] 
    
    def sample(self, shape, key):
        return random.normal(key, shape)
    
    


