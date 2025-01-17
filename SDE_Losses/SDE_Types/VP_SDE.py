import jax
from jax import random
import matplotlib.pyplot as plt
import jax.numpy as jnp
import wandb
import numpy as np
from .Base_SDE import Base_SDE_Class

### Variance Preserving SDE implemented with the framework from IMPROVED SAMPLING VIA LEARNED DIFFUSIONS
class VP_SDE_Class(Base_SDE_Class):
    def __init__(self, SDE_Type_Config, Network_Config, Energy_Class):
        self.beta_min = SDE_Type_Config["beta_min"]
        self.beta_max = SDE_Type_Config["beta_max"]
        self.config = SDE_Type_Config
        super().__init__(SDE_Type_Config, Network_Config, Energy_Class)


    def get_log_prior(self, SDE_params, x, sigma_scale_factor = 1.):
        mean = self.get_mean_prior(SDE_params)

        if(self.invariance):
            overall_sigma = self.return_prior_covar(SDE_params, sigma_scale_factor = sigma_scale_factor)
            log_pdf_vec = jax.scipy.stats.norm.logpdf(x, loc=mean, scale=overall_sigma) + 0.5*jnp.log(2 * jnp.pi * overall_sigma)/overall_sigma.shape[0]*self.Energy_Class.particle_dim
            log_pdf = jnp.sum(log_pdf_vec, axis = -1)
            return log_pdf
        if(not self.learn_covar):
            sigma= self.return_prior_covar(SDE_params, sigma_scale_factor = sigma_scale_factor)
            log_pdf_vec = jax.scipy.stats.norm.logpdf(x, loc=mean, scale=sigma) 
            log_pdf = jnp.sum(log_pdf_vec, axis = -1)
            return log_pdf
        else:
            overall_covar = self.return_prior_covar(SDE_params, sigma_scale_factor = sigma_scale_factor)
            log_pdf = jax.scipy.stats.multivariate_normal.logpdf(x, mean, overall_covar)
            return log_pdf

    
    def return_prior_covar(self, SDE_params, sigma_scale_factor = 1.):
        if(self.learn_covar):
            if(self.invariance):
                sigma, sigma_t = self.get_SDE_sigma(SDE_params)
                sigma = sigma*sigma_scale_factor
                overall_sigma = jnp.sqrt(self.beta_int(SDE_params, 1)*(sigma_t**2 - sigma**2) + sigma**2)
                return overall_sigma
            else:
                sigma, covar = self.get_SDE_sigma(SDE_params)
                sigma = sigma*sigma_scale_factor
                alpha = self.beta_int(SDE_params, 1)
                factor = jnp.exp(-alpha)[:,None]*jnp.exp(-alpha)[None, :]
                overall_covar = factor*(covar - jnp.diag(sigma)**2) + jnp.diag(sigma)**2

                return overall_covar
        else:
            if(self.invariance):
                sigma, sigma_t = self.get_SDE_sigma(SDE_params)
                sigma = sigma*sigma_scale_factor
                overall_sigma = sigma
                return overall_sigma
            else:
                sigma, covar = self.get_SDE_sigma(SDE_params)
                sigma = sigma*sigma_scale_factor
                return sigma

    
    def get_SDE_params(self):
        # "mean" is here the mean of the SDE
        if(self.invariance):
            SDE_params = {"log_beta_delta": jnp.log(self.config["beta_max"] - self.config["beta_min"]), 
            "log_beta_min": jnp.log(self.config["beta_min"]),
            "log_sigma": jnp.log(self.sigma_init), "mean": jnp.zeros((self.dim_x,)), "mean_target": jnp.zeros((self.dim_x,)),
            "log_sigma_t": jnp.log(10**-5),
            "beta_interpol_params": jnp.ones((self.n_integration_steps)),
            "repulsion_interpol_params": jnp.ones((self.n_integration_steps))}

        else:
            SDE_params = {"log_beta_delta": jnp.log(self.config["beta_max"] - self.config["beta_min"])* jnp.ones((self.dim_x,)), 
                        "log_beta_min": jnp.log(self.config["beta_min"])* jnp.ones((self.dim_x,)),
                        "log_sigma": jnp.log(self.sigma_init)* jnp.ones((self.dim_x,)), "mean": jnp.zeros((self.dim_x,)),  "mean_target": jnp.zeros((self.dim_x,)),
                        "B": -10*jnp.ones((self.dim_x,self.dim_x)) + jnp.diag((jnp.log(self.sigma_init)+10.)*jnp.ones((self.dim_x,))),
                        "beta_interpol_params": jnp.ones((self.n_integration_steps)),
                        "repulsion_interpol_params": jnp.ones((self.n_integration_steps))}
                        
        return SDE_params
    
    def get_SDE_mean(self, SDE_params):
        if(self.invariance):
            mean = jnp.zeros((self.dim_x,))
        else:
            mean = SDE_params["mean"]
        return mean
    
    def get_mean_prior(self, SDE_params):
        if(self.invariance):
            mean = jnp.zeros((self.dim_x,))
            mean_target = jnp.zeros((self.dim_x,))
            overall_mean = mean
            return overall_mean
        if(self.learn_covar):
            mean = SDE_params["mean"]
            mean_target = SDE_params["mean_target"]
            alpha = self.beta_int(SDE_params, 1)
            overall_mean = (1- jnp.exp(- alpha))*mean + jnp.exp(- alpha)*mean_target
            return overall_mean
        else:
            overall_mean = SDE_params["mean"]
            return overall_mean

    def get_SDE_sigma(self, SDE_params):
        if(self.invariance):
            sigma = jnp.exp(SDE_params["log_sigma"])*jnp.ones((self.dim_x,))
            sigma_t = jnp.exp(SDE_params["log_sigma_t"])*jnp.ones((self.dim_x,))
            return sigma, sigma_t
        else:
            sigma = jnp.exp(SDE_params["log_sigma"])
            B = SDE_params["B"]
            A = 0.5*(B + B.T)
            covar = jnp.exp(A)

            return sigma, covar

    def compute_p_xt_g_x0_statistics(self, x0, xt, t):
        mean_xt = x0 * jnp.exp(-self.beta_int(t)) 
        std_xt = jnp.sqrt(1.-1.*jnp.exp(-2*self.beta_int(t)))
        statistics_dict = {"mean": mean_xt, "std": std_xt}
        return statistics_dict
    
    def beta_int(self, SDE_params, t):
        ### TODO check if t is correct here!
        beta_min, beta_max = self.get_beta_min_and_max(SDE_params)
        beta_int_value = 0.5*t**2*(beta_max-beta_min) + t*beta_min
        return beta_int_value

    def beta(self, SDE_params, t):
        beta_min, beta_max = self.get_beta_min_and_max(SDE_params)
        return (beta_min + t * (beta_max - beta_min)) ### TODO chekc this factor 0.5

    def forward_sde(self, x, t, dt, key):
        drift = self.get_drift(x, t)
        diffusion = self.get_diffusion(x, t)

        key, subkey = random.split(key)
        noise = random.normal(subkey, shape=x.shape)
        x_next = x + drift * dt + diffusion * jnp.sqrt(dt) * noise
        return x_next, t + dt, key

    ### THIs implements drift and diffusion as in vargas papers
    def get_drift(self, SDE_params, x, t):
        mean = self.get_SDE_mean(SDE_params)
        return - self.beta(SDE_params, t)[None, :] * (x-mean[None, :])
    
    def get_div_drift(self, SDE_params, t):
        return - self.beta(SDE_params, t)
    
    def get_diffusion(self, SDE_params, x, t):
        sigma, _ = self.get_SDE_sigma(SDE_params)
        diffusion = sigma*jnp.sqrt(2*self.beta(SDE_params, t))
        return diffusion[None, :]
    
    def sample(self, shape, key):
        return random.normal(key, shape)
    
    


