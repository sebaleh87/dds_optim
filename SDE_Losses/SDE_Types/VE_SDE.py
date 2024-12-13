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
            ### if beta is learnable this also ahs to be dim(1)
            SDE_params = {"log_beta_delta": jnp.log(self.config["beta_max"] - self.config["beta_min"]), 
            "log_beta_min": jnp.log(self.config["beta_min"]),
            "log_sigma": jnp.log(self.sigma_init), "mean": jnp.zeros((self.dim_x,)), 
            "log_sigma_t": jnp.log(10**-5)}

        else:
            SDE_params = {"log_beta_delta": jnp.log(self.config["beta_max"] - self.config["beta_min"])* jnp.ones((self.dim_x,)), 
                        "log_beta_min": jnp.log(self.config["beta_min"])* jnp.ones((self.dim_x,)),
                        "log_sigma": jnp.log(self.sigma_init)* jnp.ones((self.dim_x,)), "mean": jnp.zeros((self.dim_x,)), 
                         "B": -10*jnp.ones((self.dim_x,self.dim_x)) + jnp.diag((jnp.log(self.sigma_init)+10.)*jnp.ones((self.dim_x,)))}
        return SDE_params

    def get_mean_prior(self, SDE_params):
        if(self.invariance):
            mean = jnp.zeros((self.dim_x,))
        else:
            mean = SDE_params["mean"]
        overall_mean = mean 
        return overall_mean

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
            return sigma, sigma_t
        else:
            sigma = jnp.exp(SDE_params["log_sigma"])
            B = SDE_params["B"]
            A = 0.5*(B + B.T)
            covar = jnp.exp(A)

            return sigma, covar

    def get_log_prior(self, SDE_params, x):
        mean = self.get_mean_prior(SDE_params)
        #print("VP_SDE", x.shape, mean.shape, sigma.shape)
        if(self.invariance):
            overall_sigma = self.return_prior_covar(SDE_params)
            log_pdf_vec =  jax.scipy.stats.norm.logpdf(x, loc=mean, scale=overall_sigma) + 0.5*jnp.log(2 * jnp.pi * overall_sigma)/overall_sigma.shape[0]*self.Energy_Class.particle_dim
            return jnp.sum(log_pdf_vec, axis = -1)
        if(not self.learn_covar):
            prior_sigma = self.return_prior_covar(SDE_params)
            #return jax.random.multivariate_normal(random.PRNGKey(0), mean, jnp.diag(overall_sigma**2), x.shape[0])
            log_pdf_vec = jax.scipy.stats.norm.logpdf(x, loc=mean, scale=prior_sigma) 
            log_pdf = jnp.sum(log_pdf_vec, axis = -1)
            return log_pdf
        else:
            overall_covar = self.return_prior_covar(SDE_params)
            log_pdf = jax.scipy.stats.multivariate_normal.logpdf(x, mean, overall_covar)
            return log_pdf
    

    def sample_prior(self, SDE_params, key, n_states):
        key, subkey = random.split(key)
        prior_mean = self.get_mean_prior(SDE_params)
        if(self.invariance):
            overall_sigma = self.return_prior_covar(SDE_params)
            x_prior = random.normal(subkey, shape=(n_states, self.dim_x))*overall_sigma[None, :] + prior_mean[None, :]
        elif(not self.learn_covar):
            prior_sigma = self.return_prior_covar(SDE_params)
            x_prior = random.normal(subkey, shape=(n_states, self.dim_x))*prior_sigma[None, :] + prior_mean[None, :]
        else:
            overall_covar = self.return_prior_covar(SDE_params)
            x_prior = jax.random.multivariate_normal(subkey, prior_mean, overall_covar, (n_states,))
        return x_prior, key
    
    def return_prior_covar(self, SDE_params):
        if(self.learn_covar):
            if(self.invariance):
                sigma, sigma_t = self.get_SDE_sigma(SDE_params)
                alpha = self.beta_int(SDE_params, 1)
                overall_sigma = jnp.sqrt(2*sigma**2*alpha + sigma_t**2)
                return overall_sigma
            else:
                sigma, covar = self.get_SDE_sigma(SDE_params)
                alpha = self.beta_int(SDE_params, 1)
                factor = alpha[:, None] + alpha[None, :]
                overall_covar = factor*jnp.diag(sigma**2) + covar
                return overall_covar
        else:
            if(self.invariance):
                sigma, sigma_t = self.get_SDE_sigma(SDE_params)
                alpha = self.beta_int(SDE_params, 1)
                overall_sigma = jnp.sqrt(2*sigma**2*alpha )
                return overall_sigma
            else:
                sigma, covar = self.get_SDE_sigma(SDE_params)
                alpha = self.beta_int(SDE_params, 1)
                factor = 2*alpha
                overall_covar = jnp.sqrt(factor)*sigma 
                return overall_covar

    def beta_int(self, SDE_params, t):
        beta_min, beta_max = self.get_beta_min_and_max(SDE_params)
        return 0.5*(beta_min*((beta_max/beta_min)**t))**2 ### TODO chekc this factor 0.5

    def beta(self, SDE_params, t):
        beta_min, beta_max = self.get_beta_min_and_max(SDE_params)
        return (beta_min*(beta_max/beta_min)**t)**2*(jnp.log(beta_max) - jnp.log(beta_min)) ### TODO chekc this factor 0.5


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
        sigma, _ = self.get_SDE_sigma(SDE_params)
        diffusion = sigma*jnp.sqrt(2*self.beta(SDE_params, t))
        return diffusion[None, :] 
    
    def sample(self, shape, key):
        return random.normal(key, shape)
    
    


