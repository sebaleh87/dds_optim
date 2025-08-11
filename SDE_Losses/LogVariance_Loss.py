from .Base_SDE_Loss import Base_SDE_Loss_Class
import jax
from jax import numpy as jnp
from functools import partial

class LogVariance_Loss_Class(Base_SDE_Loss_Class):

    def __init__(self, SDE_config, Optimizer_Config,  EnergyClass, Network_Config, model):
        super().__init__(SDE_config, Optimizer_Config, EnergyClass, Network_Config, model)
        self.SDE_type.stop_gradient = True
        print("Gradient over expectation is supposed to be stopped from now on")

    @partial(jax.jit, static_argnums=(0,))  
    def evaluate_loss(self, params, Energy_params, SDE_params, SDE_tracer, key, temp = 1.0):
        score = SDE_tracer["scores"]
        
        prior_mean = SDE_tracer["prior_mean"]
        prior_sigma = SDE_tracer["prior_sigma"]
        #TODO the following stop_gradients are presumably no longer relevant, since this is now done in Base_SDE 
        dW = jax.lax.stop_gradient(SDE_tracer["dW"])
        ts = SDE_tracer["ts"]
        dts = SDE_tracer["dts"][...,None]

        # tbs = jnp.repeat(ts[:,None, None], self.batch_size, axis = 1)
        # score2 = self.vmap_model(params, xs, tbs)
        # print("diff", score - score2)
        x_prior = jax.lax.stop_gradient(SDE_tracer["x_prior"])
        x_last = jax.lax.stop_gradient(SDE_tracer["x_final"])
        
        log_prior = self.vmap_get_log_prior(SDE_params, x_prior)
        mean_log_prior = jnp.mean(log_prior)

        Energy, key = self.EnergyClass.vmap_calc_energy(x_last, Energy_params, key)
        mean_Energy = jnp.mean(Energy)
        diff_factor = self.vmap_diff_factor(SDE_params, None, ts)
        drift_divergence = self.vmap_drift_divergence( SDE_params, ts)[:,None, :]
        #print("shapes", score.shape, diff_factor.shape, drift_divergence.shape)
        U = diff_factor*score

        #follows from Improved sampling via learned diffusions (6) def of R with w = stop_grad(u) and v=0
        f = (jnp.sum( U * jax.lax.stop_gradient(U) - U**2/2, axis = -1) + jnp.sum(drift_divergence, axis = -1))

        S = jnp.sum(jnp.sum(U * dW, axis = -1), axis = 0)
        R_diff = jnp.sum(dts*f  , axis = 0)
        mean_R_diff = jnp.mean(R_diff)
        Entropy = -(mean_R_diff + mean_log_prior)

        #obs = temp*R_diff + temp*S+ temp*log_prior+ Energy
        if(not self.Network_Config["model_mode"] == "latent"):
            if(self.optim_mode == "optim"):
                obs = temp*(R_diff + S+ log_prior) + Energy
            elif(self.optim_mode == "equilibrium"):
                obs = (R_diff + S+ log_prior) + Energy/temp
            else:
                raise ValueError(f"Unknown optim_mode: {self.optim}")
        else:
            log_prob_decoder = SDE_tracer["log_p_decode"]
            log_prob_encoder = SDE_tracer["log_p_encode"]

            rKL_VAE = (log_prob_decoder - log_prob_encoder)

            if(self.optim_mode == "optim"):
                obs = temp*(R_diff + S+ log_prior + rKL_VAE) + Energy
            elif(self.optim_mode == "equilibrium"):
                obs = (R_diff + S+ log_prior + rKL_VAE) + Energy/temp
            else:
                raise ValueError(f"Unknown optim_mode: {self.optim}")
            
            R_diff = R_diff + rKL_VAE

        #log_var_loss = jnp.mean((obs)**2) - jnp.mean(obs)**2#jnp.var(obs)#jnp.mean((obs)**2) - jnp.mean(obs)**2
        log_var_loss = jnp.mean((obs - jnp.mean(obs))**2)

        log_dict = {"mean_energy": mean_Energy, "Entropy": Entropy, "R_diff": R_diff, 
                      "key": key, "X_0": x_last, "mean_X_prior": jnp.mean(x_prior), "std_X_prior": jnp.mean(jnp.std(x_prior, axis = 0)), 
                       "sigma": prior_sigma,
                      "beta_min": jnp.exp(SDE_params["log_beta_min"]), "beta_delta": jnp.exp(SDE_params["log_beta_delta"]), "mean": prior_mean
                        }
        log_dict = self.compute_partition_sum(R_diff, S, log_prior, Energy, log_dict)

        # print("X-last", x_last.mean())
        # print("score", jax.lax.stop_gradient(score).mean())
        # print("Energy", jax.lax.stop_gradient(Energy).mean())
        # print("log_var_loss", jax.lax.stop_gradient(log_var_loss).mean())

        return log_var_loss, log_dict
