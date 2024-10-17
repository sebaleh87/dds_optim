from .Base_SDE_Loss import Base_SDE_Loss_Class
import jax
from jax import numpy as jnp
from functools import partial

class LogVariance_Loss_Class(Base_SDE_Loss_Class):

    def __init__(self, SDE_config, Optimizer_Config,  EnergyClass, Network_Config, model):
        super().__init__(SDE_config, Optimizer_Config, EnergyClass, Network_Config, model)
        self.SDE_type.stop_gradient = True
        print("Gradient over expectation is supposed to be stopped from now on")

    @partial(jax.jit, static_argnums=(0,), static_argnames=("n_integration_steps", "n_states", "x_dim"))  
    def compute_loss(self, params, key, n_integration_steps = 100, n_states = 10, temp = 1.0, x_dim = 2):
        SDE_tracer, key = self.SDE_type.simulate_reverse_sde_scan(self.model , params, key, n_integration_steps = n_integration_steps, n_states = n_states, x_dim = x_dim)
        score = SDE_tracer["scores"]
        dW = SDE_tracer["dW"]
        xs = SDE_tracer["xs"]
        ts = SDE_tracer["ts"]
        dt = 1./n_integration_steps

        # tbs = jnp.repeat(ts[:,None, None], self.batch_size, axis = 1)
        # score2 = self.vmap_model(params, xs, tbs)
        # print("diff", score - score2)

        x_last = SDE_tracer["x_final"]
        x_dim = x_last.shape[-1]

        U = self.SDE_type.get_diffusion(None, ts)[:,None, None]*score

        div_drift = - self.x_dim*self.SDE_type.beta(ts)[:,None, None]
        f = U * jax.lax.stop_gradient(U) - U**2/2 - div_drift
        S = jnp.sum(jnp.sum(U * dW, axis = -1), axis = 0)


        log_prior = self.SDE_type.get_log_prior( xs[0])[...,0]
        Energy = self.vmap_calc_Energy(x_last)
        Energy = Energy[...,0]
        R_diff = jnp.sum(dt*jnp.sum( f, axis = -1), axis = 0)
        obs = temp*R_diff + temp*S+ temp*log_prior + Energy
        log_var_loss = jnp.mean((obs)**2) - jnp.mean(obs)**2
        return log_var_loss, {"mean_energy": jnp.mean(Energy), "f_int": jnp.mean(R_diff), "S_int": jnp.mean(S), "likelihood_ratio": jnp.mean(obs), "key": key, "X_0": x_last}