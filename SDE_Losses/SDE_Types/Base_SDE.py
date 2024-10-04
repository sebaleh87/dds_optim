import jax
from jax import random
import matplotlib.pyplot as plt
import jax.numpy as jnp

class Base_SDE_Class:

    def __init__(self, config) -> None:
        self.config = config
        pass

    def compute_p_xt_g_x0_statistics(self, x0, xt, t):
        raise NotImplementedError("get_diffusion method not implemented")

    def get_diffusion(self, t, x):
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

    def reverse_sde(self, score, x, t, dt, key, mode = "DIS"):
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
    
    def simulate_reverse_sde_scan(self, model, params, key, n_states = 100, x_dim = 2, n_integration_steps = 1000):
        def scan_fn(carry, step):
            x, t, key = carry
            t_arr = jnp.ones((x.shape[0], 1)) * t
            score = model.apply(params, x, t_arr)
            reverse_out_dict, key = self.reverse_sde(score, x, t, dt, key)

            SDE_tracker_step = {
            "xs": reverse_out_dict["x_next"],
            "ts": reverse_out_dict["t_next"],
            "scores": score,
            "forward_drift": reverse_out_dict["forward_drift"],
            "reverse_drift": reverse_out_dict["reverse_drift"],
            "drift_ref": reverse_out_dict["drift_ref"],
            "beta_t": reverse_out_dict["beta_t"]
            }

            x = reverse_out_dict["x_next"]
            t = reverse_out_dict["t_next"]
            return (x, t, key), SDE_tracker_step

        key, subkey = random.split(key)
        x0 = random.normal(subkey, shape=(n_states, x_dim))
        t = 1.0
        dt = 1. / n_integration_steps

        (x_final, t_final, key), SDE_tracker_steps = jax.lax.scan(
            scan_fn,
            (x0, t, key),
            jnp.arange(n_integration_steps)
        )

        SDE_tracker = {
            "xs": jnp.array(SDE_tracker_steps["xs"]),
            "ts": jnp.array(SDE_tracker_steps["ts"]),
            "scores": jnp.array(SDE_tracker_steps["scores"]),
            "forward_drift": jnp.array(SDE_tracker_steps["forward_drift"]),
            "reverse_drift": jnp.array(SDE_tracker_steps["reverse_drift"]),
            "drift_ref": jnp.array(SDE_tracker_steps["drift_ref"]),
            "beta_t": jnp.array(SDE_tracker_steps["beta_t"])
        }

        return SDE_tracker, key

    def simulate_reverse_sde(self, model, params, key, n_states = 100, x_dim = 2, n_integration_steps = 1000):
        key, subkey = random.split(key)
        x0 = random.normal(subkey, shape=(n_states,x_dim))
        x = x0
        t = 1.0
        dt = 1./n_integration_steps

        SDE_tracker = {"xs": [], "ts": [], "scores": [], "forward_drift": [], "reverse_drift": [], "drift_ref":[], "beta_t": []}
        for step in range(n_integration_steps):
            t_arr = jnp.ones((x.shape[0],1))*t
            score = model.apply(params, x, t_arr)
            reverse_out_dict, key = self.reverse_sde(score,x, t, dt, key)

            SDE_tracker["xs"].append(reverse_out_dict["x_next"])
            SDE_tracker["ts"].append(reverse_out_dict["t_next"]) 
            SDE_tracker["scores"].append(score)
            SDE_tracker["drift_ref"].append(reverse_out_dict["drift_ref"])
            SDE_tracker["beta_t"].append(reverse_out_dict["beta_t"])
            SDE_tracker["forward_drift"].append(reverse_out_dict["forward_drift"])
            x = reverse_out_dict["x_next"]
            t = reverse_out_dict["t_next"]

        return SDE_tracker, key