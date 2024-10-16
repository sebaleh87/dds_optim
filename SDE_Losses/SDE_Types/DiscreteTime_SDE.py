import jax
from jax import random
import matplotlib.pyplot as plt
import jax.numpy as jnp
from jax.scipy.stats.norm import logpdf
from .Base_SDE import Base_SDE_Class
import numpy as np

class DiscreteTime_SDE_Class(Base_SDE_Class):

    def __init__(self, SDE_Type_Config, Energy_Class) -> None:
        self.stop_gradient = False
        self.n_diff_steps = SDE_Type_Config["n_diff_steps"]
        self.temp_mode = SDE_Type_Config["temp_mode"]
        self._make_beta_list()
        SDE_Type_Config["use_interpol_gradient"] = False
        self.config = SDE_Type_Config
        super().__init__(SDE_Type_Config, Energy_Class)
    
    def get_log_prior(self, x):
        return jax.scipy.stats.norm.logpdf(x, loc=0, scale=1)
    
    def _gaussian_noise_energy(self, X_prev, X_next, gamma_t_i):
        #gamma_t = jnp.clip(gamma_t_i , a_min = 0.01)
        gamma_t = gamma_t_i*jnp.ones_like(X_prev)
        #noise_loss = -jnp.mean(jnp.sum(- 0.5 * (X_prev - jnp.sqrt(1 - gamma_t) * X_next) ** 2 /gamma_t ** 2 - 0.5*jnp.log(2*jnp.pi*gamma_t**2), axis=- 1)) 
        ### TODO check why noise loss below is not working
        ### TODO 
        noise_arr = -jnp.sum(jax.scipy.stats.norm.logpdf(X_prev, loc=X_next*jnp.sqrt(1-gamma_t), scale=gamma_t*jnp.ones_like(X_next)), axis = -1)
        noise_loss = jnp.mean(noise_arr)
        return noise_loss, noise_arr
    
    def sample_from_model(self, output_dict):
        key = output_dict["key"]
        log_var = output_dict["log_var"]
        mean = output_dict["mean_x"]
        sigma = jnp.exp(0.5 * log_var)
        key, split_key = jax.random.split(key)
        if(self.temp_mode):
            eps = jax.random.normal(split_key, shape= (log_var.shape[0] , log_var.shape[-1]))*jnp.sqrt(output_dict["T_curr"])
        else:
            eps = jax.random.normal(split_key, shape= (log_var.shape[0] , log_var.shape[-1]))
            
        samples = mean + sigma * eps
        if(self.stop_gradient):
            samples = jax.lax.stop_gradient(samples)

        output_dict["samples"] = samples
        output_dict["key"] = key
        return output_dict

    def get_model_entropy(self, output_dict):
        log_var = output_dict["log_var"]
        logdet = jnp.sum(log_var, axis=-1)
        entropy_arr = 0.5 * logdet
        entropy = jnp.mean(entropy_arr)
        output_dict["entropy_arr"] = entropy_arr
        output_dict["entropy"] = entropy
        return output_dict

    def get_model_log_prob(self, output_dict):
        log_var = output_dict["log_var"]
        mean = output_dict["mean_x"] 
        samples = output_dict["samples"]

        sigma = jnp.exp(0.5 * log_var)
        log_prob = jnp.sum(logpdf(samples, loc=mean, scale=sigma), axis=-1)
        output_dict["log_prob"] = log_prob
        return output_dict

    
    def reverse_sde(self, output_dict):
        output_dict = self.sample_from_model(output_dict)
        output_dict = self.get_model_log_prob(output_dict)
        output_dict = self.get_model_entropy(output_dict)
        

        x_next = output_dict["samples"] 
        ### TODO change this to x + samples
        output_dict["x_next"] = output_dict["xs"] + x_next

        return output_dict

    def sample(self, shape, key):
        return random.normal(key, shape)
    
    def simulate_reverse_sde_scan(self, model, params, key, n_states = 100, x_dim = 2, n_integration_steps = 1000, T_curr = 1.0):
        def scan_fn(carry, step):
            x, t, dt, step, T_curr, key = carry
            t_arr = t*jnp.ones((x.shape[0], 1)) 
            in_dict = {"x": x, "t": t_arr, "grads": jnp.zeros_like(x)}
            mean_x, log_var_x = model.apply(params, in_dict)
            output_dict = {"mean_x": mean_x, "log_var": log_var_x, "xs": x, "ts": t, "T_curr": T_curr, "key": key}
            reverse_out_dict = self.reverse_sde(output_dict)
            reverse_out_dict["t_next"] = t - dt

            noise_loss_value, noise_loss_arr = self._gaussian_noise_energy(x, reverse_out_dict["x_next"], self.gamma_t_arr[step])
            reverse_out_dict["noise_loss_value"] = noise_loss_value
            reverse_out_dict["entropy_loss_value"] = -reverse_out_dict["entropy"]
            reverse_out_dict["noise_loss_arr"] = noise_loss_arr
            reverse_out_dict["entropy_loss_arr"] = -reverse_out_dict["entropy_arr"]
            key = reverse_out_dict["key"]

            SDE_tracker_step = {out_key: reverse_out_dict[out_key] for out_key in reverse_out_dict.keys() if out_key != "key"}

            x = reverse_out_dict["x_next"]
            t = reverse_out_dict["t_next"]
            step += 1
            return (x, t, dt, step, T_curr, key), SDE_tracker_step

        key, subkey = random.split(key)
        x0 = random.normal(subkey, shape=(n_states, x_dim))
        t = 1.#
        dt  = 1./n_integration_steps
        step = 0

        #print("no scan", model.apply(params, x0[0:10], t*jnp.ones((10, 1))))

        (x_final, t_final, _, _, _, key), SDE_tracker_steps = jax.lax.scan(
            scan_fn,
            (x0, t, dt, step, T_curr, key),
            jnp.arange(n_integration_steps)
        )

        SDE_tracker = {
            "entropies": SDE_tracker_steps["entropy"],
            "log_probs": SDE_tracker_steps["log_prob"],
            "xs": SDE_tracker_steps["xs"],
            "ts": SDE_tracker_steps["ts"],
            #"mean_x": SDE_tracker_steps["mean_x"],
            #"log_var_x": SDE_tracker_steps["log_var_x"],
            "noise_loss_value": SDE_tracker_steps["noise_loss_value"],
            "entropy_loss_value": SDE_tracker_steps["entropy_loss_value"],
            "noise_loss_arr": SDE_tracker_steps["noise_loss_arr"],
            "entropy_loss_arr": SDE_tracker_steps["entropy_loss_arr"]
        }
        SDE_tracker["x_final"] = x_final

        return SDE_tracker, key
    
    def _cos_func(self, t, n_steps, s=10 ** -2):
        f_t = jnp.cos(jnp.pi / 2 * ((t / n_steps) / (s + 1))) ** 2
        return f_t

    def _calc_gamma(self, alpha_hat_t, alpha_hat_t_prev, clip_value=0.99):
        return jnp.clip(1 - alpha_hat_t / alpha_hat_t_prev, a_max=clip_value, a_min=1-clip_value)

    def _make_beta_list(self):
        f_0 = self._cos_func(0, self.n_diff_steps)

        alpha_0 = 1.
        self.alpha_hat_t_list = [1.]
        self.gamma_t_list = [self._calc_gamma(alpha_0, alpha_0)]
        for i in range(self.n_diff_steps):
            j = i + 1
            alpha_hat_t = self._cos_func(j, self.n_diff_steps) / f_0
            gamma_t = self._calc_gamma(alpha_hat_t, self.alpha_hat_t_list[-1])

            self.alpha_hat_t_list.append(alpha_hat_t)
            self.gamma_t_list.append(gamma_t)

        self.gamma_t_arr = jnp.flip(jnp.array(self.gamma_t_list), axis = 0)
        self.gamma_t_arr = jnp.array(np.linspace(1, 0.05, self.n_diff_steps, endpoint = True))
        print("beta list is", self.gamma_t_arr)



    def _visualize_beta(self, n_steps=10):
        self._make_beta_list()

        from matplotlib import pyplot as plt
        steps = np.arange(0, self.n_diff_steps + 1)
        plt.figure()
        plt.plot(steps, self.alpha_hat_t_list)
        plt.ylabel("alpha")
        plt.show()

        plt.figure()
        plt.plot(steps, self.gamma_t_list)
        plt.ylabel("gamma")
        plt.show()

