from .Base_SDE_Loss import Base_SDE_Loss_Class
import jax
from jax import numpy as jnp
import numpy as np
from functools import partial
import optax

class LogVarianceLoss_MC_Class(Base_SDE_Loss_Class):

    def __init__(self, SDE_config, Optimizer_Config,  EnergyClass, Network_Config, model):
        self.minib_time_steps = SDE_config["minib_time_steps"]
        self.inner_loop_steps = int(SDE_config["n_integration_steps"]/self.minib_time_steps)
        self.lr_factor = self.inner_loop_steps
        super().__init__(SDE_config, Optimizer_Config, EnergyClass, Network_Config, model, lr_factor = self.lr_factor)
        self.SDE_type.stop_gradient = True
        print("Gradient over expectation is supposed to be stopped from now on")
        self._init_index_arrays()
        self.vmap_apply_model = jax.vmap(lambda xs, ts, params, Energy_params, SDE_params, hidden_state, key: self.SDE_type.apply_model(self.model, xs, ts, params, Energy_params, SDE_params, hidden_state, key), in_axes=(0,0,None, None, None, 0, 0))
        ### TODO find out why this is so slow!

    def _init_index_arrays(self):
        diff_step_arr = jnp.arange(0,self.n_integration_steps)
        self.T_indices = jnp.repeat(diff_step_arr[..., None], self.batch_size, axis=1)

        diff_step_arr = jnp.arange(0,self.batch_size)
        self.B_indices = jnp.repeat(diff_step_arr[None, ...], self.n_integration_steps, axis=0)


    @partial(jax.jit, static_argnums=(0,))
    def _shuffle_index_array(self, key):
        key, subkey = jax.random.split(key)
        perm_diff_array = jax.random.permutation(subkey, self.T_indices, axis=0, independent=True)

        key, subkey = jax.random.split(key)
        perm_state_array = jax.random.permutation(subkey, self.B_indices, axis=1, independent=True)

        return perm_diff_array, perm_state_array, key

    @partial(jax.jit, static_argnums=(0,2,3))
    def _split_arrays(self, arr, n_splits, axis ):
        arr_list = jnp.split(arr, n_splits, axis=axis)
        return arr_list

    @partial(jax.jit, static_argnums=(0,))
    def _preprocess_SDE_tracer(self, SDE_tracer):
        for key in SDE_tracer.keys():
            array = SDE_tracer[key]
            if array.ndim == 1:
                SDE_tracer[key] = jnp.repeat(SDE_tracer[key][:,None, None], self.batch_size, axis = 1)
        return SDE_tracer
    
    @partial(jax.jit, static_argnums=(0,))
    def _select_minibatch(self, SDE_tracer, perm_diff_array, target_keys = ["scores", "dW", "xs", "ts"]):
        ### TODO splitting of keys and hidden state wont be easy, maybe ignore key and hidden state? raise exception if they are in target_keys
        minib_SDE_tracer = {}
        batch_indices = jnp.arange(0, self.batch_size)[None, :]
        for key in SDE_tracer.keys():
            if(key in target_keys):
                array = SDE_tracer[key]
                minib_SDE_tracer[key] = array[perm_diff_array, batch_indices]

        return minib_SDE_tracer

    @partial(jax.jit, static_argnums=(0,))
    def get_data_buffer(self, params, Energy_params, SDE_params, key, n_integration_steps = 100, n_states = 10, temp = 1.0, x_dim = 2):
        SDE_tracer, key = self.SDE_type.simulate_reverse_sde_scan(self.model , params, Energy_params, SDE_params, key, n_integration_steps = n_integration_steps, n_states = n_states, x_dim = x_dim)
        return SDE_tracer, key
    
    @partial(jax.jit, static_argnums=(0,))
    def update_params_on_minibatch(self, params, Energy_params, SDE_params, opt_state, Energy_params_state, SDE_params_state, minib_SDE_tracer, T_curr, key):
        (loss_value, out_dict), (grads, SDE_params_grad) = jax.value_and_grad(self.evaluate_loss, argnums=(0, 2), has_aux = True)(params, Energy_params, SDE_params, minib_SDE_tracer, T_curr, key)
        updates, opt_state = self.optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        
        SDE_params_updates, SDE_params_state = self.SDE_params_optimizer.update(SDE_params_grad, SDE_params_state, SDE_params)
        SDE_params = optax.apply_updates(SDE_params, SDE_params_updates)

        if not self.Optimizer_Config["learn_beta_min_max"]:
            SDE_params["log_beta_min"] = jnp.log(self.SDE_type.config["beta_min"])*jnp.ones_like(SDE_params["log_beta_min"])
            SDE_params["log_beta_delta"] = jnp.log(self.SDE_type.config["beta_max"])*jnp.ones_like(SDE_params["log_beta_delta"])
        else:
            SDE_params["log_beta_min"] = jnp.log(self.SDE_type.config["beta_min"])*jnp.ones_like(SDE_params["log_beta_min"])		
        
        return params, Energy_params, SDE_params, opt_state, Energy_params_state, SDE_params_state, loss_value, out_dict


    
    #@partial(jax.jit, static_argnums=(0,))
    def update_params(self, params, Energy_params, SDE_params, opt_state, Energy_params_state, SDE_params_state, jax_key, T_curr, n_integration_steps = 100, n_states = 10, temp = 1.0, x_dim = 2):
        SDE_tracer, jax_key = self.get_data_buffer(self.model , params, Energy_params, SDE_params, jax_key, n_integration_steps = n_integration_steps, n_states = n_states, x_dim = x_dim)

        x_last = SDE_tracer["x_final"]
        Energy, jax_key = self.EnergyClass.vmap_calc_energy(x_last, Energy_params, jax_key)
        SDE_tracer["Energy"] = Energy

        perm_diff_array, _, jax_key = self._shuffle_index_array(jax_key)
        inner_loop_steps = self.inner_loop_steps
        perm_diff_array_list = self._split_arrays(perm_diff_array, inner_loop_steps, axis = 0)
        #perm_diff_array_list = [self.T_indices]

        SDE_tracer = self._preprocess_SDE_tracer(SDE_tracer)
        
        overall_out_dict = {}
        for perm_diff_array in perm_diff_array_list:
            ### TODO adjust lr schedule accordingly
            minib_SDE_tracer = self._select_minibatch(SDE_tracer, perm_diff_array)

            params, Energy_params, SDE_params, opt_state, Energy_params_state, SDE_params_state, loss_value, out_dict = self.update_params_on_minibatch(params, Energy_params, SDE_params, opt_state, 
                                                                                                                                    Energy_params_state, SDE_params_state, minib_SDE_tracer, T_curr, jax_key)

            for dict_key in out_dict.keys():
                if(dict_key != "key" and dict_key != "X_0"):
                    if (dict_key not in overall_out_dict):
                        overall_out_dict[dict_key] = []
                    overall_out_dict[dict_key].append(np.array(out_dict[dict_key]))
                elif(dict_key == "key"):
                    jax_key = out_dict["dict_key"]

        overall_out_dict["key"] = jax_key
        for key in overall_out_dict.keys():
            if (key != "key" and key != "X_0"):
                overall_out_dict[key] = np.mean(overall_out_dict[key] )

        overall_out_dict["X_0"] = x_last

        return params, Energy_params, SDE_params, opt_state, Energy_params_state, SDE_params_state, loss_value, overall_out_dict

    @partial(jax.jit, static_argnums=(0,))
    def evaluate_loss(self, params, Energy_params, SDE_params, minib_SDE_tracer, temp, key):
        ### TODO also add KL regularization term
        T = self.n_integration_steps
        control = minib_SDE_tracer["scores"]
        dW = minib_SDE_tracer["dW"]
        xs = minib_SDE_tracer["xs"]
        ts = minib_SDE_tracer["ts"]
        keys = minib_SDE_tracer["keys"]
        x_prior = minib_SDE_tracer["x_prior"]
        x_last = minib_SDE_tracer["x_last"]
        Energy = minib_SDE_tracer["Energy"]
        hidden_state = minib_SDE_tracer["hidden_state"]

        log_prior = self.vmap_get_log_prior(SDE_params, x_prior)

        score, _, _ = self.vmap_apply_model(self.model, xs, ts, params, Energy_params, SDE_params, hidden_state, keys)

        diff_factor = self.vmap_diff_factor(SDE_params, None, ts)
        drift_divergence = self.vmap_drift_divergence( SDE_params, ts)[:,None, :]
        U = diff_factor*score
        control_W = diff_factor*control

        f = jnp.sum(U * jax.lax.stop_gradient(control_W) - U**2/2 + drift_divergence, axis = -1)
        S = jnp.sum(jnp.sqrt(T)*jnp.sum(U * dW, axis = -1), axis = 0) 

        R_diff = jnp.mean(f, axis = 0)
        obs = temp*R_diff + temp*S+ temp*log_prior + Energy
        log_var_loss = jnp.mean((obs)**2) - jnp.mean(obs)**2

        ### Logging
        res_dict = self.compute_partition_sum(R_diff, S, log_prior, Energy)
        log_Z = res_dict["log_Z"]
        Free_Energy, n_eff, NLL = res_dict["Free_Energy"], res_dict["n_eff"], res_dict["NLL"]

        mean_Energy = jnp.mean(Energy)
        mean_log_prior = jnp.mean(log_prior)
        mean_R_diff = jnp.mean(R_diff)
        Entropy = -(mean_R_diff + mean_log_prior)


        return log_var_loss, {"mean_energy": mean_Energy, "Free_Energy_at_T=1": Free_Energy, "Entropy": Entropy, "R_diff": R_diff, 
                      "key": key, "X_0": x_last, "mean_X_prior": jnp.mean(x_prior), "std_X_prior": jnp.mean(jnp.std(x_prior, axis = 0)), 
                       "sigma": jnp.exp(SDE_params["log_sigma"]),
                      "beta_min": jnp.exp(SDE_params["log_beta_min"]), "beta_delta": jnp.exp(SDE_params["log_beta_delta"]), "mean": SDE_params["mean"],
                        "log_Z_at_T=1": log_Z, "n_eff": n_eff, "NLL": NLL}
    