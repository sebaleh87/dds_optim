from .Base_SDE_Loss import Base_SDE_Loss_Class
import jax
from jax import numpy as jnp
import numpy as np
from functools import partial
import optax
import time

class LogVarianceLoss_MC_Class(Base_SDE_Loss_Class):

    def __init__(self, SDE_config, Optimizer_Config,  EnergyClass, Network_Config, model):
        self.minib_time_steps = SDE_config["minib_time_steps"]
        if(self.minib_time_steps > SDE_config["n_integration_steps"]):
            raise ValueError("minib_time_steps should be smaller than n_integration_steps")

        self.inner_loop_steps = max([int(SDE_config["n_integration_steps"]/self.minib_time_steps),1])
        self.lr_factor = self.inner_loop_steps
        super().__init__(SDE_config, Optimizer_Config, EnergyClass, Network_Config, model, lr_factor = self.lr_factor)
        self.SDE_type.stop_gradient = True
        Warning("Stop gradient with respect to SDE parameters!!")
        self._init_index_arrays()
        self.vmap_apply_model = jax.vmap(lambda xs, ts, params, Energy_params, SDE_params, hidden_state, temp, key: self.SDE_type.apply_model(self.model, xs, ts, params, Energy_params, SDE_params, hidden_state, temp, key), in_axes=(0,0,None, None, None, 0, None, 0))
        ### TODO find out why this is so slow!
        self.update_params = self.update_params_MC

        self.vmap_vmap_diff_factor = jax.vmap(jax.vmap(self.SDE_type.get_diffusion, in_axes=(None, None, 0)), in_axes=(None, None, 0))
        self.vmap_vmap_drift_divergence = jax.vmap(jax.vmap(self.SDE_type.get_div_drift, in_axes = (None, 0)), in_axes=(None, 0))

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
    def _preprocess_SDE_tracer(self, SDE_tracer, exception_keys = ["hidden_states"]):

        ### can this be deleted?
        for key in SDE_tracer.keys():
            if(key not in exception_keys):
                array = SDE_tracer[key]
                if array.ndim == 1:
                    SDE_tracer[key] = jnp.repeat(SDE_tracer[key][:,None, None], self.batch_size, axis = 1)
        return SDE_tracer
    
    @partial(jax.jit, static_argnums=(0,))
    def _select_minibatch(self, SDE_tracer, perm_diff_array, target_keys = ["scores", "dW", "xs", "ts", "Energy", "x_prior", "x_final"]):
        ### TODO splitting of keys and hidden state wont be easy, maybe ignore key and hidden state? raise exception if they are in target_keys
        minib_SDE_tracer = {}
        batch_indices = jnp.arange(0, self.batch_size)[None, :]
        for key in SDE_tracer.keys():
            if(key in target_keys):
                array = SDE_tracer[key]
                #print("array shapes ",key, array.ndim)
                #print(perm_diff_array)
                if(key == "ts"):
                    minib_SDE_tracer[key] = array[perm_diff_array]
                elif array.ndim == 2 or array.ndim == 1:
                    minib_SDE_tracer[key] = array
                if array.ndim == 3:
                    minib_SDE_tracer[key] = array[perm_diff_array, batch_indices]

        #print("mini sDE shapes", [(key, minib_SDE_tracer[key].shape) for key in minib_SDE_tracer.keys()])
        return minib_SDE_tracer

    @partial(jax.jit, static_argnums=(0,), static_argnames=("n_integration_steps", "n_states", "x_dim"))
    def get_data_buffer(self, params, Energy_params, SDE_params, temp, key, n_integration_steps = 100, n_states = 10, x_dim = 2):
        SDE_tracer, key = self.SDE_type.simulate_reverse_sde_scan(self.model , params, Energy_params, SDE_params, temp, key, n_integration_steps = n_integration_steps, n_states = n_states, x_dim = x_dim)
        return SDE_tracer, key
    
    @partial(jax.jit, static_argnums=(0,))
    def update_params_on_minibatch(self, params, Interpol_params, SDE_params, opt_state, Interpol_params_state, SDE_params_state, minib_SDE_tracer, key, T_curr):
        (loss_value, out_dict), (grads, Interpol_params_grad, SDE_params_grad) = jax.value_and_grad(self.loss_fn, argnums=(0, 1, 2), has_aux = True)(params, Interpol_params, SDE_params, minib_SDE_tracer, T_curr, key)
        ### TODO add interpol params
        updates, opt_state = self.optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        
        state_dict = {"network": opt_state, "Interpol": Interpol_params_state, "SDE": SDE_params_state}
        grads_dict = {"network": grads, "Interpol": Interpol_params_grad, "SDE": SDE_params_grad}
        params_dict = {"network": params, "Interpol": Interpol_params, "SDE": SDE_params}

        state_dict, params_dict = self.apply_updates(state_dict, grads_dict, params_dict)
        SDE_params, gradients_dict = self.reset_updates(grads, Interpol_params_grad, SDE_params_grad, SDE_params)

        return params, Interpol_params, SDE_params, opt_state, Interpol_params_state, SDE_params_state, loss_value, out_dict, gradients_dict

    #@partial(jax.jit, static_argnums=(0,))
    def update_params_MC(self, params, Energy_params, SDE_params, opt_state, Energy_params_state, SDE_params_state, jax_key, T_curr):
        SDE_tracer, jax_key = self.get_data_buffer( params, Energy_params, SDE_params, T_curr, jax_key, n_integration_steps = self.n_integration_steps, n_states = self.batch_size, x_dim = self.EnergyClass.dim_x)

        ### TODO compute Q function here
        ### TODO include Value function somehow

        x_last = SDE_tracer["x_final"]
        Energy, jax_key = self.EnergyClass.vmap_calc_energy(x_last, Energy_params, jax_key)
        SDE_tracer["Energy"] = Energy

        perm_diff_array, _, jax_key = self._shuffle_index_array(jax_key)
        inner_loop_steps = self.inner_loop_steps
        perm_diff_array_list = self._split_arrays(perm_diff_array, inner_loop_steps, axis = 0)
        #perm_diff_array_list = [self.T_indices]
        ### TODO log here exact entropy, R etc
        SDE_tracer["Energy"]

        R_exact, Entropy_exact, S_exact = self.compute_R_and_entropy(SDE_tracer, SDE_params)
        
        overall_out_dict = {}
        for perm_diff_array in perm_diff_array_list:
            ### TODO adjust lr schedule accordingly
            t1 = time.time()
            minib_SDE_tracer = self._select_minibatch(SDE_tracer, perm_diff_array)

            t2 = time.time()
            params, Interpol_params, SDE_params, opt_state, Interpol_params_state, SDE_params_state, loss_value, out_dict, _ = self.update_params_on_minibatch(params, Interpol_params, 
                                                                                                                                        SDE_params, opt_state, Interpol_params_state, SDE_params_state, minib_SDE_tracer, key, T_curr)
            t3 = time.time()

            for dict_key in out_dict.keys():
                if(dict_key != "key" and dict_key != "X_0"):
                    if (dict_key not in overall_out_dict):
                        overall_out_dict[dict_key] = []
                    overall_out_dict[dict_key].append(np.array(out_dict[dict_key]))
                elif(dict_key == "key"):
                    jax_key = out_dict[dict_key]

        overall_out_dict["key"] = jax_key
        for key in overall_out_dict.keys():
            if (key != "key" and key != "X_0"):
                overall_out_dict[key] = np.mean(overall_out_dict[key] )

        overall_out_dict["X_0"] = x_last

        return params, Energy_params, SDE_params, opt_state, Energy_params_state, SDE_params_state, loss_value, overall_out_dict

    @partial(jax.jit, static_argnums=(0,))
    def evaluate_loss_for_train(self, params, Energy_params, SDE_params, SDE_tracer, key, temp = 1.0):
        pass


        return None, {}

    

    @partial(jax.jit, static_argnums=(0,))  
    def evaluate_loss(self, params, Energy_params, SDE_params, SDE_tracer, key, temp = 1.0):
        pass

        return None, {}
    