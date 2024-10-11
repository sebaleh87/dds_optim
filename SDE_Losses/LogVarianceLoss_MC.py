from .Base_SDE_Loss import Base_SDE_Loss_Class
import jax
from jax import numpy as jnp
import numpy as np
from functools import partial
import optax

class LogVarianceLoss_MC_Class(Base_SDE_Loss_Class):

    def __init__(self, SDE_config, Optimizer_Config,  EnergyClass, model):
        self.minib_time_steps = SDE_config["minib_time_steps"]
        self.inner_loop_steps = int(SDE_config["n_integration_steps"]/self.minib_time_steps)
        self.lr_factor = self.inner_loop_steps
        super().__init__(SDE_config, Optimizer_Config, EnergyClass, model, lr_factor = self.lr_factor)
        self.SDE_type.stop_gradient = True
        print("Gradient over expectation is supposed to be stopped from now on")
        self._init_index_arrays()

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

    def _preprocess_SDE_tracer(self, SDE_tracer):
        for key in SDE_tracer.keys():
            array = SDE_tracer[key]
            if array.ndim == 1:
                SDE_tracer[key] = jnp.repeat(SDE_tracer[key][:,None, None], self.batch_size, axis = 1)
        return SDE_tracer
    
    @partial(jax.jit, static_argnums=(0,))
    def _select_minibatch(self, SDE_tracer, perm_diff_array):
        minib_SDE_tracer = {}
        batch_indices = jnp.arange(0, self.batch_size)[None, :]
        for key in SDE_tracer.keys():
            array = SDE_tracer[key]
            minib_SDE_tracer[key] = array[perm_diff_array, batch_indices]

        return minib_SDE_tracer

    @partial(jax.jit, static_argnums=(0,))
    def update_params(self, params, opt_state, minib_SDE_tracer, T_curr, log_prior, Energy):
        (loss_value, out_dict), (grads,) = jax.value_and_grad(self.compute_loss, argnums=(0,), has_aux = True)(params, minib_SDE_tracer, log_prior, Energy, T_curr)
        updates, opt_state = self.optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss_value, out_dict

    #@partial(jax.jit, static_argnums=(0,))
    def update_step(self, params, opt_state, jax_key, T_curr):
        SDE_tracer, jax_key = self.SDE_type.simulate_reverse_sde_scan(self.model , params, jax_key, n_integration_steps = self.n_integration_steps, n_states = self.batch_size, x_dim = self.x_dim)
        
        xs = SDE_tracer["xs"]
        x_last = SDE_tracer["x_final"]
        log_prior = self.SDE_type.get_log_prior( xs[0])[...,0]
        Energy = self.vmap_calc_Energy(x_last)
        Energy = Energy[...,0]

        perm_diff_array, _, jax_key = self._shuffle_index_array(jax_key)
        inner_loop_steps = self.inner_loop_steps
        perm_diff_array_list = self._split_arrays(perm_diff_array, inner_loop_steps, axis = 0)
        #perm_diff_array_list = [self.T_indices]

        SDE_tracer = self._preprocess_SDE_tracer(SDE_tracer)
        
        overall_out_dict = {}
        for perm_diff_array in perm_diff_array_list:
            ### TODO adjust lr schedule accordingly
            minib_SDE_tracer = SDE_tracer#self._select_minibatch(SDE_tracer, perm_diff_array)

            params, opt_state, loss_value, out_dict = self.update_params(params, opt_state, minib_SDE_tracer, T_curr, log_prior, Energy)

            for key in out_dict.keys():
                if(key != "key" and key != "X_0"):
                    if (key not in overall_out_dict):
                        overall_out_dict[key] = []
                    overall_out_dict[key].append(out_dict[key])
                else:
                    pass

        overall_out_dict["key"] = jax_key
        for key in overall_out_dict.keys():
            if (key != "key" and key != "X_0"):
                overall_out_dict[key] = np.mean(overall_out_dict[key] )

        overall_out_dict["X_0"] = x_last

        return params, opt_state, loss_value, overall_out_dict

    @partial(jax.jit, static_argnums=(0,))
    def compute_loss(self, params, minib_SDE_tracer, log_prior, Energy, temp):
        ### TODO also add KL regularization term
        T = self.n_integration_steps
        control = minib_SDE_tracer["scores"]
        dW = minib_SDE_tracer["dW"]
        xs = minib_SDE_tracer["xs"]
        ts = minib_SDE_tracer["ts"]

        score = self.model.apply(params, xs, ts)
        #print("diff", score - control)
        #dt = 1./self.n_integration_steps

        ### TODO make importance sampling instead of simple mean
        diffusion_prefactor = self.SDE_type.get_diffusion(None, ts)
        U = diffusion_prefactor*score
        control_W = diffusion_prefactor*control

        div_drift = - self.x_dim*self.SDE_type.beta(ts)
        f = U * jax.lax.stop_gradient(control_W) - U**2/2 - div_drift
        S = jnp.mean(T*jnp.sum(U * dW, axis = -1), axis = 0)

        R_diff = jnp.mean(jnp.sum( f, axis = -1), axis = 0)
        obs = temp*R_diff + temp*S+ temp*log_prior + Energy
        log_var_loss = jnp.mean((obs)**2) - jnp.mean(obs)**2
        return log_var_loss, {"mean_energy": jnp.mean(Energy), "f_int": jnp.mean(R_diff), "S_int": jnp.mean(S), "likelihood_ratio": jnp.mean(obs)}