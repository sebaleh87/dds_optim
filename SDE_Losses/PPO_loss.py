from .Base_SDE_Loss import Base_SDE_Loss_Class
import jax
from jax import numpy as jnp
import numpy as np
from functools import partial
import optax
import time
from tqdm import tqdm

class PPO_Loss_Class(Base_SDE_Loss_Class):

    def __init__(self, SDE_config, Optimizer_Config,  EnergyClass, Network_Config, model):
        self.minib_time_steps = SDE_config["minib_time_steps"]
        if(self.minib_time_steps > SDE_config["n_integration_steps"]):
            raise ValueError("minib_time_steps should be smaller than n_integration_steps")

        self.minib_states = SDE_config.get("minib_states", SDE_config["batch_size"])
        if(self.minib_states > SDE_config["batch_size"]):
            raise ValueError("minib_states should be smaller than batch_size")

        self.splits_over_time = max([int(SDE_config["n_integration_steps"]/self.minib_time_steps),1])
        self.splits_over_states = max([int(SDE_config["batch_size"]/self.minib_states),1])
        self.inner_loop_steps = self.splits_over_time*self.splits_over_states
        Optimizer_Config["SDE_lr"] = Optimizer_Config["SDE_lr"]/ self.inner_loop_steps
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
        self.gamma = 1.
        self.lam = 0.99
        self.td_lambda = False
        #self.trace_mode = "normal" 
    
    @partial(jax.jit, static_argnums=(0, 1, 2, 3, 4))
    def get_sample_indices(self, T_size: int, B_size: int, new_t: int,new_b: int, key: jax.random.PRNGKey):
        """
        Generates random indices for sampling a JAX array.

        Args:
            T_size: The size of the T dimension of the original matrix.
            B_size: The size of the B dimension of the original matrix.
            new_t: The number of indices to sample from dimension T.
            new_b: The number of indices to sample from dimension B.
            key: A JAX PRNG key for random number generation.

        Returns:
            A tuple containing:
            - t_indices: A JAX array of shape (new_t,) with unique indices for T.
            - b_indices: A JAX array of shape (new_t, new_b) with unique indices for each B.
        """
        # Split the key for different random operations
        key, subkey = jax.random.split(key)
        t_key, b_key = jax.random.split(subkey)

        # 1. Sample `new_t` unique indices from dimension T.
        print(t_key, T_size, new_t)
        t_indices = jax.random.choice(t_key, T_size, shape=(new_t,), replace=False)

        # 2. Define a function to sample `new_b` indices for a single T-slice.
        def sample_b_indices(current_b_key):
            return jax.random.choice(current_b_key, B_size, shape=(new_b,), replace=False)

        # 3. Use `jax.vmap` to vectorize the B-sampling operation, generating a unique
        # set of `new_b` indices for each of the `new_t` selected T-slices.
        b_indices = jax.vmap(sample_b_indices)(jax.random.split(b_key, new_t))
        
        return t_indices, b_indices, key

    @partial(jax.jit, static_argnums=(0,))
    def slice_matrix_with_indices(self, matrix, t_indices, b_indices):
        """
        Slices the input matrix using the provided indices.

        Args:
            matrix: The input JAX array with shape (T, B, ...).
            t_indices: A JAX array of shape (new_t,) with indices for T.
            b_indices: A JAX array of shape (new_t, new_b) with indices for B.

        Returns:
            A new JAX array with shape (new_t, new_b, ...).
        """
        # Use advanced indexing to slice the matrix.
        # We broadcast t_indices to match the shape of b_indices.
        return matrix[t_indices[:, None], b_indices]


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
    def _select_minibatch(self, SDE_tracer, t_indices, b_indices, key, target_keys = [ "x_prev", "x_next", "ts", "dts", "rewards", "advantages", "counters", "value_target", "reverse_log_probs"]):
        minib_SDE_tracer = {}
        batch_indices = jnp.arange(0, self.batch_size)[None, :]#perm_state_array[None, :]
        for dict_key in SDE_tracer.keys():
            if(dict_key in target_keys):
                array = SDE_tracer[dict_key]
                #print("array shapes ",key, array.ndim)
                #print(perm_diff_array)
                if array.ndim == 1:
                    raise ValueError("1d arrays not supported")
                    #jax.debug.print("1d array {key} shape {shape}", key=key, shape=minib_SDE_tracer[key].shape)
                elif array.ndim == 2 :
                    #array[perm_diff_array, batch_indices]
                    minib_array = self.slice_matrix_with_indices(array, t_indices, b_indices)
                    minib_SDE_tracer[dict_key] = jnp.reshape(minib_array, (self.minib_time_steps* self.minib_states))
                elif array.ndim == 3:
                    #array[perm_diff_array, batch_indices]
                    minib_array = self.slice_matrix_with_indices(array, t_indices, b_indices)
                    minib_SDE_tracer[dict_key] = jnp.reshape(minib_array, (self.minib_time_steps* self.minib_states, -1))

        #print("mini sDE shapes", [(key, minib_SDE_tracer[key].shape) for key in minib_SDE_tracer.keys()])
        minib_SDE_tracer["log_prior"] = SDE_tracer["log_prior"]
        minib_SDE_tracer["x_0"] = SDE_tracer["xs"][0]
        minib_SDE_tracer["Free_Energy_at_T=1_per_batch"] = SDE_tracer["Free_Energy_at_T=1_per_batch"]
        return minib_SDE_tracer, key

    @partial(jax.jit, static_argnums=(0,))
    def _calc_traces(self, values, rewards):
        max_steps = rewards.shape[0]
        values = jnp.concatenate([values, jnp.zeros_like(values[0][None], dtype=values.dtype)], axis=0)
        
        # For TD(lambda) returns
        returns = jnp.zeros_like(values)
        for t in range(max_steps):
            idx = max_steps - t - 1
            delta = rewards[idx] + self.gamma * values[idx + 1] - values[idx]
            returns = returns.at[idx].set(delta + self.gamma * self.lam * returns[idx + 1])
        
        # TD(lambda) returns are the targets for value function
        td_lambda_returns = returns[0:max_steps] + values[0:max_steps]
        
        # True advantages (if needed) would be:
        advantages = returns[0:max_steps]
        
        return td_lambda_returns, advantages

    @partial(jax.jit, static_argnums=(0,))
    def compute_advantages(self, SDE_params, Interpol_params, SDE_tracer, key):
        forward_diff_log_probs = SDE_tracer["forward_diff_log_probs"]
        reverse_log_probs = SDE_tracer["reverse_log_probs"]

        x_prior = SDE_tracer["x_prior"]
        x_last = SDE_tracer["x_final"]

        log_prior = self.vmap_get_log_prior(SDE_params, x_prior)
        Energy, key = self.EnergyClass.vmap_calc_energy(x_last, Interpol_params, key)
        SDE_tracer["log_prior"] = log_prior
        SDE_tracer["Energy"] = Energy

        rewards = reverse_log_probs - forward_diff_log_probs
        rewards = rewards.at[-1].set(rewards[-1] + Energy)
        SDE_tracer["rewards"] = rewards

        SDE_tracer["Free_Energy_at_T=1"] = jnp.mean( log_prior + jnp.sum(reverse_log_probs - forward_diff_log_probs, axis = 0) + Energy)
        SDE_tracer["Free_Energy_at_T=1_per_batch"] = log_prior + jnp.sum(reverse_log_probs - forward_diff_log_probs, axis = 0) + Energy


        value_function_value = SDE_tracer["value_function_values"]
        if(self.td_lambda):
            value_target, advantages = self._calc_traces(value_function_value, rewards[..., None])
            SDE_tracer["value_target"] = value_target
            advantages = (advantages - jnp.mean(advantages)) #/ (jnp.std(advantages) + 1e-10)
        else:
            value_target = jax.lax.cumsum(rewards, axis=0, reverse=True)[..., None]
            SDE_tracer["value_target"] = value_target
            advantages = value_target - jnp.mean(value_target, axis = 1, keepdims=True)#- value_function_value


        SDE_tracer["advantages"] = advantages

        return SDE_tracer, key

    @partial(jax.jit, static_argnums=(0,), static_argnames=("n_integration_steps", "n_states", "sample_mode"))
    def get_data_buffer(self, params, Interpol_params, SDE_params, temp, key, n_integration_steps = 100, n_states = 10, sample_mode = "train"):
        SDE_tracer, key = self.SDE_type.simulate_reverse_sde_scan(self.model , params, Interpol_params, SDE_params, temp, key, n_states = n_states, sample_mode = sample_mode, n_integration_steps = n_integration_steps)
        SDE_tracer, key = self.compute_advantages(SDE_params, Interpol_params, SDE_tracer, key)
        return SDE_tracer, key
    
    @partial(jax.jit, static_argnums=(0,))
    def update_params_on_minibatch(self, params, Interpol_params, SDE_params, opt_state, Interpol_params_state, SDE_params_state, minib_SDE_tracer, key, T_curr):
        (loss_value, out_dict), (grads, Interpol_params_grad, SDE_params_grad) = jax.value_and_grad(self.evaluate_loss_for_train, argnums=(0, 1, 2), has_aux = True)( params, Interpol_params, SDE_params, minib_SDE_tracer, key, T_curr)
        ### TODO add interpol params

        # updates, opt_state = self.optimizer.update(grads, opt_state)
        # params = optax.apply_updates(params, updates)
        
        state_dict = {"network": opt_state, "Interpol": Interpol_params_state, "SDE": SDE_params_state}
        grads_dict = {"network": grads, "Interpol": Interpol_params_grad, "SDE": SDE_params_grad}
        params_dict = {"network": params, "Interpol": Interpol_params, "SDE": SDE_params}

        state_dict, params_dict = self.apply_updates(state_dict, grads_dict, params_dict)

        params = params_dict["network"]
        Interpol_params = params_dict["Interpol"]
        SDE_params = params_dict["SDE"]

        opt_state = state_dict["network"]
        Interpol_params_state = state_dict["Interpol"]
        SDE_params_state = state_dict["SDE"]

        SDE_params, gradients_dict = self.reset_updates(grads, Interpol_params_grad, SDE_params_grad, SDE_params, update_none = False)


        return params, Interpol_params, SDE_params, opt_state, Interpol_params_state, SDE_params_state, loss_value, out_dict, gradients_dict

    #@partial(jax.jit, static_argnums=(0,))
    def update_params_MC(self, params, Interpol_params, SDE_params, opt_state, Interpol_params_state, SDE_params_state, jax_key, T_curr):
        SDE_tracer, jax_key = self.get_data_buffer( params, Interpol_params, SDE_params, T_curr, jax_key, n_integration_steps = self.n_integration_steps, n_states = self.batch_size)


        x_last = SDE_tracer["x_final"]
        Energy, jax_key = self.EnergyClass.vmap_calc_energy(x_last, Interpol_params, jax_key)
        SDE_tracer["Energy"] = Energy

        
        overall_out_dict = {}
        for i in range(self.inner_loop_steps):
            #for perm_state_array in perm_state_array_list:
                ### TODO adjust lr schedule accordingly
            t1 = time.time()
            t_indices, b_indices, jax_key = self.get_sample_indices(self.n_integration_steps, self.batch_size, self.minib_time_steps, self.minib_states, jax_key)
            minib_SDE_tracer, jax_key = self._select_minibatch(SDE_tracer, t_indices, b_indices, jax_key)

            t2 = time.time()
            params, Interpol_params, SDE_params, opt_state, Interpol_params_state, SDE_params_state, loss_value, out_dict, _ = self.update_params_on_minibatch(params, Interpol_params, 
                                                                                                                                        SDE_params, opt_state, Interpol_params_state, SDE_params_state, minib_SDE_tracer, jax_key, T_curr)
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

        overall_out_dict["Free_Energy_at_T=1"] = SDE_tracer["Free_Energy_at_T=1"]

        log_dict = {"mean_energy": np.mean(Energy), "best_Energy": jnp.min(Energy), "X_0": x_last, "diffusions": SDE_tracer["diffusions"],
                      "sigma": jnp.exp(SDE_params["log_sigma"]),"beta_min": jnp.exp(SDE_params["log_beta_min"]),
                        "beta_delta": jnp.exp(SDE_params["log_beta_delta"]), "mean": SDE_tracer["prior_mean"], "sigma_prior": SDE_tracer["sigma_prior"]
                        }

        for log_dict_key in log_dict.keys():
            overall_out_dict[log_dict_key] = log_dict[log_dict_key]

        ### TODO compute some metrics here!

        return params, Interpol_params, SDE_params, opt_state, Interpol_params_state, SDE_params_state, loss_value, overall_out_dict, {}

    @partial(jax.jit, static_argnums=(0,))
    def evaluate_loss_for_train(self, params, Interpol_params, SDE_params, minib_SDE_tracer, key, temp = 1.0, alpha = 0.5, clip_value = 0.1, PPO_mode = False):

        for interpol_key in Interpol_params.keys():
            SDE_params[interpol_key] = Interpol_params[interpol_key]
        
        old_reverse_log_probs = minib_SDE_tracer["reverse_log_probs"]

        advantages = minib_SDE_tracer["advantages"]
        advantage_norm = 1.#jnp.std(advantages) + 0.1
        centered_advantages = (advantages - jnp.mean(advantages))
        value_target = minib_SDE_tracer["value_target"]
        x_prev = minib_SDE_tracer["x_prev"]
        x_next = minib_SDE_tracer["x_next"]
        time_steps = minib_SDE_tracer["ts"]
        dts = minib_SDE_tracer["dts"]
        counters = minib_SDE_tracer["counters"]
        print("counters", counters.shape)


        init_carry = jnp.zeros((x_prev.shape[0], self.Network_Config["n_hidden"]), dtype = jnp.float32) ### This is not used
        hidden_state = [(init_carry, init_carry)  for i in range(self.Network_Config["n_layers"])]

        apply_model_dict, key = self.SDE_type.apply_model(self.model, x_prev, time_steps/self.n_integration_steps, counters, params, Interpol_params, SDE_params, hidden_state, temp, key)

        score = apply_model_dict["score"]
        grad = apply_model_dict["grad"]
        SDE_params_extended =  apply_model_dict["SDE_params_extended"]

        reverse_out_dict, key = self.SDE_type.reverse_sde(SDE_params_extended, score, grad, x_prev, time_steps, dts, key)

        reverse_drift = reverse_out_dict["reverse_drift"]

        diffusion_prev = self.SDE_type.get_diffusion(SDE_params_extended, x_prev, time_steps)
        #jax.debug.print("time_steps {time_steps} dts {dts}", time_steps=time_steps, dts=dts)
        
        ## TODO compute forward log probs here
        reverse_diff_log_probs = jax.vmap(self.SDE_type.calc_diff_log_prob, in_axes=(0, 0, 0))(x_next, x_prev + reverse_drift*dts, diffusion_prev*jnp.sqrt(dts))

        
        time_steps_next = time_steps - dts ### awrrrr!!!!! time runs from T to 0!!
        if(self.SDE_Type_Config["name"] == "Bridge_SDE"):
            counters_next = counters + 1
            apply_model_dict_next, key = self.SDE_type.apply_model(self.model, x_next, time_steps_next/self.n_integration_steps, counters_next, params, Interpol_params, SDE_params, hidden_state, temp, key)

            score_next = apply_model_dict_next["score"]
            grad_next = apply_model_dict_next["grad"]
            SDE_params_extended_next =  apply_model_dict_next["SDE_params_extended"]

            reverse_out_dict_next, key = self.SDE_type.reverse_sde(SDE_params_extended_next, score_next, grad_next, x_next, time_steps_next, dts, key)
            diffusion_next = self.SDE_type.get_diffusion(SDE_params_extended_next, x_next, time_steps_next)

            forward_diff_log_probs = self.SDE_type.compute_forward_log_probs(x_next, x_prev, diffusion_next, dts, apply_model_dict_next, reverse_out_dict_next)

        else:
            diffusion_next = self.SDE_type.get_diffusion(SDE_params_extended, x_next, time_steps_next)
            forward_diff_log_probs = self.SDE_type.compute_forward_log_probs(x_next, x_prev, diffusion_next, dts)

        PPO_update, _,  ratio_diff, clipped_state = self.PPO_clipping(old_reverse_log_probs, reverse_diff_log_probs, centered_advantages[..., 0], epsilon=clip_value, PPO_mode=PPO_mode)
        Loss_actor = jnp.mean(PPO_update)*self.n_integration_steps 

        KL_loss_per_state = ratio_diff * forward_diff_log_probs
        if(PPO_mode):
            clipped_KL_loss_2 = jnp.where(clipped_state == 1, jax.lax.stop_gradient(KL_loss_per_state), KL_loss_per_state)
            rKL_loss_2 = - self.n_integration_steps * jnp.mean(clipped_KL_loss_2)
        else:
            rKL_loss_2 = - self.n_integration_steps * jnp.mean(KL_loss_per_state)


        value_function_value = apply_model_dict["value_function_value"]
        print("value function", value_function_value.shape, value_target.shape)
        Loss_critic = jnp.mean((value_function_value - value_target)**2)

        log_prior_old = minib_SDE_tracer["log_prior"]
        x_prior = minib_SDE_tracer["x_0"]
        log_prior = self.vmap_get_log_prior(SDE_params, x_prior)

        radon_nycodin_deriv = minib_SDE_tracer["Free_Energy_at_T=1_per_batch"]
        bias = jnp.mean(radon_nycodin_deriv)
        prior_PPO_loss, prob_ratios_adv, ratio, _ = self.PPO_clipping(log_prior_old, log_prior, radon_nycodin_deriv - bias, epsilon=clip_value, PPO_mode=PPO_mode)


        ### TODO maybe only do this once for the first update
        prior_loss = jnp.mean(prior_PPO_loss)

        if(PPO_mode):
            KL_pen = 2.
            KL_reg_loss = jax.lax.stop_gradient(- jnp.mean(log_prior) + jnp.mean( - reverse_diff_log_probs))
        else:
            KL_pen = 2.
            KL_reg_loss = - jnp.mean(log_prior) + jnp.mean( - reverse_diff_log_probs)

        policy_loss = alpha*(Loss_actor + rKL_loss_2 + KL_pen*KL_reg_loss + prior_loss)/advantage_norm 
        value_loss = (1-alpha)*Loss_critic
        overall_loss = policy_loss + value_loss

        log_dict = {"loss": overall_loss, "losses/Loss_critic": Loss_critic, "losses/Loss_actor": Loss_actor, "losses/prior_loss": prior_loss, "losses/KL_loss": KL_reg_loss,
                      "sigma": jnp.exp(SDE_params["log_sigma"]), "beta_min": jnp.exp(SDE_params["log_beta_min"]),
                      "beta_delta": jnp.exp(SDE_params["log_beta_delta"])
                      }

        return overall_loss, log_dict

    def PPO_clipping(self, log_prob_before, log_prob_after, advantage, epsilon = 0.1, PPO_mode = False, KL_pen = 2.):
        ### TODO check if this is correct
        ratio = jnp.exp(log_prob_after - log_prob_before)
        stopped_ratio = jax.lax.stop_gradient(ratio)
        if(PPO_mode):
            clipped_ratio = jnp.clip(ratio, 1 - epsilon, 1 + epsilon)
            PPO_loss_per_state = jnp.maximum(ratio * advantage, clipped_ratio * advantage)
            clipped_state = jnp.where(PPO_loss_per_state == clipped_ratio * advantage, 1, 0)

            return PPO_loss_per_state, jax.lax.stop_gradient(ratio * advantage), stopped_ratio, clipped_state    
        else:
            loss_per_state = stopped_ratio * advantage*log_prob_after
            overall_loss_per_state = loss_per_state 
            return overall_loss_per_state, jax.lax.stop_gradient(ratio * advantage), stopped_ratio, None

    @partial(jax.jit, static_argnums=(0,))
    def evaluate_loss(self, params, Energy_params, SDE_params, SDE_tracer, key, temp = 1.0):
        ts = SDE_tracer["ts"]
        prior_mean = SDE_tracer["prior_mean"]
        prior_sigma = SDE_tracer["prior_sigma"]
        forward_diff_log_probs = SDE_tracer["forward_diff_log_probs"]
        reverse_log_probs = SDE_tracer["reverse_log_probs"]
        log_prob_prior_scaled = SDE_tracer["log_prob_prior_scaled"]

        #average over the time dimension (axis 0)
        entropy_minus_noise = jnp.sum(reverse_log_probs - forward_diff_log_probs, axis = 0)

        x_prior = SDE_tracer["x_prior"]
        x_last = SDE_tracer["x_final"]

        log_prior = self.vmap_get_log_prior(SDE_params, x_prior)

        Energy, key = self.EnergyClass.vmap_calc_energy(x_last, Energy_params, key)
        mean_Energy = jnp.mean(Energy)

        entropy_loss = jnp.mean(jnp.sum(reverse_log_probs, axis = 0) )
        noise_loss = jnp.mean(-jnp.sum(forward_diff_log_probs, axis = 0))


        if self.SDE_type.config['use_off_policy']:  
            log_prob_prior_scaled = SDE_tracer["log_prob_prior_scaled"]
            log_prob_noise = SDE_tracer["log_prob_noise"]
            log_prob_on_policy = SDE_tracer["log_prob_on_policy"]

            delta_log_prior = log_prior - log_prob_prior_scaled 
            delta_log_prob = log_prob_on_policy - log_prob_noise
            delta_log_weights = jnp.concatenate([delta_log_prior[None, :], delta_log_prob], axis = 0)
            if(self.quantile != 0):
                quantile = self.quantile
                log_max_quantile = jnp.quantile(delta_log_weights, quantile, axis = -1)
                log_weights_max_quantile = log_max_quantile
                delta_log_weights = jnp.maximum(delta_log_weights, log_weights_max_quantile[:, None])

            log_weights = jnp.sum(self.weight_temperature* delta_log_weights, axis = 0)

            # log_weights = jnp.nan_to_num(log_weights, nan=0.0, posinf=1e10, neginf=-1e10)
            # Energy = jnp.nan_to_num(Energy, nan=1e10, posinf=1e10)
            off_policy_weights_normed = jax.lax.stop_gradient(jax.nn.softmax(log_weights, axis = -1))
            off_policy_weights_normed_times_n_samples = off_policy_weights_normed* log_prob_on_policy.shape[-1] ### multiply wiht numer of states so that mean instead of sum can be used later on
            loss, unbiased_loss, centered_loss = self.compute_rKL_log_deriv(SDE_params, log_prior, reverse_log_probs, forward_diff_log_probs, entropy_minus_noise,Energy, temp, 
                        off_policy_weights_normed_times_n_samples, use_off_policy = True)
            
        else:
            off_policy_weights_normed_times_n_samples = 1.
            loss, unbiased_loss, centered_loss = self.compute_rKL_log_deriv(SDE_params, log_prior, reverse_log_probs, forward_diff_log_probs, entropy_minus_noise,Energy, temp)


        log_dict = {"loss": loss, "mean_energy": mean_Energy, "losses/unbiased_loss": unbiased_loss, "losses/centered_loss": centered_loss,
                      "best_Energy": jnp.min(Energy), "noise_loss": noise_loss, "entropy_loss": entropy_loss, "key": key, "X_0": x_last, 
                      "sigma": jnp.exp(SDE_params["log_sigma"]),"beta_min": jnp.exp(SDE_params["log_beta_min"]),
                        "beta_delta": jnp.exp(SDE_params["log_beta_delta"]), "mean": prior_mean, "sigma_prior": prior_sigma
                        }
        
        if("fisher_grads" in SDE_tracer):
            fisher_grads = SDE_tracer["fisher_grads"]
            log_dict["fisher_grads"] = fisher_grads

        log_dict = self.compute_partition_sum(entropy_minus_noise, jnp.zeros_like(entropy_minus_noise), log_prior, Energy, log_dict, off_policy_weights = off_policy_weights_normed_times_n_samples)

        return loss, log_dict

    def compute_rKL_log_deriv(self, SDE_params, log_prior, reverse_log_probs, forward_diff_log_probs, entropy_minus_noise,Energy, temp, off_policy_weights = 1., use_off_policy = False):

        if(self.optim_mode == "optim"):
            sum_reverse_log_probs = jnp.sum(reverse_log_probs, axis = 0) + log_prior
            radon_dykodin_derivative = temp*log_prior + temp*entropy_minus_noise + Energy
            radon_nykodin_wo_reverse = -temp*jnp.sum(forward_diff_log_probs, axis = 0) + Energy

        elif(self.optim_mode == "equilibrium"):
            sum_reverse_log_probs = jnp.sum(reverse_log_probs, axis = 0) + log_prior
            radon_dykodin_derivative = log_prior + entropy_minus_noise + Energy/temp
            radon_nykodin_wo_reverse = -jnp.sum(forward_diff_log_probs, axis = 0) + Energy/temp

        if(use_off_policy):

            biased_mean = jax.lax.stop_gradient(jnp.mean(radon_dykodin_derivative, keepdims=True, axis = 0))
            reward = jax.lax.stop_gradient((radon_dykodin_derivative - biased_mean))
            L1 = jnp.mean((off_policy_weights* reward ) * sum_reverse_log_probs)
            loss = L1 + jnp.mean(off_policy_weights * radon_nykodin_wo_reverse)

            unbiased_loss = jnp.mean((off_policy_weights* reward) * sum_reverse_log_probs) + jnp.mean(off_policy_weights * radon_nykodin_wo_reverse)
            centered_loss = L1

        else:
            unbiased_mean = jax.lax.stop_gradient(jnp.mean(radon_dykodin_derivative, keepdims=True, axis = 0))
            reward = jax.lax.stop_gradient((radon_dykodin_derivative-unbiased_mean))
            L1 = jnp.mean(reward * sum_reverse_log_probs)
            #add the extra term that arises in the bridge case if the forward process does also have learnable params (see our ICLR 25 workshop paper)
            loss = L1+ jnp.mean( radon_nykodin_wo_reverse)

            unbiased_loss = jnp.mean(jax.lax.stop_gradient((radon_dykodin_derivative)) * sum_reverse_log_probs) + jnp.mean( radon_nykodin_wo_reverse)
            centered_loss = L1

        # jax.debug.print("🤯 reward {reward} 🤯", reward=jnp.mean(reward))
        # jax.debug.print("🤯 L1 {L1} 🤯", L1=L1)
        # jax.debug.print("🤯 loss {loss} 🤯", loss=loss)

        return loss, unbiased_loss, centered_loss
    