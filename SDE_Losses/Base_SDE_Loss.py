from .SDE_Types import get_SDE_Type_Class
import optax
import jax
import jax.numpy as jnp
from functools import partial
from .MovAverage import MovAvrgCalculator

class Base_SDE_Loss_Class:
    def __init__(self, SDE_config, Optimizer_Config, Energy_Class, Network_Config, model, lr_factor = 1.):
        SDE_Type_Config = SDE_config["SDE_Type_Config"]
        self.Network_Config = Network_Config
        self.Optimizer_Config = Optimizer_Config
        self.SDE_type = get_SDE_Type_Class(SDE_Type_Config, Network_Config, Energy_Class)
        
        self.lr_factor = lr_factor
        self.batch_size = SDE_config["batch_size"]
        self.n_integration_steps = SDE_config["n_integration_steps"]
        self.lr_schedule = Optimizer_Config["lr_schedule"]

        self.EnergyClass = Energy_Class
        self.model = model
        self.x_dim = self.EnergyClass.dim_x

        self.optimizer = self.initialize_optimizer()

        self.vmap_Energy_function =  jax.jit(jax.vmap(self.EnergyClass.energy_function, in_axes = (0,)))
        self.vmap_model = jax.vmap(self.model.apply, in_axes=(None,0,0))

        self.Interpol_params = self.SDE_type.get_Interpol_params()
        self.Interpol_lr = Optimizer_Config["Interpol_lr"]
        self.Interpol_params_optimizer = self.init_Interpol_params_optimizer()
        self.Interpol_params_state = self.Interpol_params_optimizer.init(self.Interpol_params)

        self.SDE_params = self.SDE_type.get_SDE_params()
        self.init_SDE_params = self.SDE_params
        self.SDE_lr = Optimizer_Config["SDE_lr"]
        self.SDE_params_optimizer = self.init_SDE_params_optimizer()
        self.SDE_params_state = self.SDE_params_optimizer.init(self.SDE_params)

        alpha = 0.01
        self.MovAvrgCalculator = MovAvrgCalculator.MovAvrgCalculator(alpha)
        self.Energy_key = jax.random.PRNGKey(0)

        if(SDE_config["update_params_mode"] == "all_in_one"):
            self.update_params = self.update_params_all_in_one
        elif(SDE_config["update_params_mode"] == "DKL"):
            self.update_params = self.update_params_DKL
        else:
            raise ValueError("Unknown update mode")
        ### TODO make initial forward pass
        ## initialize moving averages
        self.vmap_diff_factor = jax.vmap(self.SDE_type.get_diffusion, in_axes=(None, None, 0))
        self.vmap_drift_divergence = jax.vmap(self.SDE_type.get_div_drift, in_axes = (None, 0))
        self.vmap_get_log_prior = jax.vmap(self.SDE_type.get_log_prior, in_axes = (None, 0))
        self.optim_mode = "equilibrium"


    def _init_lr_schedule(self, l_max, l_start, lr_min, overall_steps, warmup_steps):
        if(self.lr_schedule == "const"):
            lr_scheudule_func = lambda step: constant_lr(step, l_max = l_max)
        elif(self.lr_schedule == "cosine_warmup"):
            lr_scheudule_func = lambda step: cos_warmup_learning_rate_schedule(step, l_max, l_start, lr_min, overall_steps, warmup_steps)
        elif(self.lr_schedule == "cosine"):
            lr_scheudule_func = lambda step: cosine_learning_rate_schedule(step, l_max, l_start, lr_min, overall_steps, 0)
        else:
            raise ValueError(f"Unknown lr_schedule: {self.lr_schedule}")

        return lr_scheudule_func
    
    def init_mov_averages(self, X_init_samples):
        self.Mov_average_dict = self.MovAvrgCalculator.initialize_averages(X_init_samples)


    @partial(jax.jit, static_argnums=(0,))
    def loss_fn(self, params, Energy_params, SDE_params, T_curr, key):
        loss, key = self.compute_loss( params, Energy_params, SDE_params, key, n_integration_steps = self.n_integration_steps, n_states = self.batch_size, temp = T_curr)
        return loss, key

    def update_step(self, params, opt_state, key, T_curr):
        params, self.Interpol_params, self.SDE_params, opt_state, self.Interpol_params_state, self.SDE_params_state, loss_value, out_dict =  self.update_params(params, self.Interpol_params, self.SDE_params
                                                                                                                                                            , opt_state, self.Interpol_params_state, self.SDE_params_state, key, T_curr)
        # for key in out_dict:
        #     print(key, jnp.mean(out_dict[key]))
        return params, opt_state, loss_value, out_dict

    @partial(jax.jit, static_argnums=(0,))
    def update_params_DKL(self, params, Energy_params, SDE_params, opt_state, Energy_params_state, SDE_params_state, key, T_curr):
        #(loss_value, out_dict), (grads, SDE_params_grad) = jax.value_and_grad(self.loss_fn, argnums=(0, 2), has_aux = True)(params, Energy_params, SDE_params, T_curr, key)
        (loss_value, out_dict), (grads) = jax.value_and_grad(self.loss_fn, argnums=(0), has_aux = True)(params, Energy_params, SDE_params, T_curr, key)
        updates, opt_state = self.optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        
        (loss_value, out_dict), (SDE_params_grad) = jax.value_and_grad(self.compute_DKL_loss, argnums=(0), has_aux = True)( SDE_params, out_dict)
        
        SDE_params_updates, SDE_params_state = self.SDE_params_optimizer.update(SDE_params_grad, SDE_params_state)
        SDE_params = optax.apply_updates(SDE_params, SDE_params_updates)
        
        return params, Energy_params, SDE_params, opt_state, Energy_params_state, SDE_params_state, loss_value, out_dict
    
    @partial(jax.jit, static_argnums=(0,))
    def update_params_all_in_one(self, params, Interpol_params, SDE_params, opt_state, Interpol_params_state, SDE_params_state, key, T_curr):
        (loss_value, out_dict), (grads, Interpol_params_grad, SDE_params_grad) = jax.value_and_grad(self.loss_fn, argnums=(0, 1, 2), has_aux = True)(params, Interpol_params, SDE_params, T_curr, key)
        updates, opt_state = self.optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        
        # check_nans(grads)
        # check_nans(SDE_params_grad)
        # print(loss_value)
        Interpol_params_updates, Interpol_params_state = self.Interpol_params_optimizer.update(Interpol_params_grad, Interpol_params_state, Interpol_params)
        Interpol_params = optax.apply_updates(Interpol_params, Interpol_params_updates)

        SDE_params_updates, SDE_params_state = self.SDE_params_optimizer.update(SDE_params_grad, SDE_params_state, SDE_params)
        SDE_params = optax.apply_updates(SDE_params, SDE_params_updates)

        if( self.Optimizer_Config["learn_SDE_params_mode"] == "all"):
            SDE_params["log_beta_min"] = self.init_SDE_params["log_beta_min"]
            SDE_params["log_beta_delta"] = self.init_SDE_params["log_beta_delta"]
        elif( self.Optimizer_Config["learn_SDE_params_mode"] == "prior_only"):
            for SDE_param_key in SDE_params.keys():
                if(SDE_param_key == "log_sigma_prior" or SDE_param_key == "mean"):
                    pass
                else:
                    SDE_params[SDE_param_key] = self.init_SDE_params[SDE_param_key]
        elif( self.Optimizer_Config["learn_SDE_params_mode"] == "all_and_beta"):
            if(self.SDE_type.config["name"] == "Bridge_SDE"):
                SDE_params["log_sigma"] = self.init_SDE_params["log_beta_delta"] ### log sigma does not exist in config as it is controled via beta
            pass			
        
        return params, Interpol_params, SDE_params, opt_state, Interpol_params_state, SDE_params_state, loss_value, out_dict
    
    @partial(jax.jit, static_argnums=(0,))
    def update_net_params_only(self, params, Energy_params, SDE_params, opt_state, Energy_params_state, SDE_params_state, key, T_curr):
        (loss_value, out_dict), (grads) = jax.value_and_grad(self.loss_fn, argnums=(0), has_aux = True)(params, Energy_params, SDE_params, T_curr, key)
        updates, opt_state = self.optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
    	
        
        return params, Energy_params, SDE_params, opt_state, Energy_params_state, SDE_params_state, loss_value, out_dict

    ### TODO move optimizers and so on here!
    def initialize_optimizer(self):
        l_start = 1e-10
        l_max = self.Optimizer_Config["lr"]
        lr_min = l_max/10
        overall_steps = self.Optimizer_Config["epochs"]*self.Optimizer_Config["steps_per_epoch"]*self.lr_factor
        warmup_steps = int(0.1 * overall_steps)
        clip_value = self.Optimizer_Config["clip_value"]
        self.schedule = self._init_lr_schedule(l_max, l_start, lr_min, overall_steps, warmup_steps)

        optimizer = optax.chain(optax.zero_nans(), optax.clip_by_global_norm(clip_value), optax.scale_by_radam(), optax.scale_by_schedule(lambda epoch: -self.schedule(epoch)))
        return optimizer
    
    def init_Interpol_params_optimizer(self):
        l_start = 1e-10
        l_max = self.Interpol_lr
        lr_min = l_max/10
        overall_steps = self.Optimizer_Config["epochs"]*self.Optimizer_Config["steps_per_epoch"]*self.lr_factor
        warmup_steps = int(0.1 * overall_steps)

        self.Interpol_schedule = self._init_lr_schedule(l_max, l_start, lr_min, overall_steps, warmup_steps)
        #optimizer = optax.adam(self.schedule)
        clip_value = self.Optimizer_Config["clip_value"]
        optimizer = optax.chain(optax.zero_nans(), optax.clip_by_global_norm(clip_value), optax.scale_by_radam(), optax.scale_by_schedule(lambda epoch: -self.Interpol_schedule(epoch)))
        return optimizer
    
    def init_SDE_params_optimizer(self):
        if(self.SDE_lr == 0.):
            l_start = 0.
        else:
            l_start = 1e-10
        l_max = self.SDE_lr
        lr_min = l_max/10
        overall_steps = self.Optimizer_Config["epochs"]*self.Optimizer_Config["steps_per_epoch"]*self.lr_factor
        warmup_steps = int(0.1 * overall_steps)
        weight_decay = self.Optimizer_Config["SDE_weight_decay"]
        clip_value = self.Optimizer_Config["clip_value"]

        self.SDE_schedule = self._init_lr_schedule(l_max, l_start, lr_min, overall_steps, warmup_steps)
        #clipping is necessary due to lennard jones instabilities for some energy functions
        optimizer = optax.chain(optax.zero_nans(), optax.clip_by_global_norm(clip_value), optax.add_decayed_weights(weight_decay), optax.scale_by_radam(), optax.scale_by_schedule(lambda epoch: -self.SDE_schedule(epoch)))
        
        return optimizer
    
    def shift_samples(self, X_samples, SDE_params, energy_key):
        shifted_samples, energy_key =  self.EnergyClass.scale_samples(X_samples, SDE_params, energy_key)
        return shifted_samples

    @partial(jax.jit, static_argnums=(0,), static_argnames=("n_integration_steps", "n_states", "sample_mode"))
    def simulate_reverse_sde_scan(self, params, Energy_params, SDE_params, temp, key, sample_mode = "train", n_states = 100, n_integration_steps = 1000):    #TODO change name! this one also applies scaling!
        SDE_tracer, key = self.SDE_type.simulate_reverse_sde_scan(self.model, params, Energy_params, SDE_params, temp, key, n_states = n_states, sample_mode = sample_mode, n_integration_steps = n_integration_steps)
        loss, out_dict = self.evaluate_loss(params, Energy_params, SDE_params, SDE_tracer, key)     #TODO add not implemented template class for self.evaluate_loss
        key, subkey = jax.random.split(key)                                                 
        batched_key = jax.random.split(subkey, n_states)
        #TODO really want to scale the output for eval?
        SDE_tracer["ys"] = jax.vmap(jax.vmap(self.shift_samples, in_axes=(0, None,0)), in_axes=(0, None, None))(SDE_tracer["xs"], Energy_params, batched_key)
        SDE_tracer["y_final"] = jax.vmap(self.shift_samples, in_axes=(0,None, 0))(SDE_tracer["x_final"], Energy_params, batched_key)
        return SDE_tracer, out_dict, key
    
    @partial(jax.jit, static_argnums=(0,), static_argnames=("n_integration_steps", "n_states"))
    def evaluate_model(self, params, Energy_params, SDE_params, key, n_states = 100, n_integration_steps = 1000):
        loss, SDE_tracer = self.compute_loss(params, Energy_params, SDE_params, key, n_integration_steps = n_integration_steps, n_states = n_states)

        return SDE_tracer, SDE_tracer["key"]
    
    def evaluate_loss(self, params, Energy_params, SDE_params, SDE_tracer, key, temp = 1.0):
        raise NotImplementedError("Not implemented yet")
    
    def compute_partition_sum(self, R_diff, S, log_prior, Energy, rec_dict, off_policy_weights = 1.):
        Z_estim = R_diff + S + log_prior + Energy
        log_Z = jax.scipy.special.logsumexp(-Z_estim) - jnp.log(Z_estim.shape[0]) ### TODO make this computation unbiased?
        #log_Z = jnp.log(jnp.mean(jnp.exp(-Z_estim)))
        #log_Z = jnp.mean(-Z_estim)
        log_weights = -Z_estim
        normed_weights = jax.nn.softmax(log_weights, axis = -1)

        n_eff = 1/(jnp.sum(normed_weights**2)*Z_estim.shape[0])

        NLL = -jnp.mean(R_diff + S + log_prior) 

        ELBO = jnp.mean(off_policy_weights*log_weights)
        EUBO = jnp.sum(off_policy_weights*normed_weights*(log_weights))
        Free_Energy = -ELBO   ### THis is the variational free energy
        res_dict = {"Free_Energy_at_T=1": Free_Energy, "normed_weights": normed_weights, "log_Z_at_T=1": log_Z, "n_eff": n_eff, "NLL": NLL, "ELBO_at_T=1": ELBO, "EUBO_at_T=1": EUBO}
        rec_dict.update(res_dict)
        return rec_dict

    @partial(jax.jit, static_argnums=(0,), static_argnames=("n_integration_steps", "n_states"))  
    def compute_loss(self, params, Energy_params, SDE_params, key, n_integration_steps = 100, n_states = 10, temp = 1.0):
        SDE_tracer, key = self.SDE_type.simulate_reverse_sde_scan(self.model , params, Energy_params, SDE_params, temp, key, n_integration_steps = n_integration_steps, n_states = n_states)
        loss, out_dict = self.evaluate_loss(params, Energy_params, SDE_params, SDE_tracer, key, temp = temp)
        out_dict["x_final"] = SDE_tracer["x_final"]
        out_dict["losses/SDE_loss"] = loss
        return loss, out_dict

    @partial(jax.jit, static_argnums=(0,))
    def compute_covar_loss(self, SDE_params, out_dict):
        X_0 = out_dict["x_final"]
        cov_X_0 = jnp.cov(X_0, rowvar = False)
        diag_cov_X_0 = jnp.diag(cov_X_0)
        out_dict["cov_X_0"] = cov_X_0
        log_diag_cov = 0.5*jnp.log(diag_cov_X_0)
        log_sigma = SDE_params["log_sigma"]
        sigma_reg = jnp.mean((log_sigma - log_diag_cov)**2)
        sigma_loss = sigma_reg
        out_dict["losses/sigma_loss"] = sigma_loss

        alpha = self.SDE_type.beta_int(SDE_params, 1.)
        mean_X0 = jnp.mean(X_0, axis = 0)
        epsilon = 10**-3
        mean_SDE = SDE_params["mean"]
        #print("mean_X0", mean_X0.shape, mean_SDE.shape)
        mean_loss = jnp.mean(( mean_X0*jax.lax.stop_gradient(jnp.exp(- alpha)) - mean_SDE)**2)
        out_dict["losses/mean_loss"] = mean_loss

        covar = cov_X_0 - jnp.diag(diag_cov_X_0)
        #print("covar", covar.shape, jnp.exp(- 2*alpha).shape, (jnp.abs(covar) * jnp.exp(- 2*alpha)).shape)
        alpha_covar = jnp.exp(-alpha)[:,None]*jnp.exp(-alpha)[None, :]
        covar_loss = jnp.mean( (jnp.abs(covar) * alpha_covar - epsilon)**2)
        #covar_loss = jnp.mean(jnp.where(jnp.abs(covar) > epsilon, (jnp.log(jnp.abs(covar) * alpha_covar) - jnp.log(epsilon))**2, 0))
        out_dict["losses/covar_loss"] = covar_loss

        overall_loss = mean_loss + covar_loss + sigma_loss
        out_dict["losses/overall_loss"] = overall_loss
        return overall_loss, out_dict
    
    @partial(jax.jit, static_argnums=(0,))
    def compute_DKL_loss(self, SDE_params, out_dict):
        X_0 = out_dict["x_final"]
        log_sigma = SDE_params["log_sigma"]
        var = jnp.exp(2*log_sigma)
        mean_SDE = SDE_params["mean"]
        alpha = self.SDE_type.beta_int(SDE_params, 1.)
        k = mean_SDE.shape[0]
        
        sigma_q = var
        diag_sigma_q = jnp.diag(sigma_q)
        cov_X_0 = jnp.cov(X_0, rowvar = False)
        sigma_p = jnp.exp(-alpha)[:,None]*jnp.exp(-alpha)[None, :]*(cov_X_0 - diag_sigma_q) + diag_sigma_q
        mean_p = jnp.exp(-alpha)*jnp.mean(X_0, axis = 0) + mean_SDE*(1- jnp.exp(-alpha))
        # DKL(p||q) = 0.5 * (tr(sigma_q^-1 sigma_p) + (mu_q - mu_p)^T sigma_q^-1 (mu_q - mu_p) - k + log(det(sigma_q) / det(sigma_p)))
        # where q is N(mean_SDE, sigma)

        first_term = jnp.trace(jnp.diag(1/sigma_q) @ sigma_p)  -k
        second_term = (mean_SDE - mean_p).transpose() @ jnp.diag(1/sigma_q) @ (mean_SDE - mean_p)
        third_term = jnp.sum(jnp.log(sigma_q)) - jnp.log(jnp.linalg.det(sigma_p))

        print("first_term", first_term, "second_term", second_term, "third_term", third_term)
        overall_loss = 0.5*(first_term + second_term + third_term)


        out_dict["losses/overall_loss"] = overall_loss
        return overall_loss, out_dict
    
    def get_param_dict(self, params):
        return {"model_params": params, "Interpol_params": self.Interpol_params, "SDE_params": self.SDE_params}
    

def constant_lr(step, l_max = 1e-4):
    return l_max

def cosine_learning_rate_schedule(step, l_max = 1e-4, l_start = 1e-5, lr_min = 1e-4, overall_steps = 1000, warmup_steps = 0):
    cosine_decay = lambda step: optax.cosine_decay_schedule(init_value=(l_max - lr_min), decay_steps=overall_steps - warmup_steps)(step) + lr_min

    return cosine_decay(step - warmup_steps)

def cos_warmup_learning_rate_schedule(step, l_max = 1e-4, l_start = 1e-5, lr_min = 1e-4, overall_steps = 1000, warmup_steps = 100):
    cosine_decay = lambda step: optax.cosine_decay_schedule(init_value=(l_max - lr_min), decay_steps=overall_steps - warmup_steps)(step) + lr_min

    return jnp.where(step < warmup_steps, l_start + (l_max - l_start) * (step / warmup_steps), cosine_decay(step - warmup_steps))

def exp_learning_rate_schedule(step, l_max = 1e-4, l_start = 1e-5, lr_min = 1e-4, overall_steps = 1000, warmup_steps = 100, lam = 5.):
    cosine_decay = lambda step: (l_max- lr_min)*jnp.exp(- 5*(step-warmup_steps)/(overall_steps-warmup_steps)) + lr_min

    return jnp.where(step < warmup_steps, l_start + (l_max - l_start) * (step / warmup_steps), cosine_decay(step - warmup_steps))



def check_nan_in_params(params, s):
    def contains_nan(x):
        return jnp.any(jnp.isnan(x))

    # Map the function across the PyTree
    nan_trees = jax.tree_util.tree_map(contains_nan, params)

    # Combine results to check if any NaNs are present
    has_nan = jax.tree_util.tree_reduce(lambda x, y: x or y, nan_trees)

    print("Does the parameter tree contain NaNs?", s, has_nan)
    print(nan_trees)


# Get the paths where NaNs appeared
def check_nan_in_key(params, s):
    # Get the paths where NaNs appeared
    nan_paths = check_nans(params)

    # Print the results
    if nan_paths:
        print("NaNs found in the following locations:")
        for path in nan_paths:
            print(path)
    else:
        print("No NaNs found in the parameter tree.")

def check_nans(tree, path=""):
    nan_paths = []

    if isinstance(tree, dict):
        # If the current node is a dictionary, recurse into its values
        for key, value in tree.items():
            new_path = f"{path}/{key}" if path else key
            nan_paths.extend(check_nans(value, new_path))
    elif isinstance(tree, (list, tuple)):
        # If the current node is a list or tuple, recurse into its elements
        for idx, value in enumerate(tree):
            new_path = f"{path}[{idx}]"
            nan_paths.extend(check_nans(value, new_path))
    else:
        # Leaf node: check if it contains NaNs
        if jnp.any(jnp.isnan(tree)):
            nan_paths.append(path)

    return nan_paths

def inverse_softplus(x):
    return jnp.log(jnp.exp(x) - 1)