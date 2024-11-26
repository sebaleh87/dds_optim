from .SDE_Types import get_SDE_Type_Class
import optax
import jax
import jax.numpy as jnp
from functools import partial
from .MovAverage import MovAvrgCalculator

class Base_SDE_Loss_Class:
    def __init__(self, SDE_config, Optimizer_Config, Energy_Class, Network_Config, model, lr_factor = 1.):
        SDE_Type_Config = SDE_config["SDE_Type_Config"]
        self.Optimizer_Config = Optimizer_Config
        self.SDE_type = get_SDE_Type_Class(SDE_Type_Config, Network_Config, Energy_Class)
        
        self.lr_factor = lr_factor
        self.batch_size = SDE_config["batch_size"]
        self.n_integration_steps = SDE_config["n_integration_steps"]

        self.EnergyClass = Energy_Class
        self.model = model
        self.x_dim = self.EnergyClass.dim_x

        self.optimizer = self.initialize_optimizer()

        self.vmap_Energy_function =  jax.jit(jax.vmap(self.EnergyClass.energy_function, in_axes = (0,)))
        self.vmap_model = jax.vmap(self.model.apply, in_axes=(None,0,0))

        self.Energy_params = self.EnergyClass.init_EnergyParams()
        self.Energy_lr = Optimizer_Config["Energy_lr"]
        self.Energy_params_optimizer = self.init_Energy_params_optimizer()
        self.Energy_params_state = self.Energy_params_optimizer.init(self.Energy_params)

        self.SDE_params = self.SDE_type.get_SDE_params()
        self.SDE_lr = Optimizer_Config["SDE_lr"]
        self.SDE_params_optimizer = self.init_SDE_params_optimizer()
        self.SDE_params_state = self.SDE_params_optimizer.init(self.SDE_params)

        alpha = 0.01
        self.MovAvrgCalculator = MovAvrgCalculator.MovAvrgCalculator(alpha)
        self.Energy_key = jax.random.PRNGKey(0)

        ### TODO make initial forward pass
        ## initialize moving averages
    
    def init_mov_averages(self, X_init_samples):
        self.Mov_average_dict = self.MovAvrgCalculator.initialize_averages(X_init_samples)


    @partial(jax.jit, static_argnums=(0,))
    def loss_fn(self, params, Energy_params, SDE_params, T_curr, key):
        loss, key = self.compute_loss( params, Energy_params, SDE_params, key, n_integration_steps = self.n_integration_steps, n_states = self.batch_size, temp = T_curr, x_dim = self.EnergyClass.dim_x)
        return loss, key

    def update_step(self, params, opt_state, key, T_curr):
        params, self.Energy_params, self.SDE_params, opt_state, self.Energy_params_state, self.SDE_params_state, loss_value, out_dict =  self.update_params(params, self.Energy_params, self.SDE_params
                                                                                                                                                            , opt_state, self.Energy_params_state, self.SDE_params_state, key, T_curr)
        # for key in out_dict:
        #     print(key, jnp.mean(out_dict[key]))
        return params, opt_state, loss_value, out_dict

    @partial(jax.jit, static_argnums=(0,))
    def update_params(self, params, Energy_params, SDE_params, opt_state, Energy_params_state, SDE_params_state, key, T_curr):
        (loss_value, out_dict), (grads, Energy_params_grad, SDE_params_grad) = jax.value_and_grad(self.loss_fn, argnums=(0, 1, 2), has_aux = True)(params, Energy_params, SDE_params, T_curr, key)
        updates, opt_state = self.optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)

        if(self.Energy_lr != 0.):
            Energy_params_updates, Energy_params_state = self.Energy_params_optimizer.update(Energy_params_grad, Energy_params_state)
            Energy_params = optax.apply_updates( Energy_params, Energy_params_updates)

        SDE_params_updates, SDE_params_state = self.SDE_params_optimizer.update(SDE_params_grad, SDE_params_state)
        SDE_params = optax.apply_updates(SDE_params, SDE_params_updates)

        SDE_params["log_beta_min"] = jnp.log(self.SDE_type.config["beta_min"])*jnp.ones_like(SDE_params["log_beta_min"])
        SDE_params["log_beta_delta"] = jnp.log(self.SDE_type.config["beta_max"])*jnp.ones_like(SDE_params["log_beta_delta"])
        
        if(False):
            flat_grads = jax.tree_util.tree_leaves(grads)
            # Sum all parameter values and count the total number of elements
            total_sum = sum([jnp.sum(param) for param in flat_grads])
            total_elements = sum([param.size for param in flat_grads])
            # Compute the average value
            average_grad = total_sum / total_elements
            out_dict["network/average_grad"] = jnp.abs(average_grad)
            X_0 = out_dict["X_0"]
            Energy_grad = jnp.mean(jnp.gradient(self.vmap_calc_Energy(X_0)))
            out_dict["network/Energy_grad"] = jnp.abs(Energy_grad)


        return params, Energy_params, SDE_params, opt_state, Energy_params_state, SDE_params_state, loss_value, out_dict

    ### TODO move optimizers and so on here!
    def initialize_optimizer(self):
        l_start = 1e-10
        l_max = self.Optimizer_Config["lr"]
        lr_min = l_max/10
        overall_steps = self.Optimizer_Config["epochs"]*self.Optimizer_Config["steps_per_epoch"]*self.lr_factor
        warmup_steps = int(0.1 * overall_steps)

        self.schedule = lambda epoch: learning_rate_schedule(epoch, l_max, l_start, lr_min, overall_steps, warmup_steps)
        #optimizer = optax.adam(self.schedule)
        optimizer = optax.chain(optax.clip_by_global_norm(1.0), optax.scale_by_radam(), optax.scale_by_schedule(lambda epoch: -self.schedule(epoch)))
        return optimizer
    
    def init_Energy_params_optimizer(self):
        l_start = 1e-10
        l_max = self.Energy_lr
        lr_min = l_max/10
        overall_steps = self.Optimizer_Config["epochs"]*self.Optimizer_Config["steps_per_epoch"]*self.lr_factor
        warmup_steps = int(0.1 * overall_steps)

        self.Energy_schedule = lambda epoch: learning_rate_schedule(epoch, l_max, l_start, lr_min, overall_steps, warmup_steps)
        #optimizer = optax.adam(self.schedule)
        optimizer = optax.chain( optax.scale_by_radam(), optax.scale_by_schedule(lambda epoch: -self.Energy_schedule(epoch)))
        return optimizer
    
    def init_SDE_params_optimizer(self):
        l_start = 1e-10
        l_max = self.SDE_lr
        lr_min = l_max/10
        overall_steps = self.Optimizer_Config["epochs"]*self.Optimizer_Config["steps_per_epoch"]*self.lr_factor
        warmup_steps = int(0.1 * overall_steps)

        self.SDE_schedule = lambda epoch: learning_rate_schedule(epoch, l_max, l_start, lr_min, overall_steps, warmup_steps)
        #optimizer = optax.radam(l_max)
        optimizer = optax.chain(optax.clip_by_global_norm(1.0), optax.scale_by_radam(), optax.scale_by_schedule(lambda epoch: -self.SDE_schedule(epoch)))
        return optimizer
    
    def shift_samples(self, X_samples, SDE_params, energy_key):
        shifted_samples, energy_key =  self.EnergyClass.scale_samples(X_samples, SDE_params, energy_key)
        return shifted_samples

    @partial(jax.jit, static_argnums=(0,), static_argnames=("n_integration_steps", "n_states"))
    def simulate_reverse_sde_scan(self, params, Energy_params, SDE_params, key, n_states = 100, n_integration_steps = 1000):
        SDE_tracer, key = self.SDE_type.simulate_reverse_sde_scan(self.model, params, Energy_params, SDE_params, key, n_states = n_states, x_dim = self.EnergyClass.dim_x, n_integration_steps = n_integration_steps)
        loss, out_dict = self.evaluate_loss(Energy_params, SDE_params, SDE_tracer, key)
        key, subkey = jax.random.split(key)
        batched_key = jax.random.split(subkey, n_states)
        SDE_tracer["ys"] = jax.vmap(jax.vmap(self.shift_samples, in_axes=(0, None,0)), in_axes=(0, None, None))(SDE_tracer["xs"], Energy_params, batched_key)
        SDE_tracer["y_final"] = jax.vmap(self.shift_samples, in_axes=(0,None, 0))(SDE_tracer["x_final"], Energy_params, batched_key)
        return SDE_tracer, out_dict, key
    
    @partial(jax.jit, static_argnums=(0,), static_argnames=("n_integration_steps", "n_states"))
    def evaluate_model(self, params, Energy_params, SDE_params, key, n_states = 100, n_integration_steps = 1000):
        loss, SDE_tracer = self.compute_loss(params, Energy_params, SDE_params, key, n_integration_steps = n_integration_steps, n_states = n_states)

        return SDE_tracer, SDE_tracer["key"]
    
    def compute_partition_sum(self, R_diff, S, log_prior, Energy):
        Z_estim = R_diff + S + log_prior + Energy
        log_Z = jnp.mean(-Z_estim)
        Free_Energy = -log_Z
        log_weights = -Z_estim
        normed_weights = jax.nn.softmax(log_weights, axis = 0)

        n_eff = 1/(jnp.sum(normed_weights**2)*Z_estim.shape[0])

        NLL = -jnp.mean(R_diff + S + log_prior) 
        return log_Z, Free_Energy, n_eff, NLL

    @partial(jax.jit, static_argnums=(0,), static_argnames=("n_integration_steps", "n_states", "x_dim"))  
    def compute_loss(self, params, Energy_params, SDE_params, key, n_integration_steps = 100, n_states = 10, temp = 1.0, x_dim = 2):
        SDE_tracer, key = self.SDE_type.simulate_reverse_sde_scan(self.model , params, Energy_params, SDE_params, key, n_integration_steps = n_integration_steps, n_states = n_states, x_dim = x_dim)
        loss, out_dict = self.evaluate_loss(Energy_params, SDE_params, SDE_tracer, key, temp = temp)

        return loss, out_dict
    
    def get_param_dict(self, params):
        return {"model_params": params, "Energy_params": self.Energy_params, "SDE_params": self.SDE_params}


    

def learning_rate_schedule(step, l_max = 1e-4, l_start = 1e-5, lr_min = 1e-4, overall_steps = 1000, warmup_steps = 100):
    cosine_decay = lambda step: optax.cosine_decay_schedule(init_value=(l_max - lr_min), decay_steps=overall_steps - warmup_steps)(step) + lr_min

    return jnp.where(step < warmup_steps, l_start + (l_max - l_start) * (step / warmup_steps), cosine_decay(step - warmup_steps))

def exp_learning_rate_schedule(step, l_max = 1e-4, l_start = 1e-5, lr_min = 1e-4, overall_steps = 1000, warmup_steps = 100, lam = 5.):
    cosine_decay = lambda step: (l_max- lr_min)*jnp.exp(- 5*(step-warmup_steps)/(overall_steps-warmup_steps)) + lr_min

    return jnp.where(step < warmup_steps, l_start + (l_max - l_start) * (step / warmup_steps), cosine_decay(step - warmup_steps))
