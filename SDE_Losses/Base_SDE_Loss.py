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


        self.SDE_params = self.EnergyClass.init_EnergyParams()

        self.sigma_lr = Optimizer_Config["sigma_lr"]
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
    def loss_fn(self, params, SDE_params, T_curr, key):
        loss, key = self.compute_loss( params, SDE_params, key, n_integration_steps = self.n_integration_steps, n_states = self.batch_size, temp = T_curr, x_dim = self.EnergyClass.dim_x)
        return loss, key

    def update_step(self, params, opt_state, key, T_curr):
        params, self.SDE_params, opt_state, self.SDE_params_state, loss_value, out_dict =  self.update_params(params, self.SDE_params, opt_state, self.SDE_params_state, key, T_curr)

        # for key in out_dict:
        #     print(key, jnp.mean(out_dict[key]))
        return params, opt_state, loss_value, out_dict

    @partial(jax.jit, static_argnums=(0,))
    def update_params(self, params, SDE_params, opt_state, SDE_params_state, key, T_curr):
        (loss_value, out_dict), (grads, SDE_params_grad) = jax.value_and_grad(self.loss_fn, argnums=(0,1), has_aux = True)(params, SDE_params, T_curr, key)
        updates, opt_state = self.optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)

        sigma_updates, SDE_params_state = self.SDE_params_optimizer.update(SDE_params_grad, SDE_params_state)
        SDE_params = optax.apply_updates( SDE_params, sigma_updates)
        
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


        return params, SDE_params, opt_state, SDE_params_state, loss_value, out_dict

    ### TODO move optimizers and so on here!
    def initialize_optimizer(self):
        l_start = 1e-10
        l_max = self.Optimizer_Config["lr"]
        overall_steps = self.Optimizer_Config["epochs"]*self.Optimizer_Config["steps_per_epoch"]*self.lr_factor
        warmup_steps = int(0.1 * overall_steps)

        self.schedule = lambda epoch: learning_rate_schedule(epoch, l_max, l_start, overall_steps, warmup_steps)
        #optimizer = optax.adam(self.schedule)
        optimizer = optax.chain(optax.clip_by_global_norm(1.0), optax.scale_by_radam(), optax.scale_by_schedule(lambda epoch: -self.schedule(epoch)))
        return optimizer
    
    def init_SDE_params_optimizer(self):
        lr = self.sigma_lr
        SDE_params_optimizer = optax.radam(lr)
        return SDE_params_optimizer
    
    def shift_samples(self, X_samples, energy_key):
        shifted_samples, energy_key =  self.EnergyClass.scale_samples(X_samples, self.SDE_params, energy_key)
        return shifted_samples

    @partial(jax.jit, static_argnums=(0,), static_argnames=("n_integration_steps", "n_states"))
    def simulate_reverse_sde_scan(self, params, key, n_states = 100, n_integration_steps = 1000):
        SDE_tracer, key = self.SDE_type.simulate_reverse_sde_scan(self.model, params, self.SDE_params, key, n_states = n_states, x_dim = self.EnergyClass.dim_x, n_integration_steps = n_integration_steps)
        
        key, subkey = jax.random.split(key)
        batched_key = jax.random.split(subkey, n_states)
        SDE_tracer["ys"] = jax.vmap(jax.vmap(self.shift_samples, in_axes=(0,0)), in_axes=(0,None))(SDE_tracer["xs"], batched_key)
        SDE_tracer["y_final"] = jax.vmap(self.shift_samples, in_axes=(0,0))(SDE_tracer["x_final"], batched_key)
        return SDE_tracer, key

    def compute_loss(self,*args, **kwargs):
        """
        Calculate the loss between predictions and targets.

        :param predictions: The predicted values.
        :param targets: The ground truth values.
        :return: The calculated loss.
        """
        raise NotImplementedError("get_loss method not implemented")
    


    

def learning_rate_schedule(step, l_max = 1e-4, l_start = 1e-5, overall_steps = 1000, warmup_steps = 100):
    lr_min = l_max/10
    cosine_decay = lambda step: optax.cosine_decay_schedule(init_value=(l_max - lr_min), decay_steps=overall_steps - warmup_steps)(step) + lr_min

    return jnp.where(step < warmup_steps, l_start + (l_max - l_start) * (step / warmup_steps), cosine_decay(step - warmup_steps))
