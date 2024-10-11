from .SDE_Types import get_SDE_Type_Class
import optax
import jax
import jax.numpy as jnp
from functools import partial

class Base_SDE_Loss_Class:
    def __init__(self, SDE_config, Optimizer_Config, Energy_Class, model, lr_factor = 1.):
        SDE_Type_Config = SDE_config["SDE_Type_Config"]
        self.Optimizer_Config = Optimizer_Config
        self.SDE_type = get_SDE_Type_Class(SDE_Type_Config)
        
        self.lr_factor = lr_factor
        self.batch_size = SDE_config["batch_size"]
        self.n_integration_steps = SDE_config["n_integration_steps"]

        self.EnergyClass = Energy_Class
        self.model = model
        self.x_dim = self.EnergyClass.dim_x

        self.optimizer = self.initialize_optimizer()

        self.vmap_calc_Energy =  jax.vmap(self.EnergyClass.calc_energy, in_axes = (0,))
        self.vmap_model = jax.vmap(self.model.apply, in_axes=(None,0,0))

    @partial(jax.jit, static_argnums=(0,))
    def loss_fn(self, params, T_curr, key):
        loss, key = self.compute_loss( params, key, n_integration_steps = self.n_integration_steps, n_states = self.batch_size, temp = T_curr, x_dim = self.EnergyClass.dim_x)
        return loss, key

    def update_step(self, params, opt_state, key, T_curr):
        return self.update_params(params, opt_state, key, T_curr)

    @partial(jax.jit, static_argnums=(0,))
    def update_params(self, params, opt_state, key, T_curr):
        (loss_value, out_dict), (grads,) = jax.value_and_grad(self.loss_fn, argnums=(0,), has_aux = True)(params, T_curr, key)
        updates, opt_state = self.optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        #p_ref_params = jax.tree_util.tree_map(lambda x, y: x - lam_p_ref * y, p_ref_params, grads_ref)
        return params, opt_state, loss_value, out_dict

    ### TODO move optimizers and so on here!
    def initialize_optimizer(self):
        l_start = 1e-10
        l_max = self.Optimizer_Config["lr"]
        overall_steps = self.Optimizer_Config["epochs"]*self.Optimizer_Config["steps_per_epoch"]*self.lr_factor
        warmup_steps = int(0.1 * overall_steps)

        self.schedule = lambda epoch: learning_rate_schedule(epoch, l_max, l_start, overall_steps, warmup_steps)
        #optimizer = optax.adam(self.schedule)
        optimizer = optax.chain(optax.clip_by_global_norm(1.0), optax.scale_by_radam(), optax.scale_by_schedule(lambda lr: -self.schedule(lr)))
        return optimizer

    def simulate_reverse_sde_scan(self, params, key, n_states = 100, n_integration_steps = 1000):
        return self.SDE_type.simulate_reverse_sde_scan(self.model, params, key, n_states = n_states, x_dim = self.EnergyClass.dim_x, n_integration_steps = n_integration_steps)

    def compute_loss(self,*args, **kwargs):
        """
        Calculate the loss between predictions and targets.

        :param predictions: The predicted values.
        :param targets: The ground truth values.
        :return: The calculated loss.
        """
        raise NotImplementedError("get_loss method not implemented")
    


    

def learning_rate_schedule(step, l_max = 1e-4, l_start = 1e-5, overall_steps = 1000, warmup_steps = 100):

    cosine_decay = optax.cosine_decay_schedule(init_value=l_max, decay_steps=overall_steps - warmup_steps)

    return jnp.where(step < warmup_steps, l_start + (l_max - l_start) * (step / warmup_steps), cosine_decay(step - warmup_steps))