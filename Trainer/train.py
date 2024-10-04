import jax
import optax
from jax import random, grad, jit
from typing import Any, Callable
from Networks.FeedForward import FeedForwardNetwork
### TODO implement energy function registry
from EnergyFunctions import get_Energy_class
from AnnealSchedules import get_AnnealSchedule_class
from SDE_Losses import get_SDE_Loss_class
from SDE_Losses.VPSDE import VPSDEClass
from functools import partial
import jax.numpy as jnp
import wandb
import numpy as np

class TrainerClass:
    def __init__(self, base_config):
        self.config = base_config

        ### TODO also make network registry
        self.model = FeedForwardNetwork(n_layers=base_config["n_layers"], hidden_dim=base_config["n_hidden"])

        AnnealConfig = base_config["Anneal_Config"]
        Energy_Config = base_config["EnergyConfig"]
        SDE_Loss_Config = base_config["SDE_Loss_Config"]
        self.dim_x = Energy_Config["dim_x"]

        self.EnergyClass = get_Energy_class(Energy_Config)
        self.AnnealClass = get_AnnealSchedule_class(AnnealConfig)
        self.SDE_LossClass = get_SDE_Loss_class(SDE_Loss_Config, self.EnergyClass, self.model)

        self.learning_rate = base_config["lr"]
        self.num_epochs = base_config["num_epochs"]
        self.batch_size = base_config["batch_size"]
        self.n_integration_steps = base_config["n_integration_steps"]

        self.optimizer = self.initialize_optimizer()
        self.params = self.model.init(random.PRNGKey(0), jnp.ones((1,self.dim_x)), jnp.ones((1,1)))
        self.opt_state = self.optimizer.init(self.params)
        self._init_wandb()
        self.EnergyClass.plot_properties()

    def _init_wandb(self):
        wandb.init(project=f"DDS_{self.EnergyClass.__class__.__name__}", config=self.config)

    # def _update_temp(self, epoch):
    #     if(self.config["anneal_mode"] == "linear"):
    #         T_curr = self.T_start - self.T_start/(self.num_epochs)*epoch
    #     elif(self.config["anneal_mode"]  == "exponential"):
    #         T_curr = self.T_start*(1/(1-self.anneal_decay**(epoch+1)))
    #     return T_curr

    def initialize_optimizer(self):
        l_start = 1e-10
        l_max = self.learning_rate
        overall_steps = self.num_epochs*self.config["steps_per_epoch"]
        warmup_steps = int(0.1 * overall_steps)

        self.schedule = lambda epoch: learning_rate_schedule(epoch, l_max, l_start, overall_steps, warmup_steps)
        optimizer = optax.adam(self.schedule)
        return optimizer

    def loss_fn(self, params, T_curr, key):
        loss, key = self.SDE_LossClass.compute_loss( params, key, n_integration_steps = self.n_integration_steps, n_states = self.batch_size, temp = T_curr, x_dim= self.dim_x)
        return loss, key

    @partial(jit, static_argnums=(0,))
    def update(self, params, opt_state, key, T_curr, lam_p_ref = 0.001):
        (loss_value, out_dict), (grads, grads_ref) = jax.value_and_grad(self.loss_fn, argnums=(0, 1), has_aux = True)(params, T_curr, key)
        updates, opt_state = self.optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        #p_ref_params = jax.tree_util.tree_map(lambda x, y: x - lam_p_ref * y, p_ref_params, grads_ref)
        return params, opt_state, loss_value, out_dict

    def train(self):

        params = self.params
        key = jax.random.PRNGKey(0)
        for epoch in range(self.num_epochs):
            SDE_tracer, key = self.SDE_LossClass.simulate_reverse_sde_scan( params, key, n_integration_steps = self.n_integration_steps, n_states = 500*2000, x_dim = self.dim_x)
            self.EnergyClass.plot_trajectories(np.array(SDE_tracer["xs"])[:,0:10,:])
            self.EnergyClass.plot_histogram(np.array(SDE_tracer["xs"])[-1,:,:])
            self.EnergyClass.plot_last_samples(np.array(SDE_tracer["xs"])[-1,:,:])


            T_curr = self.AnnealClass.update_temp()
            loss_list = []
            for i in range(self.config["steps_per_epoch"]):
                params, self.opt_state, loss, out_dict = self.update(params, self.opt_state, key, T_curr)
                key = out_dict["key"]	

                wandb.log({key: out_dict[key] for key in out_dict.keys() if (key != "key" and key != "X_0")})
                wandb.log({"X_statistics/abs_mean": np.mean(np.sqrt(np.sum(out_dict["X_0"]**2, axis = -1))), "X_statistics/mean": np.mean(np.mean(out_dict["X_0"], axis = -1))})
                loss_list.append(float(loss))
            mean_loss = np.mean(loss_list)
            lr = self.schedule(epoch*(self.config["steps_per_epoch"]))
            wandb.log({"loss": mean_loss, "schedules/temp": T_curr, "schedules/lr": lr})
            print(f"Epoch {epoch + 1} completed")
            print(mean_loss)

        return params

def learning_rate_schedule(step, l_max = 1e-4, l_start = 1e-5, overall_steps = 1000, warmup_steps = 100):

    cosine_decay = optax.cosine_decay_schedule(init_value=l_max, decay_steps=overall_steps - warmup_steps)

    return jnp.where(step < warmup_steps, l_start + (l_max - l_start) * (step / warmup_steps), cosine_decay(step - warmup_steps))
