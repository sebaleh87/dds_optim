import jax
import optax
from jax import random, grad, jit
from typing import Any, Callable
from Networks.FeedForward import FeedForwardNetwork
### TODO implement energy function registry
from EnergyFunctions import get_Energy_class
from AnnealSchedules import get_AnnealSchedule_class
from SDE_Losses import get_SDE_Loss_class
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
        self.Optimizer_Config = base_config["Optimizer_Config"]
        self.dim_x = Energy_Config["dim_x"]

        self.EnergyClass = get_Energy_class(Energy_Config)
        self.AnnealClass = get_AnnealSchedule_class(AnnealConfig)
        self.SDE_LossClass = get_SDE_Loss_class(SDE_Loss_Config, self.Optimizer_Config, self.EnergyClass, self.model)

        self.num_epochs = base_config["num_epochs"]
        self.n_integration_steps = SDE_Loss_Config["n_integration_steps"]
        self.params = self.model.init(random.PRNGKey(0), jnp.ones((1,self.dim_x)), jnp.ones((1,1)))
        self.opt_state = self.SDE_LossClass.optimizer.init(self.params)
        self._init_wandb()
        self.EnergyClass.plot_properties()

    def _init_wandb(self):
        wandb.init(project=f"DDS_{self.EnergyClass.__class__.__name__}", config=self.config)

    def train(self):

        params = self.params
        key = jax.random.PRNGKey(0)
        for epoch in range(self.num_epochs):
            SDE_tracer, key = self.SDE_LossClass.simulate_reverse_sde_scan( params, key, n_integration_steps = self.n_integration_steps, n_states = 500*2000)
            self.EnergyClass.plot_trajectories(np.array(SDE_tracer["xs"])[:,0:10,:])
            self.EnergyClass.plot_histogram(np.array(SDE_tracer["xs"])[-1,:,:])
            self.EnergyClass.plot_last_samples(np.array(SDE_tracer["xs"])[-1,:,:])


            T_curr = self.AnnealClass.update_temp()
            loss_list = []
            for i in range(self.Optimizer_Config["steps_per_epoch"]):
                params, self.opt_state, loss, out_dict = self.SDE_LossClass.update(params, self.opt_state, key, T_curr)
                key = out_dict["key"]	

                wandb.log({key: out_dict[key] for key in out_dict.keys() if (key != "key" and key != "X_0")})
                wandb.log({"X_statistics/abs_mean": np.mean(np.sqrt(np.sum(out_dict["X_0"]**2, axis = -1))), "X_statistics/mean": np.mean(np.mean(out_dict["X_0"], axis = -1))})
                loss_list.append(float(loss))
            mean_loss = np.mean(loss_list)
            lr = self.SDE_LossClass.schedule(epoch*(self.Optimizer_Config["steps_per_epoch"]))
            wandb.log({"loss": mean_loss, "schedules/temp": T_curr, "schedules/lr": lr})
            print(f"Epoch {epoch + 1} completed")
            print(mean_loss)

        return params

