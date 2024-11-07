import jax
import optax
from jax import random, grad, jit
from typing import Any, Callable
from Networks import get_network
### TODO implement energy function registry
from EnergyFunctions import get_Energy_class
from AnnealSchedules import get_AnnealSchedule_class
from SDE_Losses import get_SDE_Loss_class
from functools import partial
import jax.numpy as jnp
import wandb
import numpy as np
from tqdm.auto import trange
import time
import sys

class TrainerClass:
    def __init__(self, base_config):
        self.config = base_config
        self._init_wandb()
        AnnealConfig = base_config["Anneal_Config"]
        Energy_Config = base_config["EnergyConfig"]
        SDE_Loss_Config = base_config["SDE_Loss_Config"]
        Network_Config = base_config["Network_Config"]
        self.Optimizer_Config = base_config["Optimizer_Config"]
        self.EnergyClass = get_Energy_class(Energy_Config)
        self.AnnealClass = get_AnnealSchedule_class(AnnealConfig)

        
        self.model = get_network(Network_Config, SDE_Loss_Config)
        self.SDE_LossClass = get_SDE_Loss_class(SDE_Loss_Config, self.Optimizer_Config, self.EnergyClass, Network_Config, self.model)
        
        self.dim_x = self.EnergyClass.dim_x
        self.num_epochs = base_config["num_epochs"]
        self.n_integration_steps = SDE_Loss_Config["n_integration_steps"]
        x_init = jnp.ones((1,self.dim_x))
        init_carry = jnp.zeros((1, Network_Config["n_hidden"]))
        in_dict = {"x": x_init, "t": jnp.ones((1,1)), "grads": x_init, "hidden_state": [(init_carry, init_carry) for i in range(Network_Config["n_layers"])]}
        self.params = self.model.init(random.PRNGKey(0), in_dict)
        self.opt_state = self.SDE_LossClass.optimizer.init(self.params)
        #self._init_wandb()
        self.EnergyClass.plot_properties()

    def _init_wandb(self):
        wandb.init(config=self.config, settings=wandb.Settings(console="wrap"))
        command = " ".join(sys.argv)
        wandb.log({"run_command": command})

    def train(self):

        params = self.params
        key = jax.random.PRNGKey(0)
        Best_Energy_value_ever = np.infty

        pbar = trange(self.num_epochs)
        for epoch in pbar:
            if(epoch % 10 == 0):
                n_samples = self.config["n_eval_samples"]
                SDE_tracer, key = self.SDE_LossClass.simulate_reverse_sde_scan( params, key, n_integration_steps = self.n_integration_steps, n_states = n_samples)
                self.EnergyClass.plot_trajectories(np.array(SDE_tracer["xs"])[:,0:10,:])
                self.EnergyClass.plot_histogram(np.array(SDE_tracer["x_final"])[:,:])
                self.EnergyClass.plot_last_samples(np.array(SDE_tracer["x_final"]))
                Energy_values = self.SDE_LossClass.vmap_calc_Energy(SDE_tracer["x_final"])
                best_Energy_value = np.min(Energy_values)
                if(best_Energy_value < Best_Energy_value_ever):
                    Best_Energy_value_ever = best_Energy_value
                    #print("New best Energy value found", Best_Energy_value_ever)
                wandb.log({"Best_Energy_value": Best_Energy_value_ever})


            T_curr = self.AnnealClass.update_temp()
            loss_list = []
            for i in range(self.Optimizer_Config["steps_per_epoch"]):
                params, self.opt_state, loss, out_dict = self.SDE_LossClass.update_step(params, self.opt_state, key, T_curr)
                key = out_dict["key"]	
                #print(out_dict)
                if not hasattr(self, 'aggregated_out_dict'):
                    self.aggregated_out_dict = {k: [] for k in out_dict.keys() if k != "key" and k != "X_0"}

                for k, v in out_dict.items():
                    if k != "key" and k != "X_0":
                        #print("k", k, np.mean(v), i)
                        self.aggregated_out_dict[k].append(numpy.array(v))

                loss_list.append(float(loss))
            mean_loss = np.mean(loss_list)
            lr = self.SDE_LossClass.schedule(epoch*(self.Optimizer_Config["steps_per_epoch"]*self.SDE_LossClass.lr_factor)) ### TODO correct this for MC case
            wandb.log({"loss": mean_loss, "schedules/temp": T_curr, "schedules/lr": lr, "epoch": epoch})
            wandb.log({dict_key: np.mean(self.aggregated_out_dict[dict_key]) for dict_key in self.aggregated_out_dict})
            wandb.log({"X_statistics/abs_mean": np.mean(np.sqrt(np.sum(out_dict["X_0"]**2, axis = -1))), "X_statistics/mean": np.mean(np.mean(out_dict["X_0"], axis = -1))})
            pbar.set_description(f"mean_loss {mean_loss:.4f}, best energy: {Best_Energy_value_ever:.4f}")
            del self.aggregated_out_dict
            
        return params

