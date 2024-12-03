import jax
from jax import random
from Networks import get_network
from EnergyFunctions import get_Energy_class
from AnnealSchedules import get_AnnealSchedule_class
from SDE_Losses import get_SDE_Loss_class
import jax.numpy as jnp
import wandb
import numpy as np
from tqdm.auto import trange
import time
import pickle
import os

class TrainerClass:
    def __init__(self, base_config):
        self.config = base_config

        AnnealConfig = base_config["Anneal_Config"]
        Energy_Config = base_config["EnergyConfig"]
        SDE_Loss_Config = base_config["SDE_Loss_Config"]
        self.Network_Config = base_config["Network_Config"]
        self.Optimizer_Config = base_config["Optimizer_Config"]
        self.model = get_network(self.Network_Config, SDE_Loss_Config)

        self.EnergyClass = get_Energy_class(Energy_Config)

        self._init_wandb()
        self.AnnealClass = get_AnnealSchedule_class(AnnealConfig)
        self.SDE_LossClass = get_SDE_Loss_class(SDE_Loss_Config, self.Optimizer_Config, self.EnergyClass, self.Network_Config, self.model)

        self.dim_x = self.EnergyClass.dim_x

        self.num_epochs = base_config["num_epochs"]
        self.n_integration_steps = SDE_Loss_Config["n_integration_steps"]
        self._init_Network()
        #self.EnergyClass.plot_properties()

    def _init_Network(self):
        x_init = jnp.ones((1,self.dim_x ))
        grad_init = jnp.ones((1,self.dim_x))
        Energy_value = jnp.ones((1,1))
        init_carry = jnp.zeros((1, self.Network_Config["n_hidden"]))
        in_dict = {"x": x_init, "Energy_value": Energy_value,  "t": jnp.ones((1,1)), "grads": grad_init, "hidden_state": [(init_carry, init_carry) for i in range(self.Network_Config["n_layers"])]}
        self.params = self.model.init(random.PRNGKey(0), in_dict, train = True)
        self.opt_state = self.SDE_LossClass.optimizer.init(self.params)

        num_params = sum(x.size for x in jax.tree_util.tree_leaves(self.params))
        wandb.log({"Network/num_params": num_params})

    def _init_wandb(self):
        wandb.init(project=f"DDS_{self.EnergyClass.__class__.__name__}_{self.config['project_name']}", config=self.config)
        self.wandb_id = wandb.run.name

    def train(self):

        params = self.params
        key = jax.random.PRNGKey(0)
        Best_Energy_value_ever = np.infty
        Best_Free_Energy_value_ever = np.infty

        pbar = trange(self.num_epochs)
        for epoch in pbar:
            start_time = time.time()
            if(epoch % self.Optimizer_Config["epochs_per_eval"] == 0 and epoch):
                n_samples = self.config["n_eval_samples"]
                SDE_tracer, out_dict, key = self.SDE_LossClass.simulate_reverse_sde_scan( params, self.SDE_LossClass.Energy_params, self.SDE_LossClass.SDE_params, key, n_integration_steps = self.n_integration_steps, n_states = n_samples)
                wandb.log({ f"eval/{key}": np.mean(out_dict[key]) for key in out_dict.keys()}, step=epoch+1)

                if(self.EnergyClass.config["name"] == "DoubleMoon"):
                    n_samples = 5
                    self.EnergyClass.visualize_models(out_dict["X_0"][0:n_samples])

                fig_traj = self.EnergyClass.plot_trajectories(np.array(SDE_tracer["ys"])[:,0:10,:])
                fig_hist = self.EnergyClass.plot_histogram(np.array(SDE_tracer["y_final"]))
                fig_last_samples = self.EnergyClass.plot_last_samples(np.array(SDE_tracer["y_final"]))
                figs = {"figs/best_trajectories": fig_traj, "figs/best_histogram": fig_hist, "figs/best_last_samples": fig_last_samples}
                Energy_values = self.SDE_LossClass.vmap_Energy_function(SDE_tracer["y_final"])

                self.check_improvement(params, Best_Energy_value_ever, np.min(Energy_values), "Energy", epoch=epoch)



            T_curr = self.AnnealClass.update_temp()
            loss_list = []
            for i in range(self.Optimizer_Config["steps_per_epoch"]):
                start_grad_time = time.time()
                params, self.opt_state, loss, out_dict = self.SDE_LossClass.update_step(params, self.opt_state, key, T_curr)
                end_grad_time = time.time() 
                key = out_dict["key"]	
                #print(out_dict)
                if not hasattr(self, 'aggregated_out_dict'):
                    self.aggregated_out_dict = {k: [] for k in out_dict.keys() if k != "key" and k != "X_0"}
                    self.aggregated_out_dict["time/time_grad"] = []
                    self.aggregated_out_dict["time/time_log"] = []

                for k, v in out_dict.items():
                    if k != "key" and k != "X_0":
                        #print("k", k, np.mean(v), i)
                        self.aggregated_out_dict[k].append(np.array(v))


                loss_list.append(float(loss))
                end_log_time = time.time()
                self.aggregated_out_dict["time/time_grad"].append(end_grad_time - start_grad_time)
                self.aggregated_out_dict["time/time_log"].append(end_log_time - end_grad_time)

            epoch_time = time.time() - start_time
            mean_loss = np.mean(loss_list)
            lr = self.SDE_LossClass.schedule(epoch*(self.Optimizer_Config["steps_per_epoch"]*self.SDE_LossClass.lr_factor)) ### TODO correct this for MC case
            wandb.log({"loss": mean_loss, "schedules/temp": T_curr, "schedules/lr": lr, "time/epoch": epoch_time, "epoch": epoch}, step=epoch+1)
            wandb.log({dict_key: np.mean(self.aggregated_out_dict[dict_key]) for dict_key in self.aggregated_out_dict}, step=epoch+1)
            wandb.log({"X_statistics/mean": np.mean(out_dict["X_0"]), "X_statistics/sdt": np.mean(np.std(out_dict["X_0"], axis = 0))}, step=epoch+1)

            pbar.set_description(f"mean_loss {mean_loss:.4f}, best energy: {Best_Energy_value_ever:.4f}")

            Free_Energy_values = np.mean(self.aggregated_out_dict["Free_Energy_at_T=1"])
            self.check_improvement(params, Best_Free_Energy_value_ever, Free_Energy_values, "Free_Energy_at_T=1", epoch, figs = None)

            del self.aggregated_out_dict
            #print({key: np.exp(dict_val) for key, dict_val in self.SDE_LossClass.SDE_params.items()})

        param_dict = self.load_params_and_config(filename="best_Free_Energy_at_T=1_checkpoint.pkl")

        self.SDE_LossClass.Energy_params = param_dict["Energy_params"]
        self.SDE_LossClass.SDE_params = param_dict["SDE_params"]

        n_samples = self.config["n_eval_samples"]
        SDE_tracer, out_dict, key = self.SDE_LossClass.simulate_reverse_sde_scan( params, self.SDE_LossClass.Energy_params, self.SDE_LossClass.SDE_params, key, n_integration_steps = self.n_integration_steps, n_states = n_samples)
        fig_traj = self.EnergyClass.plot_trajectories(np.array(SDE_tracer["ys"])[:,0:10,:], panel = "best_figs")
        fig_hist = self.EnergyClass.plot_histogram(np.array(SDE_tracer["y_final"]), panel = "best_figs")
        fig_last_samples = self.EnergyClass.plot_last_samples(np.array(SDE_tracer["y_final"]), panel = "best_figs")


        return params
    
    def check_improvement(self, params, best_metric_ever, metric, metric_name, epoch,figs = None):
        best_metric_value = metric
        if(best_metric_value < best_metric_ever):
            best_metric_ever = best_metric_value

            param_dict = self.SDE_LossClass.get_param_dict(params)
            self.save_params_and_config(param_dict, self.config, filename=f"best_{metric_name}_checkpoint.pkl")
            if(figs != None):
                wandb.log(figs, step=epoch+1)

            wandb.log({f"Best_{metric_name}": best_metric_ever},step=epoch+1)

    def load_params_and_config(self, filename="params_and_config.pkl"):
        script_dir = os.path.dirname(os.path.abspath(__file__)) + "Checkpoints/" + self.wandb_id + "/"
        file_path = script_dir + filename

        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"No such file: '{file_path}'")

        with open(file_path, "rb") as f:
            data = pickle.load(f)

        return data["params"]

    def save_params_and_config(self, params, config, filename="params_and_config.pkl"):
        script_dir = os.path.dirname(os.path.abspath(__file__)) + "Checkpoints/" + self.wandb_id + "/"

        if not os.path.isdir(script_dir):
            os.makedirs(script_dir)

        data = {
            "params": params,
            "config": config
        }
        with open(script_dir + filename, "wb") as f:
            pickle.dump(data, f)

