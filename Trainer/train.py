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
from utils.rotate_vector import rotate_vector

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

        if(self.EnergyClass.invariance):
            self._test_invariance()
        #self.EnergyClass.plot_properties()

    def _init_Network(self):
        x_init = jnp.ones((1,self.dim_x ,))
        grad_init = jnp.ones((1,self.dim_x,))
        Energy_value = jnp.ones((1,1,))
        init_carry = jnp.zeros(( 1, self.Network_Config["n_hidden"],))
        in_dict = {"x": x_init, "Energy_value": Energy_value,  "t": jnp.ones((1, 1,)), "grads": grad_init, "hidden_state": [(init_carry, init_carry) for i in range(self.Network_Config["n_layers"])]}
        self.params = self.model.init(random.PRNGKey(self.Network_Config["model_seed"]), in_dict, train = True)
        self.opt_state = self.SDE_LossClass.optimizer.init(self.params)

        num_params = sum(x.size for x in jax.tree_util.tree_leaves(self.params))
        wandb.log({"Network/num_params": num_params})

        ### TODO write code that checks equivariance here!

    def _test_invariance(self):
        key = jax.random.PRNGKey(0)
        batch_size = 1
        x_init = 2*jax.random.normal(key, (batch_size,self.dim_x ))
        
        x_init_resh = x_init.reshape((batch_size, self.EnergyClass.n_particles, self.EnergyClass.particle_dim))
        x_COM = jnp.mean(x_init_resh, axis = 1, keepdims=True)
        x_centered_resh = x_init_resh - x_COM

        rotations = [np.pi, np.pi/2, np.pi/4]
        rotated_scores = []
        unrotated_scores = []
        unrotated_final_samples = []
        xs_list = []

        for rot in rotations:
            print("Rotation", rot, "is going on")
            x_centered_rot = jax.vmap(jax.vmap(rotate_vector, in_axes=(0, None)), in_axes=(0, None))(x_centered_resh, rot)
            x_centered_resh_rot =  x_centered_rot.reshape((batch_size, self.dim_x)) 

            vmap_energy, vmap_grad, key = self.SDE_LossClass.SDE_type.vmap_prior_target_grad_interpolation(x_centered_resh_rot, 0., self.SDE_LossClass.Energy_params, self.SDE_LossClass.SDE_params, key)

            grad_init = vmap_grad
            Energy_value = vmap_energy
            init_carry = jnp.zeros(( batch_size, self.Network_Config["n_hidden"],))
            in_dict = {"x": x_centered_resh_rot, "Energy_value": Energy_value,  "t": jnp.ones((batch_size, 1,)), "grads": grad_init, "hidden_state": [(init_carry, init_carry) for i in range(self.Network_Config["n_layers"])]}
            out_dict = self.model.apply(self.params, in_dict, train = True)
            score = out_dict["score"]

            x_new = x_centered_resh_rot + score
            x_new_resh = x_new.reshape((batch_size, self.EnergyClass.n_particles, self.EnergyClass.particle_dim))
            print("COM", jnp.mean(jnp.mean(x_new_resh, axis = 1, keepdims=True)))
            print("COM x_in", jnp.mean(jnp.mean(x_centered_resh_rot, axis = 1, keepdims=True)))

            rotated_scores.append(score)

            resh_scores = score.reshape((batch_size, self.EnergyClass.n_particles, self.EnergyClass.particle_dim))
            unrotated_score = jax.vmap(jax.vmap(rotate_vector, in_axes=(0, None)), in_axes=(0, None))(resh_scores, -rot)
            unrotated_scores.append(unrotated_score)


            # x_final, SDE_tracker_steps = self.SDE_LossClass.SDE_type.simulated_SDE_from_x(self.model, self.params, self.SDE_LossClass.Energy_params, self.SDE_LossClass.SDE_params, x_centered_resh_rot, key, n_integration_steps = self.n_integration_steps)
            # xs = SDE_tracker_steps["xs"]
            # print("xs",xs.shape)

            # xs_resh = xs.reshape((self.n_integration_steps, self.EnergyClass.n_particles, self.EnergyClass.particle_dim))
            # unrotated_resh_xs_resh = jax.vmap(jax.vmap(rotate_vector, in_axes=(0, None)), in_axes=(0, None))(xs_resh, -rot)
            # xs_list.append(unrotated_resh_xs_resh)
            
            # resh_x_final = x_final.reshape((batch_size, self.EnergyClass.n_particles, self.EnergyClass.particle_dim))
            # unrotated_resh_x_final = jax.vmap(jax.vmap(rotate_vector, in_axes=(0, None)), in_axes=(0, None))(resh_x_final, -rot)
            # unrotated_final_samples.append(unrotated_resh_x_final)
        if(False):
            import matplotlib.pyplot as plt

            # Assuming `xs_list` is a list of data arrays and `rot` is predefined
            fig, axs = plt.subplots(len(xs_list), 1, figsize=(8, 4 * len(xs_list)))  # Create subplots

            if len(xs_list) == 1:
                axs = [axs]  # Handle the case when there's only one subplot

            for idx, xs in enumerate(xs_list):
                ax = axs[idx]  # Get the subplot axis
                for i in range(xs.shape[1]):  # Assuming `xs` has a shape of `[n, m, 2]`
                    ax.plot(xs[:, i, 0], xs[:, i, 1], label=f'Particle {i}')
                ax.set_title(f'Trajectory of Particles {rot} (Dataset {idx})')  # Use `rot` and `idx`
                ax.set_xlabel('X')
                ax.set_ylabel('Y')
                ax.legend()

            fig.tight_layout()  # Adjust layout to avoid overlap
            wandb.log({f"fig/Trajectory Plot {rot}": wandb.Image(fig)})  # Log the single figure with subplots to Weights & Biases
            plt.close(fig)  # Close the figure after logging

        print("output scores (should be different)")
        for el in rotated_scores:
            print(el.reshape((batch_size, self.EnergyClass.n_particles, self.EnergyClass.particle_dim)))

        print("backrotated scores (should be invariant)")
        for el in unrotated_scores:
            print(el)


        print("Unrotated final samples")
        for el in unrotated_final_samples:
            print(el)

        #raise ValueError("Check if scores are invariant to rotations")


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
            if(epoch % int(self.num_epochs/self.Optimizer_Config["epochs_per_eval"]) == 0 and epoch):
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
            ### TODO SDE paramters should also be saved!
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

