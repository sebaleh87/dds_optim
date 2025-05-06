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
from matplotlib import pyplot as plt
from Metrics.optimal_transport import SD, Sinkhorn  # Import the Sinkhorn divergence class

class TrainerClass:
    def __init__(self, base_config, mode = "train"):
        self.config = base_config
        self.mode = mode

        AnnealConfig = base_config["Anneal_Config"]
        Energy_Config = base_config["EnergyConfig"]
        self.SDE_Loss_Config = base_config["SDE_Loss_Config"]
        self.Network_Config = base_config["Network_Config"]
        self.Optimizer_Config = base_config["Optimizer_Config"]
        self.model = get_network(self.Network_Config, self.SDE_Loss_Config)

        # Define which metrics should be maximized (True) or minimized (False)
        self.metric_objectives = {
            "EUBO_at_T=1": False,
            "n_eff": True,      # Higher is better for effective sample size
            "sinkhorn_divergence": False,  # Lower is better for divergences
            "Free_Energy_at_T=1": False,    # Lower is better for free energy
            "Energy": False,     # Lower is better for energy
            "ELBO_at_T=1": True,
            "Sinkhorn": False    # Lower is better for Sinkhorn divergence
        }

        self.EnergyClass = get_Energy_class(Energy_Config)

        # Generate ground truth samples for SD metric if available
        if hasattr(self.EnergyClass, 'has_tractable_distribution') and self.EnergyClass.has_tractable_distribution:
            n_samples = [self.config["n_eval_samples"]]
            print("Warning: samples have to be rescaled for GMMDistraxRandom")
            
            key = jax.random.PRNGKey(self.config["sample_seed"])
            reps = 1
            for n_sample in n_samples:
                distances = []
                for rep in range(reps):
                    self.n_sinkhorn_samples = n_sample
                    start_sample_time = time.time()
                    key, subkey =  jax.random.split(key)
                    model_samples = self.EnergyClass.generate_samples(subkey, self.n_sinkhorn_samples)
                    end_sample_time = time.time()
                    #self.sd_calculator = SD(self.EnergyClass, n_sample, key, epsilon=1e-3)
                    self.sd_calculator = Sinkhorn(self.EnergyClass, n_sample, key, epsilon=1e-3)
                    start_time = time.time()
                    #distance = self.sd_calculator.compute_SD(model_samples)
                    distance,_,_ = self.sd_calculator.compute_SD(model_samples)
                    end_time = time.time()
                    print("sample time", end_sample_time - start_sample_time)
                    print("time needed", end_time - start_time)
                    distances.append(distance)

                avg_distance = np.mean(distances)
                std_distance = np.std(distances)
                print(f"Average distance for {n_sample} samples: {avg_distance}, Std: {std_distance}")


        self._init_wandb()
        self.AnnealClass = get_AnnealSchedule_class(AnnealConfig)
        self.SDE_LossClass = get_SDE_Loss_class(self.SDE_Loss_Config, self.Optimizer_Config, self.EnergyClass, self.Network_Config, self.model)

        self.dim_x = self.EnergyClass.dim_x

        self.num_epochs = base_config["num_epochs"]
        self.n_integration_steps = self.SDE_Loss_Config["n_integration_steps"]
        self._init_Network()

        #if(self.EnergyClass.invariance):
            #self._test_invariance()
        #self.EnergyClass.plot_properties()

    def _init_Network(self):
        x_init = jnp.ones((1,self.dim_x ))
        grad_init = jnp.ones((1,self.dim_x))
        Energy_value = jnp.ones((1,1))
        if(self.Network_Config["name"] != "ADAMNetwork"):
            init_carry = jnp.zeros(( 1, self.Network_Config["n_hidden"],))
        else:
            init_carry = jnp.zeros(( 1, self.dim_x,))


        ###TODO if energy value and grads are not used it should not allocate parameters!!!!
        use_normal = self.SDE_Loss_Config["SDE_Type_Config"]["use_normal"]
        if(use_normal):
            in_dict = {"x": x_init,  "t": jnp.ones((1, 1,)), "grads": grad_init}
        else:
            in_dict = {"x": x_init, "Energy_value": Energy_value,  "t": jnp.ones((1, 1,)), "grads": grad_init, "hidden_state": [(init_carry, init_carry) for i in range(self.Network_Config["n_layers"])]}
        
        if(self.Network_Config["model_mode"] == "latent"):
            in_dict["z"] = jnp.ones((1,self.Network_Config["latent_dim"] ))
        
        self.params = self.model.init(random.PRNGKey(self.Network_Config["model_seed"]), in_dict, train = True, forw_mode = "init")
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

            vmap_energy, vmap_grad, key = self.SDE_LossClass.SDE_type.vmap_prior_target_grad_interpolation(x_centered_resh_rot, 0., self.SDE_LossClass.Interpol_params, self.SDE_LossClass.SDE_params, key)

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
        if(self.mode == "train"):
            wandb.init(project=f"DDS_{self.EnergyClass.config['name']}_{self.config['project_name']}_dim_{self.EnergyClass.dim_x}", config=self.config)
            self.wandb_id = wandb.run.name

    def plot_figures(self, SDE_tracer, epoch, sample_mode = "train"):
        overall_dict = {}
        fig_traj_dict = self.EnergyClass.plot_trajectories(np.array(SDE_tracer["ys"])[:,0:10,:])
        fig_hist_dict = self.EnergyClass.plot_histogram(np.array(SDE_tracer["y_final"]))
        fig_last_samples_dict = self.EnergyClass.plot_last_samples(np.array(SDE_tracer["y_final"]))

        if(fig_traj_dict != None):
            overall_dict.update(fig_traj_dict)
        if(fig_hist_dict != None):
            overall_dict.update(fig_hist_dict)
        if(fig_last_samples_dict != None):
            overall_dict.update(fig_last_samples_dict)

        wandb.log({f"figs_{sample_mode}/{key}": overall_dict[key] for key in overall_dict},step=epoch+1)

    def train(self):
        params = self.params
        key = jax.random.PRNGKey(self.Network_Config["model_seed"])
        Best_Energy_value_ever = np.infty
        Best_Free_Energy_value_ever = np.infty
        Best_Sinkhorn_value_ever = np.infty

        # Initialize metric history dictionary
        metric_history = {}
        best_running_avgs = {}  # Track best running averages
        
        save_metric_dict = {"Free_Energy_at_T=1": [], "ELBO_at_T=1": [], "log_Z_at_T=1": [],  "n_eff": [], "epoch": [], "EUBO_at_T=1": []}
        if hasattr(self, 'sd_calculator'):
            save_metric_dict["sinkhorn_divergence"] = []
        
        if hasattr(self.EnergyClass, 'compute_emc'):
            save_metric_dict["EMC"] = []

        pbar = trange(self.num_epochs)
        for epoch in pbar:
            T_curr = self.AnnealClass.update_temp()
            start_time = time.time()
            if((epoch % int(self.num_epochs/self.Optimizer_Config["epochs_per_eval"]) == 0 or epoch == 0) and not self.config["disable_jit"]):
                sampling_modes = ["eval"]
                for sample_mode in sampling_modes:
                    n_samples = self.config["n_eval_samples"]
                    SDE_tracer, out_dict, key = self.SDE_LossClass.simulate_reverse_sde_scan( params, self.SDE_LossClass.Interpol_params, self.SDE_LossClass.SDE_params, T_curr, key, sample_mode = sample_mode, n_integration_steps = self.n_integration_steps, n_states = n_samples)
                    
                    # Initialize metrics dict for this evaluation
                    eval_metrics = {}

                    # Store metrics in history and collect for logging
                    for metric_name, value in out_dict.items():
                        full_metric_name = f"eval_{sample_mode}/{metric_name}"
                        if full_metric_name not in metric_history:
                            metric_history[full_metric_name] = []
                        if isinstance(value, (float, np.ndarray, jnp.ndarray)):
                            metric_history[full_metric_name].append(float(np.mean(value)))
                            eval_metrics[f"eval_{sample_mode}/{metric_name}"] = np.mean(value)

                    # Calculate Sinkhorn divergence if the energy model has a tractable distribution
                    if hasattr(self, 'sd_calculator'):
                        model_samples = out_dict["X_0"][0:self.n_sinkhorn_samples]
                        distance,_,_ = self.sd_calculator.compute_SD(model_samples)
                        
                        # Store Sinkhorn metrics
                        sd_metric_name = f"eval_{sample_mode}/sinkhorn_divergence"
                        if sd_metric_name not in metric_history:
                            metric_history[sd_metric_name] = []
                        metric_history[sd_metric_name].append(float(distance))
                        
                        # Add Sinkhorn to metrics dict
                        eval_metrics[sd_metric_name] = distance

                        # Save model if this is the best Sinkhorn divergence so far
                        if sample_mode == "eval":  # Only save on eval mode
                            Best_Sinkhorn_value_ever = self.check_improvement(params, Best_Sinkhorn_value_ever, distance, "Sinkhorn", epoch=epoch)
                            save_metric_dict["sinkhorn_divergence"].append(distance)

                        if hasattr(self.EnergyClass, 'compute_emc'):
                            emc = self.EnergyClass.compute_emc(out_dict["X_0"])
                            emc_metric_name = f"eval_{sample_mode}/EMC"
                            if emc_metric_name not in metric_history:
                                metric_history[emc_metric_name] = []
                            metric_history[emc_metric_name].append(float(emc))
                            eval_metrics[emc_metric_name] = emc

                            if sample_mode == "eval": 
                                save_metric_dict["EMC"].append(emc)

                    # Calculate running averages for all metrics
                    for metric_name, values in metric_history.items():
                        if len(values) > 0:
                            # Get current value (most recent)
                            current_value = values[-1]
                            eval_metrics[f"{metric_name}"] = current_value
                            
                            # Calculate running average of last 5 values
                            last_n = values[-5:] if len(values) >= 5 else values
                            running_avg = sum(last_n) / len(last_n)
                            eval_metrics[f"{metric_name}_running_avg"] = running_avg
                            
                            # Get the base metric name without the eval_mode prefix
                            base_metric_name = metric_name.split('/')[-1] if '/' in metric_name else metric_name
                            
                            # Update best running average based on whether metric should be maximized or minimized
                            should_maximize = self.metric_objectives.get(base_metric_name, True)
                            if metric_name not in best_running_avgs:
                                best_running_avgs[metric_name] = running_avg
                            elif should_maximize and running_avg > best_running_avgs[metric_name]:
                                best_running_avgs[metric_name] = running_avg
                            elif not should_maximize and running_avg < best_running_avgs[metric_name]:
                                best_running_avgs[metric_name] = running_avg

                    # Log all metrics at once
                    wandb.log(eval_metrics, step=epoch+1)

                    self.plot_figures(SDE_tracer, epoch, sample_mode = sample_mode)

                

                # Only save based on Energy if we don't have a tractable distribution
                if not hasattr(self, 'sd_calculator'):
                    Energy_values = self.SDE_LossClass.vmap_Energy_function(SDE_tracer["y_final"])
                    Best_Free_Energy_value_ever = self.check_improvement(params, Best_Free_Energy_value_ever, np.min(Energy_values), "Energy", epoch=epoch)

                if("beta_interpol_params" in self.SDE_LossClass.Interpol_params.keys()):
                    beta_interpol_params = self.SDE_LossClass.Interpol_params["beta_interpol_params"]
                    steps = np.arange(0,len(beta_interpol_params) +1)

                    interpol_time = [self.SDE_LossClass.SDE_type.compute_energy_interpolation_time(self.SDE_LossClass.Interpol_params, t, SDE_param_key = "beta_interpol_params") for t in range(len(beta_interpol_params) + 1)]

                    fig, ax = plt.subplots()

                    ax.plot(steps, interpol_time, label='Beta Interpolation Parameters')
                    ax.set_xlabel('Steps')
                    ax.set_ylabel('Beta Interpolation Parameters')
                    ax.set_title('Beta Interpolation Parameters over Steps')
                    ax.legend()
                    wandb.log({"fig/Beta_Interpolation_Parameters": wandb.Image(fig)},step=epoch+1)
                    plt.close(fig)

                if("repulsion_interpol_params" in self.SDE_LossClass.Interpol_params.keys()):
                    #beta_interpol_params = self.SDE_LossClass.SDE_params["repulsion_interpol_params"]
                    steps = np.arange(0,len(beta_interpol_params) +1)

                    interpol_time = np.array([self.SDE_LossClass.SDE_type.compute_energy_interpolation_time(self.SDE_LossClass.Interpol_params, t, SDE_param_key = "repulsion_interpol_params") for t in range(len(beta_interpol_params) + 1)])

                    fig, ax = plt.subplots()

                    ax.plot(steps, interpol_time*(1-interpol_time), label='repulsion_interpol_params')
                    ax.set_xlabel('Steps')
                    ax.set_ylabel('repulsion_interpol_params')
                    ax.set_title('repulsion_interpol_params over Steps')
                    ax.legend()
                    wandb.log({"fig/repulsion_interpol_params": wandb.Image(fig)},step=epoch+1)
                    plt.close(fig)

                if("log_beta_over_time" in self.SDE_LossClass.SDE_params.keys()):
                    #beta_interpol_params = self.SDE_LossClass.SDE_params["repulsion_interpol_params"]
                    sigma = jnp.exp(self.SDE_LossClass.SDE_params["log_sigma"])
                    beta_per_step = sigma[None, :]*jnp.exp(self.SDE_LossClass.SDE_params["log_beta_over_time"])
                    #print(beta_per_step.shape)
                    steps = np.arange(0,len(beta_per_step) )
                    fig, ax = plt.subplots()

                    ax.plot(steps, beta_per_step, label='beta_per_step')
                    ax.set_xlabel('T = 0 (target samples) T = T prior')
                    ax.set_ylabel('beta_over_time')
                    ax.set_title('beta_over_time over Steps')
                    ax.legend()
                    wandb.log({"fig/beta_over_time": wandb.Image(fig)},step=epoch+1)
                    plt.close(fig)

                    
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
            for save_key in save_metric_dict.keys():
                if(save_key in self.aggregated_out_dict.keys()):
                    save_metric_dict[save_key].append(np.mean(self.aggregated_out_dict[save_key]))

            save_metric_dict["epoch"].append(epoch)
            self.save_metric_curves(save_metric_dict)

            # We don't need this section anymore as model saving is handled by either Sinkhorn or Energy checks
            del self.aggregated_out_dict
            #print({key: np.exp(dict_val) for key, dict_val in self.SDE_LossClass.SDE_params.items()})

        param_dict = self.SDE_LossClass.get_param_dict(params)
        self.save_params_and_config(param_dict, self.config, filename="params_and_config_train_end.pkl") ### save parameters at the end of training
        # Load the appropriate checkpoint based on whether we have a tractable distribution
        try:
            checkpoint_filename = "best_Sinkhorn_checkpoint.pkl" if hasattr(self, 'sd_calculator') else "best_Free_Energy_at_T=1_checkpoint.pkl"
            param_dict = self.load_params_and_config(filename=checkpoint_filename)

            self.SDE_LossClass.Interpol_params = param_dict["Interpol_params"]
            self.SDE_LossClass.SDE_params = param_dict["SDE_params"]

            n_samples = self.config["n_eval_samples"]
            SDE_tracer, out_dict, key = self.SDE_LossClass.simulate_reverse_sde_scan( params, self.SDE_LossClass.Interpol_params, self.SDE_LossClass.SDE_params, T_curr, key, sample_mode = "eval", n_integration_steps = self.n_integration_steps, n_states = n_samples)
            
            self.plot_figures(SDE_tracer, epoch)
        except:
            pass

        # After training loop, calculate and log running averages
        running_avg_table = self._calculate_running_averages(metric_history, best_running_avgs)
        wandb.log({"final_metrics": running_avg_table},step=epoch+1)
        wandb.finish()
        return params
    
    def check_improvement(self, params, best_metric_ever, metric, metric_name, epoch, figs = None):
        current_metric_value = metric  # Renamed for clarity
        base_metric_name = metric_name.split('/')[-1] if '/' in metric_name else metric_name
        
        should_maximize = self.metric_objectives[base_metric_name] 
        
        # Check if this is an improvement
        is_improvement = False
        if epoch == 0:
            is_improvement = True
            best_metric_ever = current_metric_value
        elif should_maximize and current_metric_value > best_metric_ever:
            is_improvement = True
            best_metric_ever = current_metric_value
        elif not should_maximize and current_metric_value < best_metric_ever:
            is_improvement = True
            best_metric_ever = current_metric_value

        if is_improvement:
            param_dict = self.SDE_LossClass.get_param_dict(params)
            self.save_params_and_config(param_dict, self.config, filename=f"best_{metric_name}_checkpoint.pkl")
            if figs is not None:
                wandb.log(figs)

            wandb.log({f"Best_{metric_name}": best_metric_ever}, step=epoch+1)
        
        return best_metric_ever  # Return the updated best value

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


    def _calculate_running_averages(self, metric_history, best_running_avgs, window_size=5):
        """Calculate running averages for all metrics and create a wandb table."""
        table = wandb.Table(columns=["Metric", f"Last {window_size} Avg", "Best Running Avg"])
        
        for metric_name, values in metric_history.items():
            if len(values) > 0:
                # Calculate running average of last window_size values
                last_n = values[-window_size:] if len(values) >= window_size else values
                running_avg = sum(last_n) / len(last_n)
                
                # Get the best running average achieved during training
                best_avg = best_running_avgs.get(metric_name, running_avg)
                
                # Add row to table
                table.add_data(metric_name, f"{running_avg:.6f}", f"{best_avg:.6f}")
        
        return table

    def save_metric_curves(self, save_metric_dict):
        script_dir = os.path.dirname(os.path.abspath(__file__)) + "Checkpoints/" + self.wandb_id + "/"
        if not os.path.isdir(script_dir):
            os.makedirs(script_dir)
        
        with open(script_dir + f"metric_dict.pkl", "wb") as f:
            pickle.dump(save_metric_dict, f)


