import os
import pickle
import wandb
import numpy as np
from tqdm.auto import tqdm
import argparse
from Trainer.train import TrainerClass
import jax
import jax.numpy as jnp
from Configs.config_loaders import run_wandb_ids


def load_params_and_config(wandb_run_name, filename = "params_and_config_train_end.pkl"):
    script_dir = os.path.dirname(os.path.abspath(__file__)) + "/TrainerCheckpoints/" + wandb_run_name + "/"
    #filename = f"best_{metric}_checkpoint.pkl"
    files = os.listdir(script_dir)
    print("Files in directory:", files)

    file_path = script_dir + filename
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"No such file: '{file_path}'")
    with open(file_path, "rb") as f:
        data = pickle.load(f)
    return data["params"], data["config"]


class EvaluatorClass(TrainerClass):
    def __init__(self, base_config, wandb_run_name, param_dict, project="DDS_evaluation"):

        self.wandb_run_name = wandb_run_name
        super().__init__(base_config, mode = "eval")
        self.params = param_dict["model_params"]
        self.SDE_LossClass.Interpol_params = param_dict["Interpol_params"]
        self.SDE_LossClass.SDE_params = param_dict["SDE_params"]
        wandb.init(
            project=project,
            config={
                "wandb_run_name": wandb_run_name,
                "n_eval_samples": base_config["n_eval_samples"],
                "config": base_config,
            }
        )


    def _init_Network(self):
        """Override parent's network initialization to prevent weight initialization"""
        pass

    def load_original_weights(self):
        """Load weights from original training run"""
        try:
            param_dict = self.load_params_and_config(filename="best_Free_Energy_at_T=1_checkpoint.pkl")
            print("Loaded best Free Energy checkpoint")
        except FileNotFoundError:
            param_dict = self.load_params_and_config(filename="best_Energy_checkpoint.pkl")
            print("Loaded best Energy checkpoint")
        self.params = param_dict["model_params"]
        self.SDE_LossClass.Energy_params = param_dict["Energy_params"]
        self.SDE_LossClass.SDE_params = param_dict["SDE_params"]

    def compute_MMD_and_Sinkhorn(self, n_eval_samples):
        pass

    def chunk_evaluate(self, n_eval_samples, chunk_size=10):
        """
        Evaluate the model in a specified number of chunks.
        """
        temp = 1.
        n_chunks = chunk_size
        samples_per_chunk = n_eval_samples
        leftover = 0# n_eval_samples % n_chunks

        aggregated_metrics = {}
        key = jax.random.PRNGKey(0)
        combined_tracer = {}
        IPM_Metrics = {"MMD": [], "Sinkhorn": []}

        for chunk_idx in tqdm(range(n_chunks), desc="Evaluating in chunks"):
            curr_chunk_size = samples_per_chunk
            if chunk_idx == n_chunks - 1 and leftover != 0:
                curr_chunk_size += leftover

            SDE_tracer, out_dict, key = self.SDE_LossClass.simulate_reverse_sde_scan(
                self.params,
                self.SDE_LossClass.Interpol_params,
                self.SDE_LossClass.SDE_params,
                temp, key, sample_mode = "train",
                n_integration_steps=self.n_integration_steps,
                n_states=curr_chunk_size
            )  

            model_samples = out_dict["X_0"]

            out_dict = self.sd_calculator.compute_MMD_and_Sinkhorn(model_samples)
            IPM_Metrics["MMD"].append(jnp.sqrt(out_dict["MMD^2"]))
            IPM_Metrics["Sinkhorn"].append(out_dict["Sinkhorn divergence"])
            
            if not combined_tracer:
                combined_tracer = {k: [] for k in SDE_tracer.keys()}

            for k, v in SDE_tracer.items():
                combined_tracer[k].append(np.array(v))

            for k, v in out_dict.items():
                if k not in aggregated_metrics:
                    aggregated_metrics[k] = []
                aggregated_metrics[k].append(v)

        for k in combined_tracer:
            combined_tracer[k] = np.concatenate(combined_tracer[k], axis=0)

        self.last_tracer = combined_tracer

        final_metrics = {}
        for k, v_list in aggregated_metrics.items():
            if all(isinstance(item, (int, float, np.number)) for item in v_list):
                final_metrics[k] = np.mean(v_list)
            else:
                try:
                    arr = np.concatenate([np.array(x) for x in v_list], axis=0)
                    final_metrics[k] = np.mean(arr)
                except ValueError:
                    final_metrics[k] = np.mean([np.mean(x) for x in v_list])

        print("Sinkhorn Distance list", IPM_Metrics["Sinkhorn"])
        final_metrics["MMD"] = {"mean": np.mean(IPM_Metrics["MMD"]), "std": np.std(IPM_Metrics["MMD"])/ jnp.sqrt(len(IPM_Metrics["MMD"]))}
        final_metrics["Sinkhorn"] = {"mean": np.mean(IPM_Metrics["Sinkhorn"]), "std": np.std(IPM_Metrics["Sinkhorn"])/ jnp.sqrt(len(IPM_Metrics["Sinkhorn"]))}

        print(final_metrics["MMD"], "MMD")
        print(final_metrics["Sinkhorn"], "Sinkhorn")
        return final_metrics

    def generate_plots(self):
        """Generate evaluation plots based on energy type"""
        if not hasattr(self, 'last_tracer'):
            raise ValueError("Must run evaluation before generating plots")
            
        if self.EnergyClass.config["name"] == "GaussianMixture":
            fig_traj = self.EnergyClass.plot_trajectories(np.array(self.last_tracer["xs"])[:,0:10,:], panel="eval_figs")
            fig_hist = self.EnergyClass.plot_histogram(np.array(self.last_tracer["x_final"]), panel="eval_figs")
            fig_last_samples = self.EnergyClass.plot_last_samples(np.array(self.last_tracer["x_final"]), panel="eval_figs")
        else:
            fig_traj = self.EnergyClass.plot_trajectories(np.array(self.last_tracer["ys"])[:,0:10,:], panel="eval_figs")
            fig_hist = self.EnergyClass.plot_histogram(np.array(self.last_tracer["y_final"]), panel="eval_figs")
            fig_last_samples = self.EnergyClass.plot_last_samples(np.array(self.last_tracer["y_final"]), panel="eval_figs")
            
        wandb.log({
            "eval/trajectories": fig_traj,
            "eval/histogram": fig_hist,
            "eval/last_samples": fig_last_samples
        })

def parse_energy_config_array(config_dict, key):
    try:
        if isinstance(config_dict[key], str):
            str_data = config_dict[key].strip('[]')
            rows = [row.strip() for row in str_data.split('\n') if row.strip()]
            array_data = []
            for row in rows:
                clean_row = row.strip('[]').strip()
                numbers = [float(num) for num in clean_row.split() if num]
                array_data.append(numbers)
            config_dict[key] = np.array(array_data, dtype=np.float32)
        elif isinstance(config_dict[key], list):
            config_dict[key] = np.array(config_dict[key], dtype=np.float32)
    except (ValueError, SyntaxError) as e:
        raise


def evaluate_on_run(wandb_run_name, n_eval_samples, chunk_size):
    params, config = load_params_and_config(wandb_run_name)
    
    if "EnergyConfig" in config:
        for key in ["means", "variances"]:
            if key in config['EnergyConfig']:
                parse_energy_config_array(config['EnergyConfig'], key)

    config.update({"n_eval_samples": n_eval_samples})

    #config = config_completer(config)
    evaluator = EvaluatorClass(config, wandb_run_name, params)

    metrics = evaluator.chunk_evaluate(n_eval_samples, chunk_size)
    #wandb.log({ f"eval/{key}": metrics[key] for key in metrics.keys()})
    #evaluator.generate_plots()
    return {"MMD": metrics["MMD"], "Sinkhorn": metrics["Sinkhorn"]}


def evaluate_runs(wandb_ids):
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_metric", default = "params_and_config_train_end.pkl", choices=["params_and_config_train_end.pkl","Free_Energy_at_T=1", "Sinkhorn"], type=str, help="Wandb run name of run to evaluate")
    parser.add_argument("--n_eval_samples", type=int, default=16000)
    parser.add_argument("--chunk_size", type=int, default=20)
    parser.add_argument("--GPU", type=int, default=0)
    args = parser.parse_args()

    # Set GPU
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID" 
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.GPU)
    
    # metric_dict = { "MMD": [], "Sinkhorn": []}

    # for wandb_id in wandb_ids:
    #     metrix_dict_per_run = evaluate_on_run(wandb_id, args.n_eval_samples, args.chunk_size)
    #     metric_dict["MMD"].append(metrix_dict_per_run["MMD"]["mean"])
    #     metric_dict["Sinkhorn"].append(metrix_dict_per_run["Sinkhorn"]["mean"])

    # print(metric_dict["Sinkhorn"])
    # Sinkhorn_metric_text = compute_average_and_variance(metric_dict["Sinkhorn"])
    # MMD_metric_text = compute_average_and_variance(metric_dict["MMD"])
    # print("Sinkhorn", Sinkhorn_metric_text)
    # print("MMD", MMD_metric_text)

    # Save MMD and Sinkhorn distances into a text file
    output_dir = os.path.dirname(os.path.abspath(__file__)) + "/Data/eval"
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "evaluation_metrics.txt")

    with open(output_file, "w") as f:
        f.write("MMD Distances:\n")
        f.write(", ".join(map(str, metric_dict["MMD"])) + "\n")
        f.write(f"{MMD_metric_text}\n\n")
        
        f.write("Sinkhorn Distances:\n")
        f.write(", ".join(map(str, metric_dict["Sinkhorn"])) + "\n")
        f.write(f"{Sinkhorn_metric_text}\n\n")

    print(f"Metrics saved to {output_file}")

def compute_average_and_variance(curve_per_seed, round_mean = 2, round_sdt = 3):
    mean_over_seeds = np.mean(curve_per_seed)
    std_over_seeds = np.std(curve_per_seed)/np.sqrt(len(curve_per_seed))
    mean_over_seeds_rounded = np.round(mean_over_seeds, round_mean)
    std_over_seeds_rounded = np.round(std_over_seeds, round_sdt)

    metric_text = f"${mean_over_seeds_rounded:.2f}"+ r"\text{\tiny{$\pm " +  f"{std_over_seeds_rounded}$" + "}}$"
    return metric_text

if __name__ == "__main__":

    ### GMM 
    # log_derivative 
    problem_list = { "MoS": run_wandb_ids.MoS, "GMM": run_wandb_ids.GMM, "GMM-DBS": run_wandb_ids.GMM_DBS, "MoS-DBS": run_wandb_ids.MoS_DBS}

    GMM_rKL_LD = ["clone-trooper-25", "legendary-bantha-29", "jedi-fighter-33", "galactic-speeder-37", "tusken-republic-41"
                    ,"grievous-carrier-45", "carbonite-lightsaber-48"]

    evaluate_runs(GMM_rKL_LD) 