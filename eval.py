import os
import pickle
import wandb
import jax
import numpy as np
from tqdm.auto import tqdm
import argparse
from Trainer.train import TrainerClass

# def go_deeper_or_add_key(config, padd_keys, padd_key):
#     if(isinstance(config[padd_key], dict)):
#         go_deeper_or_add_key(config[padd_key], padd_keys[padd_key], padd_key)
#     else:
#         for sub_key in padd_key_dict[padd_key]:
#             if sub_key not in config[key]:
#                 config[key][sub_key] = padd_key_dict[key][sub_key]
#                 print(key, sub_key, "added to config")

# def config_completer(config):
#     ### adds keys to default if key is missing
#     padd_key_dict = {"SDE_Loss_Config": {"SDE_Type_Config":{"natural_gradient": False}}}

#     for key in config:
#         if key in padd_key_dict.keys():
#             if(isinstance(config[key], dict)):
                
#             else:
#                 for sub_key in padd_key_dict[key]:
#                     if sub_key not in config[key]:
#                         config[key][sub_key] = padd_key_dict[key][sub_key]
#                         print(key, sub_key, "added to config")

#     return config


def load_params_and_config(wandb_run_name, metric = "Sinkhorn"):
    script_dir = os.path.dirname(os.path.abspath(__file__)) + "/TrainerCheckpoints/" + wandb_run_name + "/"
    filename = f"best_{metric}_checkpoint.pkl"
    # files = os.listdir(script_dir)
    # print("Files in directory:", files)

    file_path = script_dir + filename
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"No such file: '{file_path}'")
    with open(file_path, "rb") as f:
        data = pickle.load(f)
    return data["params"], data["config"]


class EvaluatorClass(TrainerClass):
    def __init__(self, base_config, wandb_run_name, param_dict, project="DDS_evaluation"):

        self.wandb_run_name = wandb_run_name
        super().__init__(base_config)
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

    def chunk_evaluate(self, n_eval_samples, chunk_size=10):
        """
        Evaluate the model in a specified number of chunks.
        """
        temp = 1.
        n_chunks = chunk_size
        samples_per_chunk = n_eval_samples // n_chunks
        leftover = n_eval_samples % n_chunks

        aggregated_metrics = {}
        key = jax.random.PRNGKey(0)
        combined_tracer = {}

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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wandb_run_name", default = "dandy-energy-6", type=str, help="Wandb run name of run to evaluate")
    parser.add_argument("--checkpoint_metric", default = "Sinkhorn", choices=["Free_Energy_at_T=1", "Sinkhorn"], type=str, help="Wandb run name of run to evaluate")
    parser.add_argument("--n_eval_samples", type=int, default=10000)
    parser.add_argument("--chunk_size", type=int, default=1000)
    parser.add_argument("--GPU", type=int, default=0)
    args = parser.parse_args()

    # Set GPU
    if args.GPU >= 0:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.GPU)

    params, config = load_params_and_config(args.wandb_run_name, metric = args.checkpoint_metric)
    
    if "EnergyConfig" in config:
        for key in ["means", "variances"]:
            if key in config['EnergyConfig']:
                parse_energy_config_array(config['EnergyConfig'], key)

    config.update({"n_eval_samples": args.n_eval_samples})

    #config = config_completer(config)
    evaluator = EvaluatorClass(config, args.wandb_run_name, params)

    metrics = evaluator.chunk_evaluate(args.n_eval_samples, args.chunk_size)
    wandb.log({ f"eval/{key}": metrics[key] for key in metrics.keys()})
    evaluator.generate_plots()

if __name__ == "__main__":
    main() 