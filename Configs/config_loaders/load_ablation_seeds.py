
import os
import pickle
import numpy as np
from itertools import cycle
import matplotlib.pyplot as plt
from loading import load_config

# def load_config(wandb_id):
#     script_dir = "/system/user/publicwork/sanokows/Denoising_diff_sampler" + "/TrainerCheckpoints/" + wandb_id + "/"
    
#     print(wandb_id)
#     if not os.path.exists(script_dir):
#         raise FileNotFoundError(f"The directory {script_dir} does not exist.")
    
    
#     with open(script_dir + "metric_dict.pkl", "rb") as f:
#         save_metric_dict = pickle.load( f)

#     return save_metric_dict

def compute_rel_error(arr, min_value):
    return np.abs(np.array(arr) - min_value) 


print("running")
if(__name__ == "__main__"):
    ablation_wandb_ids_sigma_frozen= {"wandb_ids": ["cerulean-dew-6", "northern-hill-4", "fluent-sky-2"], "sigmas": [0.3, 0.2, 0.1], "curves": []}
    ablation_wandb_ids_sigma_leared= {"wandb_ids": ["stellar-wave-5", "golden-pyramid-3", "splendid-surf-1"], "sigmas": [0.3, 0.2, 0.1], "curves": []}


    for wandb_id in ablation_wandb_ids_sigma_leared["wandb_ids"]:
        metric_dict = load_config(wandb_id)

        ablation_wandb_ids_sigma_leared["curves"].append(metric_dict["Free_Energy_at_T=1"])

    for wandb_id in ablation_wandb_ids_sigma_frozen["wandb_ids"]:
        metric_dict = load_config(wandb_id)

        ablation_wandb_ids_sigma_frozen["curves"].append(metric_dict["Free_Energy_at_T=1"])

    color_cycle = cycle(['#377eb8', '#ff7f00', '#4daf4a', '#f781bf', '#a65628', '#984ea3', '#999999', '#e41a1c', '#dede00'])
    marker_cycle = cycle(['o', 's', 'D', '^', 'v', '<', '>', 'p', '*'])
    # Filter the curves to only include values for x > 500

    x_min = 0
    y_max = 40
    step = 50
    ablation_wandb_ids_sigma_leared["curves"] = [curve[x_min::step] for curve in ablation_wandb_ids_sigma_leared["curves"]]
    ablation_wandb_ids_sigma_frozen["curves"] = [curve[x_min::step] for curve in ablation_wandb_ids_sigma_frozen["curves"]]
    min_value = 73

    for sigma, curve in zip(ablation_wandb_ids_sigma_leared["sigmas"], ablation_wandb_ids_sigma_leared["curves"]):
        rel_curve = compute_rel_error(curve, min_value)
        plt.plot(np.arange(0, len(rel_curve))*step + x_min ,rel_curve, label=r'rKL-LD $\star$ $\sigma_{\mathrm{diff, init}} = $' + f'{sigma}', color=next(color_cycle), marker=next(marker_cycle))

    for sigma, curve in zip(ablation_wandb_ids_sigma_frozen["sigmas"], ablation_wandb_ids_sigma_frozen["curves"]):
        rel_curve = compute_rel_error(curve, min_value)
        plt.plot(np.arange(0, len(rel_curve))*step + x_min , rel_curve, label=r'rKL-LD $\sigma_{\mathrm{diff, init}} = $' + f'{sigma}', color=next(color_cycle), marker=next(marker_cycle))

    plt.xticks(fontsize=14)
    plt.yticks(fontsize=16)
    plt.xlabel('Epochs', fontsize=18)
    plt.ylabel(r'$|\Delta \mathrm{ELBO}|$', fontsize=18)
    
    plt.legend(fontsize=15, loc='center left', bbox_to_anchor=(0.3, 0.55))
    plt.yscale('log')
    plt.ylim(top=y_max) #ymax is your value
    plt.grid()
    plt.tight_layout()
    plt.savefig(os.path.dirname(os.path.abspath(__file__))  + '/Figures/ablation_seeds_frozen.png', dpi=1000)
    plt.show()