import os
import pickle
import numpy as np
from itertools import cycle
import matplotlib.pyplot as plt


def load_parameters(wandb_id):
    script_dir = os.path.dirname(os.path.abspath(__file__)) + "/TrainerCheckpoints/" + wandb_id + "/"
    
    with open(script_dir + "best_Energy_checkpoint.pkl", "rb") as f:
        save_metric_dict = pickle.load( f)
    
    parameters, config = save_metric_dict["params"], save_metric_dict["config"]
    return parameters, config


if(__name__ == "__main__"):
    wandb_ids = ["leafy-silence-41", "logical-totem-34", "noble-salad-23"]
    SDE_parameter_list = []
    for wandb_id in wandb_ids:
        params, _ = load_parameters(wandb_id)
        SDE_parameters = params["SDE_params"]
        SDE_parameter_list.append(SDE_parameters)

    avg_sigma_list = []
    for SDE_parameters in SDE_parameter_list:
        log_sigma = SDE_parameters["log_sigma"]
        beta_delta = np.exp(SDE_parameters["log_beta_delta"])
        beta_min = np.exp(SDE_parameters["log_beta_min"])
        beta_max = beta_min + beta_delta
        sigma = np.exp(log_sigma) * beta_max
        avg_sigma_list.append(sigma)
    std_sigma_list = []
    for SDE_parameters in SDE_parameter_list:
        log_sigma = SDE_parameters["log_sigma"]
        beta_delta = np.exp(SDE_parameters["log_beta_delta"])
        beta_min = np.exp(SDE_parameters["log_beta_min"])
        beta_max = beta_min + beta_delta
        sigma = np.exp(log_sigma) * beta_max
        std_sigma_list.append(sigma)

    avg_sigma = np.mean(avg_sigma_list, axis=0)
    std_sigma = np.std(std_sigma_list, axis=0)

    sorted_indices = np.argsort(avg_sigma)
    sorted_avg_sigma = avg_sigma[sorted_indices]
    sorted_std_sigma = std_sigma[sorted_indices]
    plt.plot(range(len(sorted_avg_sigma)), sorted_avg_sigma , "-x", alpha=0.8)
    plt.fill_between(range(len(sorted_avg_sigma)), sorted_avg_sigma - sorted_std_sigma, sorted_avg_sigma + sorted_std_sigma, alpha=0.3)
    plt.xlabel(r'Index sorted by $\sigma$', fontsize=18)
    plt.ylabel(r'$\sigma$ in each dimension', fontsize=18)
    plt.title("Bayesian task: Brownian", fontsize=18)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.grid()
    plt.tight_layout()
    plt.savefig(os.getcwd() + '/Figures/avg_sigma_plot.png', dpi=1000)
    plt.show()





