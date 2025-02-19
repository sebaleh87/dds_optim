import os
import pickle
import numpy as np
from itertools import cycle
import matplotlib.pyplot as plt




def load_config(wandb_id):
    script_dir = "/system/user/publicwork/sanokows/Denoising_diff_sampler" + "/TrainerCheckpoints/" + wandb_id + "/"
    
    with open(script_dir + "metric_dict.pkl", "rb") as f:
        save_metric_dict = pickle.load( f)

    return save_metric_dict

def stability_brownian():

    ### TODO also show curves for divergent sigmas

    ''' LogVarLoss all SDE params: wandb_id divergent
    python main.py --SDE_Loss Bridge_LogVarLoss --Energy_Config Brownian --n_integration_steps 128 --T_start 1.0 --T_end 1. --batch_size 2000 --lr 0.005 --Energy_lr 0.0 --SDE_lr 0.005 --N_anneal 4000 --feature_dim 64 --n_hidden 64 --GPU 1 --beta_max 0.05 --beta_min 0.001 --use_interpol_gradient --Network_Type FeedForward --project_name stability_final --use_normal --SDE_Type Bridge_SDE --repulsion_strength 0.0 --sigma_init 0.5
    '''

    ''' LogVarLoss only prior: wandb_id 
    python main.py --SDE_Loss Bridge_LogVarLoss --Energy_Config Brownian --n_integration_steps 128 --T_start 1.0 --T_end 1. --batch_size 2000 --lr 0.005 --Energy_lr 0.0 --SDE_lr 0.005 --N_anneal 4000 --feature_dim 64 --n_hidden 64 --GPU 5 --beta_max 0.05 --beta_min 0.001 --use_interpol_gradient --Network_Type FeedForward --project_name stability_final --use_normal --SDE_Type Bridge_SDE --repulsion_strength 0.0 --sigma_init 0.5 --learn_SDE_params_mode prior_only
    '''

    ''' Bridge_rKL_logderiv only prior: wandb_id 
    python main.py --SDE_Loss Bridge_rKL_logderiv --Energy_Config Brownian --n_integration_steps 128 --T_start 1.0 --T_end 1. --batch_size 2000 --lr 0.005 --Energy_lr 0.0 --SDE_lr 0.005 --N_anneal 4000 --feature_dim 64 --n_hidden 64 --GPU 4 --beta_max 0.05 --beta_min 0.001 --use_interpol_gradient --Network_Type FeedForward --project_name stability_final --use_normal --SDE_Type Bridge_SDE --repulsion_strength 0.0 --sigma_init 0.5 --learn_SDE_params_mode prior_only
    '''

    rKL_all_SDE_ids_higher_sigma = ["leafy-silence-41", "logical-totem-34", "noble-salad-23"]
    ''' Bridge_rKL_logderiv learn SDE params: wandb_id "noble-salad-23"
    python main.py --SDE_Loss Bridge_rKL_logderiv --Energy_Config Brownian --n_integration_steps 128 --T_start 1.0 --T_end 1. --batch_size 2000 --lr 0.005 --Energy_lr 0.0 --SDE_lr 0.005 --N_anneal 4000 --feature_dim 64 --n_hidden 64 --GPU 0 --beta_max 0.05 --beta_min 0.001 --use_interpol_gradient --Network_Type FeedForward --project_name stability_final --use_normal --SDE_Type Bridge_SDE --repulsion_strength 0.0 --sigma_init 0.5
    '''


    #### Convergent parameters
    LV_all_SDE_ids = ["zany-vortex-45", "wobbly-serenity-40", "vocal-feather-27"]
    ''' LogVarLoss all SDE params: wandb_id = 
    python main.py --SDE_Loss Bridge_LogVarLoss --Energy_Config Brownian --n_integration_steps 128 --T_start 1.0 --T_end 1. --batch_size 2000 --lr 0.005 --Interpol_lr 0.0001 --SDE_lr 0.001 --N_anneal 4000 --feature_dim 64 --n_hidden 64 --GPU 7 --beta_max 0.025 --beta_min 0.001 --use_interpol_gradient --Network_Type FeedForward --project_name stability_final --use_normal --SDE_Type Bridge_SDE --repulsion_strength 0.0 --sigma_init 0.2 
    '''
    
    LV_only_prior = ["tough-eon-43", "cosmic-plasma-38", "devout-aardvark-31"]
    ''' LogVarLoss only prior: wandb_id = 
    python main.py --SDE_Loss Bridge_LogVarLoss --Energy_Config Brownian --n_integration_steps 128 --T_start 1.0 --T_end 1. --batch_size 2000 --lr 0.005 --Interpol_lr 0.0001 --SDE_lr 0.001 --N_anneal 4000 --feature_dim 64 --n_hidden 64 --GPU 5 --beta_max 0.025 --beta_min 0.001 --use_interpol_gradient --Network_Type FeedForward --project_name stability_final --use_normal --SDE_Type Bridge_SDE --repulsion_strength 0.0 --sigma_init 0.2 --learn_SDE_params_mode prior_only
    '''

    rKL_only_prior = ["driven-grass-42", "fearless-smoke-37", "astral-river-20"]
    ''' Bridge_rKL_logderiv only prior: wandb_id = "astral-river-20"
    python main.py --SDE_Loss Bridge_rKL_logderiv --Energy_Config Brownian --n_integration_steps 128 --T_start 1.0 --T_end 1. --batch_size 2000 --lr 0.005 --Energy_lr 0.0 --SDE_lr 0.005 --N_anneal 4000 --feature_dim 64 --n_hidden 64 --GPU 4 --beta_max 0.01 --beta_min 0.001 --use_interpol_gradient --Network_Type FeedForward --project_name stability_final --use_normal --SDE_Type Bridge_SDE --repulsion_strength 0.0 --sigma_init 0.2 --learn_SDE_params_mode prior_only
    '''

    rKL_all_SDE_ids = ["tough-sky-44","twilight-elevator-39", "lyric-fog-19"]
    ''' Bridge_rKL_logderiv learn SDE params: wandb_id = "lyric-fog-19"
    python main.py --SDE_Loss Bridge_rKL_logderiv --Energy_Config Brownian --n_integration_steps 128 --T_start 1.0 --T_end 1. --batch_size 2000 --lr 0.005 --Energy_lr 0.0 --SDE_lr 0.005 --N_anneal 4000 --feature_dim 64 --n_hidden 64 --GPU 3 --beta_max 0.01 --beta_min 0.001 --use_interpol_gradient --Network_Type FeedForward --project_name stability_final --use_normal --SDE_Type Bridge_SDE --repulsion_strength 0.0 --sigma_init 0.2
    '''
    rKL_all_SDE_average_sigma = ["serene-thunder-48", "curious-cherry-47"]
    plot_dict = {"LogVarLoss": [{"id_list": LV_all_SDE_ids,"learn_SDE_params": True, "oracle": True, "label": r"CMCD-LV"}, {"id_list": LV_only_prior, "learn_SDE_params": False, "oracle": True, "label": r"CMCD-LV $ \star$"}], 
                 "rKL_logderiv": [{"id_list": rKL_only_prior, "learn_SDE_params": False, "oracle": True, "label": r"CMCD-rKL-LD $ \star$ "}, {"id_list": rKL_all_SDE_average_sigma, "learn_SDE_params": True, "oracle": True, "label": r"CMCD-rKL-LD $ \star$ $\sigma_\mathrm{diff, avrg}$"}],
                 "rKL_logderiv_higher_sigma": [{"id_list": rKL_all_SDE_ids_higher_sigma, "learn_SDE_params": True, "oracle": True, "label": r"CMCD-rKL-LD"}]}
    

    # plot_dict = {"LogVarLoss": [{"id_list": LV_all_SDE_ids,"learn_SDE_params": True, "oracle": True, "label": r"CMCD-LV"}, {"id_list": LV_only_prior, "learn_SDE_params": False, "oracle": True, "label": r"CMCD-LV $ \star$"}], 
    #              "rKL_logderiv": [{"id_list": rKL_only_prior, "learn_SDE_params": False, "oracle": True, "label": r"CMCD-rKL w / LD $ \star$ "}], 
    #              "rKL_logderiv_higher_sigma": [{"id_list": rKL_all_SDE_ids_higher_sigma, "learn_SDE_params": True, "oracle": True, "label": r"CMCD-rKL w / LD"}]}

    def compute_average_and_error(id_list):
        all_free_energies = []
        for wandb_id in id_list:
            save_metric_dict = load_config(wandb_id)
            all_free_energies.append(save_metric_dict["Free_Energy_at_T=1"])

        min_length = min(len(free_energy) for free_energy in all_free_energies)
        all_free_energies = [free_energy[:min_length] for free_energy in all_free_energies]
        
        all_free_energies = np.array(all_free_energies)
        mean_free_energy = np.mean(all_free_energies, axis=0)
        std_free_energy = np.std(all_free_energies, axis=0)/np.sqrt(len(id_list))
        epochs = save_metric_dict["epoch"]
        
        return epochs, mean_free_energy, std_free_energy
    
    def plot_within_range(epochs, mean_free_energy, std_free_energy, label, y_min, y_max, marker = 'v', color = "#ff7f00"):
        lower_bound = mean_free_energy - std_free_energy
        upper_bound = mean_free_energy + std_free_energy
        marker = "-" + marker

        valid_indices = np.where((lower_bound >= y_min) & (upper_bound <= y_max+1))[0]
        valid_indices = valid_indices[::100]
        if len(valid_indices) > 0:
            plt.plot(valid_indices, mean_free_energy[valid_indices], marker, label=label, alpha=1., color=color)
            #plt.fill_between(valid_indices, upper_bound[valid_indices] , lower_bound[valid_indices], alpha=0.6)
            #plt.errorbar(epochs[0:len(valid_indices)], mean_free_energy[valid_indices], yerr=std_free_energy[valid_indices], label=label, capsize=5)

    color_cycle = cycle(['#377eb8', '#ff7f00', '#4daf4a', '#f781bf', '#a65628', '#984ea3', '#999999', '#e41a1c', '#dede00'])
    marker_cycle = cycle(['o', 's', 'D', '^', 'v', '<', '>', 'p', '*'])

    y_min = -1.1
    y_max = 0
    plt.figure()
    for loss_type, configs in plot_dict.items():
        for config in configs:
            id_list = config["id_list"]
            epochs, mean_free_energy, std_free_energy = compute_average_and_error(id_list)
            label = config["label"]
            color = next(color_cycle)
            marker = next(marker_cycle)
            plot_within_range(epochs, mean_free_energy, std_free_energy, label, y_min, y_max, marker=marker, color=color)
            #plt.plot(epochs, mean_free_energy, label=label, color=color, marker=marker, alpha=1.)
    plt.ylim(y_min, y_max)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=16)
    plt.xlabel('Epochs', fontsize=18)
    plt.ylabel('-ELBO', fontsize=18)
    plt.legend(fontsize=16)
    plt.grid()
    plt.tight_layout()
    print(os.path.dirname(os.path.abspath(__file__)) + '/Figures/avg_sigma_plot.png')
    plt.savefig(os.path.dirname(os.path.abspath(__file__))  + '/Figures/free_energy_plot.png', dpi=1000)
    plt.show()



if(__name__ == "__main__"):
    stability_brownian()
