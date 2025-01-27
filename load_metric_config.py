import os
import pickle





def load_config(wandb_id):
    script_dir = os.path.dirname(os.path.abspath(__file__)) + "/TrainerCheckpoints/" + wandb_id + "/"
    
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

    ''' Bridge_rKL_logderiv learn SDE params: wandb_id "noble-salad-23"
    python main.py --SDE_Loss Bridge_rKL_logderiv --Energy_Config Brownian --n_integration_steps 128 --T_start 1.0 --T_end 1. --batch_size 2000 --lr 0.005 --Energy_lr 0.0 --SDE_lr 0.005 --N_anneal 4000 --feature_dim 64 --n_hidden 64 --GPU 0 --beta_max 0.05 --beta_min 0.001 --use_interpol_gradient --Network_Type FeedForward --project_name stability_final --use_normal --SDE_Type Bridge_SDE --repulsion_strength 0.0 --sigma_init 0.5
    '''


    #### Convergent parameters

    ''' LogVarLoss all SDE params: wandb_id = 
    python main.py --SDE_Loss Bridge_LogVarLoss --Energy_Config Brownian --n_integration_steps 128 --T_start 1.0 --T_end 1. --batch_size 2000 --lr 0.005 --Interpol_lr 0.0001 --SDE_lr 0.001 --N_anneal 4000 --feature_dim 64 --n_hidden 64 --GPU 7 --beta_max 0.025 --beta_min 0.001 --use_interpol_gradient --Network_Type FeedForward --project_name stability_final --use_normal --SDE_Type Bridge_SDE --repulsion_strength 0.0 --sigma_init 0.2 
    '''

    ''' LogVarLoss only prior: wandb_id = 
    python main.py --SDE_Loss Bridge_LogVarLoss --Energy_Config Brownian --n_integration_steps 128 --T_start 1.0 --T_end 1. --batch_size 2000 --lr 0.005 --Interpol_lr 0.0001 --SDE_lr 0.001 --N_anneal 4000 --feature_dim 64 --n_hidden 64 --GPU 5 --beta_max 0.025 --beta_min 0.001 --use_interpol_gradient --Network_Type FeedForward --project_name stability_final --use_normal --SDE_Type Bridge_SDE --repulsion_strength 0.0 --sigma_init 0.2 --learn_SDE_params_mode prior_only
    '''

    ''' Bridge_rKL_logderiv only prior: wandb_id = "astral-river-20"
    python main.py --SDE_Loss Bridge_rKL_logderiv --Energy_Config Brownian --n_integration_steps 128 --T_start 1.0 --T_end 1. --batch_size 2000 --lr 0.005 --Energy_lr 0.0 --SDE_lr 0.005 --N_anneal 4000 --feature_dim 64 --n_hidden 64 --GPU 4 --beta_max 0.01 --beta_min 0.001 --use_interpol_gradient --Network_Type FeedForward --project_name stability_final --use_normal --SDE_Type Bridge_SDE --repulsion_strength 0.0 --sigma_init 0.2 --learn_SDE_params_mode prior_only
    '''


    ''' Bridge_rKL_logderiv learn SDE params: wandb_id = "lyric-fog-19"
    python main.py --SDE_Loss Bridge_rKL_logderiv --Energy_Config Brownian --n_integration_steps 128 --T_start 1.0 --T_end 1. --batch_size 2000 --lr 0.005 --Energy_lr 0.0 --SDE_lr 0.005 --N_anneal 4000 --feature_dim 64 --n_hidden 64 --GPU 3 --beta_max 0.01 --beta_min 0.001 --use_interpol_gradient --Network_Type FeedForward --project_name stability_final --use_normal --SDE_Type Bridge_SDE --repulsion_strength 0.0 --sigma_init 0.2
    '''

    plot_dict = {"LogVarLoss": [{"id": "celestial-cloud-4","learn_SDE_params": True, "oracle": True}, {"id": "blooming-fog-3", "learn_SDE_params": False, "oracle": True}], 
                 "rKL_logderiv": [{"id": "volcanic-cherry-6", "learn_SDE_params": False, "oracle": True}, {"id": "young-rain-5", "learn_SDE_params": True, "oracle": True}]}
    

    import matplotlib.pyplot as plt

    for loss_type, configs in plot_dict.items():
        for config in configs:
            wandb_id = config["id"]
            save_metric_dict = load_config(wandb_id)
            free_energy = save_metric_dict["Free_Energy_at_T=1"]
            epochs = save_metric_dict["epoch"]
            label = f"{loss_type} (oracle={config['oracle']}, learn_SDE_params={config['learn_SDE_params']})"
            plt.plot(epochs, free_energy, label=label)

    plt.xlabel('Epochs')
    plt.ylabel('Free Energy')
    plt.title('Free Energy over Epochs')
    plt.legend()
    plt.savefig(os.getcwd() + '/Figures/free_energy_plot.png', dpi=1000)
    plt.show()


if(__name__ == "__main__"):
    # Example usage:
    wandb_id = "driven-dragon-1"
    save_metric_dict = load_config(wandb_id)


    Free_Energy = save_metric_dict["Free_Energy_at_T=1"]

    import matplotlib.pyplot as plt

    epochs = save_metric_dict["epoch"]
    plt.plot(epochs, Free_Energy, label='Free Energy')
    plt.xlabel('Epochs')
    plt.ylabel('Free Energy')
    plt.title('Free Energy over Epochs')
    plt.legend()
    plt.savefig(os.getcwd() + '/Figures/free_energy_plot.png', dpi=1000)
    plt.show()