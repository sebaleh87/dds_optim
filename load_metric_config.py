import os
import pickle





def load_config(wandb_id):
    script_dir = os.path.dirname(os.path.abspath(__file__)) + "/TrainerCheckpoints/" + wandb_id + "/"
    
    with open(script_dir + "metric_dict.pkl", "rb") as f:
        save_metric_dict = pickle.load( f)

    return save_metric_dict


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