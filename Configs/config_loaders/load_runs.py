import numpy as np
from loading import load_config
import run_wandb_ids as wandb_ids
import matplotlib.pyplot as plt
import os

def compute_average_and_variance(curve_per_seed, round_mean = 2, round_sdt = 3):
    mean_over_seeds = np.mean(curve_per_seed)
    std_over_seeds = np.std(curve_per_seed)/np.sqrt(len(curve_per_seed))
    mean_over_seeds_rounded = np.round(mean_over_seeds, round_mean)
    std_over_seeds_rounded = np.round(std_over_seeds, round_sdt)
    return mean_over_seeds_rounded, std_over_seeds_rounded

if(__name__ == "__main__"):
    loss_keys = ["rKL_frozen", "rKL_logderiv", "rKL_logderiv_frozen", "LogVarLoss", "LogVarLoss_frozen"]

    problem_list = {"Seeds":  wandb_ids.Seeds, "Sonar": wandb_ids.Sonar, "Credit": wandb_ids.Credit, "Funnel": wandb_ids.Funnel, "Brownian": wandb_ids.Brownian, "MoS": wandb_ids.MoS,
                    "LGCP": wandb_ids.LGCP, "GMM": wandb_ids.GMM, "GMM-DBS": wandb_ids.GMM_DBS, "MoS-DBS": wandb_ids.MoS_DBS, 
                    "Funnel-DBS": wandb_ids.Funnel_DBS, "Sonar-DBS": wandb_ids.Sonar_DBS, "Seeds-DBS": wandb_ids.Seeds_DBS, "LGCP-DBS": wandb_ids.LGCP_DBS, 
                    "German_DBS": wandb_ids.German_DBS, "Brownian_DBS": wandb_ids.Brownian_DBS}
    k = 10
    for problem in problem_list.keys():

        if(problem == "MoS" or problem == "GMM"):
            loss_keys = wandb.loss_keys

        Curves = {}
        for loss_key in loss_keys:
            print(problem)
            Curves[loss_key] = {}
            Curves[loss_key]["sinkhorn_divergence"] = []
            Curves[loss_key]["Free_Energy_at_T=1"] = []
            Curves[loss_key]["log_Z_at_T=1"] = []
            Curves[loss_key]["EMC"] = []

            for wandb_id in problem_list[problem](loss_key):
                metric_dict = load_config(wandb_id)
                #print(metric_dict.keys())
                if("sinkhorn_divergence" in metric_dict.keys()):
                    Curves[loss_key]["sinkhorn_divergence"].append(np.array(metric_dict["sinkhorn_divergence"]))
                if("EMC" in metric_dict.keys()):
                    Curves[loss_key]["EMC"].append(np.array(metric_dict["EMC"]))
                if("log_Z_at_T=1" in metric_dict.keys()):
                    Curves[loss_key]["log_Z_at_T=1"].append(np.array(metric_dict["log_Z_at_T=1"]))
                Curves[loss_key]["Free_Energy_at_T=1"].append(np.array(metric_dict["Free_Energy_at_T=1"]))
            
            #[print(curve) for curve in Curves]
            print(problem ,loss_key)
            if("sinkhorn_divergence" in metric_dict.keys()):
                factor = int(4000/50)
                if("MoS" in problem or "GMM" in problem):
                    pad_idx = 2
                else:
                    pad_idx = 0

                #min_args = [(pad_idx + np.argmin(value[pad_idx:])) for value in Curves[loss_key]["sinkhorn_divergence"]]
                min_args = [-1 for value in Curves[loss_key]["sinkhorn_divergence"]]
                sink_value_per_seed = np.array([Curves[loss_key]["sinkhorn_divergence"][seed_idx][arg] for seed_idx, arg in enumerate(min_args)])
                Free_energy_value_per_seed = np.array([Curves[loss_key]["Free_Energy_at_T=1"][seed_idx][arg] for seed_idx, arg in enumerate(min_args)])


                sink_mean_over_seeds_rounded, sink_std_over_seeds_rounded = compute_average_and_variance(sink_value_per_seed)
                Free_energy_mean_over_seeds_rounded, Free_energy_std_over_seeds_rounded = compute_average_and_variance(Free_energy_value_per_seed)

                print(sink_value_per_seed)
                print("Sinkhorn", f"${sink_mean_over_seeds_rounded:.2f}"+ r"\text{\tiny{$\pm " +  f"{sink_std_over_seeds_rounded}$" + "}}$")
                print("ELBO", f"${-Free_energy_mean_over_seeds_rounded:.2f}"+ r"\text{\tiny{$\pm " +  f"{Free_energy_std_over_seeds_rounded}$" + "}}$")

                if(len(Curves[loss_key]["log_Z_at_T=1"]) > 0):
                    log_Z_value_per_seed = np.array([Curves[loss_key]["log_Z_at_T=1"][seed_idx][arg] for seed_idx, arg in enumerate(min_args)])
                    log_Z_mean_over_seeds_rounded, log_Z_std_over_seeds_rounded = compute_average_and_variance(log_Z_value_per_seed)
                    print("log_Z", f"${log_Z_mean_over_seeds_rounded:.2f}"+ r"\text{\tiny{$\pm " +  f"{log_Z_std_over_seeds_rounded}$" + "}}$")
                if(len(Curves[loss_key]["EMC"]) > 0):                    
                    EMC_value_per_seed = np.array([Curves[loss_key]["EMC"][seed_idx][arg] for seed_idx, arg in enumerate(min_args)])   
                    EMC_mean_over_seeds_rounded, EMC_std_over_seeds_rounded = compute_average_and_variance(EMC_value_per_seed, round_mean=4, round_sdt=5)  
                    print("EMC", f"${EMC_mean_over_seeds_rounded:.3f}"+ r"\text{\tiny{$\pm " +  f"{EMC_std_over_seeds_rounded:.3f}$" + "}}$")

                if(problem == "MoS" or problem == "GMM"):
                    average_sink_curve = np.mean(np.array([value for value in Curves[loss_key]["sinkhorn_divergence"]]), axis = 0)
                    average_free_erergy_curve = np.mean(np.array([value for value in Curves[loss_key]["Free_Energy_at_T=1"]]), axis = 0)
                    plt.figure(figsize=(10, 5))

                    plt.subplot(1, 2, 1)
                    plt.plot(average_sink_curve, label='Average Sinkhorn Divergence')
                    plt.xlabel('Iterations')
                    plt.ylabel('Sinkhorn Divergence')
                    plt.title(f'{problem} - {loss_key} Sinkhorn Divergence')
                    plt.ylim(0, 5000)
                    plt.legend()

                    plt.subplot(1, 2, 2)
                    plt.plot(average_free_erergy_curve, label='Average Free Energy at T=1')
                    plt.xlabel('Iterations')
                    plt.ylabel('Free Energy')
                    plt.title(f'{problem} - {loss_key} Free Energy at T=1')
                    plt.legend()

                    plt.tight_layout()
                    plt.savefig(os.path.dirname(os.path.abspath(__file__))+ "/Figures/" + f'/{problem}_{loss_key}_curves.png', dpi=1000)
                    plt.close()



            else:
                curve_per_seed = np.array([np.nanmin(curve) for curve in Curves[loss_key]["Free_Energy_at_T=1"]])
                mean_over_seeds_rounded, std_over_seeds_rounded = compute_average_and_variance(curve_per_seed)
                print("Free Energy", f"${-mean_over_seeds_rounded:.2f}"+ r"\text{\tiny{$\pm " +  f"{std_over_seeds_rounded}$" + "}}$")

                log_Z_value_per_seed = np.array([np.nanmin(curve) for curve in Curves[loss_key]["log_Z_at_T=1"]])
                log_Z_mean_over_seeds_rounded, log_Z_std_over_seeds_rounded = compute_average_and_variance(log_Z_value_per_seed)
                print("log_Z", f"${log_Z_mean_over_seeds_rounded:.2f}"+ r"\text{\tiny{$\pm " +  f"{log_Z_std_over_seeds_rounded}$" + "}}$")



