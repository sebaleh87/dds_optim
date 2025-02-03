import numpy as np
from loading import load_config

def Brownian():
    wandb_ids = ["leafy-silence-41", "logical-totem-34", "noble-salad-23"]
    return wandb_ids


def Seeds():
    wandb_ids = ["iconic-butterfly-5", "radiant-moon-4", "clean-bird-3"]
    return wandb_ids


def Sonar():
    wandb_ids = ["devoted-sun-13", "winter-violet-12", "wobbly-sunset-11"]
    return wandb_ids


def LGCP():
    wandb_ids = ["resilient-wind-3", "fanciful-leaf-2", "wandering-fog-1"]   
    return wandb_ids

def Credit():
    wandb_ids = ["electric-smoke-3", "gallant-cosmos-2", "daily-wood-1"]
    return wandb_ids

def Funnel():
    wandb_ids = ["amber-smoke-4", "lilac-plant-3", "crimson-eon-2", "helpful-thunder-1"]
    return wandb_ids

if(__name__ == "__main__"):

    problem_list = {"Brownian": Brownian(), "Seeds":  Seeds(), "Sonar": Sonar(), "LGCP": LGCP(), "Credit": Credit(), "Funnel": Funnel()}
    k = 10
    for problem in problem_list.keys():
        print(problem)
        Curves = []
        for wandb_id in problem_list[problem]:
            metric_dict = load_config(wandb_id)
            print(metric_dict.keys())
            if("sinkhorn_divergence" in metric_dict.keys()):
                Curves.append(np.array(metric_dict["sinkhorn_divergence"])[-1])
            else:
                Curves.append(np.array(metric_dict["Free_Energy_at_T=1"]))
        
        #[print(curve) for curve in Curves]
        
        if("sinkhorn_divergence" in metric_dict.keys()):
            curve_per_seed = np.array([value for value in Curves])
            mean_over_seeds = np.mean(curve_per_seed)
            std_over_seeds = np.std(curve_per_seed)/np.sqrt(len(curve_per_seed))
            print("sinkhorn_divergence", mean_over_seeds, "+/-", std_over_seeds)
        else:
            curve_per_seed = np.array([np.mean(curve[-k:]) for curve in Curves])
            mean_over_seeds = np.mean(curve_per_seed)
            std_over_seeds = np.std(curve_per_seed)/np.sqrt(len(curve_per_seed))
            print("Free Energy", -mean_over_seeds, "+/-", std_over_seeds)


