import numpy as np
from loading import load_config


### TODO use new runs
def Brownian(key):
    #wandb_ids = ["leafy-silence-41", "logical-totem-34", "noble-salad-23"]
    if(key == "rKL_frozen"):
        wandb_ids = ["true-cherry-4", "autumn-meadow-10", "scarlet-plant-15"]
    elif(key == "rKL_logderiv"):
        wandb_ids = ["magic-armadillo-9", "different-thunder-8", "rose-haze-1"]
    elif(key == "rKL_logderiv_frozen"):
        wandb_ids = ["lemon-forest-12", "olive-morning-3", "sage-snowflake-14"]
    elif(key == "LogVarLoss"):
        wandb_ids = ["laced-sun-13", "comfy-blaze-6", "vague-energy-2"]
    elif(key == "LogVarLoss_frozen"):
        wandb_ids = ["honest_grass-11", "eager-brook-7", "grateful-haze-5"]

    return wandb_ids


def Seeds(key):
    #wandb_ids = ["iconic-butterfly-5", "radiant-moon-4", "clean-bird-3"]
    if(key == "rKL_frozen"):
        wandb_ids = ["deft-resonance-16", "breezy-haze-9", "electric-water-6"]
    elif(key == "rKL_logderiv"):
        wandb_ids = ["apricot-galaxy-12", "glowing-oath-8", "silver-salad-1"]
    elif(key == "rKL_logderiv_frozen"):
        wandb_ids = ["tough-voice-15", "blooming-plant-11", "stellar-wildflower-4"]
    elif(key == "LogVarLoss"):
        wandb_ids = ["vital-gorge-13", "breezy-firefly-7", "deft-deluge-3"]
    elif(key == "LogVarLoss_frozen"):
        wandb_ids = ["helpful-serenity-14", "wandering-leaf-10", "lemon-armadillo-5"]
    return wandb_ids


def Sonar(key):
    #wandb_ids = ["devoted-sun-13", "winter-violet-12", "wobbly-sunset-11"]
    if(key == "rKL_frozen"):
        wandb_ids = ["graceful-valley-17", "serene-glitter-15", "vague-elevator-8"]
    elif(key == "rKL_logderiv"):
        wandb_ids = ["polished-snow-10", "giddy-firefly-5", "royal-puddle-1"]
    elif(key == "rKL_logderiv_frozen"):
        wandb_ids = ["eager-thunder-12", "trim-terrain-7", "classic-galaxy-4"]
    elif(key == "LogVarLoss"):
        wandb_ids = ["proud-fog-11", "giddy-pond-6", "northern-valley-2"]
    elif(key == "LogVarLoss_frozen"):
        wandb_ids = ["crimson-universe-16", "robust-snowball-13", "comfy-plant-9"]
    return wandb_ids


def LGCP(key):
    #wandb_ids = ["resilient-wind-3", "fanciful-leaf-2", "wandering-fog-1"] 
    if(key == "rKL_frozen"):
        wandb_ids = ["noble-glitter-8", "summer-durian-13"]
    elif(key == "rKL_logderiv"):
        wandb_ids = ["dry-gorge-4", "clear-aardvark-3", "brisk-sea-1"]
    elif(key == "rKL_logderiv_frozen"):
        wandb_ids = ["fresh-pine-10"]
    elif(key == "LogVarLoss"):
        wandb_ids = ["rosy-elevator-11", "stellar-deluge-14"]
    elif(key == "LogVarLoss_frozen"):
        wandb_ids = ["fluent-wind-9", "comic-microwave-12"]  
    return wandb_ids

def Credit(key):
    #wandb_ids = ["electric-smoke-3", "gallant-cosmos-2", "daily-wood-1"]
    if(key == "rKL_frozen"):
        wandb_ids = ["classic-dragon-16", "solar-bush-12", "vibrant-silence-4"]
    elif(key == "rKL_logderiv"):
        wandb_ids = ["rosy-wave-14", "stilted-galaxy-10", "rose-wood-1"]
    elif(key == "rKL_logderiv_frozen"):
        wandb_ids = ["happy-moon-15", "worthy-pond-9", "clean-fire-6"]
    elif(key == "LogVarLoss"):
        wandb_ids = ["honest-sound-8", "hearty-aardvark-7", "earthy-flower-2"]
    elif(key == "LogVarLoss_frozen"):
        wandb_ids = ["vibrant-leaf-13", "chocolate-frog-11", "usual-waterfall-5"]
    return wandb_ids

def Funnel(key):
    #wandb_ids = ["amber-smoke-4", "lilac-plant-3", "crimson-eon-2", "helpful-thunder-1"]
    if(key == "rKL_frozen"):
        wandb_ids = ["ethereal-pyramid-13", "smooth-oath-11", "laced-water-8"]
    elif(key == "rKL_logderiv"):
        wandb_ids = ["jolly-flower-10", "frosty-cloud-5", "misty-pond-2"]
    elif(key == "rKL_logderiv_frozen"):
        wandb_ids = ["colorful-sunset-14", "breezy-planet-9", "hopeful-butterfly-7"]
    elif(key == "LogVarLoss"):
        wandb_ids = ["light-glitter-6", "good-capybara-4", "twilight-yogurt-3"]
    elif(key == "LogVarLoss_frozen"):
        wandb_ids = ["skilled-snow-16", "cool-field-15", "faithful-brook-12"]
    return wandb_ids

if(__name__ == "__main__"):
    loss_keys = ["rKL_frozen", "rKL_logderiv", "rKL_logderiv_frozen", "LogVarLoss", "LogVarLoss_frozen"]

    problem_list = {"Seeds":  Seeds, "Sonar": Sonar, "Credit": Credit, "Funnel": Funnel} #"Brownian": Brownian, #"LGCP": LGCP
    k = 10
    for problem in problem_list.keys():
        Curves = {}
        for loss_key in loss_keys:
            print(problem)
            Curves[loss_key] = []
            for wandb_id in problem_list[problem](loss_key):
                metric_dict = load_config(wandb_id)
                #print(metric_dict.keys())
                if("sinkhorn_divergence" in metric_dict.keys()):
                    Curves[loss_key].append(np.array(metric_dict["sinkhorn_divergence"])[-1])
                else:
                    Curves[loss_key].append(np.array(metric_dict["Free_Energy_at_T=1"]))
            
            #[print(curve) for curve in Curves]
            print(problem ,loss_key)
            if("sinkhorn_divergence" in metric_dict.keys()):
                curve_per_seed = np.array([np.min(value) for value in Curves[loss_key]])
                mean_over_seeds = np.mean(curve_per_seed)
                std_over_seeds = np.std(curve_per_seed)/np.sqrt(len(curve_per_seed))
                mean_over_seeds_rounded = np.round(mean_over_seeds, 2)
                std_over_seeds_rounded = np.round(std_over_seeds, 3)
                print("Sinkhorn", f"${mean_over_seeds_rounded:.2f}"+ r"\text{\tiny{$\pm " +  f"{std_over_seeds_rounded}$" + "}}$")
            else:
                curve_per_seed = np.array([np.min(curve) for curve in Curves[loss_key]])
                mean_over_seeds = np.mean(curve_per_seed)
                std_over_seeds = np.std(curve_per_seed)/np.sqrt(len(curve_per_seed))
                mean_over_seeds_rounded = np.round(mean_over_seeds, 2)
                std_over_seeds_rounded = np.round(std_over_seeds, 3)
                print("Free Energy", f"${-mean_over_seeds_rounded:.2f}"+ r"\text{\tiny{$\pm " +  f"{std_over_seeds_rounded}$" + "}}$")


