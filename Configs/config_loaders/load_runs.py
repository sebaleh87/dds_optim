import numpy as np
from loading import load_config
import matplotlib.pyplot as plt
import os

def compute_average_and_variance(curve_per_seed, round_mean = 2, round_sdt = 3):
    mean_over_seeds = np.mean(curve_per_seed)
    std_over_seeds = np.std(curve_per_seed)/np.sqrt(len(curve_per_seed))
    mean_over_seeds_rounded = np.round(mean_over_seeds, round_mean)
    std_over_seeds_rounded = np.round(std_over_seeds, round_sdt)
    return mean_over_seeds_rounded, std_over_seeds_rounded

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
        wandb_ids = ["honest-grass-11", "eager-brook-7", "grateful-haze-5"]

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
        wandb_ids = ["noble-glitter-8", "summer-durian-13", "sage-tree-19"]
    elif(key == "rKL_logderiv"):
        wandb_ids = ["dry-gorge-4", "clear-aardvark-3", "brisk-sea-1"]
    elif(key == "rKL_logderiv_frozen"):
        wandb_ids = ["fresh-pine-10", "polished-water-15", "resilient-yogurt-20"]
    elif(key == "LogVarLoss"):
        wandb_ids = ["rosy-elevator-11", "stellar-deluge-14", "generous-pyramid-17"]
    elif(key == "LogVarLoss_frozen"):
        wandb_ids = ["fluent-wind-9", "comic-microwave-12", "autumn-firebrand-18"]  
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
        wandb_ids = ["stilted-forest-24", "graceful-bush-21", "mild-river-18"]
    elif(key == "rKL_logderiv_frozen"):
        wandb_ids = ["colorful-sunset-14", "breezy-planet-9", "hopeful-butterfly-7"]
    elif(key == "LogVarLoss"):
        wandb_ids = ["clean-durian-23", "iconic-puddle-20", "major-frog-19"]
    elif(key == "LogVarLoss_frozen"):
        wandb_ids = ["skilled-snow-16", "cool-field-15", "sleek-yogurt-17"]
    return wandb_ids

def MoS(key):
    #wandb_ids = ["amber-smoke-4", "lilac-plant-3", "crimson-eon-2", "helpful-thunder-1"]
    if(key == "rKL_frozen"):
        wandb_ids = ["elderberry-meringue-10", "buttermilk-brownie-4"]
    elif(key == "rKL_logderiv"):
        wandb_ids = ["bumbleberry-cake-13", "chocolate-strudel-8", "peach-tart-1"]
    elif(key == "rKL_logderiv_frozen"):
        wandb_ids = ["bumbleberry-pastry-11", "carrant-flambee-6", "hershey-flan-3"]
    elif(key == "LogVarLoss"):
        wandb_ids = ["custard_brulee-14", "pumpkin-pastry-9", "key-lime-crumble-2"]
    elif(key == "LogVarLoss_frozen"):
        wandb_ids = ["butterscotch-brulee-12", "spiced-bun-7", "buttermilk-cake-5"]
    return wandb_ids

def GMM(key):
    #wandb_ids = ["amber-smoke-4", "lilac-plant-3", "crimson-eon-2", "helpful-thunder-1"]
    if(key == "rKL_frozen"):
        wandb_ids = ["stilted-wildflower-21", "dry-silence-17", "celestial-feather-11"]
    elif(key == "rKL_logderiv"):
        wandb_ids = ["gallant-monkey-1", "fresh-dream-9", "rosy-fire-15"]
    elif(key == "rKL_logderiv_frozen"):
        wandb_ids = ["smart-forest-20", "fluent-terrain-16", "fearless-frost-10"]
    # elif(key == "rKL_logderiv_off_policy"):
    #     wandb_ids = ["curious-deluge-18", "prime-tarrain-17", "peach-sound-16"]
    elif(key == "LogVarLoss"):
        wandb_ids = ["valiant-serenity-19", "misunderstood-dragon-14", "splendid-grass-8"]#["generous-sun-4", "pious-deluge-3", "ruby-blaze-2"]
    elif(key == "LogVarLoss_frozen"):
        wandb_ids = ["prime-totem-18", "curious-bee-13", "solar-violet-12"]
    return wandb_ids

def MoS_DBS(key):
    #wandb_ids = ["amber-smoke-4", "lilac-plant-3", "crimson-eon-2", "helpful-thunder-1"]
    if(key == "rKL_frozen"):
        wandb_ids = ["tough-waterfall-17", "skilled-disco-9", "fiery-sun-4"]
    elif(key == "rKL_logderiv"):
        wandb_ids = ["drawn-fire-34", "sage-eon-32", "likely-oath-29"]
    elif(key == "rKL_logderiv_frozen"):
        wandb_ids = ["robust-cosmos-33", "restful-sound-31", "polar-aardvark-30"]
    elif(key == "LogVarLoss"):
        wandb_ids = ["swept-wind-16", "eager-eon-10", "graceful-eon-5"]
    elif(key == "LogVarLoss_frozen"):
        wandb_ids = ["misty-microwave-15", "major-planet-6", "colorful-morning-2"]
    return wandb_ids

def GMM_DBS(key):
    #wandb_ids = ["amber-smoke-4", "lilac-plant-3", "crimson-eon-2", "helpful-thunder-1"]
    if(key == "rKL_frozen"):
        wandb_ids = ["vivid-forest-40", "peachy-morning-37", "restful-vortex-31"]
    elif(key == "rKL_logderiv"):
        wandb_ids = ["sweet-monkey-34", "efficient-sky-27", "chocolate-hill-17"]
    elif(key == "rKL_logderiv_frozen"):
        wandb_ids = ["true-fog-33", "spring-pond-26", "prime-wood-14"]
    elif(key == "LogVarLoss"):
        wandb_ids = ["pious-plasma-35", "swept-snowflake-38", "trim-wildflower-32"]
    elif(key == "LogVarLoss_frozen"):
        wandb_ids = ["fanciful-firefly-39", "confused-tree-36", "playful-haze-30"]
    return wandb_ids

def Funnel_DBS(key):
    #wandb_ids = ["amber-smoke-4", "lilac-plant-3", "crimson-eon-2", "helpful-thunder-1"]
    if(key == "rKL_frozen"):
        wandb_ids = ["iconic-pine-20", "golden-dawn-15", "glamorous-dragon-10"]
    elif(key == "rKL_logderiv"):
        wandb_ids = ["pious-terrain-16", "feasible-aardvark-11", "splendid-wildflower-8"]
    elif(key == "rKL_logderiv_frozen"):
        wandb_ids = ["young-forest-18", "silver-salad-13", "misty-plant-7"]
    elif(key == "LogVarLoss"):
        wandb_ids = ["faithful-yogurt-17", "fine-silence-12", "smart-leaf-6"]
    elif(key == "LogVarLoss_frozen"):
        wandb_ids = ["leafy-haze-19", "gentle-sponge-14", "wise-lake-9"]
    return wandb_ids

if(__name__ == "__main__"):
    loss_keys = ["rKL_frozen", "rKL_logderiv", "rKL_logderiv_frozen", "LogVarLoss", "LogVarLoss_frozen"]

    problem_list = {"Seeds":  Seeds, "Sonar": Sonar, "Credit": Credit, "Funnel": Funnel, "Brownian": Brownian, 
                    "LGCP": LGCP, "GMM": GMM, "GMM-DBS": GMM_DBS, "MoS-DBS": MoS_DBS, "Funnel-DBS": Funnel_DBS}#"MoS": MoS} #"Brownian": Brownian, #"LGCP": LGCP
    k = 10
    for problem in problem_list.keys():

        if(problem == "MoS" or problem == "GMM"):
            loss_keys = [ "rKL_logderiv", "rKL_logderiv_frozen", "LogVarLoss", "LogVarLoss_frozen", "rKL_frozen"]

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

                print("Sinkhorn", f"${sink_mean_over_seeds_rounded:.2f}"+ r"\text{\tiny{$\pm " +  f"{sink_std_over_seeds_rounded}$" + "}}$")
                print("Free Energy", f"${-Free_energy_mean_over_seeds_rounded:.2f}"+ r"\text{\tiny{$\pm " +  f"{Free_energy_std_over_seeds_rounded}$" + "}}$")

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
                    plt.savefig(os.path.dirname(os.path.abspath(__file__)) + f'/{problem}_{loss_key}_curves.png', dpi=1000)
                    plt.close()



            else:
                curve_per_seed = np.array([np.nanmin(curve) for curve in Curves[loss_key]["Free_Energy_at_T=1"]])
                mean_over_seeds_rounded, std_over_seeds_rounded = compute_average_and_variance(curve_per_seed)
                print("Free Energy", f"${-mean_over_seeds_rounded:.2f}"+ r"\text{\tiny{$\pm " +  f"{std_over_seeds_rounded}$" + "}}$")

                log_Z_value_per_seed = np.array([np.nanmin(curve) for curve in Curves[loss_key]["log_Z_at_T=1"]])
                log_Z_mean_over_seeds_rounded, log_Z_std_over_seeds_rounded = compute_average_and_variance(log_Z_value_per_seed)
                print("log_Z", f"${log_Z_mean_over_seeds_rounded:.2f}"+ r"\text{\tiny{$\pm " +  f"{log_Z_mean_over_seeds_rounded}$" + "}}$")



