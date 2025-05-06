
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
        wandb_ids = ["elderberry-meringue-10", "buttermilk-brownie-4", "custard-bun-15"]
    elif(key == "rKL_logderiv"):
        wandb_ids = ["bumbleberry-cake-13", "chocolate-strudel-8", "peach-tart-1", "star-force-16", "rebel-fleet-20", "carbonite-wars-24",
                     "galactic-admiral-28", "holographic-commander-32", "stellar-commander-36", "holographic-wars-40"]
    elif(key == "rKL_logderiv_frozen"):
        wandb_ids = ["bumbleberry-pastry-11", "currant-flambee-6", "hershey-flan-3", "scruffy-looking-pilot-18", "stellar-podracer-21", "sith-womprat-25", "scruffy-looking-rancor-29",
                     "ancient-womprat-33", "civilized-nexu-37", "hokey-wookie-41"]
    elif(key == "LogVarLoss"):
        wandb_ids = ["custard-brulee-14", "pumpkin-pastry-9", "key-lime-crumble-2", "forgotten-wars-17", "rebel-tie-fighter-23", "grievous-speeder-27", "hokey-rancor-31",
                     "dark-nexu-35", "mythical-jawa-39", "civilized-lightsaber-43"]
    elif(key == "LogVarLoss_frozen"):
        wandb_ids = ["butterscotch-brulee-12", "spiced-bun-7", "buttermilk-cake-5", "jedi-bothan-19", "old-pilot-22", "light-parsec-26", "legendary-droid-30", "old-council-34"
                     , "scruffy-looking-republic-38", "jedi-emperor-42"]
    return wandb_ids

def GMM(key):
    #wandb_ids = ["amber-smoke-4", "lilac-plant-3", "crimson-eon-2", "helpful-thunder-1"]
    if(key == "rKL_frozen"):
        wandb_ids = ["stilted-wildflower-21", "dry-silence-17", "celestial-feather-11"]
    elif(key == "rKL_logderiv"):
        wandb_ids = ["gallant-monkey-1", "fresh-dream-9", "rosy-fire-15", "clone-trooper-25", "legendary-bantha-29", "jedi-fighter-33", "galactic-speeder-37", "tusken-republic-41"
                    ,"grievous-carrier-45", "carbonite-lightsaber-48"]
    elif(key == "rKL_logderiv_frozen"):
        wandb_ids = ["smart-forest-20", "fluent-terrain-16", "fearless-frost-10", "ancient-parsec-23", "jedi-womprat-31", "star-council-35", "imperial-wookie-39",
                     "forgotten-bantha-43", "imperial-republic-49", "mythical-ewok-51"]
    # elif(key == "rKL_logderiv_off_policy"):
    #     wandb_ids = ["curious-deluge-18", "prime-tarrain-17", "peach-sound-16"]
    elif(key == "LogVarLoss"):
        wandb_ids = ["valiant-serenity-19", "misunderstood-dragon-14", "splendid-grass-8", "forgotten-bothan-24", "jedi-ewok-28",
                     "grievous-parsec-32", "mythical-xwing-36", "dark-parsec-40", "star-shuttle-44", "jedi-admiral-47"]
    elif(key == "LogVarLoss_frozen"):
        wandb_ids = ["prime-totem-18", "curious-bee-13", "solar-violet-12", "light-trooper-26", "imperial-midichlorian-30", "legendary-force-34", "star-speeder-38",
                     "elegant-wookie-42", "carbonite-tie-fighter-46", "hokey-midichlorian-50"]
    return wandb_ids

def MoS_DBS(key):
    #wandb_ids = ["amber-smoke-4", "lilac-plant-3", "crimson-eon-2", "helpful-thunder-1"]
    if(key == "rKL_frozen"):
        wandb_ids = ["tough-waterfall-17", "skilled-disco-9", "fiery-sun-4"]
    elif(key == "rKL_logderiv"):
        wandb_ids = ["drawn-fire-34", "sage-eon-32", "likely-oath-29", "sith-wars-35", "hokey-nexu-39", "ancient-podracer-43", "clean-forest-47",
                     "cerulean-disco-51", "robust-bee-55", "chocolate-leaf-59"]
    elif(key == "rKL_logderiv_frozen"):
        wandb_ids = ["robust-cosmos-33", "restful-sound-31", "polar-aardvark-30", "scruffy-looking-republic-37", "sith-republic-41", "stellar-droid-45", "woven-river-49",
                     "misunderstood-glade-53", "twilight-blaze-57", "cool-salad-62"]
    elif(key == "LogVarLoss"):
        wandb_ids = ["swept-wind-16", "eager-eon-10", "graceful-eon-5", "rebel-shuttle-36", "forgotten-nerf-herder-40", "civilized-parsec-44", "autumn-armadillo-48", "balmy-vortex-52"
                     , "grateful-salad-56", "gentle-music-60"]
    elif(key == "LogVarLoss_frozen"):
        wandb_ids = ["misty-microwave-15", "major-planet-6", "colorful-morning-2", "rogue-master-38", "old-shuttle-42", "clone-cantina-46", "honest-glade-50", "sunny-gorge-54",
                     "whole-firefly-58", "deft-moon-61"]
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

def Seeds_DBS(key):
    #wandb_ids = ["amber-smoke-4", "lilac-plant-3", "crimson-eon-2", "helpful-thunder-1"]
    if(key == "rKL_frozen"):
        wandb_ids = ["vivid-forest-14", "vivid-disco-11", "distinctive-dragon-9"]
    elif(key == "rKL_logderiv"):
        wandb_ids = ["rose-surf-6", "devout-shadow-4", "daily-frost-2"]
    elif(key == "rKL_logderiv_frozen"):
        wandb_ids = ["hopeful-dew-8", "eternal-spaceship-7", "robust-frog-5"]
    elif(key == "LogVarLoss"):
        wandb_ids = ["sage-frog-17", "rose-monkey-16", "eager-terrain-15"]
    elif(key == "LogVarLoss_frozen"):
        wandb_ids = ["stilted-valley-13", "chocolate-sea-12", "treasured-vortex-10"]
    return wandb_ids

def Sonar_DBS(key):
    #wandb_ids = ["amber-smoke-4", "lilac-plant-3", "crimson-eon-2", "helpful-thunder-1"]
    if(key == "rKL_frozen"):
        wandb_ids = ["clean-disco-26", "robust-grass-25", "giddy-yogurt-24"]
    elif(key == "rKL_logderiv"):
        wandb_ids = ["solar-terrain-11", "icy-wood-7", "fanciful-fire-2"]
    elif(key == "rKL_logderiv_frozen"):
        wandb_ids = ["leafy-snowball-23", "vocal-glade-22", "dainty-plant-20"]
    elif(key == "LogVarLoss"):
        wandb_ids = ["twilight-morning-13", "mild-glitter-10", "wild-pond-6"]
    elif(key == "LogVarLoss_frozen"):
        wandb_ids = ["lively-plasma-14", "fanciful-pyramid-9", "atomic-grass-5"]
    return wandb_ids

def LGCP_DBS(key):
    #wandb_ids = ["amber-smoke-4", "lilac-plant-3", "crimson-eon-2", "helpful-thunder-1"]
    if(key == "rKL_frozen"):
        wandb_ids = ["fancy-fog-13", "exalted-blaze-8", "vital-wind-3"]
    elif(key == "rKL_logderiv"):
        wandb_ids = ["genial-forest-14", "wandering-meadow-9", "worthy-forest-4"]
    elif(key == "rKL_logderiv_frozen"):
        wandb_ids = ["deep-snowball-12", "firm-waterfall-7", "wise-bee-2"]
    elif(key == "LogVarLoss"):
        wandb_ids = ["wobbly-pond-15", "comic-energy-10", "rose-lake-5"]
    elif(key == "LogVarLoss_frozen"):
        wandb_ids = ["bright-frog-11", "wobbly-serenity-6", "fearless-snowflake-1"]
    return wandb_ids

def Brownian_DBS(key):
    if(key == "rKL_frozen"):
        wandb_ids = ["daily-bee-15", "warm-deluge-11", "warm-fog-7"]
    elif(key == "rKL_logderiv"):
        wandb_ids = ["grateful-aardvark-3", "sandy-wildflower-2", "ethereal-bee-1"]
    elif(key == "rKL_logderiv_frozen"):
        wandb_ids = ["glamorous-bird-14", "kind-gorge-10", "glad-snowball-6"]
    elif(key == "LogVarLoss"):
        wandb_ids = ["winter-yogurt-13", "prime-breeze-8", "stoic-dew-4"]
    elif(key == "LogVarLoss_frozen"):
        wandb_ids = ["hearty-darkness-12", "snowy-dew-9", "fanciful-sun-5"]
    return wandb_ids

def German_DBS(key):
    if(key == "rKL_frozen"):
        wandb_ids = ["logical-hill-15", "easy-dust-12", "eager-vortex-6"]
    elif(key == "rKL_logderiv"):
        wandb_ids = []
    elif(key == "rKL_logderiv_frozen"):
        wandb_ids = ["logical-eon-14", "fiery-glade-11", "major-waterfall-5"]
    elif(key == "LogVarLoss"):
        wandb_ids = ["splendid-voice-8", "apricot-dragon-3", "celestial-sun-2"]
    elif(key == "LogVarLoss_frozen"):
        wandb_ids = ["pious-butterfly-18", "radiant-snow-17", "dutiful-lake-16"]
    return wandb_ids

