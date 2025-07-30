
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
        wandb_ids = ["splendid-hill-46", "sunny-resonance-45", "proud-valley-44"]
    elif(key == "rKL_logderiv"):
        wandb_ids = [ "star-force-16", "rebel-fleet-20", "carbonite-wars-24",
                     "galactic-admiral-28", "holographic-commander-32", "stellar-commander-36", "holographic-wars-40"] #"bumbleberry-cake-13", "chocolate-strudel-8", "peach-tart-1",
    elif(key == "rKL_logderiv_frozen"):
        wandb_ids = [ "scruffy-looking-pilot-18", "stellar-podracer-21", "sith-womprat-25", "scruffy-looking-rancor-29",
                     "ancient-womprat-33", "civilized-nexu-37", "hokey-wookie-41"] #"bumbleberry-pastry-11", "currant-flambee-6", "hershey-flan-3",
    elif(key == "LogVarLoss"):
        wandb_ids = [ "forgotten-wars-17", "rebel-tie-fighter-23", "grievous-speeder-27", "hokey-rancor-31",
                     "dark-nexu-35", "mythical-jawa-39", "civilized-lightsaber-43"] #"custard-brulee-14", "pumpkin-pastry-9", "key-lime-crumble-2",
    elif(key == "LogVarLoss_frozen"):
        wandb_ids = [ "jedi-bothan-19", "old-pilot-22", "light-parsec-26", "legendary-droid-30", "old-council-34"
                     , "scruffy-looking-republic-38", "jedi-emperor-42"] #"butterscotch-brulee-12", "spiced-bun-7", "buttermilk-cake-5",
    return wandb_ids

def GMM(key):
    #wandb_ids = ["amber-smoke-4", "lilac-plant-3", "crimson-eon-2", "helpful-thunder-1"]
    if(key == "rKL_frozen"):
        wandb_ids = ["hearty-mountain-54", "upbeat-valley-53", "decent-eon-52"]
    elif(key == "rKL_logderiv"):
        wandb_ids = [ "clone-trooper-25", "legendary-bantha-29", "jedi-fighter-33", "galactic-speeder-37", "tusken-republic-41"
                    ,"grievous-carrier-45", "carbonite-lightsaber-48"] #"gallant-monkey-1", "fresh-dream-9", "rosy-fire-15",
    elif(key == "rKL_logderiv_frozen"):
        wandb_ids = [ "ancient-parsec-23", "jedi-womprat-31", "star-council-35", "imperial-wookie-39",
                     "forgotten-bantha-43", "imperial-republic-49", "mythical-ewok-51"] #"smart-forest-20", "fluent-terrain-16", "fearless-frost-10",
    # elif(key == "rKL_logderiv_off_policy"):
    #     wandb_ids = ["curious-deluge-18", "prime-tarrain-17", "peach-sound-16"]
    elif(key == "LogVarLoss"):
        wandb_ids = [ "forgotten-bothan-24", "jedi-ewok-28",
                     "grievous-parsec-32", "mythical-xwing-36", "dark-parsec-40", "star-shuttle-44", "jedi-admiral-47"] #"valiant-serenity-19", "misunderstood-dragon-14", "splendid-grass-8",
    elif(key == "LogVarLoss_frozen"):
        wandb_ids = [ "light-trooper-26", "imperial-midichlorian-30", "legendary-force-34", "star-speeder-38",
                     "elegant-wookie-42", "carbonite-tie-fighter-46", "hokey-midichlorian-50"] # "prime-totem-18", "curious-bee-13", "solar-violet-12",
    return wandb_ids

def MoS_DBS(key):
    #wandb_ids = ["amber-smoke-4", "lilac-plant-3", "crimson-eon-2", "helpful-thunder-1"]
    if(key == "rKL_frozen"):
        wandb_ids = ["devoted-blaze-79", "resilient-pine-78", "deft-thunder-77"]
    elif(key == "rKL_logderiv"):
        wandb_ids = [ "sith-wars-35", "hokey-nexu-39", "ancient-podracer-43", "clean-forest-47",
                     "cerulean-disco-51", "robust-bee-55", "chocolate-leaf-59"] #"drawn-fire-34", "sage-eon-32", "likely-oath-29",
    elif(key == "rKL_logderiv_frozen"):
        wandb_ids = [ "wise-universe-64", "fiery-grass-66", "woven-cloud-68", "polished-shape-70", "sandy-armadillo-72", "faithful-tree-73", "balmy-glitter-75"] #"robust-cosmos-33", "restful-sound-31", "polar-aardvark-30",
    elif(key == "LogVarLoss"):
        wandb_ids = ["prime-monkey-63", "brisk-pine-65", "winter-music-67", "deft-disco-69", "balmy-music-71", "dandy-glitter-74", "rosy-elevator-76" ] # "swept-wind-16", "eager-eon-10", "graceful-eon-5",
    elif(key == "LogVarLoss_frozen"):
        wandb_ids = [ "rogue-master-38", "old-shuttle-42", "clone-cantina-46", "honest-glade-50", "sunny-gorge-54",
                     "whole-firefly-58", "deft-moon-61"] #"misty-microwave-15", "major-planet-6", "colorful-morning-2",
    return wandb_ids

def GMM_DBS(key):
    #wandb_ids = ["amber-smoke-4", "lilac-plant-3", "crimson-eon-2", "helpful-thunder-1"]
    if(key == "rKL_frozen"):
        wandb_ids = ["sandy-lion-71", "dutiful-serenity-70", "dazzling-flower-69"]
    elif(key == "rKL_logderiv"):
        wandb_ids = ["summer-wildflower-66", "comic-monkey-62", "devoted-elevator-58", "wise-star-54", "sith-council-50", "forgotten-rancor-46", "dark-force-43"]
    elif(key == "rKL_logderiv_frozen"):
        wandb_ids = ["elegant-republic-41", "sith-xwing-48", "fearless-energy-52", "fine-serenity-56", "good-vortex-60", "worthy-darkness-64", "atomic-morning-68"]
    elif(key == "LogVarLoss"):
        wandb_ids = ["floral-butterfly-65", "jumping-paper-61", "fallen-tree-57", "valiant-rain-53", "imperial-astromech-49", "elegant-lightsaber-45", "hokey-nerf-herder-44"]
    elif(key == "LogVarLoss_frozen"):
        wandb_ids = ["noble-sound-67", "pleasant-dew-63", "helpful-grass-59", "rural-grass-55", "tusken-wookie-51", "jedi-wookie-47", "dark-republic-42"]
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

def MW54(key):
    if(key == "rKL_frozen"):
        wandb_ids = ["splendid-paper-54", "jolly-sun-53", "easy-night-51"]
    elif(key == "rKL_logderiv"):
        wandb_ids = ["radiant-armadillo-32", "wandering-grass-28", "iconic-serenity-21"]
    elif(key == "rKL_logderiv_frozen"):
        wandb_ids = ["effortless-yogurt-31", "fanciful-donkey-27", "bright-capybara-22"]
    elif(key == "LogVarLoss"):
        wandb_ids = ["wild-bee-29", "misunderstood-tree-25", "robust-energy-23"]
    elif(key == "LogVarLoss_frozen"):
        wandb_ids = ["neat-totem-37", "grateful-sky-30", "comic-dawn-26"]
    return wandb_ids

def MW54_DBS(key):
    if(key == "rKL_frozen"):
        wandb_ids = ["skilled-star-58", "graceful-armadillo-57", "restful-disco-55"]
    elif(key == "rKL_logderiv"):
        wandb_ids = ["dry-totem-47", "curious-monkey-43", "visionary-river-39"]
    elif(key == "rKL_logderiv_frozen"):
        wandb_ids = ["twilight-donkey-48", "swift-field-44", "dulcet-water-40"]
    elif(key == "LogVarLoss"):
        wandb_ids = ["whole-fog-46", "firm-wood-42", "unique-glade-38"]
    elif(key == "LogVarLoss_frozen"):
        wandb_ids = ["eager-darkness-49", "driven-tree-45", "icy-tree-41"]
    return wandb_ids

def GMM_100D_CMCD(key):
    if(key == "rKL_frozen"):
        wandb_ids = ["leafy-paper-39", "deep-sound-35", "frosty-lion-31", "ethereal-bee-26", "zany-dawn-23", "fragrant-elevator-19", 
                     "rare-energy-15", "eager-disco-11", "ancient-universe-7", "crisp-grass-3"]
    elif(key == "rKL_logderiv"):
        wandb_ids = ["lilac-bee-36", "valiant-morning-32", "playful-yogurt-28", "silver-frost-24", "stilted-firefly-22", "stilted-puddle-18",
                     "serene-flower-16", "celestial-star-12", "super-frog-8", "volcanic-thunder-4"]
    elif(key == "rKL_logderiv_frozen"):
        wandb_ids = ["icy-sun-40", "deep-flower-37", "logical-gorge-33", "brisk-dream-29", "snowy-universe-25", "driven-plant-20", "kind-waterfall-13",
                     "youthful-surf-9", "bright-firefly-5", "deep-sunset-1"]
    elif(key == "LogVarLoss"):
        wandb_ids = ["wise-snowball-43", "denim-energy-41", "devoted-microwave-42"]
    elif(key == "LogVarLoss_frozen"):
        wandb_ids = ["major-wave-38", "robust-dew-34", "usual-night-30", "devoted-resonance-27", "fanciful-voice-21", 
                     "polar-frost-17", "soft-night-14", "fast-sky-10", "swept-spaceship-6", "wise-terrain-2"]
    return wandb_ids


def GMM_100D_DBS(key):
    if(key == "rKL_frozen"):
        wandb_ids = ["winter-sponge-18", "effortless-bird-15", "flowing-shape-8", "exalted-star-3"]
    elif(key == "rKL_logderiv"):
        wandb_ids = ["hopeful-silence-26", "dashing-glade-20", "glorious-totem-17", "glamorous-lion-12", "fallen-microwave-7"]
    elif(key == "rKL_logderiv_frozen"):
        wandb_ids = ["crisp-leaf-31", "woven-sky-29", "fresh-valley-27", "quiet-haze-23", "happy-galaxy-21", "unique-dragon-13", "major-breeze-9",
                      "stoic-pyramid-4", "ethereal-dew-1"]
    elif(key == "LogVarLoss"):
        wandb_ids = ["polished-water-25", "wild-wave-19", "upbeat-firefly-16", "floral-sky-11", "jolly-night-6"]
    elif(key == "LogVarLoss_frozen"):
        wandb_ids = ["fiery-brook-33", "absurd-feather-32", "genial-cosmos-30", "mild-deluge-28", "gentle-puddle-24", 
                     "crisp-resonance-22", "dandy-capybara-14", "olive-energy-10", "desert-river-5", "icy-eon-2"]
    return wandb_ids

loss_keys = [ "rKL_logderiv", "rKL_logderiv_frozen", "LogVarLoss", "LogVarLoss_frozen", "rKL_frozen"]