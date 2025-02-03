
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