import numpyro
import pickle
import jax.numpy as np
import numpyro.distributions as pydist
from .data_utils import standardize_and_pad
import numpyro
import os

def load_data(dset):
    current_folder = os.path.dirname(os.path.abspath(__file__))
    if dset == "Australian":
        with open(current_folder + "/Datasets/australian_full.pkl", "rb") as f:
            X, Y = pickle.load(f)
        Y = (Y + 1) // 2
    if dset == "Ionosphere":
        with open(current_folder + "/Datasets/ionosphere_full.pkl", "rb") as f:
            X, Y = pickle.load(f)
        Y = (Y + 1) // 2
    if dset == "Sonar":
        with open(current_folder + "/Datasets/sonar_full.pkl", "rb") as f:
            X, Y = pickle.load(f)
        Y = (Y + 1) // 2
    if dset == "A1a":
        with open(current_folder + "/Datasets/a1a_full.pkl", "rb") as f:
            X, Y = pickle.load(f)
        Y = (Y + 1) // 2
    if dset == "Madelon":
        with open(current_folder + "/Datasets/madelon_full.pkl", "rb") as f:
            X, Y = pickle.load(f)
        Y = (Y + 1) // 2
    X = standardize_and_pad(X)
    return X, Y


def load_model_lr(dset):
    def model(Y):
        w = numpyro.sample("weights", pydist.Normal(np.zeros(dim), np.ones(dim)))
        logits = np.dot(X, w)
        with numpyro.plate("J", n_data):
            y = numpyro.sample("y", pydist.BernoulliLogits(logits), obs=Y)

    X, Y = load_data(dset)
    dim = X.shape[1]
    n_data = X.shape[0]
    model_args = (Y,)
    return model, model_args