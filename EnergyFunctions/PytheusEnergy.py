from .BaseEnergy import EnergyModelClass
import jax
import jax.numpy as jnp
from functools import partial

from scipy.optimize import minimize
from scipy.spatial.distance import cosine as cos_dist
from pytheus import theseus as th
from pytheus.fancy_classes import Graph, State
import numpy as np
#import optax



class PytheusEnergyClass(EnergyModelClass):
    def __init__(self, config, a=1.0):
        """
        Initialize the Mexican Hat Potential.
        
        :param a: Position of the minima.

        """
        print("Start initializing PytheusEnergyClass")
        challenge_index = config["challenge_index"]
        challenges = [(4,4,2),(5,4,4),(6,4,6),(7,4,8),(8,4,10),(9,4,12)]

        dim, n_ph, anc = challenges[challenge_index]
        halo_state = [str(ii)*n_ph+'0'*anc for ii in range(dim)]
        target_state = State(halo_state)

        dimensions = [dim]*n_ph+[1]*anc
        all_edges = th.buildAllEdges(dimensions)

        mathings_catalog = th.allPerfectMatchings(dimensions)
        SPACE_BASIS = list(mathings_catalog.keys())

        PERFECT_MATCHINGS = {} 
        for ket, pm_list in mathings_catalog.items():
            PERFECT_MATCHINGS[ket] = [[all_edges.index(edge) for edge in pm] for pm in pm_list]
        PERFECT_MATCHINGS = np.array(list(PERFECT_MATCHINGS.values()))

        target_unnormed = np.array([(key in target_state.kets)*1.0 for key in SPACE_BASIS])
        TARGET_NORMED = target_unnormed / np.sqrt(target_unnormed@target_unnormed)

        PERFECT_MATCHINGS = jnp.array(PERFECT_MATCHINGS)

        TARGET_NORMED = jnp.array(TARGET_NORMED)

        eps = 1e-10
        graph_state = lambda edges: edges[PERFECT_MATCHINGS].prod(axis=-1).sum(axis=-1)
        normed_state = lambda state: state / (eps + jnp.sqrt(state @ state))
        fidelity = lambda state: (state @ TARGET_NORMED)**2
        loss_fun = lambda x: - fidelity(normed_state(graph_state(x)))

        self.x_dim = len(all_edges)
        self.loss_fun = jax.jit(loss_fun)

        config["dim_x"] = self.x_dim
        print("End initializing PytheusEnergyClass")
        super().__init__(config)

    @partial(jax.jit, static_argnums=(0,))
    def calc_energy(self, x):
        """
        Calculate the energy of the Mexican Hat Potential.
        
        :param x: Input array.
        :return: Energy value.
        """

        return self.loss_fun(x)