from .BaseEnergy import EnergyModelClass
import jax
import jax.numpy as jnp
from functools import partial
import pickle
from scipy.optimize import minimize
from scipy.spatial.distance import cosine as cos_dist
from pytheus import theseus as th
from pytheus.fancy_classes import Graph, State
import numpy as np
#import optax
from flax import linen as nn

@jax.jit
def compute_graph_state_stable(edges, PERFECT_MATCHINGS):
    #max_edges = jax.lax.stop_gradient(jnp.max(edges))
    #edges = edges/max_edges
    edges = edges[PERFECT_MATCHINGS]
    log_edges = jnp.log(jnp.abs(edges))
    log_product = jnp.sum(log_edges, axis=-1)

    product = jnp.exp(log_product) #jax.nn.softmax(log_edges, axis = -1)#
    minus_sign = (-1)**(jnp.sum(jnp.where(edges < 0, 1, 0), axis = -1))

    graph_state = jnp.sum(minus_sign*product, axis=-1)
    return graph_state

@jax.jit
def numerically_stable_norm(state, eps = 10**-20):  ### TODO check if an epsilon is necssary here?
    max_state = jax.lax.stop_gradient(jnp.max(jnp.abs(state)))
    state = state/(max_state + eps)
    state_norm = jnp.linalg.norm(state) + eps
    normed_state = state / state_norm
    return normed_state

class PytheusEnergyClass(EnergyModelClass):
    def __init__(self, config):
        """
        Initialize the Mexican Hat Potential.
        
        :param a: Position of the minima.

        """
        print(f'Start initializing PytheusEnergyClass with challange index {config["challenge_index"]}')
        challenge_index = config["challenge_index"]
        challenges = [(4,4,2),(5,4,4),(6,4,6),(7,4,8),(8,4,10),(9,4,12)] # color, nodes , anc

        dim, n_ph, anc = challenges[challenge_index]
        halo_state = [str(ii)*n_ph+'0'*anc for ii in range(dim)]
        target_state = State(halo_state)

        dimensions = [dim]*n_ph+[1]*anc 
        all_edges = th.buildAllEdges(dimensions) # list of tuples (n1, n2, c1, c2)

        if challenge_index != 3:
            matchings_catalog = th.allPerfectMatchings(dimensions)
            SPACE_BASIS = list(matchings_catalog.keys())

            PERFECT_MATCHINGS = {} 
            for ket, pm_list in matchings_catalog.items():
                PERFECT_MATCHINGS[ket] = [[all_edges.index(edge) for edge in pm] for pm in pm_list]
            PERFECT_MATCHINGS = np.array(list(PERFECT_MATCHINGS.values()))

            target_unnormed = np.array([(key in target_state.kets)*1.0 for key in SPACE_BASIS])
            TARGET_NORMED = target_unnormed / np.sqrt(target_unnormed@target_unnormed)

            PERFECT_MATCHINGS = jnp.array(PERFECT_MATCHINGS)
            TARGET_NORMED = jnp.array(TARGET_NORMED)

        else:
            print("loading pytheus files for challenge 3")
            with open("/system/user/slehner/pytheus_files/matchings_3.pkl", "rb") as f:
                PERFECT_MATCHINGS = pickle.load(f)

            with open("/system/user/slehner/pytheus_files/target_normed_3.pkl", "rb") as f:
                TARGET_NORMED = pickle.load(f)
            
        
        self.PERFECT_MATCHINGS = PERFECT_MATCHINGS
        self.TARGET_NORMED = TARGET_NORMED

        self.eps = 1e-20
        graph_state = lambda edges: edges[PERFECT_MATCHINGS].prod(axis=-1).sum(axis=-1) # use logsumexp?!
        normed_state = lambda state, state_norm: state / (self.eps + state_norm)
        fidelity = lambda state: (state @ self.TARGET_NORMED)**2
        #loss_fun = lambda x: - fidelity(normed_state(graph_state(x)))

        self.x_dim = len(all_edges)
        self.fidelity = jax.jit(fidelity)
        self.graph_state = jax.jit(graph_state)
        self.norm_state = jax.jit(normed_state)

        config["dim_x"] = self.x_dim
        print("End initializing PytheusEnergyClass", "variable size is", self.x_dim)
        super().__init__(config)

    @partial(jax.jit, static_argnums=(0,))
    def energy_function(self, x):
        """
        Calculate the energy of the Mexican Hat Potential.
        
        :param x: Input array.
        :return: Energy value.
        """
        graph_state = self.graph_state(x)#compute_graph_state_stable(x, self.PERFECT_MATCHINGS)
        normed_state = numerically_stable_norm(graph_state)
        # graph_state = self.graph_state(x)
        # state_norm = jnp.linalg.norm(graph_state)
        # normed_state = self.norm_state(graph_state, state_norm)
        self.energy_value = -self.fidelity(normed_state) + 1

        #vec_norm = state_norm
        return self.energy_value# + vec_norm**2 *jnp.heaviside(vec_norm - 1.0, 0.0)
    


    #@jax.custom_gradient
#def sample_from_beta( a,b, subkey):
 #   samples = jax.random.beta(subkey, a,b, )
  #  pdf = jax.scipy.stats.beta.pdf(samples, a, b)
   # return samples, lambda g: (-g*jax.grad(tfp.math.betainc, argnums=0)(a, b, samples)/pdf, -g*jax.grad(tfp.math.betainc, argnums=1)(a, b, samples)/pdf, None)
