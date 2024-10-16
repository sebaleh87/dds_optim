
import numpy as np
import jax.numpy as jnp

def compute_graph_state_stable(edges):
    print("edges", edges.shape)
    #max_edges = jax.lax.stop_gradient(jnp.max(edges))
    #edges = edges/max_edges
    log_edges = jnp.log(np.abs(edges))
    log_product = jnp.sum(log_edges, axis=-1)
    print(log_product)
    product = jnp.exp(log_product) #jax.nn.softmax(log_edges, axis = -1)#
    minus_sign = (-1)**(jnp.sum(jnp.where(edges < 0, 1, 0), axis = -1))
    print("edges", product.shape)
    graph_state = jnp.sum(minus_sign*product, axis=-1)
    return graph_state


if(__name__ == "__main__"):

    graph_state = lambda edges: edges.prod(axis=-1).sum(axis=-1) 

    edges = jnp.array(np.random.normal(size = (3,100)))
    #edges = jnp.ones((3,100))
    edges = jnp.array(np.random.uniform(size = (3,100)))

    print("numerically stable", compute_graph_state_stable(edges))
    print("unsable", graph_state(edges))