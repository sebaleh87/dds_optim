from scipy.optimize import minimize
from scipy.spatial.distance import cosine as cos_dist
from pytheus import theseus as th
from pytheus.fancy_classes import Graph, State
import numpy as np
import jax.numpy as jnp
import jax
import optax
import os

if(__name__ == '__main__'):
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
    os.environ["CUDA_VISIBLE_DEVICES"]=f"{5}"

    # def gaussian(x, mean, variance):
    #     return -jnp.exp(jnp.sum(-0.5 * ((x[None,:] - mean) ** 2 / variance) - 0.5*jnp.log(2 * jnp.pi * variance), axis = -1))

    # means = jnp.array([[0., 10.], [11,0]])
    # variances = jnp.array([[1., 1.], [1.,1.]])
    # xs = [jnp.array([11,0])]
    # for x in xs:
    #     print(-jnp.exp(-0.5 * jnp.sum((x - means[1]) ** 2 / variances[1], axis = -1) )/jnp.sqrt(4*jnp.pi**2*variances[1,0]*jnp.pi*variances[1,1]))
    #     value = jnp.sum(gaussian(x, means, variances), axis=0)
    #     print(x, value)
    # raise ValueError("")

    challenges = [(4,4,2),(5,4,4),(6,4,6),(7,4,8),(8,4,10),(9,4,12)]


    test_arr = jnp.arange(0,100)

    cumsum = jax.lax.cumsum(test_arr, axis=0)

    dim, n_ph, anc = challenges[1]
    halo_state = [str(ii)*n_ph+'0'*anc for ii in range(dim)]
    target_state = State(halo_state)
    target_state

    dimensions = [dim]*n_ph+[1]*anc
    all_edges = th.buildAllEdges(dimensions)
    len(all_edges)

    mathings_catalog = th.allPerfectMatchings(dimensions)
    SPACE_BASIS = list(mathings_catalog.keys())

    PERFECT_MATCHINGS = {} 
    for ket, pm_list in mathings_catalog.items():
        PERFECT_MATCHINGS[ket] = [[all_edges.index(edge) for edge in pm] for pm in pm_list]
    PERFECT_MATCHINGS = np.array(list(PERFECT_MATCHINGS.values()))

    target_unnormed = np.array([(key in target_state.kets)*1.0 for key in SPACE_BASIS])
    TARGET_NORMED = target_unnormed / np.sqrt(target_unnormed@target_unnormed)

    PERFECT_MATCHINGS = np.array(PERFECT_MATCHINGS)
    TARGET_NORMED = np.array(TARGET_NORMED)

    eps = 1e-24
    graph_state = lambda edges: edges[PERFECT_MATCHINGS].prod(axis=-1).sum(axis=-1)
    normed_state = lambda state: state / (eps + np.sqrt(state @ state))
    #normed_state = lambda state: state / max([eps,  np.sqrt(state @ state)])
    fidelity = lambda state: (state @ TARGET_NORMED)**2
    loss_fun = lambda x: - fidelity(normed_state(graph_state(x)))


    # boundaries = [(-1,1)] * len(all_edges)
    # epochs = 1000
    # result_list = []
    # for epoch in range(epochs):
    #     x0 = np.random.rand(len(all_edges))
    #     result = minimize(loss_fun, #lambda x: loss_fun(x) + sum(x**2)*1e-10,
    #                     x0, bounds= boundaries,method='L-BFGS-B')
    #     result_list.append(result)
    #     print("epoch", epoch, "result", result.fun)
    #     print("best result", min([result.fun for result in result_list]))


    PERFECT_MATCHINGS = jnp.array(PERFECT_MATCHINGS)

    TARGET_NORMED = jnp.array(TARGET_NORMED)

    eps = 1e-10
    graph_state = lambda edges: edges[PERFECT_MATCHINGS].prod(axis=-1).sum(axis=-1)
    normed_state = lambda state: state / (eps + jnp.sqrt(state @ state))
    fidelity = lambda state: (state @ TARGET_NORMED)**2
    loss_fun = lambda x: - fidelity(normed_state(graph_state(x)))
    x0 = np.random.rand(len(all_edges))
    x0 = jnp.array(x0)

    graph_state = jax.jit(graph_state)
    state0 = graph_state(x0)

    normed_state = jax.jit(normed_state)
    normed1 = normed_state(state0)

    fidelity = jax.jit(fidelity)
    fid0 = fidelity(normed1)

    loss_fun = jax.jit(loss_fun)
    loss0 = loss_fun(x0)

    loss_fun(x0)

    x0 = jnp.array(np.random.rand(len(all_edges)))

    optimizer = optax.adam(learning_rate=0.1)

    x = jnp.array(x0)  # Initial value of x
    opt_state = optimizer.init(x)

    def update_step(x, opt_state):
        # Compute the gradient of the objective function w.r.t. x
        loss, grads = jax.value_and_grad(loss_fun)(x)
        
        # Compute updates using the optimizer
        updates, opt_state = optimizer.update(grads, opt_state)
        
        # Apply the updates to x
        x = optax.apply_updates(x, updates)
        
        return x, opt_state, loss
    
    reps = 1000
    result_list = []
    for rep in range(reps):
        x0 = jnp.array(np.random.rand(len(all_edges)))
        
        optimizer = optax.adam(learning_rate=0.1)
        x = jnp.array(x0)  # Initial value of x
        opt_state = optimizer.init(x)

        num_steps = 1000
        for step in range(num_steps):
            x, opt_state, loss = update_step(x, opt_state)
            result_list.append(np.array(loss))
            if step % 10 == 0:
                print(f"Step {step}: loss = {loss}.")#, x = {x}")
        print("Best value is", min(result_list))
        print(jnp.min(jnp.abs(x)), jnp.max(jnp.abs(x)))

    print(f"Final optimized x, with loss {loss},\n{x}")

