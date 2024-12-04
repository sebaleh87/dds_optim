import jax
from jax import grad, jit
import optax

shift = 5

def my_function(x):
    A = 10
    x = x - shift
    energy_value = A * x.shape[-1] + jnp.sum(x**2 - A * jnp.cos(2 * jnp.pi * x), axis = -1)
    return energy_value


if(__name__ == "__main__"):
    import jax.numpy as jnp
    import os
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
    os.environ["CUDA_VISIBLE_DEVICES"]=f"{7}"

    # Define the gradient of the function
    import jax
    import jax.numpy as jnp
    import optax
    import numpy as np
    from matplotlib import pyplot as plt

    # Set up the optimizer
    learning_rates =  [1., 0.1, 0.01, 0.001, 0.0001]
    for lr in learning_rates:
        optimizer = optax.adam(lr)
        n = 1000*200

        # Initialize the parameters and optimizer state
        x = jnp.array(np.random.normal(size = (n, 2)))  # Initial guess for x
        opt_state = jax.vmap(optimizer.init, in_axes=(0,))(x)

        # Define the loss function
        def loss_fn(x):
            return my_function(x)

        # Create a function to update the parameters
        @jax.jit
        def update(x, opt_state):
            # Compute the gradient of the loss function
            grads = jax.grad(loss_fn)(x)
            # Update the parameters using the optimizer
            updates, opt_state = optimizer.update(grads, opt_state)
            new_x = optax.apply_updates(x, updates)
            return new_x, opt_state

        vmap_update = jax.vmap(update, in_axes=(0, 0))

        # Optimization loop
        num_iterations = 100
        for i in range(num_iterations):
            x, opt_state = vmap_update(x, opt_state)
            if i % 10 == 0:
                print(f"Iteration {i}: x = {x}, my_function(x) = {my_function(x)}")

        # Final result
        last_samples = x
        print(f"Optimal x: {x}")
        print(f"Minimum value of the function: {my_function(x)}")

        # Create a grid of points
        range_ = 4
        x = np.linspace(-range_ + shift, range_+ shift, 400)
        y = np.linspace(-range_+ shift, range_+ shift, 400)
        X, Y = np.meshgrid(x, y)
        pos = jnp.concatenate([X[...,None], Y[...,None]], axis=-1)
        Z = my_function(pos)

        # Plot the contour
        plt.figure(figsize=(8, 6))
        contour = plt.contour(X, Y, Z, levels=50)
        plt.colorbar(contour)

        # Plot the trajectory
        plt.plot(last_samples[:,  0], last_samples[:,  1], 'x', color = "red", label='Newton\'s Method Trajectory')
        # Add labels and title
        plt.xlim(-range_ , range_+ shift)
        plt.ylim(-range_ , range_+ shift)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title(f'Contour plot of my_function with Newton\'s Method Trajectory, average minima')
        try:
            plt.savefig(os.getcwd() + f"/Denoising_diff_sampler/Figures/adam_{lr}.png")
        except:
            plt.savefig(os.getcwd() + f"//Figures/adam_{lr}.png")     
