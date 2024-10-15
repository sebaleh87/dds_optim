import jax
import jax.numpy as jnp
from jax import grad, jacfwd, jacrev
import numpy as np


def compute_grad(f, x0):
    return grad(f)(x0)

def compute_hessian(f, x0):
    return hessian(f)(x0)

def compute_nat_grad(f, x, tol=1e-6):
    grad_f = grad(f)(x)  
    hess_f = hessian(f)(x)  

    #hess_inv = jnp.linalg.inv(hess_f)
    hess_inv = jnp.where(jnp.linalg.cond(hess_f) > 1 / tol, jnp.linalg.pinv(hess_f), jnp.linalg.inv(hess_f))
    # Newton's update rule
    return hess_inv @ grad_f

def hessian(f):
    """Computes the Hessian matrix of a scalar-valued function `f`."""
    return jacfwd(jacrev(f))

def newtons_method(f, x0, max_iters=100, tol=1e-6, lr = 10**-1, eps = 10**-6, hessian_mode = True):
    """Performs Newton's method to find a local minimum of `f` starting from `x0`."""
    x = x0
    x_list = [x0]
    for i in range(max_iters):
        grad_f = grad(f)(x)  # Gradient of the function
        hess_f = hessian(f)(x)  # Hessian matrix
        hess_f += jnp.eye(hess_f.shape[0]) * eps 
        
        # Check if the Hessian is singular or nearly singular
        # if jnp.linalg.cond(hess_f) > 1 / tol:
        #     print("Hessian is singular or nearly singular, using pseudo-inverse.")
        #     hess_inv = jnp.linalg.pinv(hess_f)
        # else:
        #     hess_inv = jnp.linalg.inv(hess_f)
        #hess_inv = jnp.where(jnp.linalg.cond(hess_f) > 1 / tol, jnp.linalg.pinv(hess_f), jnp.linalg.inv(hess_f))
        hess_inv = jnp.where(jnp.linalg.cond(hess_f) > 1 / tol, jnp.linalg.pinv(hess_f), jnp.linalg.inv(hess_f))
        # Newton's update rule
        if(hessian_mode):
            delta_x = hess_inv @ grad_f
        else:
            delta_x = grad_f
        x = x - lr*delta_x

        x_list.append(x)

        # Check for convergence
    #     if jnp.linalg.norm(delta_x) < tol:
    #         print(f"Converged in {i+1} iterations.")
    #         break
    # else:
    #     print("Reached maximum iterations without convergence.")

    return x, x_list

# Example usage
def my_function(x):
    A = 10
    energy_value = A * x.shape[-1] + jnp.sum(x**2 - A * jnp.cos(2 * jnp.pi * x), axis = -1)
    return energy_value

if(__name__ == "__main__"):
    import os
    from jax import vmap
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
    os.environ["CUDA_VISIBLE_DEVICES"]=f"{7}"

    # Vectorize newtons_method to handle multiple initial guesses
    n = 10
    initial_guesses = jnp.array(np.random.normal(size = (n, 2)))
    vectorized_newtons_method = vmap(lambda a,b: newtons_method(a,b, hessian_mode = True, lr = 10**-2), in_axes=(None, 0))

    minima, trajectories = vectorized_newtons_method(my_function, initial_guesses)
    print("Found minima at:", minima)

    import matplotlib.pyplot as plt

    x = np.linspace(-2, 2, 50)
    y = np.linspace(-2, 2, 50)
    X, Y = np.meshgrid(x, y)
    pos = jnp.concatenate([X[...,None], Y[...,None]], axis=-1)
    vmap_compute_grad = jax.vmap(jax.vmap(compute_grad, in_axes = (None, 0)), in_axes = (None, 0))
    vmap_compute_hess = jax.vmap(jax.vmap(compute_nat_grad, in_axes = (None, 0)), in_axes = (None, 0))
    grads = vmap_compute_grad(my_function, pos)
    hessians = vmap_compute_hess(my_function, pos)
    plt.figure()
    # Plot the gradient direction as quiver plot
    plt.quiver(X, Y, grads[:,:, 0], grads[:,:,1], color='black', alpha=1)
    x_list_np = np.array(trajectories)
        # Plot the trajectory
    plt.plot(x_list_np[:, :,  0], x_list_np[:, :,  1], '-x', color = "green", label='Newton\'s Method Trajectory')
    plt.scatter(x_list_np[0, :,  0], x_list_np[0, :,  1], color='red')
    try:
        plt.savefig(os.getcwd() + "/Denoising_diff_sampler/Figures/grads.png")
    except:
        plt.savefig(os.getcwd() + "/Figures/grads.png")

    plt.figure()
    # Plot the gradient direction as quiver plot
    plt.quiver(X, Y, hessians[:,:, 0], hessians[:,:,1], color='blue', alpha=1)
    x_list_np = np.array(trajectories)
        # Plot the trajectory
    plt.plot(x_list_np[:, :,  0], x_list_np[:, :,  1], '-x', color = "green", label='Newton\'s Method Trajectory')
    plt.scatter(x_list_np[0, :,  0], x_list_np[0, :,  1], color='red')
    try:
        plt.savefig(os.getcwd() + "/Denoising_diff_sampler/Figures/hessian.png")
    except:
        plt.savefig(os.getcwd() + "/Figures/grahessiands.png")

    # Create a grid of points
    x = np.linspace(-2, 2, 400)
    y = np.linspace(-2, 2, 400)
    X, Y = np.meshgrid(x, y)
    pos = jnp.concatenate([X[...,None], Y[...,None]], axis=-1)
    Z = my_function(pos)

    # Convert x_list to numpy array for plotting
    x_list_np = np.array(trajectories)

    # Plot the contour
    plt.figure(figsize=(8, 6))
    contour = plt.contour(X, Y, Z, levels=50)
    plt.colorbar(contour)

    # Plot the trajectory
    plt.plot(x_list_np[:, :,  0], x_list_np[:, :,  1], '-x', color = "green", label='Newton\'s Method Trajectory')
    plt.scatter(x_list_np[0, :,  0], x_list_np[0, :,  1], color='red')
    # Add labels and title
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(f'Contour plot of my_function with Newton\'s Method Trajectory, average minima {np.mean(my_function(minima))}')
    #plt.legend()
    try:
        plt.savefig(os.getcwd() + f"/Denoising_diff_sampler/Figures/hessian_cont.png")
    except:
        plt.savefig(os.getcwd() + f"//Figures/hessian_cont.png")          


    print("init Energy", np.mean(my_function(initial_guesses)))
    print("final Energy", np.mean(my_function(minima)))
    # Plot the trajectory with zoomed-in versions
    for i in range(n):
        plt.figure(figsize=(8, 6))
        x = np.linspace(x_list_np[:, i, 0].min() - 0.1, x_list_np[:, i, 0].max() + 0.1, 400)
        y = np.linspace(x_list_np[:, i, 1].min() - 0.1, x_list_np[:, i, 1].max() + 0.1, 400)
        X, Y = np.meshgrid(x, y)   
        # Compute the gradient at each point in the grid

        Z = my_function(jnp.concatenate([X[...,None], Y[...,None]], axis=-1))
        contour = plt.contour(X, Y, Z, levels=50)
        plt.colorbar(contour)

        # Plot the trajectory for the i-th initial guess
        plt.plot(x_list_np[:, i, 0], x_list_np[:, i, 1], '-x', label=f'Trajectory {i+1}')
        plt.scatter(x_list_np[0, i, 0], x_list_np[0, i, 1], color='red')

        x = np.linspace(x_list_np[:, i, 0].min() - 0.1, x_list_np[:, i, 0].max() + 0.1, 50)
        y = np.linspace(x_list_np[:, i, 1].min() - 0.1, x_list_np[:, i, 1].max() + 0.1, 50)
        X, Y = np.meshgrid(x, y)
        pos = jnp.concatenate([X[...,None], Y[...,None]], axis=-1)
        grads = vmap_compute_grad(my_function, pos)
        # Plot the gradient direction as quiver plot
        plt.quiver(X, Y, grads[:,:, 0], grads[:,:,1], color='black', alpha=1)
        hessians = vmap_compute_hess(my_function, pos)
        plt.quiver(X, Y, hessians[:,:, 0], hessians[:,:,1], color='blue', alpha=1)

        # Add labels and title
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title(f'Zoomed Contour plot of my_function with Newton\'s Method Trajectory {i+1}')
        plt.legend()

        # Set the zoomed-in limits
        plt.xlim(x_list_np[:, i, 0].min() - 0.1, x_list_np[:, i, 0].max() + 0.1)
        plt.ylim(x_list_np[:, i, 1].min() - 0.1, x_list_np[:, i, 1].max() + 0.1)

        # Save the zoomed-in plot
        try:
            plt.savefig(os.getcwd() + f"/Denoising_diff_sampler/Figures/hessian_zoomed_{i+1}.png")
        except:
            plt.savefig(os.getcwd() + f"//Figures/hessian_zoomed_{i+1}.png")            
        plt.close()

