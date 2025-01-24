from .BaseEnergy import EnergyModelClass
import jax
import jax.numpy as jnp
import jax.scipy.linalg as slinalg
import numpy as np
from functools import partial
import os


def get_bin_counts(array_in, num_bins_per_dim):
    """Divide 2D input space into a grid and count points in each bin."""
    scaled_array = array_in * num_bins_per_dim
    counts = np.zeros((num_bins_per_dim, num_bins_per_dim))
    for elem in scaled_array:
        flt_row, col_row = np.floor(elem)
        row = int(min(flt_row, num_bins_per_dim - 1))
        col = int(min(col_row, num_bins_per_dim - 1))
        counts[row, col] += 1
    return counts


def get_bin_vals(num_bins):
    """Get grid coordinates for all bins."""
    grid_indices = jnp.arange(num_bins)
    bin_vals = jnp.array([(i, j) for i in grid_indices for j in grid_indices])
    return bin_vals


def kernel_func(x, y, signal_variance, num_grid_per_dim, raw_length_scale):
    """Compute covariance kernel function."""
    normalized_distance = jnp.linalg.norm(x - y) / (num_grid_per_dim * raw_length_scale)
    return signal_variance * jnp.exp(-normalized_distance)


def gram(kernel, xs):
    """Compute gram matrix given kernel function and points."""
    return jax.vmap(lambda x: jax.vmap(lambda y: kernel(x, y))(xs))(xs)


class LGCPClass(EnergyModelClass):
    def __init__(self, config):
        """
        Initialize the Log Gaussian Cox Process model.
        
        :param config: Dictionary containing:
            - num_grid_per_dim: number of grid points per dimension (default 40)
            - use_whitened: whether to use whitened parameterization (default False)
        """
        super().__init__(config)
        
        self.num_grid_per_dim = config.get("num_grid_per_dim", 40)
        self.num_latents = self.num_grid_per_dim ** 2
        self.use_whitened = config.get("use_whitened", False)
        
        # Load data points from CSV using relative path
        current_dir = os.path.dirname(os.path.abspath(__file__))
        data_path = os.path.join(current_dir, 'EnergyData', 'Datasets', 'pines.csv')
        pines_array = self.get_pines_points(data_path)[1:, 1:]  # Skip header row and first column
        bin_counts = get_bin_counts(pines_array, self.num_grid_per_dim)
        self.flat_bin_counts = jnp.reshape(bin_counts, (self.num_latents,))
        
        # Set LGCP parameters as in Moller et al, 1998
        self.poisson_a = 1.0 / self.num_latents
        self.signal_variance = 1.91
        self.beta = 1.0 / 33
        
        # Compute gram matrix and its Cholesky decomposition
        self.bin_vals = get_bin_vals(self.num_grid_per_dim)
        kernel = lambda x, y: kernel_func(x, y, self.signal_variance, 
                                        self.num_grid_per_dim, self.beta)
        self.gram_matrix = gram(kernel, self.bin_vals)
        self.cholesky_gram = jnp.linalg.cholesky(self.gram_matrix)
        
        # Set mean function (constant)
        self.mu_zero = jnp.log(126.) - 0.5 * self.signal_variance

    def get_pines_points(self, file_path):
        """Get the pines data points from CSV file."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Pines dataset not found at: {file_path}")
            
        with open(file_path, "rt") as input_file:
            data = np.genfromtxt(input_file, delimiter=",")
        return data

    @partial(jax.jit, static_argnums=(0,))
    def energy_function(self, x):
        """
        Calculate the negative log probability (energy) of the LGCP.
        
        :param x: Input array of shape (num_latents,)
        :return: Energy value
        """
        if self.use_whitened:
            return -self.whitened_posterior_log_density(x)
        else:
            return -self.unwhitened_posterior_log_density(x)

    def whitened_posterior_log_density(self, white):
        """Compute log density in whitened parameterization."""
        # Prior term
        quadratic_term = -0.5 * jnp.sum(white ** 2)
        white_gaussian_log_normalizer = -0.5 * self.num_latents * jnp.log(2.0 * jnp.pi)
        prior_log_density = white_gaussian_log_normalizer + quadratic_term
        
        # Likelihood term
        latent_function = jnp.matmul(self.cholesky_gram, white) + self.mu_zero
        likelihood = jnp.sum(latent_function * self.flat_bin_counts - 
                           self.poisson_a * jnp.exp(latent_function))
        
        return prior_log_density + likelihood

    def unwhitened_posterior_log_density(self, latents):
        """Compute log density in unwhitened parameterization."""
        # Transform to whitened space
        white = slinalg.solve_triangular(
            self.cholesky_gram, latents - self.mu_zero, lower=True)
        
        # Prior term
        half_log_det_gram = jnp.sum(jnp.log(jnp.abs(jnp.diag(self.cholesky_gram))))
        unwhitened_gaussian_log_normalizer = (-0.5 * self.num_latents * jnp.log(2.0 * jnp.pi) 
                                            - half_log_det_gram)
        prior_log_density = -0.5 * jnp.sum(white ** 2) + unwhitened_gaussian_log_normalizer
        
        # Likelihood term
        likelihood = jnp.sum(latents * self.flat_bin_counts - 
                           self.poisson_a * jnp.exp(latents))
        
        return prior_log_density + likelihood 
    

if __name__ == "__main__":
    config = {
        "num_grid_per_dim": 40,
        "use_whitened": False
    }
    lgcp_model = LGCPClass(config)
    print(lgcp_model.energy_function(jnp.zeros(lgcp_model.num_latents)))