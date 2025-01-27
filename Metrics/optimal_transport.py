# Core imports
import jax
import jax.numpy as jnp
from functools import partial

# OTT-JAX imports
from ott.problems.linear import linear_problem
from ott.solvers.linear import sinkhorn
from ott.tools import sinkhorn_divergence
from ott.geometry import pointcloud


class OT:
    """Base Optimal Transport class for computing transport costs between sample sets."""
    
    def __init__(self, gt_samples, epsilon=1e-3):
        """
        Initialize with ground truth samples and regularization parameter.
        
        Args:
            gt_samples: Ground truth samples to compare against
            epsilon: Regularization parameter for numerical stability
        """
        self.groundtruth = gt_samples
        self.epsilon = epsilon

    def compute_OT(self, model_samples, entropy_reg=True):
        """
        Compute optimal transport cost between ground truth and model samples.
        
        Args:
            model_samples: Samples from the model to compare against ground truth
            entropy_reg: If True, return entropy regularized cost, else unregularized cost
            
        Returns:
            float: The optimal transport cost
        """
        geom = pointcloud.PointCloud(self.groundtruth, model_samples, epsilon=self.epsilon)
        ot_prob = linear_problem.LinearProblem(geom)
        solver = sinkhorn.Sinkhorn()
        ot = solver(ot_prob)
        
        if entropy_reg:
            return ot.reg_ot_cost
        return jnp.sum(ot.matrix * ot.geom.cost_matrix)


class SD:
    """Sinkhorn Divergence class for computing distribution distances."""
    
    def __init__(self, gt_samples, epsilon=1e-3):
        """
        Initialize with ground truth samples and regularization parameter.
        
        Args:
            gt_samples: Ground truth samples to compare against
            epsilon: Regularization parameter for numerical stability
        """
        self.groundtruth = gt_samples
        self.epsilon = epsilon

    def compute_SD(self, model_samples):
        """
        Compute Sinkhorn divergence between ground truth and model samples.
        
        Args:
            model_samples: Samples from the model to compare against ground truth
            
        Returns:
            float: The Sinkhorn divergence
        """
        geom = pointcloud.PointCloud(self.groundtruth, model_samples, epsilon=self.epsilon)
        sd = sinkhorn_divergence.sinkhorn_divergence(geom, x=geom.x, y=geom.y)
        return sd[1].divergence
    
    def compute_approximate_W2(self, model_samples):
        """
        Compute approximation of W2 distance via square root of Sinkhorn divergence.
        As epsilon → 0, this converges to the true W2 distance.
        
        Args:
            model_samples: Samples from the model to compare against ground truth
            
        Returns:
            float: Approximate W2 distance
        """
        return jnp.sqrt(self.compute_SD(model_samples))


if __name__ == "__main__":
    # Add necessary paths
    import sys
    sys.path.append('/system/user/publicwork/bartmann/DDS_Optim')
    sys.path.append('/system/user/publicwork/bartmann/DDS_Optim/EnergyFunctions')
    
    # Import energy function
    from EnergyFunctions.FunnelDistrax import FunnelDistraxClass
    
    # Print JAX configuration
    print("JAX Configuration:")
    print(f"enable_x64: {jax.config.read('jax_enable_x64')}")
    print(f"default device: {jax.default_backend()}")
    
    # Test parameters
    n_samples = 2000
    seeds = range(42, 51)  # Seeds from 42 to 50
    
    for seed in seeds:
        print(f"\nTesting with seed {seed}")
        
        # Generate samples
        key = jax.random.PRNGKey(seed)
        key1, key2 = jax.random.split(key)
        
        funnel = FunnelDistraxClass({"dim_x": 10, 'scaling': 1.0})
        samples1 = funnel.generate_samples(key1, n_samples)
        samples2 = funnel.generate_samples(key2, n_samples)
        
        # Print sample statistics
        # print(f"Sample sums: {jnp.sum(samples1):.6f}, {jnp.sum(samples2):.6f}")
        # print(f"Max norms: {jnp.max(jnp.linalg.norm(samples1, axis=-1)):.6f}, "
        #       f"{jnp.max(jnp.linalg.norm(samples2, axis=-1)):.6f}")
        
        # Compute distances
        sd = SD(samples1, epsilon=1e-3)
        ot = OT(samples1, epsilon=1e-3)
        
        # Print results
        print("\nDistance Metrics:")
        print(f"Sinkhorn divergence: {sd.compute_SD(samples2):.6f}")
        print(f"Approximate W2 distance: {sd.compute_approximate_W2(samples2):.6f}")
        print(f"OT cost (with entropy reg): {ot.compute_OT(samples2, entropy_reg=True):.6f}")
        print(f"OT cost (without entropy reg): {ot.compute_OT(samples2, entropy_reg=False):.6f}")



