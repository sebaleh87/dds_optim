# Core imports
import jax
import jax.numpy as jnp
from functools import partial

# OTT-JAX imports
from ott.problems.linear import linear_problem
from ott.solvers.linear import sinkhorn
from ott.tools import sinkhorn_divergence
from ott.geometry import pointcloud
import torch

# POT imports
#import ot as pot
import numpy as np


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
    
    def __init__(self, Energy_function, n_samples, key, epsilon=1e-3):
        """
        Initialize with ground truth samples and regularization parameter.
        
        Args:
            gt_samples: Ground truth samples to compare against
            epsilon: Regularization parameter for numerical stability
        """
        self.epsilon = epsilon
        self.energy_function = Energy_function
        self.key = key
        self.n_samples = n_samples

    def resample_energy(self, key, n_samples):
        return self.energy_function.generate_samples(key, n_samples)

    def compute_SD(self, model_samples, gt_samples = None):
        """
        Compute Sinkhorn divergence between ground truth and model samples.
        
        Args:
            model_samples: Samples from the model to compare against ground truth
            
        Returns:
            float: The Sinkhorn divergence
        """
        self.key, subkey = jax.random.split(self.key)

        self.groundtruth = self.resample_energy(subkey, model_samples.shape[0])
        geom = pointcloud.PointCloud(self.groundtruth, model_samples, epsilon=self.epsilon)
        sd = sinkhorn_divergence.sinkhorn_divergence(geom, x=geom.x, y=geom.y)
        return sd[1].divergence


    # JAX implementation
    #@partial(jax.jit, static_argnums=(0,-1,-2))
    def mmd_loss(self, source, target, kernel_mul=2.0, kernel_num=10, fix_sigma = 100):
        batch_size = source.shape[0]
        total = jnp.concatenate([source, target], axis=0)
        total0 = jnp.expand_dims(total, 0)
        total1 = jnp.expand_dims(total, 1)
        L2_distance = jnp.sum((total0 - total1) ** 2, axis=2)

        n_samples = source.shape[0] + target.shape[0]

        bandwidth = fix_sigma#jnp.sum(L2_distance) / (n_samples**2 - n_samples + 1e-8)

        # jax.debug.print("kernel_mul: {kernel_mul}", kernel_mul = kernel_mul)
        # jax.debug.print("kernel_mul: {kernel_num}", kernel_num = kernel_num)
        # jax.debug.print("🤯 kernel_num {kernel_num} 🤯", kernel_num=kernel_num)
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]
        kernel_vals = jnp.array([jnp.exp(-L2_distance / bw) for bw in bandwidth_list])
        kernels = jnp.sum(kernel_vals, axis = 0)
        XX = kernels[:batch_size, :batch_size]
        YY = kernels[batch_size:, batch_size:]
        XY = kernels[:batch_size, batch_size:]
        YX = kernels[batch_size:, :batch_size]
        return jnp.mean(XX + YY - XY - YX)


    def mmd_loss_jax(self, model_samples, kernel_mul=2.0, kernel_num=10, fix_sigma=None, n_samples = 2000):
        ### TODO pay attention the output is MMD^2!
        self.key, subkey = jax.random.split(self.key)
        self.groundtruth = self.resample_energy(subkey, n_samples)
        mmd_loss = self.mmd_loss(model_samples[0:n_samples], self.groundtruth, kernel_mul, kernel_num)
        return mmd_loss
    
    def compute_MMD_and_Sinkhorn(self, model_samples, kernel_mul=2.0, kernel_num=10, fix_sigma=None, n_MMD_samples = 4000):
        mmd_loss = self.mmd_loss_jax(model_samples[0:n_MMD_samples], kernel_mul, kernel_num, fix_sigma, n_MMD_samples)
        sd = self.compute_SD(model_samples)
        out_dict = {"MMD^2": mmd_loss, "Sinkhorn divergence": sd}
        return out_dict



if __name__ == "__main__":
    # Add necessary paths
    import sys
    sys.path.append('/system/user/publicwork/bartmann/DDS_Optim')
    sys.path.append('/system/user/publicwork/bartmann/DDS_Optim/EnergyFunctions')
    
    # Import energy function
    from EnergyFunctions.FunnelDistrax import FunnelDistraxClass
    from EnergyFunctions.StudentTMixture import StudentTMixtureClass
    
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
        
        # funnel = FunnelDistraxClass({"dim_x": 10, 'scaling': 1.0})
        # samples1 = funnel.generate_samples(key1, n_samples)
        # samples2 = funnel.generate_samples(key2, n_samples)

        n_eval_samples = 2000
        dim = 50
        num_components = 10
        df = 2.0



        Energy_Config = {
            "name": "StudentTMixture",
            "dim_x": dim,
            "num_components": num_components,
            "df": df,
            "scaling": 1.0
        }

        student_t = StudentTMixtureClass(Energy_Config)
        samples1 = student_t.generate_samples(key1, n_samples)
        samples2 = student_t.generate_samples(key2, n_samples)
        
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
        #print(f"Sinkhorn divergence (POT): {sd.compute_SD_POT(samples2):.6f}")
        #print(f"Approximate W2 distance: {sd.compute_approximate_W2(samples2):.6f}")
        print(f"OT cost (with entropy reg): {ot.compute_OT(samples2, entropy_reg=True):.6f}")
        print(f"OT cost (without entropy reg): {ot.compute_OT(samples2, entropy_reg=False):.6f}")



