from __future__ import annotations

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

import pickle
import pykeops.torch as keops
import torch
import tqdm

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
    
    def __init__(
        self,
        energy_function,
        n_samples,
        key,
        p: float = 2,
        epsilon: float = 1e-3,
        max_iters: int = 100,
        stop_thresh: float = 1e-5,
        verbose: bool = False,
        n_max: int | None = None,
        **kwargs,
    ):
        if not isinstance(p, int):
            raise TypeError(f"p must be an integer greater than 0, got {p}")
        if p <= 0:
            raise ValueError(f"p must be an integer greater than 0, got {p}")
        self.p = p

        if epsilon <= 0:
            raise ValueError("Entropy regularization term eps must be > 0")
        self.eps = epsilon

        if not isinstance(max_iters, int) or max_iters <= 0:
            raise TypeError(f"max_iters must be an integer > 0, got {max_iters}")
        self.max_iters = max_iters

        if not isinstance(stop_thresh, float):
            raise TypeError(f"stop_thresh must be a float, got {stop_thresh}")
        self.stop_thresh = stop_thresh

        self.n_max = n_max
        self.verbose = verbose
        self.energy_function = energy_function
        self.key = key
        self.n_samples = n_samples

    def resample_energy(self, key, n_samples):
        return self.energy_function.generate_samples(key, n_samples)
    
    def compute_SD(self, model_samples, gt_samples = None):
        if("MW" in self.energy_function.config['name']):
            return self.compute_SD_many_well(model_samples, gt_samples)
        else:
            return self.compute_SD_others(model_samples, gt_samples)
    
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

    def compute_SD_others(self, model_samples, gt_samples = None):
        """
        Compute Sinkhorn divergence between ground truth and model samples.
        
        Args:
            model_samples: Samples from the model to compare against ground truth
            
        Returns:
            float: The Sinkhorn divergence
        """
        self.key, subkey = jax.random.split(self.key)
        self.groundtruth = self.resample_energy(subkey, model_samples.shape[0])
        
        geom = pointcloud.PointCloud(self.groundtruth, model_samples, epsilon=self.eps)
        sd = sinkhorn_divergence.sinkhorn_divergence(geom, x=geom.x, y=geom.y)
        return sd[1].divergence

    def compute_SD_many_well(
        self,
        model_samples,
        w_x: torch.Tensor | None = None,
        w_y: torch.Tensor | None = None,
    ):
        self.key, subkey = jax.random.split(self.key)
        self.groundtruth = self.resample_energy(subkey, self.n_samples)
        x = self.groundtruth
        y = model_samples
        if len(x.shape) != 2:
            raise ValueError(f"x must be an [n, d] tensor but got shape {x.shape}")
        if len(y.shape) != 2:
            raise ValueError(f"x must be an [m, d] tensor but got shape {y.shape}")
        if x.shape[1] != y.shape[1]:
            raise ValueError(
                f"x and y must match in the last dimension (i.e. x.shape=[n, d], "
                f"y.shape[m, d]) but got x.shape = {x.shape}, y.shape={y.shape}"
            )

        if w_x is not None:
            if w_y is None:
                raise ValueError("If w_x is not None, w_y must also be not None")
            if len(w_x.shape) > 1:
                w_x = w_x.squeeze()
            if len(w_x.shape) != 1:
                raise ValueError(
                    f"w_x must have shape [n,] or [n, 1] "
                    f"where x.shape = [n, d], but got w_x.shape = {w_x.shape}"
                )
            if w_x.shape[0] != x.shape[0]:
                raise ValueError(
                    f"w_x must match the shape of x in dimension 0 but got "
                    f"x.shape = {x.shape} and w_x.shape = {w_x.shape}"
                )
        if w_y is not None:
            if w_x is None:
                raise ValueError("If w_y is not None, w_x must also be not None")
            if len(w_y.shape) > 1:
                w_y = w_y.squeeze()
            if len(w_y.shape) != 1:
                raise ValueError(
                    f"w_y must have shape [n,] or [n, 1] "
                    f"where x.shape = [n, d], but got w_y.shape = {w_y.shape}"
                )
            if w_x.shape[0] != x.shape[0]:
                raise ValueError(
                    f"w_y must match the shape of y in dimension 0 but got "
                    f"y.shape = {y.shape} and w_y.shape = {w_y.shape}"
                )

        # Distance matrix [n, m]
        x = torch.tensor(np.array(x))#.cpu()
        y = torch.tensor(np.array(y))#.cpu()
        x_i = keops.Vi(x)  # [n, 1, d]
        y_j = keops.Vj(y)  # [i, m, d]
        if self.p == 1:
            M_ij = ((x_i - y_j) ** self.p).abs().sum(dim=2)  # [n, m]
        else:
            M_ij = ((x_i - y_j) ** self.p).sum(dim=2) ** (1.0 / self.p)  # [n, m]

        # Weights [n,] and [m,]
        if w_x is None and w_y is None:
            w_x = torch.ones(x.shape[0]).to(x) / x.shape[0]
            w_y = torch.ones(y.shape[0]).to(x) / y.shape[0]
            w_y *= w_x.shape[0] / w_y.shape[0]

        sum_w_x = w_x.sum().item()
        sum_w_y = w_y.sum().item()
        if abs(sum_w_x - sum_w_y) > 1e-5:
            raise ValueError(
                f"Weights w_x and w_y do not sum to the same value, "
                f"got w_x.sum() = {sum_w_x} and w_y.sum() = {sum_w_y} "
                f"(absolute difference = {abs(sum_w_x - sum_w_y)}"
            )

        log_a = torch.log(w_x)  # [n]
        log_b = torch.log(w_y)  # [m]

        # Initialize the iteration with the change of variable
        u = torch.zeros_like(w_x)
        v = self.eps * torch.log(w_y)

        u_i = keops.Vi(u.unsqueeze(-1))
        v_j = keops.Vj(v.unsqueeze(-1))

        if self.verbose:
            pbar = tqdm.trange(self.max_iters)
        else:
            pbar = range(self.max_iters)

        for _ in pbar:
            u_prev = u
            v_prev = v

            summand_u = (-M_ij + v_j) / self.eps
            u = self.eps * (log_a - summand_u.logsumexp(dim=1).squeeze())
            u_i = keops.Vi(u.unsqueeze(-1))

            summand_v = (-M_ij + u_i) / self.eps
            v = self.eps * (log_b - summand_v.logsumexp(dim=0).squeeze())
            v_j = keops.Vj(v.unsqueeze(-1))

            max_err_u = torch.max(torch.abs(u_prev - u))
            max_err_v = torch.max(torch.abs(v_prev - v))
            if self.verbose:
                pbar.set_postfix(
                    {"Current Max Error": max(max_err_u, max_err_v).item()}
                )
            if max_err_u < self.stop_thresh and max_err_v < self.stop_thresh:
                break

        P_ij = ((-M_ij + u_i + v_j) / self.eps).exp()

        approx_corr_1 = P_ij.argmax(dim=1).squeeze(-1)
        approx_corr_2 = P_ij.argmax(dim=0).squeeze(-1)

        if u.shape[0] > v.shape[0]:
            distance = (P_ij * M_ij).sum(dim=1).sum()
        else:
            distance = (P_ij * M_ij).sum(dim=0).sum()
        return distance

    

    # def compute_approximate_W2(self, model_samples):
    #     """
    #     Compute approximation of W2 distance via square root of Sinkhorn divergence.
    #     As epsilon → 0, this converges to the true W2 distance.
        
    #     Args:
    #         model_samples: Samples from the model to compare against ground truth
            
    #     Returns:
    #         float: Approximate W2 distance
    #     """
    #     return jnp.sqrt(self.compute_SD(model_samples))
    
    # def compute_SD_POT(self, model_samples):
    #     """
    #     Compute Sinkhorn divergence using POT library for comparison.
        
    #     Args:
    #         model_samples: Samples from the model to compare against ground truth
            
    #     Returns:
    #         float: The Sinkhorn divergence computed via POT
    #     """
    #     # Convert to numpy as POT doesn't work with JAX arrays
    #     a = np.ones((len(self.groundtruth),)) / len(self.groundtruth)
    #     b = np.ones((len(model_samples),)) / len(model_samples)
        
    #     # Compute cost matrix (squared Euclidean)
    #     M = pot.dist(np.array(self.groundtruth), np.array(model_samples), metric='sqeuclidean')
        
    #     # Compute Sinkhorn divergence
    #     div = pot.sinkhorn(
    #         a, b, 
    #         M,
    #         reg=self.epsilon,
    #         numItermax=1000,
    #         log=True
    #     )
    #     return float(div[0])  # Extract just the transport cost



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


