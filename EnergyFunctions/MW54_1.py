from __future__ import annotations

from matplotlib import pyplot as plt
from .BaseEnergy import EnergyModelClass
import jax
import jax.numpy as jnp
import numpy as np
from functools import partial
from jax.scipy.special import logsumexp
import chex
import os
import pickle
from .base import Distribution, rejection_sampling
from .gauss import GMM, IsotropicGauss

import math
#import plotly.graph_objects as go
import torch
#from sde_sampler.eval.plots import plot_marginal



class DoubleWell(Distribution):
    def __init__(
        self,
        dim: int = 1,
        separation: float = 2.0,
        shift: float = 0.0,
        grid_points: int = 2001,
        rejection_sampling_scaling: float = 3.0,
        domain_delta: float = 2.5,
        **kwargs,
    ):
        if not dim == 1:
            raise ValueError("`dim` needs to be `1`. Consider using `MultiWell`.")
        super().__init__(dim=1, grid_points=grid_points, **kwargs)
        self.rejection_sampling_scaling = rejection_sampling_scaling
        self.register_buffer("separation", torch.tensor(separation), persistent=False)
        self.register_buffer("shift", torch.tensor(shift), persistent=False)

        # Set domain
        if self.domain is None:
            domain = self.shift + (
                self.separation.sqrt() + domain_delta
            ) * torch.tensor([[-1.0, 1.0]])
            self.set_domain(domain)

    def unnorm_log_prob(self, x: torch.Tensor) -> torch.Tensor:
        x = x - self.shift
        return -((x**2 - self.separation) ** 2)

    def score(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        x = x - self.shift
        return -4.0 * (x**2 - self.separation) * x

    def marginal(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        return self.pdf(x)

    def get_proposal_distr(self):
        device = self.domain.device
        loc = self.shift + self.separation.sqrt() * torch.tensor(
            [[-1.0], [1.0]], device=device
        )
        scale = 1 / self.separation.sqrt() * torch.ones(2, 1, device=device)
        proposal = GMM(
            dim=1,
            loc=loc,
            scale=scale,
            mixture_weights=torch.ones(2, device=device),
            domain_tol=None,
        )
        proposal.to(device)
        return proposal

    def sample(self, shape: tuple | None = None) -> torch.Tensor:
        if shape is None:
            shape = tuple()
        proposal = self.get_proposal_distr()
        return rejection_sampling(
            shape=shape,
            target=self,
            proposal=proposal,
            scaling=self.rejection_sampling_scaling,
        )
    '''
    def plots(self, samples, nbins=100) -> torch.Tensor:
        samples = self.sample((samples.shape[0],))
        fig = plot_marginal(
            x=samples,
            marginal=lambda x, **kwargs: self.pdf(x),
            dim=0,
            nbins=nbins,
            domain=self.domain,
        )
    
        x = torch.linspace(*self.domain[0], steps=nbins, device=self.domain.device)
        y = (
            self.get_proposal_distr().pdf(x.unsqueeze(-1))
            * self.rejection_sampling_scaling
        )
        fig.add_trace(
            go.Scatter(
                x=x.cpu(),
                y=y.squeeze(-1).cpu(),
                mode="lines",
                name="proposal",
            )
        )
        return {"plots/rejection_sampling": fig}
    '''

class MultiWell(Distribution):
    def __init__(
        self,
        dim: int = 2,
        n_double_wells: int = 1,
        separation: float = 2.0,
        shift: float = 0.0,
        domain_dw_delta: float = 2.5,
        domain_gauss_scale: float = 5.0,
        **kwargs,
    ):
        super().__init__(dim=dim, **kwargs)
        # Define parameters
        self.separation = separation
        if n_double_wells > dim or n_double_wells == 0:
            raise ValueError(f"Please specify between 1 and {dim} double wells.")
        self.n_double_wells = n_double_wells
        self.n_gauss = self.dim - self.n_double_wells

        # Initialize distributions
        self.double_well = DoubleWell(
            separation=separation, shift=shift, domain_delta=domain_dw_delta
        )
        domain = self.double_well.domain.repeat(self.n_double_wells, 1)
        self.gauss = None
        if self.n_gauss > 0:
            self.gauss = IsotropicGauss(
                dim=self.n_gauss,
                loc=shift,
                log_norm_const=0.5 * math.log(2.0 * math.pi) * self.n_gauss,
                domain_scale=domain_gauss_scale,
            )
            domain = torch.cat([domain, self.gauss.domain])

        # Set domain
        self.set_domain(domain)

    def _initialize_distr(self):
        if self.gauss is not None:
            return self.gauss._initialize_distr()

    def compute_stats(self):
        # Double well
        self.double_well.compute_stats()
        self.log_norm_const = self.double_well.log_norm_const * self.n_double_wells
        self.expectations = {
            name: exp * self.n_double_wells
            for name, exp in self.double_well.expectations.items()
        }
        self.stddevs = torch.cat([self.double_well.stddevs] * self.n_double_wells)

        # Gauss
        if self.gauss is not None:
            self.gauss.compute_stats()
            self.log_norm_const += self.gauss.log_norm_const
            for name in self.expectations:
                # This assumes that the expectations are reducing the dim via a sum
                self.expectations[name] += self.gauss.expectations[name]
            self.stddevs = torch.cat([self.stddevs, self.gauss.stddevs])

        assert (self.pdf(self.domain.T) < 1e-5).all()

    def unnorm_log_prob(self, x: torch.Tensor) -> torch.Tensor:
        log_prob = self.double_well.unnorm_log_prob(x[:, : self.n_double_wells]).sum(
            dim=-1, keepdim=True
        )
        if self.gauss is not None:
            log_prob += self.gauss.unnorm_log_prob(x[:, self.n_double_wells :])
        assert log_prob.shape == (*x.shape[:-1], 1)
        return log_prob

    def score(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        score = self.double_well.score(x[:, : self.n_double_wells])
        if self.gauss is not None:
            score_gauss = self.gauss.score(x[:, self.n_double_wells :])
            score = torch.cat([score, score_gauss], dim=-1)
        return score

    def marginal(self, x: torch.Tensor, dim: int = 0) -> torch.Tensor:
        if dim < self.n_double_wells:
            return self.double_well.marginal(x)
        return self.gauss.marginal(x)

    def sample(self, shape: tuple | None = None) -> torch.Tensor:
        if shape is None:
            shape = tuple()
        samples = self.double_well.sample(shape + (self.n_double_wells,)).squeeze(-1)
        if self.gauss is not None:
            samples_gauss = self.gauss.sample(shape)
            samples = torch.cat([samples, samples_gauss], dim=-1)
        return samples


class ManyWellClass1(EnergyModelClass):
    def __init__(self, config):
        """

        """

        super().__init__(config)
        self.d = self.config["d"]
        self.m = self.config["m"]
        self.b = 1
        self.c = 0.5
        self.dim_x = self.d
        self.has_tractable_distribution = True
        #self.chosen_energy_function = self.energy_function_richter
        self.invariance = False

        self.double_well = DoubleWell(separation=4.0, shift=0.0, domain_delta=2.5)
        self.double_well.compute_stats()
        #domain = self.double_well.domain.repeat(self.m, 1)
        self.d_0 = 4.0
        zer = np.array([np.sqrt(self.d_0), -np.sqrt(self.d_0)])
        grids = np.meshgrid(*([zer] * self.m))
        self.means = jnp.array(np.stack(grids, axis=-1).reshape(-1, self.m))


    @partial(jax.jit, static_argnums=(0,))
    def energy_function(self, x):    
        """

        """
        #d_0 = 4.0

        energy = self.b *jnp.sum((x[:self.m]**2 - self.d_0)**2) + self.c*jnp.sum(x[self.m:]**2)
        
        return energy
    
    def log_prob(self, x: chex.Array) -> chex.Array:
        """
        Calculate the log probability (negative energy)
        """
        return -self.energy_function(x)
    
    def sample(self, seed: chex.PRNGKey, sample_shape: chex.Shape = ()) -> chex.Array:
        """
        Generate samples from the Many Well distribution
        by multiple double well rejection sampling
        """
        samples = jnp.array(self.double_well.sample(sample_shape + (self.m,)).squeeze(-1))
        return samples

    def generate_samples(self, key, n_samples):
        """
        Generate multiple samples.
        """
        return self.sample(key, sample_shape=(n_samples,))
    
    def visualise(self, samples: chex.Array = None, axes=None, show=False, prefix='') -> dict:
        """
        Visualize samples from the Double Well distribution.
        Shows the first two dimensions if m >= 2, or first dimension vs first Gaussian dimension if m == 1.
        
        :param samples: Optional array of samples to plot
        :param axes: Optional matplotlib axes for plotting
        :param show: Whether to show the plot
        :param prefix: Prefix for saving the plot
        :return: Dictionary with wandb image
        """
        plt.close()
        fig = plt.figure(figsize=(10, 8))
        if axes is None:
            ax = fig.add_subplot(111)
        else:
            ax = axes
            
        if samples is not None:
            # Plot first two dimensions
            if self.m >= 2:
                ax.scatter(samples[:, 0], samples[:, 1], alpha=0.5, s=10)
                ax.set_xlabel('x₁')
                ax.set_ylabel('x₂')
            else:
                # Plot first double well dimension vs first Gaussian dimension
                ax.scatter(samples[:, 0], samples[:, self.m], alpha=0.5, s=10)
                ax.set_xlabel('x₁ (Double Well)')
                ax.set_ylabel('x_{m+1} (Gaussian)')
        
        # Set plot bounds based on the double well modes
        d_0 = 4.0
        bound = 2.5 * jnp.sqrt(d_0)
        ax.set_xlim(-bound, bound)
        ax.set_ylim(-bound, bound)
        
        plt.title(f"Double Well Samples (m={self.m}, d={self.d})")
        plt.grid(True)
        
        # Remove ticks for cleaner visualization
        plt.xticks([])
        plt.yticks([])
        
        plt.savefig(os.path.join('EnergyFunctions', 'EnergyData', 'Plots', 'vis.png'))
        
        if show:
            plt.show()
        



if __name__ == '__main__':
    # Test configurations
    configs = [
        {"d": 5, "m": 5, "dim_x": 10, 'scaling': 1.0}, 
    ]
    
    for config in configs:
        # Initialize model
        double_well = ManyWellClass1(config)
        
        # Generate samples
        key = jax.random.PRNGKey(0)
        samples = double_well.generate_samples(key, 2000)
        
        # Visualize
        double_well.visualise(samples, show=True)
        
        # Print some statistics about the first dimensions
        if config["m"] >= 2:
            print(f"\nStatistics for 2D Double Well (m={config['m']}, d={config['d']}):")
            print(f"Mean of first two dims: ({jnp.mean(samples[:, 0]):.3f}, {jnp.mean(samples[:, 1]):.3f})")
            print(f"Std of first two dims: ({jnp.std(samples[:, 0]):.3f}, {jnp.std(samples[:, 1]):.3f})")
        else:
            print(f"\nStatistics for 1D Double Well (m={config['m']}, d={config['d']}):")
            print(f"Mean of double well dim: {jnp.mean(samples[:, 0]):.3f}")
            print(f"Std of double well dim: {jnp.std(samples[:, 0]):.3f}")
            print(f"Mean of first Gaussian dim: {jnp.mean(samples[:, 1]):.3f}")
            print(f"Std of first Gaussian dim: {jnp.std(samples[:, 1]):.3f}")
    
    
    
