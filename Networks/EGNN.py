
import flax
from flax import linen as nn
from functools import partial
import jax.numpy as jnp
import jax



class EGNNNetwork(nn.Module):
    n_layers: int = 2
    hidden_dim: int = 32
    feature_dim: int = 32
    n_particles: int = 1
    out_dim: int = 2

    @nn.compact
    def __call__(self, in_dict):
        
        x_0 = in_dict["x"]
        h = in_dict["h"]
        x = x_0

        orig_x_shapes = x_0.shape
        orig_h_shapes = h.shape
        x = x.reshape((self.n_particles, self.out_dim))
        h = jnp.repeat(h[None, :], self.n_particles, axis = 0)

        for i in range(self.n_layers):
            net = EGNNLayer( hidden_dim = self.hidden_dim, feature_dim = self.feature_dim, n_particles = self.n_particles,out_dim = self.out_dim)
            x, h = net(x, h)

        x = x.reshape(orig_x_shapes)
        x_hidden = nn.Dense(1, kernel_init=nn.initializers.he_normal())(h)
        x_hidden = jnp.repeat(x_hidden, self.out_dim, axis = -1)
        x_hidden = x_hidden.reshape(orig_x_shapes)
        out_dict = {"x": x-x_0,  "x_hidden": x_hidden}
        return out_dict

        


class EGNNLayer(nn.Module):
    hidden_dim: int = 32
    feature_dim: int = 32
    n_particles: int = 1
    out_dim: int = 2

    def setup(self):

        self.net_m = nn.Sequential([
            nn.Dense(self.hidden_dim, kernel_init=nn.initializers.he_normal(),
                                 bias_init=nn.initializers.zeros),
            nn.LayerNorm(),
            nn.silu,
            nn.Dense(self.hidden_dim, kernel_init=nn.initializers.he_normal(),
                                 bias_init=nn.initializers.zeros),
            nn.LayerNorm(),
            nn.silu,
        ]
        )

        self.net_e = nn.Sequential([
            nn.Dense(self.hidden_dim, kernel_init=nn.initializers.he_normal(),
                                 bias_init=nn.initializers.zeros),
            nn.LayerNorm(),
            nn.silu,
            nn.Dense(self.hidden_dim, kernel_init=nn.initializers.he_normal(),
                                 bias_init=nn.initializers.zeros),
            nn.LayerNorm(),
            nn.silu,
                    ]
        )

        self.net_d = nn.Sequential([
            nn.Dense(self.hidden_dim, kernel_init=nn.initializers.he_normal(),
                                 bias_init=nn.initializers.zeros),
            nn.LayerNorm(),
            nn.silu,
            nn.Dense(1, kernel_init=nn.initializers.he_normal(),
                                 bias_init=nn.initializers.zeros),
            nn.silu        ]
        )

        self.net_h = nn.Sequential([
            nn.Dense(self.feature_dim, kernel_init=nn.initializers.he_normal(),
                                 bias_init=nn.initializers.zeros),
            nn.LayerNorm(),
            nn.silu,        
            nn.Dense(self.feature_dim, kernel_init=nn.initializers.he_normal(),
                                 bias_init=nn.initializers.zeros),
            nn.LayerNorm(),
            nn.silu        ]
        )
        

    #@nn.compact
    @partial(flax.linen.jit, static_argnums=(0,))
    def __call__(self, x_prev, h_prev ):
        ### TODO use triu to increase efficency
        mask = 1-jnp.eye(x_prev.shape[0])
        flattened_mask = mask.reshape(-1)
        x = x_prev
        h = h_prev

        dist_squared = mask*jnp.sum((x[:, None, : ] - x[ None,:, :])**2, axis = -1)
        #print("mask", mask)
        #flattened_dist_squared = dist_squared.reshape(self.n_particles**2, -1)

        h1 = mask[...,None]*jnp.repeat(h[:, None, ...], self.n_particles, axis = 1)
        h2 = mask[...,None]*jnp.repeat(h[None, :, :], self.n_particles, axis = 0)

        mass_mat = mask[...,None]*self.net_e(jnp.concatenate([h1, h2 ,dist_squared[...,None]], axis = -1))  ### concatenate and flatten and unflatten
        flattened_mass_mat  = mass_mat.reshape(self.n_particles**2, -1)


        #m_transformed_ij = flattened_mask[...,None]*self.net_m(flattened_mass_mat)*flattened_mass_mat ### old version
        m_transformed_ij = flattened_mask[...,None]*self.net_m(flattened_mass_mat)
        m_transformed_ij_resh = m_transformed_ij.reshape(self.n_particles, self.n_particles, -1)
        m_transformed_i = jnp.sum(m_transformed_ij_resh, axis = -2)

        d_net_out = self.net_d(flattened_mass_mat).reshape(self.n_particles, self.n_particles, -1)
        h_next = self.net_h(jnp.concatenate([h, m_transformed_i], axis = -1)) ### concatenate
        #x_next = x + jnp.sum( mask[...,None]*(x[:, None, : ] - x[None, :, :])/(jnp.sqrt(dist_squared[..., None] + 10**-3)+1)*d_net_out, axis = 1)
        delta_x = jnp.sum( mask[...,None]*(x[:, None, : ] - x[None, :, :])*d_net_out, axis = -2)
        x_next = x + delta_x
        # print("x_next", jax.lax.stop_gradient(jnp.mean(x_next)))
        # print("h_next", jax.lax.stop_gradient(jnp.mean(h_next)))
        # print(delta_x.shape,((x[:, None, : ] - x[None, :, :])*d_net_out).shape, mask[...,None].shape, "shapes")
        # print(d_net_out.shape,(x[:, None, : ] - x[None, :, :]).shape,  "d_net_out")
        # print(jax.lax.stop_gradient(d_net_out))
        # print("averages", jax.lax.stop_gradient(jnp.mean(d_net_out)), jax.lax.stop_gradient(jnp.mean(m_transformed_i)))
        # print("xs")
        # print(jax.lax.stop_gradient(jnp.mean(delta_x)), jax.lax.stop_gradient(jnp.mean(x_next)))
        return x_next, h_next