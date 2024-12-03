




class EGNNLayer(nn.Module):

    def setup(self):

        pass

    def __call__(self, x_prev, h_prev, ):
        pass


class EGNNLayer(nn.Module):
    n_layers: int = 2
    hidden_dim: int = 32
    out_dim : int = 1
    n_particles: int = 1

    def setup(self):

        self.net_m = nn.sequential(
            nn.Dense(self.hidden_dim, kernel_init=nn.initializers.he_normal(),
                                 bias_init=nn.initializers.zeros),
            nn.silu,
            nn.Dense(self.out_dim, kernel_init=nn.initializers.he_normal(),
                                 bias_init=nn.initializers.zeros),
            nn.silu
        )

        self.net_e = nn.sequential(
            nn.Dense(self.hidden_dim, kernel_init=nn.initializers.he_normal(),
                                 bias_init=nn.initializers.zeros),
            nn.silu,
            nn.Dense(self.out_dim, kernel_init=nn.initializers.he_normal(),
                                 bias_init=nn.initializers.zeros),
            nn.silu
        )

        self.net_d = nn.sequential(
            nn.Dense(self.hidden_dim, kernel_init=nn.initializers.he_normal(),
                                 bias_init=nn.initializers.zeros),
            nn.silu,
            nn.Dense(self.out_dim, kernel_init=nn.initializers.he_normal(),
                                 bias_init=nn.initializers.zeros),
            nn.silu
        )

        self.net_h = nn.sequential(
            nn.Dense(self.hidden_dim, kernel_init=nn.initializers.he_normal(),
                                 bias_init=nn.initializers.zeros),
            nn.silu
        )
        

    @nn.compact
    #@partial(flax.linen.jit, static_argnums=(0,))
    def __call__(self, x_prev, h_prev, ):
        ### TODO use triu to increase efficency
        mask = jnp.eye(d_ij_squared.shape[0])
        flattened_mask = mask.reshape(-1)
        x = x_prev
        h = h_prev
        x = x.reshape(self.n_paricles, -1)
        h = h.reshape(self.n_paricles, -1)
        dist_squared = mask*np.sum((x[None, :, : ] - x[:, None, :])**2, axis = -1)
        flattened_dist_squared = dist_squared.reshape(self.n_particles**2, -1)

        h1 = mask[...,None]*jnp.repeat(h[None, ...], self.n_particles, axis = 0)
        h2 = mask[...,None]*jnp.repeat(h[:, None, :], self.n_particles, axis = 1)

        mass_mat = mask[...,None]*self.net_e(h1, h2 ,dist_squared)  ### concatenate and flatten and unflatten
        flattened_mass_mat  = mass_mat.reshape(self.n_particles**2, -1)


        m = jnp.sum(jnp.sum(flattened_mask[...,None]*self.mass_network(flattened_mass_mat)*flattened_mass_mat, axis = 0)) ### TODO flatten and unflatten
        h_next = self.net_h(jnp.concatenate([h, m]), axis = -1) ### concatenate
        x_next = x + jnp.sum(jnp.sum( mask[...,None]*(x[None, :, : ] - x[:, None, :])/(jnp.sqrt(dist_squared)+1)*self.net_d(flattened_mass_mat).reshape(self.n_particles, self.n_particles, -1), axis = 0), axis = 1)


        return x_next, h_next