
import jax
import jax.numpy as jnp

def init_params(**kwargs):
  """
  """
  key = jax.random.PRNGKey(42)
  
  dims_in = kwargs['hidden_dims'][:-1]
  dims_out = kwargs['hidden_dims'][1:]

  pars = {}
  pars['nn'] = [ {} for _ in range(len(kwargs['hidden_dims'])-1) ]

  for i in range(len(kwargs['hidden_dims'])-1):

    key, subkey = jax.random.split(key)
    shape = ( dims_in[i], dims_out[i] )
    scale = jnp.sqrt( float(dims_in[i]) )
    pars['nn'][i]['w'] = ( jax.random.normal(subkey, shape=shape) / scale )

    key, subkey = jax.random.split(key)
    shape = ( dims_out[i], )
    pars['nn'][i]['b'] = jax.random.normal(subkey, shape=shape)

  return pars

def linear_layer(x, w, b = None):
  y = jnp.dot(x, w)
  return y + b if b is not None else y

def wavefunction(pars, pos):
  """
  """
  h = pos
  for i in range(len(pars['nn'])-1):
    h = jnp.tanh(linear_layer(h, **pars['nn'][i]))
  h = linear_layer(h, **pars['nn'][-1])
#  decay = jnp.log(2.*jax.nn.sigmoid(-1e-4*jnp.mean(pos**2)))
#  decay = jnp.log(jax.nn.sigmoid(1e2-1e-2*jnp.mean(pos**2)))
#  return h.sum() + decay
#  return -jnp.sum(jnp.abs(h))
  return -jnp.sum(jax.nn.softplus(h))
