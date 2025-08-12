
import numpy as np
import jax
import jax.numpy as jnp
import chex

from absl import logging
from ml_collections import ConfigDict

from src import utils

def get_device_shape(nwalkers: int):
  """
  """

  # GPUs logging
  num_devices = jax.device_count()
  num_hosts = num_devices // jax.local_device_count()
  kind_device = jax.devices()[0].platform
  logging.info('Simulations are going to use %i %s. Devices are diveded among %i hosts', 
                num_devices, kind_device, num_hosts)

  if nwalkers % num_devices != 0:
    raise ValueError('Number of walkers must be divisible by number of devices')
 
  # Defining the shape of data shared across devides 
  nw_per_dev = nwalkers // num_devices
  shape = (num_devices // num_hosts, nw_per_dev)

  return shape

def init_positions_randomly(cfg: ConfigDict, shape: tuple, key: chex.PRNGKey):
  """
  """

  # include number of particles in the shape
  np_shape = shape + (cfg.npart, 1)

  # make sure data on hosts are initialized differently
  key = jax.random.fold_in(key, jax.process_index())
  key, subkey = jax.random.split(key)

  if cfg.ndim == 1:

    pos = cfg.init_width * jax.random.uniform(subkey, shape=np_shape[:-1], minval=-1., maxval=1.)

  else:

    phi = jax.random.uniform(subkey, shape=np_shape, minval=0., maxval=2.*jnp.pi)
    key, subkey = jax.random.split(key)

    if cfg.ndim == 3:

      costheta = jax.random.uniform(subkey, shape=np_shape, minval=-1., maxval=+1.)

      key, subkey = jax.random.split(key)
      r = cfg.init_width * jnp.cbrt(jax.random.uniform(subkey, shape=np_shape, minval=0., maxval=1.))

      x = r * jnp.sqrt(1-costheta**2) * jnp.cos(phi)                                                                     
      y = r * jnp.sqrt(1-costheta**2) * jnp.sin(phi)
      z = r * costheta
                                             
      pos = jnp.concatenate([x, y, z], axis=-1).reshape(shape + (3*cfg.npart,))

    elif cfg.ndim == 2:

      r = cfg.init_width * jnp.sqrt(jax.random.uniform(subkey, shape=np_shape, minval=0., maxval=1.))

      x = r * jnp.cos(phi)
      y = r * jnp.sin(phi)

      pos = jnp.concatenate([x, y], axis=-1).reshape(shape + (2*cfg.npart,))

    else:
      raise ValueError("Number of spatial dimensions must be 1, 2 or 3.")

  pos = utils.broadcast(pos)

  return pos
