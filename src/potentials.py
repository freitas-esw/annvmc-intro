
import jax.numpy as jnp

def free_particle(q, r, **kwargs):
  """
  """
  del q, r, kwargs
  return 0.0

def harmonic_oscillator(q, r, **kwargs):
  """
  """
  del r
  return 0.5 * jnp.sum(kwargs['omega']**2 * q**2)

def morse_oscillator(q, r, **kwargs):
  """
  """
  del r
  short_range = jnp.exp(-2. * q)
  long_range = -2. * jnp.exp(-q)
  return kwargs['D'] * jnp.sum(short_range + long_range)

def hydrogen_atom(q, r, **kwargs):
  """
  """
  del q, kwargs
  return -jnp.sum(1. / r)

def hydrogen_molecule(q, r, **kwargs):
  """
  """
  q_H = jnp.array([0., 0., kwargs['R']/2])[None,:]
  dr_H = jnp.linalg.norm(2. * q_H)
  dr_p = jnp.linalg.norm(q - q_H, axis=-1)
  dr_m = jnp.linalg.norm(q + q_H, axis=-1)
  return -jnp.sum(1./dr_p + 1./dr_m) + 1./dr_H

def poschl_teller(q, r, **kwargs):
  """
  """
  del r
  return - jnp.sum(kwargs['V0'] / jnp.cosh(q)**2)

def yukawa_potential(q, r, **kwargs):
  """
  """
  del q
  return - kwargs['alpha'] * jnp.sum(jnp.exp(-r * kwargs['delta']) / r)

def coulomb_interaction(dq, dr, **kwargs):
  """
  """
  del dq, kwargs
  return jnp.sum(1. / dr) 
