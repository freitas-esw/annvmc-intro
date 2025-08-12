
import chex
import jax
import jax.numpy as jnp

from jax import lax
from typing import Callable
from ml_collections import ConfigDict

from src import utils

@chex.dataclass
class SupData:
  """ Supplementary data  """
  local_energy: jax.Array
  kin: jax.Array
  kjf: jax.Array
  Vext: jax.Array
  Vint: jax.Array


def basic_features(x, npart, ndim):
  """
  """
  
  q = jnp.reshape(x, [npart, ndim])
  r = jnp.linalg.norm(q, axis=-1)

  dq = jnp.reshape(q, [1, npart, ndim]) - jnp.reshape(q, [npart, 1, ndim])
  dr = jnp.linalg.norm(dq + jnp.eye(npart)[..., None], axis=-1) * (1 - jnp.eye(npart))

  return q, r, dq, dr

def make_local_energy(npart: int, 
                      ndim: int, 
                      wavefunction: Callable, 
                      external: Callable, 
                      interaction: Callable, 
                      **kwargs):
  """
  """

  nq = npart * ndim
  umat = jnp.eye(nq)
  grad_psi = jax.grad(wavefunction, argnums=1)
  
  def kinetic_energy(pars, pos):

    pseudoforce = lambda y: grad_psi(pars, y)

    def _fn(i, val):
      primal, tangent = jax.jvp(pseudoforce, (pos,), (umat[i],))
      kjf = primal[i]**2
      kin = - tangent[i] - kjf 
      return val + jnp.array([kin, kjf])

    return 0.5 * lax.fori_loop(0, nq, _fn, jnp.zeros(2))

  def potential_energy(pos):
    q, r, dq, dr = basic_features(pos, npart, ndim)
    Vext = external(q, r, **kwargs)
    i, j = jnp.triu_indices(npart, k=1)
    Vint = interaction(dq[i,j,:], dr[i,j], **kwargs)
    return Vext, Vint

  def local_energy(pars, pos):
    kin, kjf = kinetic_energy(pars, pos)
    Vext, Vint = potential_energy(pos)
    return kin, kjf, Vext, Vint

  return local_energy

def make_total_energy(cfg: ConfigDict, wavefunction: Callable):
  """
  wavefunction: wavefunction callable
  """ 

  # Create and vectorize the local energy
  el = make_local_energy(npart = cfg.npart,
                         ndim = cfg.ndim, 
                         wavefunction = wavefunction,
                         external = cfg.external,
                         interaction = cfg.interaction,
                         **cfg.args.to_dict())
  local_energy = jax.vmap(el, in_axes=(None, 0), out_axes=0)
  v_wf = jax.vmap(wavefunction, in_axes=(None, 0), out_axes=0)
  
  @jax.custom_jvp
  def total_energy(pars, x):
    kin, kjf, Vext, Vint = local_energy(pars, x) 
    ave_kin = utils.pmean(jnp.mean(kin))
    ave_kjf = utils.pmean(jnp.mean(kjf))
    ave_Vext = utils.pmean(jnp.mean(Vext))
    ave_Vint = utils.pmean(jnp.mean(Vint))
    energy = ave_kin + ave_Vext + ave_Vint
    return energy, SupData(local_energy = kin + Vext + Vint, 
                           kin = ave_kin, 
                           kjf = ave_kjf,
                           Vext = ave_Vext,
                           Vint = ave_Vint)
  
  @total_energy.defjvp
  def total_energy_jvp(primals, tangents):

    pars, x = primals
    energy, data = total_energy(pars, x)

    diff = data.local_energy - energy

    psi_primal, psi_tangent = jax.jvp(v_wf, primals, tangents)
    primals_out = energy, data
    tangents_out = (jnp.dot(psi_tangent, diff), data)

    return primals_out, tangents_out

  return total_energy
