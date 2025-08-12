
import jax
import jax.numpy as jnp

from jax import lax

from src import utils

def move_all_metropolis_hastings(
    pars,
    probability,
    pos,
    key,
    logprob,
    num_accepts,
    stddev
):
  """
  """

  key, subkey = jax.random.split(key)
  pos_trial = pos + stddev * jax.random.normal(subkey, shape=pos.shape)
  logprob_trial = 2. * probability(pars, pos_trial)
  ratio = logprob_trial - logprob

  key, subkey = jax.random.split(key)
  rnd = jnp.log(jax.random.uniform(subkey, shape=logprob.shape))
  cond = ratio > rnd
  pos_new = jnp.where(cond[...,None], pos_trial, pos)
  logprob_new = jnp.where(cond, logprob_trial, logprob)
  num_accepts += jnp.sum(cond)

  return pos_new, key, logprob_new, num_accepts

def make_mcmc_step(sampler, probability, steps, nwalkers, **kwargs):
  """
  probability: half log probability callable
  """

  vprob = jax.vmap(probability, in_axes=(None, 0), out_axes=0)

  @jax.jit
  def mcmc_step(pars, pos, key, width):

    def step_fn(i, x):
      return sampler(pars, vprob, *x, stddev=width)

    logprob = 2. * vprob(pars, pos)
    pos, key, _, num_accepts = lax.fori_loop(0, steps, step_fn,
                                              (pos, key, logprob, 0.))
    pmove = jnp.sum(num_accepts) / (steps * nwalkers)
    pmove = utils.pmean(pmove)

    return pos, pmove

  return mcmc_step

