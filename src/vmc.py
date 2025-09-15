
import optax
import chex
import jax
import kfac_jax

import jax.numpy as jnp
import numpy as np

from absl import logging
from ml_collections import ConfigDict

from src import utils
from src import setup
from src import stats
from src import operators
from src import samplers

def make_mc_integration_step(mcmc, total_energy):
  """
  """

  def step(pars, pos, key, width):
    """
    """
    pos, pmove = mcmc(pars, pos, key, width)
    energy, stats = total_energy(pars, pos)
    return pos, energy, stats, pmove

  return step

def make_gradient_descent_step(mcmc, loss_and_grad, optimizer):
  """
  """

  def update(pars, pos, state):
    """
    """
    (loss, stats), grad = loss_and_grad(pars, pos)
    grad = utils.pmean(grad)
    updates, new_state = optimizer.update(grad, state, pars)
    new_pars = optax.apply_updates(pars, updates)
    return loss, stats, new_pars, new_state

  def step(pars, pos, state, key, width):
    """
    """
    pos, pmove = mcmc(pars, pos, key, width)
    loss, stats, new_pars, new_state = update(pars, pos, state)
    return pos, loss, stats, new_pars, new_state, pmove

  return step

def run(cfg: ConfigDict):
  """
  """

  # Logging the method
  logging.info('The variational Monte Carlo (VMC) was the chosen method.')

  # Setup the data shape according to available devices 
  shape = setup.get_device_shape(cfg.nwalkers)
 
  # Defining the seed for RNG 
  logging.info('Seed number: %d', cfg.debug.seed)
  key = jax.random.PRNGKey(cfg.debug.seed) 

  # Initialize particle positions and variational parameters
  key, subkey = jax.random.split(key)
  pos = cfg.system.init_positions(cfg.system, shape, subkey)
  pars = utils.replicate(cfg.wavefunction.init_params(**cfg.wavefunction.args.to_dict()))
  mcmc_width = utils.replicate(jnp.asarray(cfg.mcmc.width))

  # Shard the key over devices
  sharded_key = utils.shard_key(key)

  # Create a callble to compute the wavefunction
  wf = cfg.wavefunction.logpsi

  # Create the function to compute the total energy
  total_energy = operators.make_total_energy(cfg.system, wf)

  # Create the function that realizes the sampling step
  mcmc = samplers.make_mcmc_step(sampler = cfg.mcmc.sampler, 
                                 probability = wf, 
                                 steps = cfg.mcmc.steps,
                                 nwalkers = shape[1], # nwalkers per device
                                 **cfg.mcmc.args.to_dict())

  # Defining learning rate schedule
  def learning_rate(t):
    return cfg.vmc.hp.lr * jnp.power(1. / (1. + t / cfg.vmc.hp.delay), cfg.vmc.hp.decay)

  if cfg.vmc.penalty:
    penalty = utils.pmap(jax.vmap(cfg.vmc.penalty, in_axes=(None, 0), out_axes=0))
    evaluate_loss = lambda pars, pos: total_energy(pars, pos) + penalty(pars, pos)
  else:
    evaluate_loss = total_energy
    
  # Differentiate wrt parameters (argument 0)
  loss_and_grad = jax.value_and_grad(evaluate_loss, argnums=0, has_aux=True)

  # Create optimizer
  optimizer = optax.chain(
    optax.scale_by_adam(b1 = cfg.vmc.grad.b1, 
                        b2 = cfg.vmc.grad.b2, 
                        eps = cfg.vmc.grad.eps, 
                        eps_root = cfg.vmc.grad.eps_root),
    optax.add_decayed_weights(weight_decay = cfg.vmc.grad.weight_decay, mask = None),
    optax.scale_by_schedule(learning_rate),
    optax.scale(-1.),
    )
  state = jax.pmap(optimizer.init)(pars)
  step = make_gradient_descent_step(mcmc, loss_and_grad, optimizer)
  pstep = utils.pmap(step, donate_argnums=(0, 1, 2))

  # Create the file to save the data stats of each step
  scheme = ['t', 'W.Ene.', 'W.Ene.Var.', 'Energy', 'Kinetic', 'KJF', 'External', 'Interaction', 'Potential', 'Accept']
  stats_file = utils.create_file(cfg.log.save_path, 'vmc-stats', scheme=scheme)

  # Define the variable to store the data stats
  observables = stats.StatDataVMC(alpha = cfg.vmc.alpha)

  # Performs the Variational Monte Carlo optimization
  logging.info('Starting Variational Monte Carlo optimization.')
  for t in range(cfg.vmc.iterations):

    sharded_key, subkeys = utils.p_split(sharded_key)
    pos, loss, data, pars, state, pmove = pstep(pars, pos, state, subkeys, mcmc_width)

    observables.save_stats(loss, data)
    observables.vmc_log_stats(stats_file, t, pmove)
    observables.logging(t, pmove)
 
    # Adjusting the trial move step
    if t % cfg.mcmc.adapt_frequency == 0:
      if pmove[0] > 0.5:
        mcmc_width *= 1.1
      elif pmove[0] < 0.45:
        mcmc_width /= 1.1
    else:
      if pmove[0] > 0.999:
        mcmc_width *= 1.1
      elif pmove[0] < 0.199:
        mcmc_width /= 1.1

    if cfg.debug.check_nan:
      tree = {'pars': pars, 'loss': loss, 'state': state, 'data': data, 'pos': pos}
      chex.assert_tree_all_finite(tree)

    # Checkpointing
    if (t+1) % cfg.log.save_frequency == 0 or (t+1) == cfg.vmc.iterations:
      utils.save_ckpt(cfg.log.save_path, 
                      f'vmc-ckpt-{t:06d}',
                      t=t, 
                      pos=pos, 
                      pars=pars,
                    #  state=state, # inhomogeneous part issue 
                      width=mcmc_width,
                      key=sharded_key, 
                      config=cfg.__repr__())

  stats_file.close()

  # Create the Monte Carlo integration step
  step = make_mc_integration_step(mcmc, total_energy)
  pstep = utils.pmap(step, donate_argnums=(1))

  # Create the file to save the data stats of each Monte Carlo integration step
  stats_file = utils.create_file(cfg.log.save_path, 'mc-integration', scheme=scheme)
  walkers_file = open(cfg.log.save_path+'/walkers.npy', 'wb')

  # Reset the variable to store the data stats
  observables = stats.StatDataVMC(alpha = cfg.vmc.alpha)

  # Performs the Variational Monte Carlo integration
  logging.info('Starting Variational Monte Carlo integration.')
  for t in range(cfg.vmc.mci_steps):

    sharded_key, subkeys = utils.p_split(sharded_key)
    pos, energy, data, pmove = pstep(pars, pos, subkeys, mcmc_width)

    observables.save_stats(energy, data)
    observables.vmc_log_stats(stats_file, t, pmove)
    observables.logging(t, pmove)

    np.save(walkers_file, np.array(pos))

  stats_file.close()
  walkers_file.close()

  return
