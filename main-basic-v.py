# Basic version of the code applied to the harmonic oscillator

# Importing libraries
from absl import app
from absl import logging

import optax
import jax 
import jax.numpy as jnp

from jax import lax

#####################################################################
# the following variables can be changed according to their meaning #
#####################################################################

# define the number of optimization iterations
iterations = 1000

# define the number of particles, spatial dimensions, and number of configurations
npart = 1
ndim = 1
nconf = 4096

# define the hidden layers width
widths = [32, 32, 32,]

# define the box size where the initial coordinates will be
Lbox = 2.

# initial trial move step for Metropolis algorithm
mcmc_width = 0.1

######################################################################
# only change the following variables if you know what you are doing #
######################################################################

# defining double precision
jax.config.update("jax_enable_x64", True)

# define the number of degrees of freedom (that is equal to the size of input features n_0)
n0 = npart * ndim

# define the vector n of widths
n = [n0,] + widths + [1,]

# define a identity matrix of size n0
Imat = jnp.eye(n0)
  
delay = iterations // 2
decay = 1.0
rate = 0.005

def learning_rate(t):
  return rate * jnp.power(1. / (1. + t / delay), decay)

def metropolis(pars, pos, key, log_p, num_accepts, delta):
  """ 
  Performs a trial move of all coordinates and accepts/rejects 
  according to metropolis algorithm
  """

  # split the keys before generating randum numbers
  key, subkey = jax.random.split(key)
  # propose a trial configuration by moving the older one
  pos_t = pos + delta * jax.random.normal(subkey, shape=pos.shape)
  # compute \ln |\psi(x_t)|^2 for trial configuration
  log_p_t = 2. * v_wf(pars, pos_t)
  # compute the probability ratio \ln (\psi(x_t) / \psi (x))^2
  ratio = log_p_t - log_p

  # split the keys before generating randum numbers
  key, subkey = jax.random.split(key)
  # generate uniform random number for each configuration to apply acceptation criteria 
  rnd = jnp.log(jax.random.uniform(subkey, shape=log_p.shape))
  # define logical variable with acceptance condition
  cond = ratio > rnd
  # carry out the acceptation/rejection 
  pos_n = jnp.where(cond[...,None], pos_t, pos)
  log_p_n = jnp.where(cond, log_p_t, log_p)
  # count the number of accepted moves
  num_accepts += jnp.sum(cond)

  return pos_n, key, log_p_n, num_accepts

@jax.jit
def mcmc(pars, pos, key, delta):
  """
  Performs the Metropolis algorithm 5 times and compute acceptation rate
  """
  # define a wrapper function for metropolis in order to use jax.fori_loop 
  def step_fn(i, x):
    return metropolis(pars, *x, delta)

  # compute \ln |\psi (x)|^2 for current configurations
  log_p = 2. * v_wf(pars, pos)
  # perform accept/reject step 5 times
  pos, key, _, num_accepts = lax.fori_loop(0, 5, step_fn, (pos, key, log_p, 0.))
  # compute acceptation rate
  pmove = jnp.sum(num_accepts) / (5. * nconf)

  return pos, pmove

def init_params(key):
  """
  Initializes the variational parameters
  Args:
    key: RNG key
  Returns:
    pars: all weights and biases in a Python dictionary
  """
  
  dims_in = n[:-1]
  dims_out = n[1:]

  pars = {}
  pars['nn'] = [ {} for _ in range(len(n)-1) ]

  for i in range(len(n)-1):

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
  Function to compute the trial wave function value for a given
  configuration pos and a set of variational parameters pars
  Args:
    pos: coordinate positions of the particles
    pars: variational parameters
  Returns:
    ln [psi_pars (pos)]
  """
  h = pos
  for i in range(len(pars['nn'])-1):
    h = jnp.tanh(linear_layer(h, **pars['nn'][i]))
  h = linear_layer(h, **pars['nn'][-1])
  return -jnp.sum(jax.nn.softplus(h))
  
# generate a vectorized version of wavefunciton that computes the 
# log psi of a "batch" of configurations
v_wf = jax.vmap(wavefunction, in_axes=(None, 0), out_axes=0)
# more info in https://docs.jax.dev/en/latest/_autosummary/jax.vmap.html#jax.vmap
  
# Use automatic differentiation to generate a gradient of the
# wavefunction with respect to the variational parameters
grad_t_psi = jax.grad(wavefunction, argnums=0)
# Use automatic differentiation to generate a gradient of the
# wavefunction with respect to the particle coordinates
grad_x_psi = jax.grad(wavefunction, argnums=1)
# more info in https://docs.jax.dev/en/latest/_autosummary/jax.grad.html

def potential_energy(pos):
  """
  Computes the potential energy for a given configuration pos
  """
  return 0.5 * jnp.sum(pos**2)

def kinetic_energy(pars, pos):
  """
  Computes the local kinetic energy ({\nabla^2 \psi} / {2\psi}) for a given configuration pos
  """

  # to call grad_x_psi you need to pass two variables, for example, grad_x_psi(v1, v2)
  # but I will always use the variable pars for v1, then I need to
  # create a wrapper of grad_x_psi 
  pseudoforce = lambda y: grad_x_psi(pars, y)
  # now, pseudoforce(pos) evaluate directly grad_x_psi(pars, pos)

  # the definition of _fn needs to be inside kinetic_energy because 
  # pseudoforce use the variable pars 
  def _fn(i, val):
    # jax.jvp is automatic differentiation to compute Jacobians
    # more info in https://docs.jax.dev/en/latest/_autosummary/jax.jvp.html
    primal, tangent = jax.jvp(pseudoforce, (pos,), (Imat[i],))
    # primal is equal to pseudoforce(pos), which is an vector that contains derivatives w.r.t. coordinates 
    # tangent is equal to grad_pseudoforce * (column i of Imat), which is the second derivative of the i-th component.

    # note that grad_pseudoforce is a matrix, where the component G_{ij} = {\del^2 \psi} / {\del_i \del_j}
    # and i and j are index for all coordinates of all particles. The multiplication with Imat[i] return G_{ii}

    # for the local kinetic energy we need to compute  [{\del^2 \psi} / {\del x_i^2}] / \psi
    # since wavefunction returns log psi, second derivative w.r.t. to component i is given by
    # {\del^2 \ln \psi} / {\del x_i^2} + [{\del \ln \psi} / {\del x_i}]^2
    kin = - tangent[i] - primal[i]**2
    return val + kin

  # lax.fori_loop execute the following sequence of code
  # val = 0.0
  # for i = 0, n0
  #   val += fn(i, val)
  # return val
  return 0.5 * lax.fori_loop(0, n0, _fn, 0.0)

def el(pars, pos):
  """
  Computes the local energy for a single configuration
  """
  kin = kinetic_energy(pars, pos)
  pot = potential_energy(pos)
  return kin + pot
  
# generate a vectorized version of el that computes the 
# local energy of a "batch" of configurations
local_energy = jax.vmap(el, in_axes=(None, 0), out_axes=0)
# more info in https://docs.jax.dev/en/latest/_autosummary/jax.vmap.html#jax.vmap

@jax.custom_jvp
def total_energy(pars, pos):
  """
  Computes the expected value of the Hamiltonian by Monte Carlo integration
  """
  loc_ene = local_energy(pars, pos) 
  energy = jnp.mean(loc_ene)
  return energy, loc_ene
 
# define a custom Jacobian for the total energy
@total_energy.defjvp
def total_energy_jvp(primals, tangents):

  pars, pos = primals
  energy, loc_ene = total_energy(pars, pos)

  diff = loc_ene - energy

  # compute psi_primal = psi(pars, pos) and psi_tangent = {\del \ln \psi}{\del \theta}
  psi_primal, psi_tangent = jax.jvp(v_wf, primals, tangents)

  primals_out = energy, loc_ene
  tangents_out = (jnp.dot(psi_tangent, diff), loc_ene)
  # note that jnp.dot(psi_tangent, diff) is equal to gradient of the total energy 
  # w.r.t. variational parameters

  return primals_out, tangents_out

# define a function that computes the total energy and its gradients w.r.t. variational 
# parameters by using automatic differentiation
ene_and_grad = jax.value_and_grad(total_energy, argnums=0, has_aux=True)
# more info in https://docs.jax.dev/en/latest/_autosummary/jax.value_and_grad.html#jax.value_and_grad

def main(_):
  logging.info('Welcome to Neural Network-based Variational Monte Carlo Simulations!')
 
  delta = mcmc_width

  # create a key for generating random numbers
  # more info in https://docs.jax.dev/en/latest/jax.random.html
  key = jax.random.PRNGKey(42) 

  # always split the key before generating new random numbers
  key, subkey = jax.random.split(key)
  # initialize variational parameters
  pars = init_params(subkey)

  key, subkey = jax.random.split(key)
  # initialize particle coordinates
  pos = Lbox * (0.5 - jax.random.uniform(subkey, shape=(nconf, npart*ndim)))

  # create the ADAM optimizer
  optimizer = optax.chain(
    optax.scale_by_adam(b1 = 0.9, b2 = 0.999, eps = 1e-8, eps_root = 0.0),
    optax.scale_by_schedule(learning_rate),
    optax.scale(-1.),
    )
  # create variable to store the optimizer state
  state = optimizer.init(pars)

  logging.info('Starting Variational Monte Carlo optimization.')

  for t in range(iterations):
  
    key, subkey = jax.random.split(key)

    # perform Metropolis algorithm
    pos, pmove = mcmc(pars, pos, subkey, delta)
    # compute total energy and its gradients w.r.t. parameters
    (ene, loc_ene), grad = ene_and_grad(pars, pos)
    # update parameters
    updates, state = optimizer.update(grad, state, pars)
    pars = optax.apply_updates(pars, updates)
    
    # Adjusting the trial move step
    if t % 10 == 0:
      if pmove > 0.5:
        delta *= 1.1
      elif pmove < 0.45:
        delta /= 1.1

    logging.info('Step %05d: Energy=%03.4f, pmove=%03.2f', t, ene, pmove)

  return

if __name__ == '__main__':
  app.run(main)
