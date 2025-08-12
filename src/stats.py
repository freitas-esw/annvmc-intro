
import chex
import jax
import jax.numpy as jnp

from absl import logging

@chex.dataclass
class StatDataVMC:

  t: int
  alpha: jax.Array

  ene:  jax.Array
  kin:  jax.Array
  kjf:  jax.Array
  pot:  jax.Array
  Vext: jax.Array
  Vint: jax.Array

  wg_loss: jax.Array
  wv_loss: jax.Array

  def __init__(self, alpha):

    self.flag = None
    self.alpha = alpha
    
    self.ene  = jnp.zeros(1)
    self.kin  = jnp.zeros(1)
    self.kjf  = jnp.zeros(1)
    self.pot  = jnp.zeros(1)
    self.Vext = jnp.zeros(1)
    self.Vint = jnp.zeros(1)

    self.wg_loss = jnp.zeros(1)
    self.wv_loss = jnp.zeros(1)

    return

  def save_stats(self, loss, data):

    pot = data.Vext + data.Vint
    ene = data.kin + pot
 
    self.ene  = ene
    self.kin  = data.kin
    self.kjf  = data.kjf
    self.pot  = pot
    self.Vext = data.Vext
    self.Vint = data.Vint
 
    if self.flag:
      incr_loss = self.alpha * (loss  - self.wg_loss)
      self.wg_loss += incr_loss
      self.wv_loss = (1. - self.alpha) * (self.wv_loss + incr_loss**2 / self.alpha)
    else:
      self.wg_loss = loss
      self.wv_loss = 0. * loss
      self.flag = True

    return

  def vmc_log_stats(self, file, t, pmove):
    row = [t,
           self.wg_loss[0], 
           self.wv_loss[0], 
           self.ene[0], 
           self.kin[0], 
           self.kjf[0], 
           self.Vext[0], 
           self.Vint[0], 
           self.pot[0], 
           pmove[0]]
    file.write(','.join(map(str, row)) + '\n')
    return

  def logging(self, t, pmove):
    msg = 'Step %05d: W.Ene.=%03.4f; W.Ene.Var.=%03.4f; Energy=%03.4f; Kinetic=%03.4f; Potential=%03.4f, pmove=%0.2f'
    logging.info(msg, t, self.wg_loss[0], self.wv_loss[0], self.ene[0], self.kin[0], self.pot[0], pmove[0])
    return
