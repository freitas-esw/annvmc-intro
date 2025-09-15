
from absl import app
from absl import logging

from pyinstrument import Profiler
from ml_collections import ConfigDict
from jax import config

from src import vmc

from src.inputs import ho
from src.inputs import mo
from src.inputs import pt
from src.inputs import yu
from src.inputs import h
from src.inputs import h2
from src.inputs import h2p

# defining double precision
config.update("jax_enable_x64", True)

def main(_):
  
  # Greetings to start the simulation
  logging.info('Welcome to Neural Network-based Variational Monte Carlo Simulations!')

  # 1D harmonic oscillator
  logging.info('Starting the simulation for 1D harmonic oscillator!')
  cfg = ho.get_config() 
  logging.info('System config:\n\n%s', cfg)
  vmc.run(cfg)
  logging.info('Simulations done 1/7...')

  # 1D Morse oscillator
  logging.info('Starting the simulation for 1D Morse oscillator!')
  cfg = mo.get_config() 
  logging.info('System config:\n\n%s', cfg)
  vmc.run(cfg)
  logging.info('Simulations done 2/7...')

  # 1D Poschl-Teller potential
  logging.info('Starting the simulation for 1D Poschl-Teller potential!')
  cfg = pt.get_config() 
  logging.info('System config:\n\n%s', cfg)
  vmc.run(cfg)
  logging.info('Simulations done 3/7...')

  # 3D Yukawa potential
  logging.info('Starting the simulation for 3D Yukawa potential!')
  cfg = yu.get_config() 
  logging.info('System config:\n\n%s', cfg)
  vmc.run(cfg)
  logging.info('Simulations done 4/7...')

  # Hydrogen atom
  logging.info('Starting the simulation for the hydrogen atom!')
  cfg = h.get_config() 
  logging.info('System config:\n\n%s', cfg)
  vmc.run(cfg)
  logging.info('Simulations done 5/7...')

  # Ionic hydrogen molecule
  logging.info('Starting the simulation for the ion hydrogen molecule!')
  cfg = h2p.get_config() 
  logging.info('System config:\n\n%s', cfg)
  vmc.run(cfg)
  logging.info('Simulations done 6/7...')

  # Hydrogen molecule
  logging.info('Starting the simulation for the hydrogen molecule!')
  cfg = h2.get_config() 
  logging.info('System config:\n\n%s', cfg)
  vmc.run(cfg)
  logging.info('Simulations done 7/7.')
  logging.info('End of simulations.')

if __name__ == '__main__':
  app.run(main)
