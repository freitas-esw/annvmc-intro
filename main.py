
from absl import app
from absl import flags
from absl import logging

from pyinstrument import Profiler
from ml_collections import ConfigDict
from ml_collections.config_flags import config_flags
from jax import config

from src import vmc

# defining double precision
config.update("jax_enable_x64", True)

# internal imports
FLAGS = flags.FLAGS
config_flags.DEFINE_config_file('config', None, 'Path to config file.')

def main(_):
  
  cfg = FLAGS.config

  # Log the configuration of the simulation
  logging.info('System config:\n\n%s', cfg)
  
  # Greetings to start the simulation
  logging.info('Welcome to Neural Network-based Variational Monte Carlo Simulations!')

  # Either to start the program profiler or not
  if cfg.debug.profile:
    profiler = Profiler()
    profiler.start()

  vmc.run(cfg)
  
  if cfg.debug.profile:
    profiler.stop()
    profiler.print() 

if __name__ == '__main__':
  app.run(main)
