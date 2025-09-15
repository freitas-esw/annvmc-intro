
from src import base
from src import potentials

def get_config():

  cfg = base.default()

  cfg.system.npart = 1
  cfg.system.ndim = 1

  # External and interaction potentials are:
  cfg.system.external = potentials.morse_oscillator
  cfg.system.interaction = potentials.free_particle

  # Hamiltonian parameters
  cfg.system.args.D = 0.5 

  # Initialization width for particle positions
  cfg.system.init_width = 0.5

  # Artificial neural network architecture
  in_size = cfg.system.ndim * cfg.system.npart
  cfg.wavefunction.args.hidden_dims = [in_size, 24, 32, 16, 1]

  # Optimization hyperparameters
  cfg.vmc.hp.lr = 0.005
  cfg.vmc.hp.delay = 3000
  cfg.vmc.hp.decay = 1.01

  # Monte Carlo integration steps
  cfg.vmc.iterations = 5000
  cfg.vmc.mci_steps = 100

  # Initial trial move step
  cfg.mcmc.width = 0.0005

  # Log labels and checkpoint frequency
  cfg.log.label = 'mo'
  cfg.log.save_path = 'workspace/morse/'
  cfg.log.save_frequency = 5000

  # Set the seed for RNG
  cfg.debug.seed = 42

  return cfg
