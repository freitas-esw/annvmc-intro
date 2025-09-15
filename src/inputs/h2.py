
from src import base
from src import potentials

def get_config():

  cfg = base.default()

  cfg.system.npart = 2
  cfg.system.ndim = 3

  # External and interaction potentials are:
  cfg.system.external = potentials.hydrogen_molecule
  cfg.system.interaction = potentials.coulomb_interaction

  # Hamiltonian parameters
  cfg.system.args.R = 1.399 # Hydrogen distance for the H2 molecule

  # Initialization width for particle positions
  cfg.system.init_width = 5.

  # Artificial neural network architecture
  in_size = cfg.system.ndim * cfg.system.npart
  cfg.wavefunction.args.hidden_dims = [in_size, 64, 64, 64, 1] 

  # Optimization hyperparameters
  cfg.vmc.hp.lr = 0.004
  cfg.vmc.hp.delay = 3000
  cfg.vmc.hp.decay = 1.01

  # Monte Carlo integration steps
  cfg.vmc.iterations = 5000
  cfg.vmc.mci_steps = 100

  # Initial trial move step
  cfg.mcmc.width = 0.005

  # Log labels and checkpoint frequency
  cfg.log.label = 'h2'
  cfg.log.save_path = 'workspace/h2-mol/'
  cfg.log.save_frequency = 5000

  # Set the seed for RNG
  cfg.debug.seed = 42

  return cfg
