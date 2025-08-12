
from src import base
from src import potentials

def get_config():

  cfg = base.default()

  cfg.system.npart = 1
  cfg.system.ndim = 1

  # External and interaction potentials are:
  cfg.system.external = potentials.poschl_teller
  cfg.system.interaction = potentials.free_particle

  # Hamiltonian parameters
  n = 1.0 # Number of bound states for Poschl-Teller potential
  cfg.system.args.V0 = n*(n+1)/2 # Strength V_0 of Poschl-Teller potential

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
  cfg.mcmc.width = 0.01

  # Log labels and checkpoint frequency
  cfg.log.label = 'pt'
  cfg.log.save_path = 'workspace/poschl-teller/'
  cfg.log.save_frequency = 5000

  # Set the seed for RNG
  cfg.debug.seed = 42

  return cfg
