
from src import base
from src import potentials

def get_config():
  """
  """

  cfg = base.default()

  cfg.system.npart = 1
  cfg.system.ndim = 1

  # Available external potentials are:
  #  - free_particle 
  #  - harmonic_oscillator
  #  - morse_oscilator
  #  - hydrogen_atom
  #  - hydrogen_molecule
  #  - poschl_teller
  #  - yukawa_potential 
  cfg.system.external = potentials.harmonic_oscillator

  # Available interaction potentials are:
  #  - free_particle (non interacting particles) 
  #  - coulomb_interaction
  cfg.system.interaction = potentials.free_particle

  # Hamiltonian parameters
  cfg.system.args.omega = 1.0       # Trap frequency in harmonic oscillator units
#  cfg.system.args.D = 0.5          # Morse potential parameter 
#  cfg.system.args.R = 2.004        # Hydrogen distance for the H2+ ion
#  cfg.system.args.R = 1.399        # Hydrogen distance for the H2 molecule
#  n = 1.0                          # Number of bound states for Poschl-Teller potential
#  cfg.system.args.V0 = n*(n+1)/2   # Strength V_0 of Poschl-Teller potential
#  cfg.system.args.alpha = 1.0      # Strength of Yukawa potential (usually 1)
#  cfg.system.args.delta = 0.2      # Screenning factor

  # Initialization width for particle positions
  cfg.system.init_width = 4.

  # Artificial neural network architecture
  in_size = cfg.system.ndim * cfg.system.npart
#  cfg.wavefunction.args.hidden_dims = [in_size, 64, 64, 64, 1]  # for hydrogen molecule
  cfg.wavefunction.args.hidden_dims = [in_size, 24, 32, 16, 1]   # for all other systems

  # Optimization hyperparameters
  cfg.vmc.hp.lr = 0.005
  cfg.vmc.hp.delay = 3000
  cfg.vmc.hp.decay = 1.01 

  cfg.vmc.iterations = 5000
  cfg.vmc.mci_steps = 100

  cfg.mcmc.width = 0.01

  # Log labels and checkpoint frequency
  cfg.log.label = 'ho'
  cfg.log.save_path = 'workspace/oscillator/'
  cfg.log.save_frequency = 5000

  cfg.debug.seed = 42

  return cfg
