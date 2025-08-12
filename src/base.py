
from src import setup
from src import potentials
from src import ann
from src import samplers

from ml_collections import ConfigDict

def default() -> ConfigDict:

  cfg = ConfigDict({
    
    'nwalkers': 4096,

    'system': {
      'init_positions': setup.init_positions_randomly,
      'npart': 1,
      'ndim': 1,               
      'init_width': 1.0,
      'interaction': potentials.free_particle,
      'external': potentials.harmonic_oscillator,
      'args': {
        'omega': 1.0, # harmonic oscillator frequency
        'D': 0.5,     # morse oscillator parameter
        'R': 1.48,    # atom separation for hydrogen molecule 
      },
    },

    'wavefunction': {
      'init_params': ann.init_params,
      'logpsi': ann.wavefunction,
      'args': {
        'hidden_dims': [1, 16, 32, 16, 1],
      },
    },

    'mcmc': {
      'sampler': samplers.move_all_metropolis_hastings,
      'steps': 5,
      'width': 0.1,
      'adapt_frequency': 10,
      'args': {}
    },

    'vmc': {
      'iterations': 20000,
      'mci_steps': 100,
      # weighted stats factor
      'alpha': 0.1,
      'penalty': None,
      'hp': {
        'lr': 0.01,
        'delay': 10000,
        'decay': 0.99,
      },
      'grad': {
        'b1': 0.9,
        'b2': 0.999,
        'eps': 1e-8,
        'eps_root': 0.0,
        'weight_decay': 0.01,
      },
    },

    'debug': {
      'seed': 42,
      'profile': False,
      'check_nan': False,
    },

    'log': {
      'save_path': 'workspace/',
      'label': 'harmonic-oscillator',
      'save_frequency': 20000,
    },

  })

  return cfg
