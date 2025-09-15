import os
import math as mt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from jax import vmap
from src import ann 

prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']

# Set plot config - APS style
plt.rcParams['figure.dpi']     = 100
plt.rcParams['font.size']      = 17
plt.rcParams['axes.linewidth'] = 1.25

plt.rcParams['font.weight']        = 'normal'
plt.rcParams['axes.labelweight']   = 'normal'
plt.rcParams['axes.titleweight']   = 'normal'

plt.rcParams['legend.frameon'] = False
plt.rcParams['legend.loc']     = 'upper center'

plt.rcParams['xtick.labelsize']     = 15
plt.rcParams['xtick.direction']     = 'in'
plt.rcParams['xtick.major.width']   = 1.25
plt.rcParams['xtick.major.size']    = 5
plt.rcParams['xtick.minor.visible'] = True
plt.rcParams['xtick.minor.width']   = 1.25
plt.rcParams['xtick.minor.size']    = 3.5
plt.rcParams['xtick.top']           = 'on'

plt.rcParams['ytick.labelsize']     = 15
plt.rcParams['ytick.direction']     = 'in'
plt.rcParams['ytick.major.width']   = 1.25
plt.rcParams['ytick.major.size']    = 5
plt.rcParams['ytick.minor.visible'] = True
plt.rcParams['ytick.minor.width']   = 1.25
plt.rcParams['ytick.minor.size']    = 3.5
plt.rcParams['ytick.right']         = 'on'

# Latex options
plt.rcParams['text.usetex'] = False
plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['font.family'] = 'STIXGeneral'

def result_str(value, error):
  d = - int(mt.floor(mt.log10(np.abs(error))))
  e = int(np.round(error, d) * 10**d)
  v = np.round(value , d)
  return f'{v:.{d}f}'+'({:n})'.format(e)

def gswf_ho(x):
  return np.exp(-x**2/2.)/np.pi**(1./4.)

def gswf_mo(x):
  return np.sqrt(2.)*np.exp(-np.exp(-x)-x/2.)

def gswf_pt(x):
  #return np.sech(x)/np.sqrt(2.)
  return np.cosh(x)**(-1)/np.sqrt(2.)

def gswf_h(x):
  r = x #  r=np.linalg.norm(x)
  return np.exp(-r)/np.sqrt(np.pi)

def gswf_yu(x):
  r = x #  r=np.linalg.norm(x)
  delta=0.2
  expo = -r + delta**2 * (3. - 2.*delta) * r**2/ 12. - delta**3 * r**3 / 18.
  norm = np.sqrt(np.pi * (1. + 3.*delta**2/2. - 11.*delta**3/6.))
  return np.exp(expo)/norm

if __name__ == "__main__":

  gswf_ann = vmap(ann.wavefunction, in_axes=(None, 0))

  directory = ['oscillator/', 'morse/', 'poschl-teller/', 'hydrogen/', 'yukawa/', 'h2-ion/', 'h2-mol/']
  dir_1d = ['oscillator/', 'morse/', 'poschl-teller/']
  dir_coulomb = ['hydrogen/', 'yukawa/']
  dir_h2 = ['h2-ion/', 'h2-mol/']
  path = 'workspace/'
  Eunits = ['$\\hbar\\omega$', 
            '$\\hbar^2 / m a_{\\rm m}$', 
            '$\\hbar^2 / m a_{\\rm pt}$',
            '$\\hbar^2 / m a_{\\rm B}$',
            '$\\hbar^2 / m a_{\\rm B}$',
            '$\\hbar^2 / m a_{\\rm B}$',
            '$\\hbar^2 / m a_{\\rm B}$']
  Lunits = ['a_{\\rm ho}', 
            'a_{\\rm m}', 
            'a_{\\rm pt}',
            'a_{\\rm B}',
            'a_{\\rm B}',
            'a_{\\rm B}',
            'a_{\\rm B}']
  Eref = [0.5, -0.125, -0.5, -0.5, -0.326, -0.59724, -1.1645]  
    
  file_i = 'mc-integration-01.csv'
  file_o = 'vmc-stats-01.csv'
  file_f = 'vmc-ckpt-004999.npz'
  file_w = 'walkers.npy'

  i = 0 #25
  f = -1 #1025
  s = 5 #15

  os.chdir(path)
  for i in range(len(directory)):
    os.chdir(directory[i])

    data_i = pd.read_csv(file_i)
    data_o = pd.read_csv(file_o)
    pars = np.load(file_f, allow_pickle=True)['pars'].tolist()
    data_w = open(file_w, 'rb')

    pos = np.load(data_w)
    flag = True

    while flag:
      try:
        pos = np.concatenate([pos, np.load(data_w)], axis=0)
      except:
        break

    x = data_o['t'].values[i:f:s]
    y = data_o['W.Ene.'].values[i:f:s]
    e = data_o['W.Ene.Var.'].values[i:f:s]

    fig, ax = plt.subplots(figsize=[5.5,4.5], dpi=300)
    ax.errorbar(x, y, yerr=np.sqrt(e), marker='.', ms=1, ls='', c=colors[0], zorder=0)
    ax.plot(x, np.ones_like(x)*Eref[i], ls='--', c=colors[1], zorder=1)
    ax.set_xlabel('Optimization iterations')
    ax.set_ylabel('Energy ['+Eunits[i]+']')

    fig.tight_layout()
    fig.savefig('opt-ene.png', transparent=True)

    print('System: ', directory[i])
    print('Energy: ', result_str(data_i['W.Ene.'].values[-1], 
                        np.sqrt(data_i['W.Ene.Var.'].values[-1])))
    print()

    if directory[i] in dir_1d:

      ave_pos = np.mean(pos)
      var_pos = np.var(pos)
      xrange = (ave_pos-4*np.sqrt(var_pos), ave_pos+4*np.sqrt(var_pos))
      y, x_ed = np.histogram(pos.reshape([-1,]), bins=101, density=True, range=xrange)
      x = (x_ed[:-1] + x_ed[1:])/2. 

      if directory[i] == 'oscillator/':
        gswf = gswf_ho 
      elif directory[i] == 'morse/':
        gswf = gswf_mo 
      elif directory[i] == 'poschl-teller/':
        gswf = gswf_pt 

      fig, ax = plt.subplots(figsize=[5.5,4.5], dpi=300)
      ax.scatter(x, y, s=10, marker='.', ls='', c=colors[0], zorder=1, label='VMC simulation')
      ax.plot(x, gswf(x)**2, ls='--', c=colors[1], zorder=0, label='analytical result')
      ax.set_xlabel('$x$ ['+Lunits[i]+']')
      ax.set_ylabel('$\\rho (x)$ ['+Lunits[i]+'$^{-1}$]')

      fig.legend(loc='upper right', fontsize='x-small', bbox_to_anchor=(0.95, 0.925))
      fig.tight_layout()
      fig.savefig('density.png', transparent=True)

      xrange = (ave_pos-10*np.sqrt(var_pos), ave_pos+10*np.sqrt(var_pos))
      x = np.linspace(xrange[0], xrange[1], 1001)
      psi = np.exp(gswf_ann(pars, x))
      norm = np.sum((x[1]-x[0])*psi**2)

      xrange = (ave_pos-4*np.sqrt(var_pos), ave_pos+4*np.sqrt(var_pos))
      x = np.linspace(xrange[0], xrange[1], 101)
      psi = np.exp(gswf_ann(pars, x))/np.sqrt(norm)

      fig, ax = plt.subplots(figsize=[5.5,4.5], dpi=300)
      ax.scatter(x, psi, s=10, marker='.', ls='', c=colors[0], zorder=1, label='VMC simulation')
      ax.plot(x, gswf(x), ls='--', c=colors[1], zorder=0, label='analytical result')
      ax.set_xlabel('$x$ [$'+Lunits[i]+'$]')
      ax.set_ylabel('$\\psi (x)$ [$1/\\sqrt{'+Lunits[i]+'}$ ]')

      fig.legend(loc='upper right', fontsize='x-small', bbox_to_anchor=(0.95, 0.925))
      fig.tight_layout()
      fig.savefig('gswf.png', transparent=True)

    if directory[i] in dir_coulomb:

      r = np.linalg.norm(pos, axis=-1)
      ave_r = np.mean(r)
      var_r = np.var(r)
      xrange = (0.0, 6.0)
      y, x_ed = np.histogram(r.reshape([-1,]), bins=101, density=True, range=xrange)
      y = y / (4. * np.pi)
      x = (x_ed[:-1] + x_ed[1:])/2. 

      if directory[i] == 'hydrogen/':
        gswf = gswf_h 
      elif directory[i] == 'yukawa/':
        gswf = gswf_yu 

      fig, ax = plt.subplots(figsize=[5.5,4.5], dpi=300)
      ax.scatter(x, y, s=10, marker='.', ls='', c=colors[0], zorder=1, label='VMC simulation')
      ax.plot(x, x**2*gswf(x)**2, ls='--', c=colors[1], zorder=0, label='analytical result')
      ax.set_xlabel('$r$ [$'+Lunits[i]+'$]')
      ax.set_ylabel('$r^2 \\rho (r)$ [$'+Lunits[i]+'^{-1}$]')

      fig.legend(loc='upper right', fontsize='x-small', bbox_to_anchor=(0.95, 0.925))
      fig.tight_layout()
      fig.savefig('density.png', transparent=True)

    if directory[i] in dir_h2:

      q = pos.reshape(pos.shape[:-1]+(-1, 3))
      x = q[..., 0].reshape([-1,])
      z = q[..., 2].reshape([-1,])

      hist, x_ed, z_ed = np.histogram2d(x, z, bins=50, range=[[-2.1,2.1],[-2.1,2.1]], density=True)
      hist = hist.T 

      fig, ax = plt.subplots(figsize=[5.5,4.5], dpi=300)
      im = ax.imshow(hist, origin='lower', extent=[x_ed[0], x_ed[-1], z_ed[0], z_ed[-1]], 
                     cmap='viridis', aspect='auto', interpolation='bilinear')
      ax.set_xlabel('$x$ [$'+Lunits[i]+'$]')
      ax.set_ylabel('$z$ [$'+Lunits[i]+'$]')

      fig.colorbar(im, label='$\\rho(x,z)$')
      fig.tight_layout()
      fig.savefig('density.png', transparent=True)
 
    os.chdir('..')

  print('All plots done successfully.')
