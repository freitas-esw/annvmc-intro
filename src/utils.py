
import os
import jax
import kfac_jax
import numpy as np

from absl import logging
from functools import partial

replicate = kfac_jax.utils.replicate_all_local_devices

broadcast = kfac_jax.utils.broadcast_all_local_devices

shard_key = kfac_jax.utils.make_different_rng_key_on_all_devices

p_split = kfac_jax.utils.p_split

PMAP_AXIS_NAME = 'qmc_pmap_axis'

pmean = partial(kfac_jax.utils.pmean_if_pmap, axis_name=PMAP_AXIS_NAME)

pmap = partial(jax.pmap, axis_name=PMAP_AXIS_NAME)

def create_directory(path: str):
  """
  """
  dir_path = os.path.join(os.getcwd(), path)
  if not os.path.isdir(dir_path):                                                                                      
    os.makedirs(dir_path)     
  return dir_path

def create_file(path: str, name: str, scheme: list = []):
  """
  """
  dir_path = create_directory(path)
  filename = os.path.join(dir_path, name)
  i = 1
  while os.path.isfile(filename+f'-{i:02d}.csv') and i < 99: i+=1
  if i > 99: raise RuntimeError("Directory has too much files.")
  file = open(filename+f'-{i:02d}.csv', 'w', encoding='UTF-8')
  file.write(','.join(scheme) + '\n')
  return file

def save_ckpt(path: str, name: str, **data):
  """
  """
  dir_path = create_directory(path)
  filename = os.path.join(dir_path, name) + '.npz'
  logging.info('Saving checkpoint %s', filename)
  if os.path.isfile(filename):
    raise RuntimeError("File already exits.")
  file = open(filename, 'wb')
  np.savez(file, **data)
  return 

