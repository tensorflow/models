"""
This file contains functions used to load the Pytorch Mesh R-CNN checkpoint
weights into the Tensorflow model.
"""

import numpy as np
import tensorflow as tf
from torch import load
from official.vision.beta.projects.mesh_rcnn.modeling.heads.z_head import \
    ZHead
# from z_head import ZHead
from official.vision.beta.projects.mesh_rcnn.utils.weight_utils.config_data import \
    ZHeadConfigData
# from config_data import ZHeadConfigData


def pth_to_dict(pth_path):
  """ Converts a TF checkpoint into a nested dictionary of weights.
  Args:
  pth_path: String, indicating filepath of the Pytorch checkpoint
  Returns:
  Dictionary where the checkpoint weights are stored
  Number of weights read
  """
  print("\nConverting model checkpoint from {} to weights dictionary\n".format(
      pth_path))
  
  pth_dict = load(pth_path, map_location='cpu') # may need to delete cpu arg
  weights_dict = {}
  n_read = 0

  for key in pth_dict["model"]:
    keys = key.split('.')
    curr_dict = weights_dict
    for newKey in keys[:-1]:
      if newKey not in curr_dict:
        curr_dict[newKey] = {}
      curr_dict = curr_dict[newKey]

    np_tensor = pth_dict["model"][key].cpu().numpy()
    # transpose from Pytorch to use in TF
    if len(np_tensor.shape) == 4:
      np_tensor = np.transpose(np_tensor, [2, 3, 1, 0])

    if len(np_tensor.shape) == 2:
      np_tensor = np.transpose(np_tensor, [1, 0])

    tf_tensor = tf.convert_to_tensor(np_tensor)
    curr_dict[keys[-1]] = tf_tensor

    n_read += np.prod(np_tensor.shape)

  print("Successfully read {} checkpoint weights\n".format(n_read))
  return weights_dict, n_read


def get_zhead_layer_cfgs(weights_dict, zhead_name):
  """ Fetches the config classes for the mesh head.
  This function generates a list of config classes corresponding to
  each building block in the mesh head.
  Args:
    weights_dict: Dictionary that stores the backbone model weights.
    backbone_name: String, indicating the desired mesh head configuration.
  Returns:
    A list containing the config classes of the mesh head building block.
  """

  print("Fetching z head config classes for {}\n".format(zhead_name))
  cfgs = ZHeadConfigData(weights_dict).get_cfg_list(zhead_name)
  return cfgs


def load_weights_zhead(zhead, weights_dict, zhead_name):
  """ Loads the weights defined in the weights_dict into the backbone.
  This function loads the backbone weights by first fetching the necesary
  config classes for the backbone, then loads them in one by one for
  each layer that has weights associated with it.
  Args:
    zhead: keras.Model
    weights_dict: Dictionary that stores the zhead model weights.
    zhead_name: String, indicating the desired zhead configuration.
  Returns:
    Number of weights loaded in.
  """
  print("Loading z head weights\n")
  cfgs = get_zhead_layer_cfgs(weights_dict, zhead_name)
  cfgs.load_weights(zhead)
  return