# Copyright 2024 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Functions used to load the ODAPI CenterNet checkpoint."""

from official.projects.centernet.modeling.layers import cn_nn_blocks
from official.projects.centernet.utils.checkpoints import config_classes
from official.projects.centernet.utils.checkpoints import config_data
from official.vision.modeling.backbones import mobilenet
from official.vision.modeling.layers import nn_blocks

Conv2DBNCFG = config_classes.Conv2DBNCFG
HeadConvCFG = config_classes.HeadConvCFG
ResidualBlockCFG = config_classes.ResidualBlockCFG
HourglassCFG = config_classes.HourglassCFG

BackboneConfigData = config_data.BackboneConfigData
HeadConfigData = config_data.HeadConfigData


def get_backbone_layer_cfgs(weights_dict, backbone_name):
  """Fetches the config classes for the backbone.

  This function generates a list of config classes corresponding to
  each building block in the backbone.

  Args:
    weights_dict: Dictionary that stores the backbone model weights.
    backbone_name: String, indicating the desired backbone configuration.

  Returns:
    A list containing the config classe of the backbone building block
  """

  print("Fetching backbone config classes for {}\n".format(backbone_name))
  cfgs = BackboneConfigData(weights_dict=weights_dict).get_cfg_list(
      backbone_name)
  return cfgs


def load_weights_backbone(backbone, weights_dict, backbone_name):
  """Loads the weights defined in the weights_dict into the backbone.

  This function loads the backbone weights by first fetching the necessary
  config classes for the backbone, then loads them in one by one for
  each layer that has weights associated with it.

  Args:
    backbone: keras.Model backbone.
    weights_dict: Dictionary that stores the backbone model weights.
    backbone_name: String, indicating the desired backbone configuration.

  Returns:
    Number of weights loaded in
  """
  print("Loading backbone weights\n")
  backbone_layers = backbone.layers
  cfgs = get_backbone_layer_cfgs(weights_dict, backbone_name)
  n_weights_total = 0

  cfg = cfgs.pop(0)
  for i in range(len(backbone_layers)):
    layer = backbone_layers[i]
    if isinstance(layer,
                  (mobilenet.Conv2DBNBlock,
                   cn_nn_blocks.HourglassBlock,
                   nn_blocks.ResidualBlock)):
      n_weights = cfg.load_weights(layer)
      print("Loading weights for: {}, weights loaded: {}".format(
          cfg, n_weights))
      n_weights_total += n_weights
      # pylint: disable=g-explicit-length-test
      if len(cfgs) == 0:
        print("{} Weights have been loaded for {} / {} layers\n".format(
            n_weights_total, i + 1, len(backbone_layers)))
        return n_weights_total
      cfg = cfgs.pop(0)
  return n_weights_total


def get_head_layer_cfgs(weights_dict, head_name):
  """Fetches the config classes for the head.

  This function generates a list of config classes corresponding to
  each building block in the head.

  Args:
    weights_dict: Dictionary that stores the decoder model weights.
    head_name: String, indicating the desired head configuration.

  Returns:
    A list containing the config classes of the backbone building block
  """
  print("Fetching head config classes for {}\n".format(head_name))

  cfgs = HeadConfigData(weights_dict=weights_dict).get_cfg_list(head_name)
  return cfgs


def load_weights_head(head, weights_dict, head_name):
  """Loads the weights defined in the weights_dict into the head.

  This function loads the head weights by first fetching the necessary
  config classes for the decoder, then loads them in one by one for
  each layer that has weights associated with it.

  Args:
    head: keras.Model head.
    weights_dict: Dictionary that stores the decoder model weights.
    head_name: String, indicating the desired head configuration.

  Returns:
    Number of weights loaded in
  """
  print("Loading head weights\n")
  head_layers = head.layers
  cfgs = get_head_layer_cfgs(weights_dict, head_name)
  n_weights_total = 0

  cfg = cfgs.pop(0)
  for i in range(len(head_layers)):
    layer = head_layers[i]
    if isinstance(layer, cn_nn_blocks.CenterNetHeadConv):
      n_weights = cfg.load_weights(layer)
      print("Loading weights for: {}, weights loaded: {}".format(
          cfg, n_weights))
      n_weights_total += n_weights
      # pylint: disable=g-explicit-length-test
      if len(cfgs) == 0:
        print("{} Weights have been loaded for {} / {} layers\n".format(
            n_weights_total, i + 1, len(head_layers)))
        return n_weights_total
      cfg = cfgs.pop(0)
  return n_weights_total


def load_weights_model(model, weights_dict, backbone_name, head_name):
  """Loads weights into the model.

  Args:
    model: keras.Model to load weights into.
    weights_dict: Dictionary that stores the weights of the model.
    backbone_name: String, indicating the desired backbone configuration.
    head_name: String, indicating the desired head configuration.

  Returns:

  """
  print("Loading model weights\n")
  n_weights = 0
  if backbone_name:
    n_weights += load_weights_backbone(
        model.backbone,
        weights_dict["model"]["_feature_extractor"]["_network"],
        backbone_name)
  if head_name:
    n_weights += load_weights_head(
        model.head,
        weights_dict["model"]["_prediction_head_dict"],
        head_name)
  print("Successfully loaded {} model weights.\n".format(n_weights))
  return model
