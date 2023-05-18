# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
# ==============================================================================
"""Functions to generate bidirectional feature pyramids based on image features.

Provides bidirectional feature pyramid network (BiFPN) generators that can be
used to build object detection feature extractors, as proposed by Tan et al.
See https://arxiv.org/abs/1911.09070 for more details.
"""
import collections
import functools
from six.moves import range
from six.moves import zip
import tensorflow as tf

from object_detection.utils import bifpn_utils


def _create_bifpn_input_config(fpn_min_level,
                               fpn_max_level,
                               input_max_level,
                               level_scales=None):
  """Creates a BiFPN input config for the input levels from a backbone network.

  Args:
    fpn_min_level: the minimum pyramid level (highest feature map resolution) to
      use in the BiFPN.
    fpn_max_level: the maximum pyramid level (lowest feature map resolution) to
      use in the BiFPN.
    input_max_level: the maximum pyramid level that will be provided as input to
      the BiFPN. Accordingly, the BiFPN will compute additional pyramid levels
      from input_max_level, up to the desired fpn_max_level.
    level_scales: a list of pyramid level scale factors. If 'None', each level's
      scale is set to 2^level by default, which corresponds to each successive
      feature map scaling by a factor of 2.

  Returns:
    A list of dictionaries for each feature map expected as input to the BiFPN,
    where each has entries for the feature map 'name' and 'scale'.
  """
  if not level_scales:
    level_scales = [2**i for i in range(fpn_min_level, fpn_max_level + 1)]

  bifpn_input_params = []
  for i in range(fpn_min_level, min(fpn_max_level, input_max_level) + 1):
    bifpn_input_params.append({
        'name': '0_up_lvl_{}'.format(i),
        'scale': level_scales[i - fpn_min_level]
    })

  return bifpn_input_params


def _get_bifpn_output_node_names(fpn_min_level, fpn_max_level, node_config):
  """Returns a list of BiFPN output node names, given a BiFPN node config.

  Args:
    fpn_min_level: the minimum pyramid level (highest feature map resolution)
      used by the BiFPN.
    fpn_max_level: the maximum pyramid level (lowest feature map resolution)
      used by the BiFPN.
    node_config: the BiFPN node_config, a list of dictionaries corresponding to
      each node in the BiFPN computation graph, where each entry should have an
      associated 'name'.

  Returns:
    A list of strings corresponding to the names of the output BiFPN nodes.
  """
  num_output_nodes = fpn_max_level - fpn_min_level + 1
  return [node['name'] for node in node_config[-num_output_nodes:]]


def _create_bifpn_node_config(bifpn_num_iterations,
                              bifpn_num_filters,
                              fpn_min_level,
                              fpn_max_level,
                              input_max_level,
                              bifpn_node_params=None,
                              level_scales=None,
                              use_native_resize_op=False):
  """Creates a config specifying a bidirectional feature pyramid network.

  Args:
    bifpn_num_iterations: the number of top-down bottom-up feature computations
      to repeat in the BiFPN.
    bifpn_num_filters: the number of filters (channels) for every feature map
      used in the BiFPN.
    fpn_min_level: the minimum pyramid level (highest feature map resolution) to
      use in the BiFPN.
    fpn_max_level: the maximum pyramid level (lowest feature map resolution) to
      use in the BiFPN.
    input_max_level: the maximum pyramid level that will be provided as input to
      the BiFPN. Accordingly, the BiFPN will compute additional pyramid levels
      from input_max_level, up to the desired fpn_max_level.
    bifpn_node_params: If not 'None', a dictionary of additional default BiFPN
      node parameters that will be applied to all BiFPN nodes.
    level_scales: a list of pyramid level scale factors. If 'None', each level's
      scale is set to 2^level by default, which corresponds to each successive
      feature map scaling by a factor of 2.
    use_native_resize_op: If true, will use
      tf.compat.v1.image.resize_nearest_neighbor for unsampling.

  Returns:
    A list of dictionaries used to define nodes in the BiFPN computation graph,
    as proposed by EfficientDet, Tan et al (https://arxiv.org/abs/1911.09070).
    Each node's entry has the corresponding keys:
      name: String. The name of this node in the BiFPN. The node name follows
        the format '{bifpn_iteration}_{dn|up}_lvl_{pyramid_level}', where 'dn'
        or 'up' refers to whether the node is in the top-down or bottom-up
        portion of a single BiFPN iteration.
      scale: the scale factor for this node, by default 2^level.
      inputs: A list of names of nodes which are inputs to this node.
      num_channels: The number of channels for this node.
      combine_method: String. Name of the method used to combine input
        node feature maps, 'fast_attention' by default for nodes which have more
        than one input. Otherwise, 'None' for nodes with only one input node.
      input_op: A (partial) function which is called to construct the layers
        that will be applied to this BiFPN node's inputs. This function is
        called with the arguments:
          input_op(name, input_scale, input_num_channels, output_scale,
                   output_num_channels, conv_hyperparams, is_training,
                   freeze_batchnorm)
      post_combine_op: A (partial) function which is called to construct the
        layers that will be applied to the result of the combine operation for
        this BiFPN node. This function will be called with the arguments:
          post_combine_op(name, conv_hyperparams, is_training, freeze_batchnorm)
        If 'None', then no layers will be applied after the combine operation
        for this node.
  """
  if not level_scales:
    level_scales = [2**i for i in range(fpn_min_level, fpn_max_level + 1)]

  default_node_params = {
      'num_channels':
          bifpn_num_filters,
      'combine_method':
          'fast_attention',
      'input_op':
          functools.partial(
              _create_bifpn_resample_block,
              downsample_method='max_pooling',
              use_native_resize_op=use_native_resize_op),
      'post_combine_op':
          functools.partial(
              bifpn_utils.create_conv_block,
              num_filters=bifpn_num_filters,
              kernel_size=3,
              strides=1,
              padding='SAME',
              use_separable=True,
              apply_batchnorm=True,
              apply_activation=True,
              conv_bn_act_pattern=False),
  }
  if bifpn_node_params:
    default_node_params.update(bifpn_node_params)

  bifpn_node_params = []
  # Create additional base pyramid levels not provided as input to the BiFPN.
  # Note, combine_method and post_combine_op are set to None for additional
  # base pyramid levels because they do not combine multiple input BiFPN nodes.
  for i in range(input_max_level + 1, fpn_max_level + 1):
    node_params = dict(default_node_params)
    node_params.update({
        'name': '0_up_lvl_{}'.format(i),
        'scale': level_scales[i - fpn_min_level],
        'inputs': ['0_up_lvl_{}'.format(i - 1)],
        'combine_method': None,
        'post_combine_op': None,
    })
    bifpn_node_params.append(node_params)

  for i in range(bifpn_num_iterations):
    # The first bottom-up feature pyramid (which includes the input pyramid
    # levels from the backbone network and the additional base pyramid levels)
    # is indexed at 0. So, the first top-down bottom-up pass of the BiFPN is
    # indexed from 1, and repeated for bifpn_num_iterations iterations.
    bifpn_i = i + 1

    # Create top-down nodes.
    for level_i in reversed(range(fpn_min_level, fpn_max_level)):
      inputs = []
      # BiFPN nodes in the top-down pass receive input from the corresponding
      # level from the previous BiFPN iteration's bottom-up pass, except for the
      # bottom-most (min) level node, which is computed once in the initial
      # bottom-up pass, and is afterwards only computed in each top-down pass.
      if level_i > fpn_min_level or bifpn_i == 1:
        inputs.append('{}_up_lvl_{}'.format(bifpn_i - 1, level_i))
      else:
        inputs.append('{}_dn_lvl_{}'.format(bifpn_i - 1, level_i))
      inputs.append(bifpn_node_params[-1]['name'])
      node_params = dict(default_node_params)
      node_params.update({
          'name': '{}_dn_lvl_{}'.format(bifpn_i, level_i),
          'scale': level_scales[level_i - fpn_min_level],
          'inputs': inputs
      })
      bifpn_node_params.append(node_params)

    # Create bottom-up nodes.
    for level_i in range(fpn_min_level + 1, fpn_max_level + 1):
      # BiFPN nodes in the bottom-up pass receive input from the corresponding
      # level from the preceding top-down pass, except for the top (max) level
      # which does not have a corresponding node in the top-down pass.
      inputs = ['{}_up_lvl_{}'.format(bifpn_i - 1, level_i)]
      if level_i < fpn_max_level:
        inputs.append('{}_dn_lvl_{}'.format(bifpn_i, level_i))
      inputs.append(bifpn_node_params[-1]['name'])
      node_params = dict(default_node_params)
      node_params.update({
          'name': '{}_up_lvl_{}'.format(bifpn_i, level_i),
          'scale': level_scales[level_i - fpn_min_level],
          'inputs': inputs
      })
      bifpn_node_params.append(node_params)

  return bifpn_node_params


def _create_bifpn_resample_block(name,
                                 input_scale,
                                 input_num_channels,
                                 output_scale,
                                 output_num_channels,
                                 conv_hyperparams,
                                 is_training,
                                 freeze_batchnorm,
                                 downsample_method=None,
                                 use_native_resize_op=False,
                                 maybe_apply_1x1_conv=True,
                                 apply_1x1_pre_sampling=True,
                                 apply_1x1_post_sampling=False):
  """Creates resample block layers for input feature maps to BiFPN nodes.

  Args:
    name: String. Name used for this block of layers.
    input_scale: Scale factor of the input feature map.
    input_num_channels: Number of channels in the input feature map.
    output_scale: Scale factor of the output feature map.
    output_num_channels: Number of channels in the output feature map.
    conv_hyperparams: A `hyperparams_builder.KerasLayerHyperparams` object
      containing hyperparameters for convolution ops.
    is_training: Indicates whether the feature generator is in training mode.
    freeze_batchnorm: Bool. Whether to freeze batch norm parameters during
      training or not. When training with a small batch size (e.g. 1), it is
      desirable to freeze batch norm update and use pretrained batch norm
      params.
    downsample_method: String. Method to use when downsampling feature maps.
    use_native_resize_op: Bool. Whether to use the native resize up when
      upsampling feature maps.
    maybe_apply_1x1_conv: Bool. If 'True', a 1x1 convolution will only be
      applied if the input_num_channels differs from the output_num_channels.
    apply_1x1_pre_sampling: Bool. Whether a 1x1 convolution will be applied to
      the input feature map before the up/down-sampling operation.
    apply_1x1_post_sampling: Bool. Whether a 1x1 convolution will be applied to
      the input feature map after the up/down-sampling operation.

  Returns:
    A list of layers which may be applied to the input feature maps in order to
    compute feature maps with the specified scale and number of channels.
  """
  # By default, 1x1 convolutions are only applied before sampling when the
  # number of input and output channels differ.
  if maybe_apply_1x1_conv and output_num_channels == input_num_channels:
    apply_1x1_pre_sampling = False
    apply_1x1_post_sampling = False

  apply_bn_for_resampling = True
  layers = []
  if apply_1x1_pre_sampling:
    layers.extend(
        bifpn_utils.create_conv_block(
            name=name + '1x1_pre_sample/',
            num_filters=output_num_channels,
            kernel_size=1,
            strides=1,
            padding='SAME',
            use_separable=False,
            apply_batchnorm=apply_bn_for_resampling,
            apply_activation=False,
            conv_hyperparams=conv_hyperparams,
            is_training=is_training,
            freeze_batchnorm=freeze_batchnorm))

  layers.extend(
      bifpn_utils.create_resample_feature_map_ops(input_scale, output_scale,
                                                  downsample_method,
                                                  use_native_resize_op,
                                                  conv_hyperparams, is_training,
                                                  freeze_batchnorm, name))

  if apply_1x1_post_sampling:
    layers.extend(
        bifpn_utils.create_conv_block(
            name=name + '1x1_post_sample/',
            num_filters=output_num_channels,
            kernel_size=1,
            strides=1,
            padding='SAME',
            use_separable=False,
            apply_batchnorm=apply_bn_for_resampling,
            apply_activation=False,
            conv_hyperparams=conv_hyperparams,
            is_training=is_training,
            freeze_batchnorm=freeze_batchnorm))

  return layers


def _create_bifpn_combine_op(num_inputs, name, combine_method):
  """Creates a BiFPN output config, a list of the output BiFPN node names.

  Args:
    num_inputs: The number of inputs to this combine operation.
    name: String. The name of this combine operation.
    combine_method: String. The method used to combine input feature maps.

  Returns:
    A function which may be called with a list of num_inputs feature maps
    and which will return a single feature map.
  """

  combine_op = None
  if num_inputs < 1:
    raise ValueError('Expected at least 1 input for BiFPN combine.')
  elif num_inputs == 1:
    combine_op = lambda x: x[0]
  else:
    combine_op = bifpn_utils.BiFPNCombineLayer(
        combine_method=combine_method, name=name)
  return combine_op


class KerasBiFpnFeatureMaps(tf.keras.Model):
  """Generates Keras based BiFPN feature maps from an input feature map pyramid.

  A Keras model that generates multi-scale feature maps for detection by
  iteratively computing top-down and bottom-up feature pyramids, as in the
  EfficientDet paper by Tan et al, see arxiv.org/abs/1911.09070 for details.
  """

  def __init__(self,
               bifpn_num_iterations,
               bifpn_num_filters,
               fpn_min_level,
               fpn_max_level,
               input_max_level,
               is_training,
               conv_hyperparams,
               freeze_batchnorm,
               bifpn_node_params=None,
               use_native_resize_op=False,
               name=None):
    """Constructor.

    Args:
      bifpn_num_iterations: The number of top-down bottom-up iterations.
      bifpn_num_filters: The number of filters (channels) to be used for all
        feature maps in this BiFPN.
      fpn_min_level: The minimum pyramid level (highest feature map resolution)
        to use in the BiFPN.
      fpn_max_level: The maximum pyramid level (lowest feature map resolution)
        to use in the BiFPN.
      input_max_level: The maximum pyramid level that will be provided as input
        to the BiFPN. Accordingly, the BiFPN will compute any additional pyramid
        levels from input_max_level up to the desired fpn_max_level, with each
        successivel level downsampling by a scale factor of 2 by default.
      is_training: Indicates whether the feature generator is in training mode.
      conv_hyperparams: A `hyperparams_builder.KerasLayerHyperparams` object
        containing hyperparameters for convolution ops.
      freeze_batchnorm: Bool. Whether to freeze batch norm parameters during
        training or not. When training with a small batch size (e.g. 1), it is
        desirable to freeze batch norm update and use pretrained batch norm
        params.
      bifpn_node_params: An optional dictionary that may be used to specify
        default parameters for BiFPN nodes, without the need to provide a custom
        bifpn_node_config. For example, if '{ combine_method: 'sum' }', then all
        BiFPN nodes will combine input feature maps by summation, rather than
        by the default fast attention method.
      use_native_resize_op: If True, will use
        tf.compat.v1.image.resize_nearest_neighbor for unsampling.
      name: A string name scope to assign to the model. If 'None', Keras
        will auto-generate one from the class name.
    """
    super(KerasBiFpnFeatureMaps, self).__init__(name=name)
    bifpn_node_config = _create_bifpn_node_config(
        bifpn_num_iterations,
        bifpn_num_filters,
        fpn_min_level,
        fpn_max_level,
        input_max_level,
        bifpn_node_params,
        use_native_resize_op=use_native_resize_op)
    bifpn_input_config = _create_bifpn_input_config(fpn_min_level,
                                                    fpn_max_level,
                                                    input_max_level)
    bifpn_output_node_names = _get_bifpn_output_node_names(
        fpn_min_level, fpn_max_level, bifpn_node_config)

    self.bifpn_node_config = bifpn_node_config
    self.bifpn_output_node_names = bifpn_output_node_names
    self.node_input_blocks = []
    self.node_combine_op = []
    self.node_post_combine_block = []

    all_node_params = bifpn_input_config
    all_node_names = [node['name'] for node in all_node_params]
    for node_config in bifpn_node_config:
      # Maybe transform and/or resample input feature maps.
      input_blocks = []
      for input_name in node_config['inputs']:
        if input_name not in all_node_names:
          raise ValueError(
              'Input feature map ({}) does not exist:'.format(input_name))
        input_index = all_node_names.index(input_name)
        input_params = all_node_params[input_index]
        input_block = node_config['input_op'](
            name='{}/input_{}/'.format(node_config['name'], input_name),
            input_scale=input_params['scale'],
            input_num_channels=input_params.get('num_channels', None),
            output_scale=node_config['scale'],
            output_num_channels=node_config['num_channels'],
            conv_hyperparams=conv_hyperparams,
            is_training=is_training,
            freeze_batchnorm=freeze_batchnorm)
        input_blocks.append((input_index, input_block))

      # Combine input feature maps.
      combine_op = _create_bifpn_combine_op(
          num_inputs=len(input_blocks),
          name=(node_config['name'] + '/combine'),
          combine_method=node_config['combine_method'])

      # Post-combine layers.
      post_combine_block = []
      if node_config['post_combine_op']:
        post_combine_block.extend(node_config['post_combine_op'](
            name=node_config['name'] + '/post_combine/',
            conv_hyperparams=conv_hyperparams,
            is_training=is_training,
            freeze_batchnorm=freeze_batchnorm))

      self.node_input_blocks.append(input_blocks)
      self.node_combine_op.append(combine_op)
      self.node_post_combine_block.append(post_combine_block)
      all_node_params.append(node_config)
      all_node_names.append(node_config['name'])

  def call(self, feature_pyramid):
    """Compute BiFPN feature maps from input feature pyramid.

    Executed when calling the `.__call__` method on input.

    Args:
      feature_pyramid: list of tuples of (tensor_name, image_feature_tensor).

    Returns:
      feature_maps: an OrderedDict mapping keys (feature map names) to
        tensors where each tensor has shape [batch, height_i, width_i, depth_i].
    """
    feature_maps = [el[1] for el in feature_pyramid]
    output_feature_maps = [None for node in self.bifpn_output_node_names]

    for index, node in enumerate(self.bifpn_node_config):
      node_scope = 'node_{:02d}'.format(index)
      with tf.name_scope(node_scope):
        # Apply layer blocks to this node's input feature maps.
        input_block_results = []
        for input_index, input_block in self.node_input_blocks[index]:
          block_result = feature_maps[input_index]
          for layer in input_block:
            block_result = layer(block_result)
          input_block_results.append(block_result)

        # Combine the resulting feature maps.
        node_result = self.node_combine_op[index](input_block_results)

        # Apply post-combine layer block if applicable.
        for layer in self.node_post_combine_block[index]:
          node_result = layer(node_result)

        feature_maps.append(node_result)

        if node['name'] in self.bifpn_output_node_names:
          index = self.bifpn_output_node_names.index(node['name'])
          output_feature_maps[index] = node_result

    return collections.OrderedDict(
        zip(self.bifpn_output_node_names, output_feature_maps))
