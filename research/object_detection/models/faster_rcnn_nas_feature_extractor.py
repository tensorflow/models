# Lint as: python2, python3
# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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

"""NASNet Faster R-CNN implementation.

Learning Transferable Architectures for Scalable Image Recognition
Barret Zoph, Vijay Vasudevan, Jonathon Shlens, Quoc V. Le
https://arxiv.org/abs/1707.07012
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from six.moves import range
import tensorflow.compat.v1 as tf
import tf_slim as slim

from object_detection.meta_architectures import faster_rcnn_meta_arch
from object_detection.utils import variables_helper

# pylint: disable=g-import-not-at-top
try:
  from nets.nasnet import nasnet
  from nets.nasnet import nasnet_utils
except:  # pylint: disable=bare-except
  pass
# pylint: enable=g-import-not-at-top

arg_scope = slim.arg_scope


def nasnet_large_arg_scope_for_detection(is_batch_norm_training=False):
  """Defines the default arg scope for the NASNet-A Large for object detection.

  This provides a small edit to switch batch norm training on and off.

  Args:
    is_batch_norm_training: Boolean indicating whether to train with batch norm.

  Returns:
    An `arg_scope` to use for the NASNet Large Model.
  """
  imagenet_scope = nasnet.nasnet_large_arg_scope()
  with arg_scope(imagenet_scope):
    with arg_scope([slim.batch_norm], is_training=is_batch_norm_training) as sc:
      return sc


# Note: This is largely a copy of _build_nasnet_base inside nasnet.py but
# with special edits to remove instantiation of the stem and the special
# ability to receive as input a pair of hidden states.
def _build_nasnet_base(hidden_previous,
                       hidden,
                       normal_cell,
                       reduction_cell,
                       hparams,
                       true_cell_num,
                       start_cell_num):
  """Constructs a NASNet image model."""

  # Find where to place the reduction cells or stride normal cells
  reduction_indices = nasnet_utils.calc_reduction_layers(
      hparams.num_cells, hparams.num_reduction_layers)

  # Note: The None is prepended to match the behavior of _imagenet_stem()
  cell_outputs = [None, hidden_previous, hidden]
  net = hidden

  # NOTE: In the nasnet.py code, filter_scaling starts at 1.0. We instead
  # start at 2.0 because 1 reduction cell has been created which would
  # update the filter_scaling to 2.0.
  filter_scaling = 2.0

  # Run the cells
  for cell_num in range(start_cell_num, hparams.num_cells):
    stride = 1
    if hparams.skip_reduction_layer_input:
      prev_layer = cell_outputs[-2]
    if cell_num in reduction_indices:
      filter_scaling *= hparams.filter_scaling_rate
      net = reduction_cell(
          net,
          scope='reduction_cell_{}'.format(reduction_indices.index(cell_num)),
          filter_scaling=filter_scaling,
          stride=2,
          prev_layer=cell_outputs[-2],
          cell_num=true_cell_num)
      true_cell_num += 1
      cell_outputs.append(net)
    if not hparams.skip_reduction_layer_input:
      prev_layer = cell_outputs[-2]
    net = normal_cell(
        net,
        scope='cell_{}'.format(cell_num),
        filter_scaling=filter_scaling,
        stride=stride,
        prev_layer=prev_layer,
        cell_num=true_cell_num)
    true_cell_num += 1
    cell_outputs.append(net)

  # Final nonlinearity.
  # Note that we have dropped the final pooling, dropout and softmax layers
  # from the default nasnet version.
  with tf.variable_scope('final_layer'):
    net = tf.nn.relu(net)
  return net


# TODO(shlens): Only fixed_shape_resizer is currently supported for NASNet
# featurization. The reason for this is that nasnet.py only supports
# inputs with fully known shapes. We need to update nasnet.py to handle
# shapes not known at compile time.
class FasterRCNNNASFeatureExtractor(
    faster_rcnn_meta_arch.FasterRCNNFeatureExtractor):
  """Faster R-CNN with NASNet-A feature extractor implementation."""

  def __init__(self,
               is_training,
               first_stage_features_stride,
               batch_norm_trainable=False,
               reuse_weights=None,
               weight_decay=0.0):
    """Constructor.

    Args:
      is_training: See base class.
      first_stage_features_stride: See base class.
      batch_norm_trainable: See base class.
      reuse_weights: See base class.
      weight_decay: See base class.

    Raises:
      ValueError: If `first_stage_features_stride` is not 16.
    """
    if first_stage_features_stride != 16:
      raise ValueError('`first_stage_features_stride` must be 16.')
    super(FasterRCNNNASFeatureExtractor, self).__init__(
        is_training, first_stage_features_stride, batch_norm_trainable,
        reuse_weights, weight_decay)

  def preprocess(self, resized_inputs):
    """Faster R-CNN with NAS preprocessing.

    Maps pixel values to the range [-1, 1].

    Args:
      resized_inputs: A [batch, height_in, width_in, channels] float32 tensor
        representing a batch of images with values between 0 and 255.0.

    Returns:
      preprocessed_inputs: A [batch, height_out, width_out, channels] float32
        tensor representing a batch of images.

    """
    return (2.0 / 255.0) * resized_inputs - 1.0

  def _extract_proposal_features(self, preprocessed_inputs, scope):
    """Extracts first stage RPN features.

    Extracts features using the first half of the NASNet network.
    We construct the network in `align_feature_maps=True` mode, which means
    that all VALID paddings in the network are changed to SAME padding so that
    the feature maps are aligned.

    Args:
      preprocessed_inputs: A [batch, height, width, channels] float32 tensor
        representing a batch of images.
      scope: A scope name.

    Returns:
      rpn_feature_map: A tensor with shape [batch, height, width, depth]
      end_points: A dictionary mapping feature extractor tensor names to tensors

    Raises:
      ValueError: If the created network is missing the required activation.
    """
    del scope

    if len(preprocessed_inputs.get_shape().as_list()) != 4:
      raise ValueError('`preprocessed_inputs` must be 4 dimensional, got a '
                       'tensor of shape %s' % preprocessed_inputs.get_shape())

    with slim.arg_scope(nasnet_large_arg_scope_for_detection(
        is_batch_norm_training=self._train_batch_norm)):
      with arg_scope([slim.conv2d,
                      slim.batch_norm,
                      slim.separable_conv2d],
                     reuse=self._reuse_weights):
        _, end_points = nasnet.build_nasnet_large(
            preprocessed_inputs, num_classes=None,
            is_training=self._is_training,
            final_endpoint='Cell_11')

    # Note that both 'Cell_10' and 'Cell_11' have equal depth = 2016.
    rpn_feature_map = tf.concat([end_points['Cell_10'],
                                 end_points['Cell_11']], 3)

    # nasnet.py does not maintain the batch size in the first dimension.
    # This work around permits us retaining the batch for below.
    batch = preprocessed_inputs.get_shape().as_list()[0]
    shape_without_batch = rpn_feature_map.get_shape().as_list()[1:]
    rpn_feature_map_shape = [batch] + shape_without_batch
    rpn_feature_map.set_shape(rpn_feature_map_shape)

    return rpn_feature_map, end_points

  def _extract_box_classifier_features(self, proposal_feature_maps, scope):
    """Extracts second stage box classifier features.

    This function reconstructs the "second half" of the NASNet-A
    network after the part defined in `_extract_proposal_features`.

    Args:
      proposal_feature_maps: A 4-D float tensor with shape
        [batch_size * self.max_num_proposals, crop_height, crop_width, depth]
        representing the feature map cropped to each proposal.
      scope: A scope name.

    Returns:
      proposal_classifier_features: A 4-D float tensor with shape
        [batch_size * self.max_num_proposals, height, width, depth]
        representing box classifier features for each proposal.
    """
    del scope

    # Note that we always feed into 2 layers of equal depth
    # where the first N channels corresponds to previous hidden layer
    # and the second N channels correspond to the final hidden layer.
    hidden_previous, hidden = tf.split(proposal_feature_maps, 2, axis=3)

    # Note that what follows is largely a copy of build_nasnet_large() within
    # nasnet.py. We are copying to minimize code pollution in slim.

    # TODO(shlens,skornblith): Determine the appropriate drop path schedule.
    # For now the schedule is the default (1.0->0.7 over 250,000 train steps).
    hparams = nasnet.large_imagenet_config()
    if not self._is_training:
      hparams.set_hparam('drop_path_keep_prob', 1.0)

    # Calculate the total number of cells in the network
    # -- Add 2 for the reduction cells.
    total_num_cells = hparams.num_cells + 2
    # -- And add 2 for the stem cells for ImageNet training.
    total_num_cells += 2

    normal_cell = nasnet_utils.NasNetANormalCell(
        hparams.num_conv_filters, hparams.drop_path_keep_prob,
        total_num_cells, hparams.total_training_steps)
    reduction_cell = nasnet_utils.NasNetAReductionCell(
        hparams.num_conv_filters, hparams.drop_path_keep_prob,
        total_num_cells, hparams.total_training_steps)
    with arg_scope([slim.dropout, nasnet_utils.drop_path],
                   is_training=self._is_training):
      with arg_scope([slim.batch_norm], is_training=self._train_batch_norm):
        with arg_scope([slim.avg_pool2d,
                        slim.max_pool2d,
                        slim.conv2d,
                        slim.batch_norm,
                        slim.separable_conv2d,
                        nasnet_utils.factorized_reduction,
                        nasnet_utils.global_avg_pool,
                        nasnet_utils.get_channel_index,
                        nasnet_utils.get_channel_dim],
                       data_format=hparams.data_format):

          # This corresponds to the cell number just past 'Cell_11' used by
          # by _extract_proposal_features().
          start_cell_num = 12
          # Note that this number equals:
          #  start_cell_num + 2 stem cells + 1 reduction cell
          true_cell_num = 15

          with slim.arg_scope(nasnet.nasnet_large_arg_scope()):
            net = _build_nasnet_base(hidden_previous,
                                     hidden,
                                     normal_cell=normal_cell,
                                     reduction_cell=reduction_cell,
                                     hparams=hparams,
                                     true_cell_num=true_cell_num,
                                     start_cell_num=start_cell_num)

    proposal_classifier_features = net
    return proposal_classifier_features

  def restore_from_classification_checkpoint_fn(
      self,
      first_stage_feature_extractor_scope,
      second_stage_feature_extractor_scope):
    """Returns a map of variables to load from a foreign checkpoint.

    Note that this overrides the default implementation in
    faster_rcnn_meta_arch.FasterRCNNFeatureExtractor which does not work for
    NASNet-A checkpoints.

    Args:
      first_stage_feature_extractor_scope: A scope name for the first stage
        feature extractor.
      second_stage_feature_extractor_scope: A scope name for the second stage
        feature extractor.

    Returns:
      A dict mapping variable names (to load from a checkpoint) to variables in
      the model graph.
    """
    # Note that the NAS checkpoint only contains the moving average version of
    # the Variables so we need to generate an appropriate dictionary mapping.
    variables_to_restore = {}
    for variable in variables_helper.get_global_variables_safely():
      if variable.op.name.startswith(
          first_stage_feature_extractor_scope):
        var_name = variable.op.name.replace(
            first_stage_feature_extractor_scope + '/', '')
        var_name += '/ExponentialMovingAverage'
        variables_to_restore[var_name] = variable
      if variable.op.name.startswith(
          second_stage_feature_extractor_scope):
        var_name = variable.op.name.replace(
            second_stage_feature_extractor_scope + '/', '')
        var_name += '/ExponentialMovingAverage'
        variables_to_restore[var_name] = variable
    return variables_to_restore
