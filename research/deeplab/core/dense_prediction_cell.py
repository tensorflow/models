# Copyright 2018 The TensorFlow Authors All Rights Reserved.
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

"""Dense Prediction Cell class that can be evolved in semantic segmentation.

DensePredictionCell is used as a `layer` in semantic segmentation whose
architecture is determined by the `config`, a dictionary specifying
the architecture.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from deeplab.core import utils

slim = tf.contrib.slim

# Local constants.
_META_ARCHITECTURE_SCOPE = 'meta_architecture'
_CONCAT_PROJECTION_SCOPE = 'concat_projection'
_OP = 'op'
_CONV = 'conv'
_PYRAMID_POOLING = 'pyramid_pooling'
_KERNEL = 'kernel'
_RATE = 'rate'
_GRID_SIZE = 'grid_size'
_TARGET_SIZE = 'target_size'
_INPUT = 'input'


def dense_prediction_cell_hparams():
  """DensePredictionCell HParams.

  Returns:
    A dictionary of hyper-parameters used for dense prediction cell with keys:
      - reduction_size: Integer, the number of output filters for each operation
          inside the cell.
      - dropout_on_concat_features: Boolean, apply dropout on the concatenated
          features or not.
      - dropout_on_projection_features: Boolean, apply dropout on the projection
          features or not.
      - dropout_keep_prob: Float, when `dropout_on_concat_features' or
          `dropout_on_projection_features' is True, the `keep_prob` value used
          in the dropout operation.
      - concat_channels: Integer, the concatenated features will be
          channel-reduced to `concat_channels` channels.
      - conv_rate_multiplier: Integer, used to multiply the convolution rates.
          This is useful in the case when the output_stride is changed from 16
          to 8, we need to double the convolution rates correspondingly.
  """
  return {
    'reduction_size': 256,
    'dropout_on_concat_features': True,
    'dropout_on_projection_features': False,
    'dropout_keep_prob': 0.9,
    'concat_channels': 256,
    'conv_rate_multiplier': 1,
  }


class DensePredictionCell(object):
  """DensePredictionCell class used as a 'layer' in semantic segmentation."""

  def __init__(self, config, hparams=None):
    """Initializes the dense prediction cell.

    Args:
      config: A dictionary storing the architecture of a dense prediction cell.
      hparams: A dictionary of hyper-parameters, provided by users. This
        dictionary will be used to update the default dictionary returned by
        dense_prediction_cell_hparams().

    Raises:
       ValueError: If `conv_rate_multiplier` has value < 1.
    """
    self.hparams = dense_prediction_cell_hparams()
    if hparams is not None:
      self.hparams.update(hparams)
    self.config = config

    # Check values in hparams are valid or not.
    if self.hparams['conv_rate_multiplier'] < 1:
      raise ValueError('conv_rate_multiplier cannot have value < 1.')

  def _get_pyramid_pooling_arguments(
      self, crop_size, output_stride, image_grid, image_pooling_crop_size=None):
    """Gets arguments for pyramid pooling.

    Args:
      crop_size: A list of two integers, [crop_height, crop_width] specifying
        whole patch crop size.
      output_stride: Integer, output stride value for extracted features.
      image_grid: A list of two integers, [image_grid_height, image_grid_width],
        specifying the grid size of how the pyramid pooling will be performed.
      image_pooling_crop_size: A list of two integers, [crop_height, crop_width]
        specifying the crop size for image pooling operations. Note that we
        decouple whole patch crop_size and image_pooling_crop_size as one could
        perform the image_pooling with different crop sizes.

    Returns:
      A list of (resize_value, pooled_kernel)
    """
    resize_height = utils.scale_dimension(crop_size[0], 1. / output_stride)
    resize_width = utils.scale_dimension(crop_size[1], 1. / output_stride)
    # If image_pooling_crop_size is not specified, use crop_size.
    if image_pooling_crop_size is None:
      image_pooling_crop_size = crop_size
    pooled_height = utils.scale_dimension(
        image_pooling_crop_size[0], 1. / (output_stride * image_grid[0]))
    pooled_width = utils.scale_dimension(
        image_pooling_crop_size[1], 1. / (output_stride * image_grid[1]))
    return ([resize_height, resize_width], [pooled_height, pooled_width])

  def _parse_operation(self, config, crop_size, output_stride,
      image_pooling_crop_size=None):
    """Parses one operation.

    When 'operation' is 'pyramid_pooling', we compute the required
    hyper-parameters and save in config.

    Args:
      config: A dictionary storing required hyper-parameters for one
        operation.
      crop_size: A list of two integers, [crop_height, crop_width] specifying
        whole patch crop size.
      output_stride: Integer, output stride value for extracted features.
      image_pooling_crop_size: A list of two integers, [crop_height, crop_width]
        specifying the crop size for image pooling operations. Note that we
        decouple whole patch crop_size and image_pooling_crop_size as one could
        perform the image_pooling with different crop sizes.

    Returns:
      A dictionary stores the related information for the operation.
    """
    if config[_OP] == _PYRAMID_POOLING:
      (config[_TARGET_SIZE],
       config[_KERNEL]) = self._get_pyramid_pooling_arguments(
          crop_size=crop_size,
          output_stride=output_stride,
          image_grid=config[_GRID_SIZE],
          image_pooling_crop_size=image_pooling_crop_size)

    return config

  def build_cell(self,
      features,
      output_stride=16,
      crop_size=None,
      image_pooling_crop_size=None,
      weight_decay=0.00004,
      reuse=None,
      is_training=False,
      fine_tune_batch_norm=False,
      scope=None):
    """Builds the dense prediction cell based on the config.

    Args:
      features: Input feature map of size [batch, height, width, channels].
      output_stride: Int, output stride at which the features were extracted.
      crop_size: A list [crop_height, crop_width], determining the input
        features resolution.
      image_pooling_crop_size: A list of two integers, [crop_height, crop_width]
        specifying the crop size for image pooling operations. Note that we
        decouple whole patch crop_size and image_pooling_crop_size as one could
        perform the image_pooling with different crop sizes.
      weight_decay: Float, the weight decay for model variables.
      reuse: Reuse the model variables or not.
      is_training: Boolean, is training or not.
      fine_tune_batch_norm: Boolean, fine-tuning batch norm parameters or not.
      scope: Optional string, specifying the variable scope.

    Returns:
      Features after passing through the constructed dense prediction cell with
        shape = [batch, height, width, channels] where channels are determined
        by `reduction_size` returned by dense_prediction_cell_hparams().

    Raises:
      ValueError: Use Convolution with kernel size not equal to 1x1 or 3x3 or
        the operation is not recognized.
    """
    batch_norm_params = {
      'is_training': is_training and fine_tune_batch_norm,
      'decay': 0.9997,
      'epsilon': 1e-5,
      'scale': True,
    }
    hparams = self.hparams
    with slim.arg_scope(
        [slim.conv2d, slim.separable_conv2d],
        weights_regularizer=slim.l2_regularizer(weight_decay),
        activation_fn=tf.nn.relu,
        normalizer_fn=slim.batch_norm,
        padding='SAME',
        stride=1,
        reuse=reuse):
      with slim.arg_scope([slim.batch_norm], **batch_norm_params):
        with tf.variable_scope(scope, _META_ARCHITECTURE_SCOPE, [features]):
          depth = hparams['reduction_size']
          branch_logits = []
          for i, current_config in enumerate(self.config):
            scope = 'branch%d' % i
            current_config = self._parse_operation(
                config=current_config,
                crop_size=crop_size,
                output_stride=output_stride,
                image_pooling_crop_size=image_pooling_crop_size)
            tf.logging.info(current_config)
            if current_config[_INPUT] < 0:
              operation_input = features
            else:
              operation_input = branch_logits[current_config[_INPUT]]
            if current_config[_OP] == _CONV:
              if current_config[_KERNEL] == [1, 1] or current_config[
                _KERNEL] == 1:
                branch_logits.append(
                    slim.conv2d(operation_input, depth, 1, scope=scope))
              else:
                conv_rate = [r * hparams['conv_rate_multiplier']
                             for r in current_config[_RATE]]
                branch_logits.append(
                    utils.split_separable_conv2d(
                        operation_input,
                        filters=depth,
                        kernel_size=current_config[_KERNEL],
                        rate=conv_rate,
                        weight_decay=weight_decay,
                        scope=scope))
            elif current_config[_OP] == _PYRAMID_POOLING:
              pooled_features = slim.avg_pool2d(
                  operation_input,
                  kernel_size=current_config[_KERNEL],
                  stride=[1, 1],
                  padding='VALID')
              pooled_features = slim.conv2d(
                  pooled_features,
                  depth,
                  1,
                  scope=scope)
              pooled_features = tf.image.resize_bilinear(
                  pooled_features,
                  current_config[_TARGET_SIZE],
                  align_corners=True)
              # Set shape for resize_height/resize_width if they are not Tensor.
              resize_height = current_config[_TARGET_SIZE][0]
              resize_width = current_config[_TARGET_SIZE][1]
              if isinstance(resize_height, tf.Tensor):
                resize_height = None
              if isinstance(resize_width, tf.Tensor):
                resize_width = None
              pooled_features.set_shape(
                  [None, resize_height, resize_width, depth])
              branch_logits.append(pooled_features)
            else:
              raise ValueError('Unrecognized operation.')
          # Merge branch logits.
          concat_logits = tf.concat(branch_logits, 3)
          if self.hparams['dropout_on_concat_features']:
            concat_logits = slim.dropout(
                concat_logits,
                keep_prob=self.hparams['dropout_keep_prob'],
                is_training=is_training,
                scope=_CONCAT_PROJECTION_SCOPE + '_dropout')
          concat_logits = slim.conv2d(concat_logits,
                                      self.hparams['concat_channels'],
                                      1,
                                      scope=_CONCAT_PROJECTION_SCOPE)
          if self.hparams['dropout_on_projection_features']:
            concat_logits = slim.dropout(
                concat_logits,
                keep_prob=self.hparams['dropout_keep_prob'],
                is_training=is_training,
                scope=_CONCAT_PROJECTION_SCOPE + '_dropout')
          return concat_logits