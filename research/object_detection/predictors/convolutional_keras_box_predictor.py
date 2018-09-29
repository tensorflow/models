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

"""Convolutional Box Predictors with and without weight sharing."""
import collections

import tensorflow as tf

from object_detection.core import box_predictor
from object_detection.utils import static_shape

keras = tf.keras.layers

BOX_ENCODINGS = box_predictor.BOX_ENCODINGS
CLASS_PREDICTIONS_WITH_BACKGROUND = (
    box_predictor.CLASS_PREDICTIONS_WITH_BACKGROUND)
MASK_PREDICTIONS = box_predictor.MASK_PREDICTIONS


class _NoopVariableScope(object):
  """A dummy class that does not push any scope."""

  def __enter__(self):
    return None

  def __exit__(self, exc_type, exc_value, traceback):
    return False


class ConvolutionalBoxPredictor(box_predictor.KerasBoxPredictor):
  """Convolutional Keras Box Predictor.

  Optionally add an intermediate 1x1 convolutional layer after features and
  predict in parallel branches box_encodings and
  class_predictions_with_background.

  Currently this box predictor assumes that predictions are "shared" across
  classes --- that is each anchor makes box predictions which do not depend
  on class.
  """

  def __init__(self,
               is_training,
               num_classes,
               box_prediction_heads,
               class_prediction_heads,
               other_heads,
               conv_hyperparams,
               num_layers_before_predictor,
               min_depth,
               max_depth,
               freeze_batchnorm,
               inplace_batchnorm_update,
               name=None):
    """Constructor.

    Args:
      is_training: Indicates whether the BoxPredictor is in training mode.
      num_classes: number of classes.  Note that num_classes *does not*
        include the background category, so if groundtruth labels take values
        in {0, 1, .., K-1}, num_classes=K (and not K+1, even though the
        assigned classification targets can range from {0,... K}).
      box_prediction_heads: A list of heads that predict the boxes.
      class_prediction_heads: A list of heads that predict the classes.
      other_heads: A dictionary mapping head names to lists of convolutional
        heads.
      conv_hyperparams: A `hyperparams_builder.KerasLayerHyperparams` object
        containing hyperparameters for convolution ops.
      num_layers_before_predictor: Number of the additional conv layers before
        the predictor.
      min_depth: Minimum feature depth prior to predicting box encodings
        and class predictions.
      max_depth: Maximum feature depth prior to predicting box encodings
        and class predictions. If max_depth is set to 0, no additional
        feature map will be inserted before location and class predictions.
      freeze_batchnorm: Whether to freeze batch norm parameters during
        training or not. When training with a small batch size (e.g. 1), it is
        desirable to freeze batch norm update and use pretrained batch norm
        params.
      inplace_batchnorm_update: Whether to update batch norm moving average
        values inplace. When this is false train op must add a control
        dependency on tf.graphkeys.UPDATE_OPS collection in order to update
        batch norm statistics.
      name: A string name scope to assign to the model. If `None`, Keras
        will auto-generate one from the class name.

    Raises:
      ValueError: if min_depth > max_depth.
    """
    super(ConvolutionalBoxPredictor, self).__init__(
        is_training, num_classes, freeze_batchnorm=freeze_batchnorm,
        inplace_batchnorm_update=inplace_batchnorm_update,
        name=name)
    if min_depth > max_depth:
      raise ValueError('min_depth should be less than or equal to max_depth')
    if len(box_prediction_heads) != len(class_prediction_heads):
      raise ValueError('All lists of heads must be the same length.')
    for other_head_list in other_heads.values():
      if len(box_prediction_heads) != len(other_head_list):
        raise ValueError('All lists of heads must be the same length.')

    self._prediction_heads = {
        BOX_ENCODINGS: box_prediction_heads,
        CLASS_PREDICTIONS_WITH_BACKGROUND: class_prediction_heads,
    }

    if other_heads:
      self._prediction_heads.update(other_heads)

    self._conv_hyperparams = conv_hyperparams
    self._min_depth = min_depth
    self._max_depth = max_depth
    self._num_layers_before_predictor = num_layers_before_predictor

    self._shared_nets = []

  def build(self, input_shapes):
    """Creates the variables of the layer."""
    if len(input_shapes) != len(self._prediction_heads[BOX_ENCODINGS]):
      raise ValueError('This box predictor was constructed with %d heads,'
                       'but there are %d inputs.' %
                       (len(self._prediction_heads[BOX_ENCODINGS]),
                        len(input_shapes)))
    for stack_index, input_shape in enumerate(input_shapes):
      net = tf.keras.Sequential(name='PreHeadConvolutions_%d' % stack_index)
      self._shared_nets.append(net)

      # Add additional conv layers before the class predictor.
      features_depth = static_shape.get_depth(input_shape)
      depth = max(min(features_depth, self._max_depth), self._min_depth)
      tf.logging.info(
          'depth of additional conv before box predictor: {}'.format(depth))
      if depth > 0 and self._num_layers_before_predictor > 0:
        for i in range(self._num_layers_before_predictor):
          net.add(keras.Conv2D(depth, [1, 1],
                               name='Conv2d_%d_1x1_%d' % (i, depth),
                               padding='SAME',
                               **self._conv_hyperparams.params()))
          net.add(self._conv_hyperparams.build_batch_norm(
              training=(self._is_training and not self._freeze_batchnorm),
              name='Conv2d_%d_1x1_%d_norm' % (i, depth)))
          net.add(self._conv_hyperparams.build_activation_layer(
              name='Conv2d_%d_1x1_%d_activation' % (i, depth),
          ))
    self.built = True

  def _predict(self, image_features):
    """Computes encoded object locations and corresponding confidences.

    Args:
      image_features: A list of float tensors of shape [batch_size, height_i,
        width_i, channels_i] containing features for a batch of images.

    Returns:
      box_encodings: A list of float tensors of shape
        [batch_size, num_anchors_i, q, code_size] representing the location of
        the objects, where q is 1 or the number of classes. Each entry in the
        list corresponds to a feature map in the input `image_features` list.
      class_predictions_with_background: A list of float tensors of shape
        [batch_size, num_anchors_i, num_classes + 1] representing the class
        predictions for the proposals. Each entry in the list corresponds to a
        feature map in the input `image_features` list.
    """
    predictions = collections.defaultdict(list)

    for (index, image_feature) in enumerate(image_features):

      # Apply shared conv layers before the head predictors.
      net = self._shared_nets[index](image_feature)

      for head_name in self._prediction_heads:
        head_obj = self._prediction_heads[head_name][index]
        prediction = head_obj(net)
        predictions[head_name].append(prediction)

    return predictions
