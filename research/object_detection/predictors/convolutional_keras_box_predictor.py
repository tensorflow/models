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
from object_detection.utils import shape_utils
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

    # We generate a consistent ordering for the prediction head names,
    # So that all workers build the model in the exact same order
    self._sorted_head_names = sorted(self._prediction_heads.keys())

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
      net = []

      # Add additional conv layers before the class predictor.
      features_depth = static_shape.get_depth(input_shape)
      depth = max(min(features_depth, self._max_depth), self._min_depth)
      tf.logging.info(
          'depth of additional conv before box predictor: {}'.format(depth))

      if depth > 0 and self._num_layers_before_predictor > 0:
        for i in range(self._num_layers_before_predictor):
          net.append(keras.Conv2D(depth, [1, 1],
                                  name='SharedConvolutions_%d/Conv2d_%d_1x1_%d'
                                  % (stack_index, i, depth),
                                  padding='SAME',
                                  **self._conv_hyperparams.params()))
          net.append(self._conv_hyperparams.build_batch_norm(
              training=(self._is_training and not self._freeze_batchnorm),
              name='SharedConvolutions_%d/Conv2d_%d_1x1_%d_norm'
              % (stack_index, i, depth)))
          net.append(self._conv_hyperparams.build_activation_layer(
              name='SharedConvolutions_%d/Conv2d_%d_1x1_%d_activation'
              % (stack_index, i, depth),
          ))
      # Until certain bugs are fixed in checkpointable lists,
      # this net must be appended only once it's been filled with layers
      self._shared_nets.append(net)
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

    for (index, net) in enumerate(image_features):

      # Apply shared conv layers before the head predictors.
      for layer in self._shared_nets[index]:
        net = layer(net)

      for head_name in self._sorted_head_names:
        head_obj = self._prediction_heads[head_name][index]
        prediction = head_obj(net)
        predictions[head_name].append(prediction)

    return predictions


class WeightSharedConvolutionalBoxPredictor(box_predictor.KerasBoxPredictor):
  """Convolutional Box Predictor with weight sharing based on Keras.

  Defines the box predictor as defined in
  https://arxiv.org/abs/1708.02002. This class differs from
  ConvolutionalBoxPredictor in that it shares weights and biases while
  predicting from different feature maps. However, batch_norm parameters are not
  shared because the statistics of the activations vary among the different
  feature maps.

  Also note that separate multi-layer towers are constructed for the box
  encoding and class predictors respectively.
  """

  def __init__(self,
               is_training,
               num_classes,
               box_prediction_head,
               class_prediction_head,
               other_heads,
               conv_hyperparams,
               depth,
               num_layers_before_predictor,
               freeze_batchnorm,
               inplace_batchnorm_update,
               kernel_size=3,
               apply_batch_norm=False,
               share_prediction_tower=False,
               use_depthwise=False,
               name=None):
    """Constructor.

    Args:
      is_training: Indicates whether the BoxPredictor is in training mode.
      num_classes: number of classes.  Note that num_classes *does not*
        include the background category, so if groundtruth labels take values
        in {0, 1, .., K-1}, num_classes=K (and not K+1, even though the
        assigned classification targets can range from {0,... K}).
      box_prediction_head: The head that predicts the boxes.
      class_prediction_head: The head that predicts the classes.
      other_heads: A dictionary mapping head names to convolutional
        head classes.
      conv_hyperparams: A `hyperparams_builder.KerasLayerHyperparams` object
        containing hyperparameters for convolution ops.
      depth: depth of conv layers.
      num_layers_before_predictor: Number of the additional conv layers before
        the predictor.
      freeze_batchnorm: Whether to freeze batch norm parameters during
        training or not. When training with a small batch size (e.g. 1), it is
        desirable to freeze batch norm update and use pretrained batch norm
        params.
      inplace_batchnorm_update: Whether to update batch norm moving average
        values inplace. When this is false train op must add a control
        dependency on tf.graphkeys.UPDATE_OPS collection in order to update
        batch norm statistics.
      kernel_size: Size of final convolution kernel.
      apply_batch_norm: Whether to apply batch normalization to conv layers in
        this predictor.
      share_prediction_tower: Whether to share the multi-layer tower among box
        prediction head, class prediction head and other heads.
      use_depthwise: Whether to use depthwise separable conv2d instead of
       regular conv2d.
      name: A string name scope to assign to the model. If `None`, Keras
        will auto-generate one from the class name.
    """
    super(WeightSharedConvolutionalBoxPredictor, self).__init__(
        is_training, num_classes, freeze_batchnorm=freeze_batchnorm,
        inplace_batchnorm_update=inplace_batchnorm_update,
        name=name)

    self._box_prediction_head = box_prediction_head
    self._prediction_heads = {
        CLASS_PREDICTIONS_WITH_BACKGROUND: class_prediction_head,
    }
    if other_heads:
      self._prediction_heads.update(other_heads)
    # We generate a consistent ordering for the prediction head names,
    # so that all workers build the model in the exact same order.
    self._sorted_head_names = sorted(self._prediction_heads.keys())

    self._conv_hyperparams = conv_hyperparams
    self._depth = depth
    self._num_layers_before_predictor = num_layers_before_predictor
    self._kernel_size = kernel_size
    self._apply_batch_norm = apply_batch_norm
    self._share_prediction_tower = share_prediction_tower
    self._use_depthwise = use_depthwise

    # Additional projection layers to bring all feature maps to uniform
    # channels.
    self._additional_projection_layers = []
    # The base tower layers for each head.
    self._base_tower_layers_for_heads = {
        BOX_ENCODINGS: [],
        CLASS_PREDICTIONS_WITH_BACKGROUND: [],
    }
    for head_name in other_heads.keys():
      self._base_tower_layers_for_heads[head_name] = []

    # A dict maps the tower_name_scope of each head to the shared conv layers in
    # the base tower for different feature map levels.
    self._head_scope_conv_layers = {}

  def _insert_additional_projection_layer(
      self, inserted_layer_counter, target_channel):
    projection_layers = []
    if inserted_layer_counter >= 0:
      use_bias = False if self._apply_batch_norm else True
      projection_layers.append(keras.Conv2D(
          target_channel, [1, 1], strides=1, padding='SAME',
          name='ProjectionLayer/conv2d_{}'.format(inserted_layer_counter),
          **self._conv_hyperparams.params(use_bias=use_bias)))
      if self._apply_batch_norm:
        projection_layers.append(self._conv_hyperparams.build_batch_norm(
            training=(self._is_training and not self._freeze_batchnorm),
            name='ProjectionLayer/conv2d_{}/BatchNorm'.format(
                inserted_layer_counter)))
      inserted_layer_counter += 1
    return inserted_layer_counter, projection_layers

  def _compute_base_tower(self, tower_name_scope, feature_index):
    conv_layers = []
    batch_norm_layers = []
    activation_layers = []
    use_bias = False if self._apply_batch_norm else True
    for additional_conv_layer_idx in range(self._num_layers_before_predictor):
      layer_name = '{}/conv2d_{}'.format(
          tower_name_scope, additional_conv_layer_idx)
      if tower_name_scope not in self._head_scope_conv_layers:
        if self._use_depthwise:
          conv_layers.append(
              tf.keras.layers.SeparableConv2D(
                  self._depth,
                  [self._kernel_size, self._kernel_size],
                  padding='SAME',
                  name=layer_name,
                  **self._conv_hyperparams.params(use_bias=use_bias)))
        else:
          conv_layers.append(
              tf.keras.layers.Conv2D(
                  self._depth,
                  [self._kernel_size, self._kernel_size],
                  padding='SAME',
                  name=layer_name,
                  **self._conv_hyperparams.params(use_bias=use_bias)))
      # Each feature gets a separate batchnorm parameter even though they share
      # the same convolution weights.
      if self._apply_batch_norm:
        batch_norm_layers.append(self._conv_hyperparams.build_batch_norm(
            training=(self._is_training and not self._freeze_batchnorm),
            name='{}/conv2d_{}/BatchNorm/feature_{}'.format(
                tower_name_scope, additional_conv_layer_idx, feature_index)))
      activation_layers.append(tf.keras.layers.Lambda(tf.nn.relu6))

    # Set conv layers as the shared conv layers for different feature maps with
    # the same tower_name_scope.
    if tower_name_scope in self._head_scope_conv_layers:
      conv_layers = self._head_scope_conv_layers[tower_name_scope]

    # Stack the base_tower_layers in the order of conv_layer, batch_norm_layer
    # and activation_layer
    base_tower_layers = []
    for i in range(self._num_layers_before_predictor):
      base_tower_layers.extend([conv_layers[i]])
      if self._apply_batch_norm:
        base_tower_layers.extend([batch_norm_layers[i]])
      base_tower_layers.extend([activation_layers[i]])
    return conv_layers, base_tower_layers

  def build(self, input_shapes):
    """Creates the variables of the layer."""
    feature_channels = [
        shape_utils.get_dim_as_int(input_shape[3])
        for input_shape in input_shapes
    ]
    has_different_feature_channels = len(set(feature_channels)) > 1
    if has_different_feature_channels:
      inserted_layer_counter = 0
      target_channel = max(set(feature_channels), key=feature_channels.count)
      tf.logging.info('Not all feature maps have the same number of '
                      'channels, found: {}, appending additional projection '
                      'layers to bring all feature maps to uniformly have {} '
                      'channels.'.format(feature_channels, target_channel))
    else:
      # Place holder variables if has_different_feature_channels is False.
      target_channel = -1
      inserted_layer_counter = -1

    def _build_layers(tower_name_scope, feature_index):
      conv_layers, base_tower_layers = self._compute_base_tower(
          tower_name_scope=tower_name_scope, feature_index=feature_index)
      if tower_name_scope not in self._head_scope_conv_layers:
        self._head_scope_conv_layers[tower_name_scope] = conv_layers
      return base_tower_layers

    for feature_index, input_shape in enumerate(input_shapes):
      # Additional projection layers should not be shared as input channels
      # (and thus weight shapes) are different
      inserted_layer_counter, projection_layers = (
          self._insert_additional_projection_layer(
              inserted_layer_counter, target_channel))
      self._additional_projection_layers.append(projection_layers)

      if self._share_prediction_tower:
        box_tower_scope = 'PredictionTower'
      else:
        box_tower_scope = 'BoxPredictionTower'
      # For box tower base
      box_tower_layers = _build_layers(box_tower_scope, feature_index)
      self._base_tower_layers_for_heads[BOX_ENCODINGS].append(box_tower_layers)

      for head_name in self._sorted_head_names:
        if head_name == CLASS_PREDICTIONS_WITH_BACKGROUND:
          tower_name_scope = 'ClassPredictionTower'
        else:
          tower_name_scope = '{}PredictionTower'.format(head_name)
        box_tower_layers = _build_layers(tower_name_scope, feature_index)
        self._base_tower_layers_for_heads[head_name].append(box_tower_layers)

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

    def _apply_layers(base_tower_layers, image_feature):
      for layer in base_tower_layers:
        image_feature = layer(image_feature)
      return image_feature

    for (index, image_feature) in enumerate(image_features):
      # Apply additional projection layers to image features
      for layer in self._additional_projection_layers[index]:
        image_feature = layer(image_feature)

      # Apply box tower layers.
      box_tower_feature = _apply_layers(
          self._base_tower_layers_for_heads[BOX_ENCODINGS][index],
          image_feature)
      box_encodings = self._box_prediction_head(box_tower_feature)
      predictions[BOX_ENCODINGS].append(box_encodings)

      for head_name in self._sorted_head_names:
        head_obj = self._prediction_heads[head_name]
        if self._share_prediction_tower:
          head_tower_feature = box_tower_feature
        else:
          head_tower_feature = _apply_layers(
              self._base_tower_layers_for_heads[head_name][index],
              image_feature)
        prediction = head_obj(head_tower_feature)
        predictions[head_name].append(prediction)
    return predictions
