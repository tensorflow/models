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

"""RFCN Box Predictor."""
import tensorflow.compat.v1 as tf
from object_detection.core import box_predictor
from object_detection.utils import ops

BOX_ENCODINGS = box_predictor.BOX_ENCODINGS
CLASS_PREDICTIONS_WITH_BACKGROUND = (
    box_predictor.CLASS_PREDICTIONS_WITH_BACKGROUND)
MASK_PREDICTIONS = box_predictor.MASK_PREDICTIONS


class RfcnKerasBoxPredictor(box_predictor.KerasBoxPredictor):
  """RFCN Box Predictor.

  Applies a position sensitive ROI pooling on position sensitive feature maps to
  predict classes and refined locations. See https://arxiv.org/abs/1605.06409
  for details.

  This is used for the second stage of the RFCN meta architecture. Notice that
  locations are *not* shared across classes, thus for each anchor, a separate
  prediction is made for each class.
  """

  def __init__(self,
               is_training,
               num_classes,
               conv_hyperparams,
               freeze_batchnorm,
               num_spatial_bins,
               depth,
               crop_size,
               box_code_size,
               name=None):
    """Constructor.

    Args:
      is_training: Indicates whether the BoxPredictor is in training mode.
      num_classes: number of classes.  Note that num_classes *does not*
        include the background category, so if groundtruth labels take values
        in {0, 1, .., K-1}, num_classes=K (and not K+1, even though the
        assigned classification targets can range from {0,... K}).
      conv_hyperparams: A `hyperparams_builder.KerasLayerHyperparams` object
        containing hyperparameters for convolution ops.
      freeze_batchnorm: Whether to freeze batch norm parameters during
        training or not. When training with a small batch size (e.g. 1), it is
        desirable to freeze batch norm update and use pretrained batch norm
        params.
      num_spatial_bins: A list of two integers `[spatial_bins_y,
        spatial_bins_x]`.
      depth: Target depth to reduce the input feature maps to.
      crop_size: A list of two integers `[crop_height, crop_width]`.
      box_code_size: Size of encoding for each box.
      name: A string name scope to assign to the box predictor. If `None`, Keras
        will auto-generate one from the class name.
    """
    super(RfcnKerasBoxPredictor, self).__init__(
        is_training, num_classes, freeze_batchnorm=freeze_batchnorm,
        inplace_batchnorm_update=False, name=name)
    self._freeze_batchnorm = freeze_batchnorm
    self._conv_hyperparams = conv_hyperparams
    self._num_spatial_bins = num_spatial_bins
    self._depth = depth
    self._crop_size = crop_size
    self._box_code_size = box_code_size

    # Build the shared layers used for both heads
    self._shared_conv_layers = []
    self._shared_conv_layers.append(
        tf.keras.layers.Conv2D(
            self._depth,
            [1, 1],
            padding='SAME',
            name='reduce_depth_conv',
            **self._conv_hyperparams.params()))
    self._shared_conv_layers.append(
        self._conv_hyperparams.build_batch_norm(
            training=(self._is_training and not self._freeze_batchnorm),
            name='reduce_depth_batchnorm'))
    self._shared_conv_layers.append(
        self._conv_hyperparams.build_activation_layer(
            name='reduce_depth_activation'))

    self._box_encoder_layers = []
    location_feature_map_depth = (self._num_spatial_bins[0] *
                                  self._num_spatial_bins[1] *
                                  self.num_classes *
                                  self._box_code_size)
    self._box_encoder_layers.append(
        tf.keras.layers.Conv2D(
            location_feature_map_depth,
            [1, 1],
            padding='SAME',
            name='refined_locations_conv',
            **self._conv_hyperparams.params()))
    self._box_encoder_layers.append(
        self._conv_hyperparams.build_batch_norm(
            training=(self._is_training and not self._freeze_batchnorm),
            name='refined_locations_batchnorm'))

    self._class_predictor_layers = []
    self._total_classes = self.num_classes + 1  # Account for background class.
    class_feature_map_depth = (self._num_spatial_bins[0] *
                               self._num_spatial_bins[1] *
                               self._total_classes)
    self._class_predictor_layers.append(
        tf.keras.layers.Conv2D(
            class_feature_map_depth,
            [1, 1],
            padding='SAME',
            name='class_predictions_conv',
            **self._conv_hyperparams.params()))
    self._class_predictor_layers.append(
        self._conv_hyperparams.build_batch_norm(
            training=(self._is_training and not self._freeze_batchnorm),
            name='class_predictions_batchnorm'))

  @property
  def num_classes(self):
    return self._num_classes

  def _predict(self, image_features, proposal_boxes, **kwargs):
    """Computes encoded object locations and corresponding confidences.

    Args:
      image_features: A list of float tensors of shape [batch_size, height_i,
      width_i, channels_i] containing features for a batch of images.
      proposal_boxes: A float tensor of shape [batch_size, num_proposals,
        box_code_size].
      **kwargs: Unused Keyword args

    Returns:
      box_encodings: A list of float tensors of shape
        [batch_size, num_anchors_i, q, code_size] representing the location of
        the objects, where q is 1 or the number of classes. Each entry in the
        list corresponds to a feature map in the input `image_features` list.
      class_predictions_with_background: A list of float tensors of shape
        [batch_size, num_anchors_i, num_classes + 1] representing the class
        predictions for the proposals. Each entry in the list corresponds to a
        feature map in the input `image_features` list.

    Raises:
      ValueError: if num_predictions_per_location is not 1 or if
        len(image_features) is not 1.
    """
    if len(image_features) != 1:
      raise ValueError('length of `image_features` must be 1. Found {}'.
                       format(len(image_features)))
    image_feature = image_features[0]
    batch_size = tf.shape(proposal_boxes)[0]
    num_boxes = tf.shape(proposal_boxes)[1]
    net = image_feature
    for layer in self._shared_conv_layers:
      net = layer(net)

    # Location predictions.
    box_net = net
    for layer in self._box_encoder_layers:
      box_net = layer(box_net)
    box_encodings = ops.batch_position_sensitive_crop_regions(
        box_net,
        boxes=proposal_boxes,
        crop_size=self._crop_size,
        num_spatial_bins=self._num_spatial_bins,
        global_pool=True)
    box_encodings = tf.squeeze(box_encodings, axis=[2, 3])
    box_encodings = tf.reshape(box_encodings,
                               [batch_size * num_boxes, 1, self.num_classes,
                                self._box_code_size])

    # Class predictions.
    class_net = net
    for layer in self._class_predictor_layers:
      class_net = layer(class_net)
    class_predictions_with_background = (
        ops.batch_position_sensitive_crop_regions(
            class_net,
            boxes=proposal_boxes,
            crop_size=self._crop_size,
            num_spatial_bins=self._num_spatial_bins,
            global_pool=True))
    class_predictions_with_background = tf.squeeze(
        class_predictions_with_background, axis=[2, 3])
    class_predictions_with_background = tf.reshape(
        class_predictions_with_background,
        [batch_size * num_boxes, 1, self._total_classes])

    return {BOX_ENCODINGS: [box_encodings],
            CLASS_PREDICTIONS_WITH_BACKGROUND:
            [class_predictions_with_background]}
