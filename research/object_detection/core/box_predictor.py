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

"""Box predictor for object detectors.

Box predictors are classes that take a high level
image feature map as input and produce two predictions,
(1) a tensor encoding box locations, and
(2) a tensor encoding classes for each box.

These components are passed directly to loss functions
in our detection models.

These modules are separated from the main model since the same
few box predictor architectures are shared across many models.
"""
from abc import abstractmethod
import tensorflow as tf

BOX_ENCODINGS = 'box_encodings'
CLASS_PREDICTIONS_WITH_BACKGROUND = 'class_predictions_with_background'
MASK_PREDICTIONS = 'mask_predictions'


class BoxPredictor(object):
  """BoxPredictor."""

  def __init__(self, is_training, num_classes):
    """Constructor.

    Args:
      is_training: Indicates whether the BoxPredictor is in training mode.
      num_classes: number of classes.  Note that num_classes *does not*
        include the background category, so if groundtruth labels take values
        in {0, 1, .., K-1}, num_classes=K (and not K+1, even though the
        assigned classification targets can range from {0,... K}).
    """
    self._is_training = is_training
    self._num_classes = num_classes

  @property
  def num_classes(self):
    return self._num_classes

  def predict(self, image_features, num_predictions_per_location,
              scope=None, **params):
    """Computes encoded object locations and corresponding confidences.

    Takes a list of high level image feature maps as input and produces a list
    of box encodings and a list of class scores where each element in the output
    lists correspond to the feature maps in the input list.

    Args:
      image_features: A list of float tensors of shape [batch_size, height_i,
      width_i, channels_i] containing features for a batch of images.
      num_predictions_per_location: A list of integers representing the number
        of box predictions to be made per spatial location for each feature map.
      scope: Variable and Op scope name.
      **params: Additional keyword arguments for specific implementations of
              BoxPredictor.

    Returns:
      A dictionary containing at least the following tensors.
        box_encodings: A list of float tensors. Each entry in the list
          corresponds to a feature map in the input `image_features` list. All
          tensors in the list have one of the two following shapes:
          a. [batch_size, num_anchors_i, q, code_size] representing the location
            of the objects, where q is 1 or the number of classes.
          b. [batch_size, num_anchors_i, code_size].
        class_predictions_with_background: A list of float tensors of shape
          [batch_size, num_anchors_i, num_classes + 1] representing the class
          predictions for the proposals. Each entry in the list corresponds to a
          feature map in the input `image_features` list.

    Raises:
      ValueError: If length of `image_features` is not equal to length of
        `num_predictions_per_location`.
    """
    if len(image_features) != len(num_predictions_per_location):
      raise ValueError('image_feature and num_predictions_per_location must '
                       'be of same length, found: {} vs {}'.
                       format(len(image_features),
                              len(num_predictions_per_location)))
    if scope is not None:
      with tf.variable_scope(scope):
        return self._predict(image_features, num_predictions_per_location,
                             **params)
    return self._predict(image_features, num_predictions_per_location,
                         **params)

  # TODO(rathodv): num_predictions_per_location could be moved to constructor.
  # This is currently only used by ConvolutionalBoxPredictor.
  @abstractmethod
  def _predict(self, image_features, num_predictions_per_location, **params):
    """Implementations must override this method.

    Args:
      image_features: A list of float tensors of shape [batch_size, height_i,
        width_i, channels_i] containing features for a batch of images.
      num_predictions_per_location: A list of integers representing the number
        of box predictions to be made per spatial location for each feature map.
      **params: Additional keyword arguments for specific implementations of
              BoxPredictor.

    Returns:
      A dictionary containing at least the following tensors.
        box_encodings: A list of float tensors. Each entry in the list
          corresponds to a feature map in the input `image_features` list. All
          tensors in the list have one of the two following shapes:
          a. [batch_size, num_anchors_i, q, code_size] representing the location
            of the objects, where q is 1 or the number of classes.
          b. [batch_size, num_anchors_i, code_size].
        class_predictions_with_background: A list of float tensors of shape
          [batch_size, num_anchors_i, num_classes + 1] representing the class
          predictions for the proposals. Each entry in the list corresponds to a
          feature map in the input `image_features` list.
    """
    pass

