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

"""Mask R-CNN Box Predictor."""
from object_detection.core import box_predictor


BOX_ENCODINGS = box_predictor.BOX_ENCODINGS
CLASS_PREDICTIONS_WITH_BACKGROUND = (
    box_predictor.CLASS_PREDICTIONS_WITH_BACKGROUND)
MASK_PREDICTIONS = box_predictor.MASK_PREDICTIONS


class MaskRCNNBoxPredictor(box_predictor.BoxPredictor):
  """Mask R-CNN Box Predictor.

  See Mask R-CNN: He, K., Gkioxari, G., Dollar, P., & Girshick, R. (2017).
  Mask R-CNN. arXiv preprint arXiv:1703.06870.

  This is used for the second stage of the Mask R-CNN detector where proposals
  cropped from an image are arranged along the batch dimension of the input
  image_features tensor. Notice that locations are *not* shared across classes,
  thus for each anchor, a separate prediction is made for each class.

  In addition to predicting boxes and classes, optionally this class allows
  predicting masks and/or keypoints inside detection boxes.

  Currently this box predictor makes per-class predictions; that is, each
  anchor makes a separate box prediction for each class.
  """

  def __init__(self,
               is_training,
               num_classes,
               box_prediction_head,
               class_prediction_head,
               third_stage_heads):
    """Constructor.

    Args:
      is_training: Indicates whether the BoxPredictor is in training mode.
      num_classes: number of classes.  Note that num_classes *does not*
        include the background category, so if groundtruth labels take values
        in {0, 1, .., K-1}, num_classes=K (and not K+1, even though the
        assigned classification targets can range from {0,... K}).
      box_prediction_head: The head that predicts the boxes in second stage.
      class_prediction_head: The head that predicts the classes in second stage.
      third_stage_heads: A dictionary mapping head names to mask rcnn head
        classes.
    """
    super(MaskRCNNBoxPredictor, self).__init__(is_training, num_classes)
    self._box_prediction_head = box_prediction_head
    self._class_prediction_head = class_prediction_head
    self._third_stage_heads = third_stage_heads

  @property
  def num_classes(self):
    return self._num_classes

  def get_second_stage_prediction_heads(self):
    return BOX_ENCODINGS, CLASS_PREDICTIONS_WITH_BACKGROUND

  def get_third_stage_prediction_heads(self):
    return sorted(self._third_stage_heads.keys())

  def _predict(self,
               image_features,
               num_predictions_per_location,
               prediction_stage=2):
    """Optionally computes encoded object locations, confidences, and masks.

    Predicts the heads belonging to the given prediction stage.

    Args:
      image_features: A list of float tensors of shape
        [batch_size, height_i, width_i, channels_i] containing roi pooled
        features for each image. The length of the list should be 1 otherwise
        a ValueError will be raised.
      num_predictions_per_location: A list of integers representing the number
        of box predictions to be made per spatial location for each feature map.
        Currently, this must be set to [1], or an error will be raised.
      prediction_stage: Prediction stage. Acceptable values are 2 and 3.

    Returns:
      A dictionary containing the predicted tensors that are listed in
      self._prediction_heads. A subset of the following keys will exist in the
      dictionary:
        BOX_ENCODINGS: A float tensor of shape
          [batch_size, 1, num_classes, code_size] representing the
          location of the objects.
        CLASS_PREDICTIONS_WITH_BACKGROUND: A float tensor of shape
          [batch_size, 1, num_classes + 1] representing the class
          predictions for the proposals.
        MASK_PREDICTIONS: A float tensor of shape
          [batch_size, 1, num_classes, image_height, image_width]

    Raises:
      ValueError: If num_predictions_per_location is not 1 or if
        len(image_features) is not 1.
      ValueError: if prediction_stage is not 2 or 3.
    """
    if (len(num_predictions_per_location) != 1 or
        num_predictions_per_location[0] != 1):
      raise ValueError('Currently FullyConnectedBoxPredictor only supports '
                       'predicting a single box per class per location.')
    if len(image_features) != 1:
      raise ValueError('length of `image_features` must be 1. Found {}'.format(
          len(image_features)))
    image_feature = image_features[0]
    predictions_dict = {}

    if prediction_stage == 2:
      predictions_dict[BOX_ENCODINGS] = self._box_prediction_head.predict(
          features=image_feature,
          num_predictions_per_location=num_predictions_per_location[0])
      predictions_dict[CLASS_PREDICTIONS_WITH_BACKGROUND] = (
          self._class_prediction_head.predict(
              features=image_feature,
              num_predictions_per_location=num_predictions_per_location[0]))
    elif prediction_stage == 3:
      for prediction_head in self.get_third_stage_prediction_heads():
        head_object = self._third_stage_heads[prediction_head]
        predictions_dict[prediction_head] = head_object.predict(
            features=image_feature,
            num_predictions_per_location=num_predictions_per_location[0])
    else:
      raise ValueError('prediction_stage should be either 2 or 3.')

    return predictions_dict
