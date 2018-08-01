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

"""Base Mask RCNN head class."""
from abc import abstractmethod


class MaskRCNNHead(object):
  """Mask RCNN head base class."""

  def __init__(self):
    """Constructor."""

  def predict(self, roi_pooled_features):
    """Returns the head's predictions.

    Args:
      roi_pooled_features: A float tensor of shape
        [batch_size, height, width, channels] containing ROI pooled features
        from a batch of boxes.
    """
    return self._predict(roi_pooled_features)

  @abstractmethod
  def _predict(self, roi_pooled_features):
    """The abstract internal prediction function that needs to be overloaded.

    Args:
      roi_pooled_features: A float tensor of shape
        [batch_size, height, width, channels] containing ROI pooled features
        from a batch of boxes.
    """
