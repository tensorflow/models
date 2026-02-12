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
"""DETR (Detection Transformer) meta-architecture definition.

See https://arxiv.org/abs/2005.12872 for more information.
"""

import abc

class DETRKerasFeatureExtractor(object):
  """Keras-based DETR Feature Extractor definition."""

  def __init__(self,
               is_training,
               features_stride,
               batch_norm_trainable=False,
               weight_decay=0.0):
    """Constructor.

    Args:
      is_training: A boolean indicating whether the training version of the
        computation graph should be constructed.
      features_stride: Output stride of first stage feature map.
      batch_norm_trainable: Whether to update batch norm parameters during
        training or not. When training with a relative large batch size
        (e.g. 8), it could be desirable to enable batch norm update.
      weight_decay: float weight decay for feature extractor (default: 0.0).
    """
    self._is_training = is_training
    self.features_stride = features_stride
    self._train_batch_norm = (batch_norm_trainable and is_training)
    self._weight_decay = weight_decay

  @abc.abstractmethod
  def preprocess(self, resized_inputs):
    """Feature-extractor specific preprocessing (minus image resizing)."""
    pass

  @abc.abstractmethod
  def get_proposal_feature_extractor_model(self, name):
    """Get model that extracts first stage RPN features, to be overridden."""
    pass