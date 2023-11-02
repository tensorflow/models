# Copyright 2023 The TensorFlow Authors. All Rights Reserved.
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

"""YT8M prediction model definition."""

import functools
from typing import Any, Optional

from absl import logging
import tensorflow as tf, tf_keras

from official.projects.yt8m.configs import yt8m as yt8m_cfg
from official.projects.yt8m.modeling import backbones  # pylint: disable=unused-import
from official.projects.yt8m.modeling import heads
from official.vision.modeling.backbones import factory


layers = tf_keras.layers


class VideoClassificationModel(tf_keras.Model):
  """A video classification model class builder.

  The model consists of a backbone (dbof) and a classification head.
  The dbof backbone projects features for each frame into a higher dimensional
  'clustering' space, pools across frames in that space, and then
  uses a configurable video-level model to classify the now aggregated features.
  The model will randomly sample either frames or sequences of frames during
  training to speed up convergence.
  """

  def __init__(
      self,
      params: yt8m_cfg.VideoClassificationModel,
      backbone: Optional[tf_keras.Model] = None,
      num_classes: int = 3862,
      input_specs: layers.InputSpec = layers.InputSpec(
          shape=[None, None, 1152]
      ),
      l2_weight_decay: Optional[float] = None,
      **kwargs,
  ):
    """YT8M video classification model initialization function.

    Args:
      params: Model configuration parameters.
      backbone: Optional backbone model. Will build a backbone if None.
      num_classes: `int` number of classes in dataset.
      input_specs: `tf_keras.layers.InputSpec` specs of the input tensor.
        [batch_size x num_frames x num_features]
      l2_weight_decay: An optional `float` of kernel regularizer weight decay.
      **kwargs: keyword arguments to be passed.
    """
    super().__init__()
    self._params = params
    self._num_classes = num_classes
    self._input_specs = input_specs
    self._l2_weight_decay = l2_weight_decay
    self._config_dict = {
        "params": params,
        "input_specs": input_specs,
        "num_classes": num_classes,
        "l2_weight_decay": l2_weight_decay,
    }

    if backbone is None:
      # Divide weight decay by 2.0 to match the implementation of tf.nn.l2_loss.
      # (https://www.tensorflow.org/api_docs/python/tf/keras/regularizers/l2)
      # (https://www.tensorflow.org/api_docs/python/tf/nn/l2_loss)
      l2_regularizer = (
          tf_keras.regularizers.l2(l2_weight_decay / 2.0)
          if l2_weight_decay
          else None
      )
      backbone = factory.build_backbone(
          input_specs=input_specs,
          backbone_config=params.backbone,
          norm_activation_config=params.norm_activation,
          l2_regularizer=l2_regularizer,
          **kwargs,
      )

    self.backbone = backbone
    self.build_head()

  def build_head(self):
    logging.info("Build DbofModel with %s.", self._params.head.type)
    head_cfg = self._params.head.get()
    if self._params.head.type == "moe":
      normalizer_params = dict(
          synchronized=self._params.norm_activation.use_sync_bn,
          momentum=self._params.norm_activation.norm_momentum,
          epsilon=self._params.norm_activation.norm_epsilon,
      )
      aggregation_head = functools.partial(
          heads.MoeModel, normalizer_params=normalizer_params
      )
    elif self._params.head.type == "logistic":
      aggregation_head = heads.LogisticModel
    else:
      logging.warn("Skip build head type: %s", self._params.head.type)
      return

    l2_regularizer = (
        tf_keras.regularizers.l2(self._l2_weight_decay / 2.0)
        if self._l2_weight_decay
        else None
    )
    self.head = aggregation_head(
        vocab_size=self._num_classes,
        l2_regularizer=l2_regularizer,
        **head_cfg.as_dict(),
    )

  def get_config(self):
    return self._config_dict

  @classmethod
  def from_config(cls, config):
    return cls(**config)

  def call(
      self,
      inputs: tf.Tensor,
      num_frames: Any = None,
      training: Any = None,
  ) -> dict[str, tf.Tensor]:
    features = self.backbone(
        inputs,
        num_frames=num_frames,
        training=training,
    )
    outputs = self.head(features, training=training)
    return outputs

  @property
  def checkpoint_items(self) -> dict[str, Any]:
    """Returns a dictionary of items to be additionally checkpointed."""
    return dict(backbone=self.backbone, head=self.head)
