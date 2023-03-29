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
import tensorflow as tf

from official.projects.yt8m.configs import yt8m as yt8m_cfg
from official.projects.yt8m.modeling import nn_layers


layers = tf.keras.layers


class DbofModel(tf.keras.Model):
  """A YT8M model class builder.

  Creates a Deep Bag of Frames model.
  The model projects the features for each frame into a higher dimensional
  'clustering' space, pools across frames in that space, and then
  uses a configurable video-level model to classify the now aggregated features.
  The model will randomly sample either frames or sequences of frames during
  training to speed up convergence.
  """

  def __init__(
      self,
      params: yt8m_cfg.DbofModel,
      num_classes: int = 3862,
      input_specs: layers.InputSpec = layers.InputSpec(
          shape=[None, None, 1152]),
      l2_weight_decay: Optional[float] = None,
      **kwargs,
  ):
    """YT8M Dbof model initialization function.

    Args:
      params: model configuration parameters
      num_classes: `int` number of classes in dataset.
      input_specs: `tf.keras.layers.InputSpec` specs of the input tensor.
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

    self.dbof_backbone = nn_layers.Dbof(
        params,
        num_classes,
        input_specs,
        l2_weight_decay,
        **kwargs,
    )

    logging.info("Build DbofModel with %s.", params.agg_classifier_model)
    if hasattr(nn_layers, params.agg_classifier_model):
      aggregation_head = getattr(nn_layers, params.agg_classifier_model)
      if params.agg_classifier_model == "MoeModel":
        normalizer_params = dict(
            synchronized=params.norm_activation.use_sync_bn,
            momentum=params.norm_activation.norm_momentum,
            epsilon=params.norm_activation.norm_epsilon,
        )
        aggregation_head = functools.partial(
            aggregation_head, normalizer_params=normalizer_params)

      if params.agg_model is not None:
        kwargs.update(params.agg_model.as_dict())
      self.head = aggregation_head(
          input_specs=layers.InputSpec(shape=[None, params.hidden_size]),
          vocab_size=num_classes,
          l2_penalty=l2_weight_decay,
          **kwargs)

  def get_config(self):
    return self._config_dict

  @classmethod
  def from_config(cls, config):
    return cls(**config)

  def call(
      self, inputs: tf.Tensor, training: Any = None, mask: Any = None
  ) -> tf.Tensor:
    features = self.dbof_backbone(inputs)
    outputs = self.head(features)
    return outputs["predictions"]
