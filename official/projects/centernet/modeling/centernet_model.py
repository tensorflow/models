# Copyright 2024 The TensorFlow Authors. All Rights Reserved.
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

"""Centernet detection models."""

from typing import Mapping, Union, Any

import tensorflow as tf, tf_keras


class CenterNetModel(tf_keras.Model):
  """CenterNet Model."""

  def __init__(self,
               backbone: tf_keras.Model,
               head: tf_keras.Model,
               detection_generator: tf_keras.layers.Layer,
               **kwargs):
    """CenterNet Model.

    Args:
      backbone: a backbone network.
      head: a projection head for centernet.
      detection_generator: a detection generator for centernet.
      **kwargs: keyword arguments to be passed.
    """
    super(CenterNetModel, self).__init__(**kwargs)
    # model components
    self._backbone = backbone
    self._detection_generator = detection_generator
    self._head = head

  def call(self,  # pytype: disable=annotation-type-mismatch,signature-mismatch
           inputs: tf.Tensor,
           training: bool = None,
           **kwargs) -> Mapping[str, tf.Tensor]:
    features = self._backbone(inputs)
    raw_outputs = self._head(features)
    model_outputs = {'raw_output': raw_outputs}
    if not training:
      predictions = self._detection_generator(raw_outputs)
      model_outputs.update(predictions)
    return model_outputs

  @property
  def checkpoint_items(
      self) -> Mapping[str, Union[tf_keras.Model, tf_keras.layers.Layer]]:
    """Returns a dictionary of items to be additionally checkpointed."""
    items = dict(backbone=self.backbone, head=self.head)

    return items

  @property
  def backbone(self):
    return self._backbone

  @property
  def detection_generator(self):
    return self._detection_generator

  @property
  def head(self):
    return self._head

  def get_config(self) -> Mapping[str, Any]:
    config_dict = {
        'backbone': self._backbone,
        'head': self._head,
        'detection_generator': self._detection_generator,
    }
    return config_dict

  @classmethod
  def from_config(cls, config, custom_objects=None):
    return cls(**config)
