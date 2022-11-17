# Copyright 2022 The TensorFlow Authors. All Rights Reserved.
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

"""Yolo models."""

from typing import Mapping, Union, Any, Dict
import tensorflow as tf
from official.projects.yolo.modeling.layers import nn_blocks


class Yolo(tf.keras.Model):
  """The YOLO model class."""

  def __init__(self,
               backbone,
               decoder,
               head,
               detection_generator,
               **kwargs):
    """Detection initialization function.

    Args:
      backbone: `tf.keras.Model` a backbone network.
      decoder: `tf.keras.Model` a decoder network.
      head: `RetinaNetHead`, the RetinaNet head.
      detection_generator: the detection generator.
      **kwargs: keyword arguments to be passed.
    """
    super(Yolo, self).__init__(**kwargs)

    self._config_dict = {
        'backbone': backbone,
        'decoder': decoder,
        'head': head,
        'detection_generator': detection_generator
    }

    # model components
    self._backbone = backbone
    self._decoder = decoder
    self._head = head
    self._detection_generator = detection_generator
    self._fused = False
    return

  def call(self,
           inputs: tf.Tensor,
           training: bool = None,
           mask: Any = None) -> Dict[str, tf.Tensor]:
    maps = self.backbone(inputs)
    decoded_maps = self.decoder(maps)
    raw_predictions = self.head(decoded_maps)
    if training:
      return {'raw_output': raw_predictions}
    else:
      # Post-processing.
      predictions = self.detection_generator(raw_predictions)
      predictions.update({'raw_output': raw_predictions})
      return predictions

  @property
  def backbone(self):
    return self._backbone

  @property
  def decoder(self):
    return self._decoder

  @property
  def head(self):
    return self._head

  @property
  def detection_generator(self):
    return self._detection_generator

  def get_config(self):
    return self._config_dict

  @classmethod
  def from_config(cls, config):
    return cls(**config)

  @property
  def checkpoint_items(
      self) -> Mapping[str, Union[tf.keras.Model, tf.keras.layers.Layer]]:
    """Returns a dictionary of items to be additionally checkpointed."""
    items = dict(backbone=self.backbone, head=self.head)
    if self.decoder is not None:
      items.update(decoder=self.decoder)
    return items

  def fuse(self):
    """Fuses all Convolution and Batchnorm layers to get better latency."""
    print('Fusing Conv Batch Norm Layers.')
    if not self._fused:
      self._fused = True
      for layer in self.submodules:
        if isinstance(layer, nn_blocks.ConvBN):
          layer.fuse()
      self.summary()
    return
