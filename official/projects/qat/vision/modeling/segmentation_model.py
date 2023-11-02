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

"""Build segmentation models."""
from typing import Any, Mapping, Union

# Import libraries
import tensorflow as tf, tf_keras

layers = tf_keras.layers


@tf_keras.utils.register_keras_serializable(package='Vision')
class SegmentationModelQuantized(tf_keras.Model):
  """A Segmentation class model.

  Input images are passed through backbone first. Decoder network is then
  applied, and finally, segmentation head is applied on the output of the
  decoder network. Layers such as ASPP should be part of decoder. Any feature
  fusion is done as part of the segmentation head (i.e. deeplabv3+ feature
  fusion is not part of the decoder, instead it is part of the segmentation
  head). This way, different feature fusion techniques can be combined with
  different backbones, and decoders.
  """

  def __init__(self, backbone: tf_keras.Model, decoder: tf_keras.layers.Layer,
               head: tf_keras.layers.Layer,
               input_specs: tf_keras.layers.InputSpec, **kwargs):
    """Segmentation initialization function.

    Args:
      backbone: a backbone network.
      decoder: a decoder network. E.g. FPN.
      head: segmentation head.
      input_specs: The shape specifications of input tensor.
      **kwargs: keyword arguments to be passed.
    """
    inputs = tf_keras.Input(shape=input_specs.shape[1:], name=input_specs.name)
    backbone_features = backbone(inputs)

    if decoder:
      backbone_feature = backbone_features[str(decoder.get_config()['level'])]
      decoder_feature = decoder(backbone_feature)
    else:
      decoder_feature = backbone_features

    backbone_feature = backbone_features[str(head.get_config()['low_level'])]
    x = {'logits': head((backbone_feature, decoder_feature))}
    super().__init__(inputs=inputs, outputs=x, **kwargs)
    self._config_dict = {
        'backbone': backbone,
        'decoder': decoder,
        'head': head,
    }
    self.backbone = backbone
    self.decoder = decoder
    self.head = head

  @property
  def checkpoint_items(
      self) -> Mapping[str, Union[tf_keras.Model, tf_keras.layers.Layer]]:
    """Returns a dictionary of items to be additionally checkpointed."""
    items = dict(backbone=self.backbone, head=self.head)
    if self.decoder is not None:
      items.update(decoder=self.decoder)
    return items

  def get_config(self) -> Mapping[str, Any]:
    return self._config_dict

  @classmethod
  def from_config(cls, config, custom_objects=None):
    return cls(**config)
