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
"""Build segmentation models."""

# Import libraries
import tensorflow as tf

layers = tf.keras.layers


@tf.keras.utils.register_keras_serializable(package='Vision')
class SegmentationModel(tf.keras.Model):
  """A Segmentation class model.

  Input images are passed through backbone first. Decoder network is then
  applied, and finally, segmentation head is applied on the output of the
  decoder network. Layers such as ASPP should be part of decoder. Any feature
  fusion is done as part of the segmentation head (i.e. deeplabv3+ feature
  fusion is not part of the decoder, instead it is part of the segmentation
  head). This way, different feature fusion techniques can be combined with
  different backbones, and decoders.
  """

  def __init__(self,
               backbone,
               decoder,
               head,
               **kwargs):
    """Segmentation initialization function.

    Args:
      backbone: a backbone network.
      decoder: a decoder network. E.g. FPN.
      head: segmentation head.
      **kwargs: keyword arguments to be passed.
    """
    super(SegmentationModel, self).__init__(**kwargs)
    self._config_dict = {
        'backbone': backbone,
        'decoder': decoder,
        'head': head,
    }
    self.backbone = backbone
    self.decoder = decoder
    self.head = head

  def call(self, inputs, training=None):
    backbone_features = self.backbone(inputs)

    if self.decoder:
      decoder_features = self.decoder(backbone_features)
    else:
      decoder_features = backbone_features

    return self.head(backbone_features, decoder_features)

  @property
  def checkpoint_items(self):
    """Returns a dictionary of items to be additionally checkpointed."""
    items = dict(backbone=self.backbone, head=self.head)
    if self.decoder is not None:
      items.update(decoder=self.decoder)
    return items

  def get_config(self):
    return self._config_dict

  @classmethod
  def from_config(cls, config, custom_objects=None):
    return cls(**config)
