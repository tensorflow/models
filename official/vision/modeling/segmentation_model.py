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

"""Build segmentation models."""
from typing import Any, Mapping, Union, Optional, Dict

import tensorflow as tf, tf_keras

layers = tf_keras.layers


@tf_keras.utils.register_keras_serializable(package='Vision')
class SegmentationModel(tf_keras.Model):
  """A Segmentation class model.

  Input images are passed through backbone first. Decoder network is then
  applied, and finally, segmentation head is applied on the output of the
  decoder network. Layers such as ASPP should be part of decoder. Any feature
  fusion is done as part of the segmentation head (i.e. deeplabv3+ feature
  fusion is not part of the decoder, instead it is part of the segmentation
  head). This way, different feature fusion techniques can be combined with
  different backbones, and decoders.
  """

  def __init__(self, backbone: tf_keras.Model, decoder: tf_keras.Model,
               head: tf_keras.layers.Layer,
               mask_scoring_head: Optional[tf_keras.layers.Layer] = None,
               **kwargs):
    """Segmentation initialization function.

    Args:
      backbone: a backbone network.
      decoder: a decoder network. E.g. FPN.
      head: segmentation head.
      mask_scoring_head: mask scoring head.
      **kwargs: keyword arguments to be passed.
    """
    super(SegmentationModel, self).__init__(**kwargs)
    self._config_dict = {
        'backbone': backbone,
        'decoder': decoder,
        'head': head,
        'mask_scoring_head': mask_scoring_head,
    }
    self.backbone = backbone
    self.decoder = decoder
    self.head = head
    self.mask_scoring_head = mask_scoring_head

  def call(
      self, inputs: tf.Tensor, training: bool = None  # pytype: disable=annotation-type-mismatch,signature-mismatch
  ) -> Dict[str, tf.Tensor]:
    backbone_features = self.backbone(inputs)

    if self.decoder:
      decoder_features = self.decoder(backbone_features)
    else:
      decoder_features = backbone_features

    logits = self.head((backbone_features, decoder_features))
    outputs = {'logits': logits}
    if self.mask_scoring_head:
      mask_scores = self.mask_scoring_head(logits)
      outputs.update({'mask_scores': mask_scores})
    return outputs

  @property
  def checkpoint_items(
      self) -> Mapping[str, Union[tf_keras.Model, tf_keras.layers.Layer]]:
    """Returns a dictionary of items to be additionally checkpointed."""
    items = dict(backbone=self.backbone, head=self.head)
    if self.decoder is not None:
      items.update(decoder=self.decoder)
    if self.mask_scoring_head is not None:
      items.update(mask_scoring_head=self.mask_scoring_head)
    return items

  def get_config(self) -> Mapping[str, Any]:
    return self._config_dict

  @classmethod
  def from_config(cls, config, custom_objects=None):
    return cls(**config)
