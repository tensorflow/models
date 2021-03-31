# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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

"""RetinaNet."""

# Import libraries
import tensorflow as tf


@tf.keras.utils.register_keras_serializable(package='Vision')
class RetinaNetModel(tf.keras.Model):
  """The RetinaNet model class."""

  def __init__(self,
               backbone,
               decoder,
               head,
               detection_generator,
               **kwargs):
    """Classification initialization function.

    Args:
      backbone: `tf.keras.Model` a backbone network.
      decoder: `tf.keras.Model` a decoder network.
      head: `RetinaNetHead`, the RetinaNet head.
      detection_generator: the detection generator.
      **kwargs: keyword arguments to be passed.
    """
    super(RetinaNetModel, self).__init__(**kwargs)
    self._config_dict = {
        'backbone': backbone,
        'decoder': decoder,
        'head': head,
        'detection_generator': detection_generator,
    }
    self._backbone = backbone
    self._decoder = decoder
    self._head = head
    self._detection_generator = detection_generator

  def call(self,
           images,
           image_shape=None,
           anchor_boxes=None,
           training=None):
    """Forward pass of the RetinaNet model.

    Args:
      images: `Tensor`, the input batched images, whose shape is
        [batch, height, width, 3].
      image_shape: `Tensor`, the actual shape of the input images, whose shape
        is [batch, 2] where the last dimension is [height, width]. Note that
        this is the actual image shape excluding paddings. For example, images
        in the batch may be resized into different shapes before padding to the
        fixed size.
      anchor_boxes: a dict of tensors which includes multilevel anchors.
        - key: `str`, the level of the multilevel predictions.
        - values: `Tensor`, the anchor coordinates of a particular feature
            level, whose shape is [height_l, width_l, num_anchors_per_location].
      training: `bool`, indicating whether it is in training mode.

    Returns:
      scores: a dict of tensors which includes scores of the predictions.
        - key: `str`, the level of the multilevel predictions.
        - values: `Tensor`, the box scores predicted from a particular feature
            level, whose shape is
            [batch, height_l, width_l, num_classes * num_anchors_per_location].
      boxes: a dict of tensors which includes coordinates of the predictions.
        - key: `str`, the level of the multilevel predictions.
        - values: `Tensor`, the box coordinates predicted from a particular
            feature level, whose shape is
            [batch, height_l, width_l, 4 * num_anchors_per_location].
      attributes: a dict of (attribute_name, attribute_predictions). Each
        attribute prediction is a dict that includes:
        - key: `str`, the level of the multilevel predictions.
        - values: `Tensor`, the attribute predictions from a particular
            feature level, whose shape is
            [batch, height_l, width_l, att_size * num_anchors_per_location].
    """
    # Feature extraction.
    features = self.backbone(images)
    if self.decoder:
      features = self.decoder(features)

    # Dense prediction. `raw_attributes` can be empty.
    raw_scores, raw_boxes, raw_attributes = self.head(features)

    if training:
      outputs = {
          'cls_outputs': raw_scores,
          'box_outputs': raw_boxes,
      }
      if raw_attributes:
        outputs.update({'att_outputs': raw_attributes})
      return outputs
    else:
      # Post-processing.
      final_results = self.detection_generator(
          raw_boxes, raw_scores, anchor_boxes, image_shape, raw_attributes)
      outputs = {
          'detection_boxes': final_results['detection_boxes'],
          'detection_scores': final_results['detection_scores'],
          'detection_classes': final_results['detection_classes'],
          'num_detections': final_results['num_detections'],
          'cls_outputs': raw_scores,
          'box_outputs': raw_boxes,
      }
      if raw_attributes:
        outputs.update({
            'att_outputs': raw_attributes,
            'detection_attributes': final_results['detection_attributes'],
        })
      return outputs

  @property
  def checkpoint_items(self):
    """Returns a dictionary of items to be additionally checkpointed."""
    items = dict(backbone=self.backbone, head=self.head)
    if self.decoder is not None:
      items.update(decoder=self.decoder)

    return items

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
