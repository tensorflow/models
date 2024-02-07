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

"""PointPillars Model."""
from typing import Any, Dict, List, Mapping, Optional, Tuple, Union

import tensorflow as tf, tf_keras

from official.projects.pointpillars.utils import utils


@tf_keras.utils.register_keras_serializable(package='Vision')
class PointPillarsModel(tf_keras.Model):
  """The PointPillars model class."""

  def __init__(self,
               featurizer: tf_keras.layers.Layer,
               backbone: tf_keras.Model,
               decoder: tf_keras.Model,
               head: tf_keras.layers.Layer,
               detection_generator: tf_keras.layers.Layer,
               min_level: int,
               max_level: int,
               image_size: Tuple[int, int],
               anchor_sizes: List[Tuple[float, float]],
               **kwargs):
    """Initialize the model class.

    Args:
      featurizer: A `tf_keras.layers.Layer` to extract features from pillars.
      backbone: A `tf_keras.Model` to downsample feature images.
      decoder: A `tf_keras.Model` to upsample feature images.
      head: A `tf_keras.layers.Layer` to predict targets.
      detection_generator: A `tf_keras.layers.Layer` to generate detections.
      min_level: An `int` minimum level of multiscale outputs.
      max_level: An `int` maximum level of multiscale outputs.
      image_size: A tuple (height, width) of image size.
      anchor_sizes: A list of tuple (length, width) of anchor boxes.
      **kwargs: Additional keyword arguments to be passed.
    """
    super(PointPillarsModel, self).__init__(**kwargs)
    self._featurizer = featurizer
    self._backbone = backbone
    self._decoder = decoder
    self._head = head
    self._detection_generator = detection_generator
    self._min_level = min_level
    self._max_level = max_level
    self._image_size = image_size
    self._anchor_sizes = anchor_sizes

  def generate_outputs(
      self,
      raw_scores: Dict[str, tf.Tensor],
      raw_boxes: Dict[str, tf.Tensor],
      raw_attributes: Dict[str, Dict[str, tf.Tensor]],
      image_shape: Optional[tf.Tensor] = None,
      anchor_boxes: Optional[Mapping[str, tf.Tensor]] = None,
      generate_detections: bool = False) -> Mapping[str, Any]:
    if not raw_attributes:
      raise ValueError('PointPillars model needs attribute heads.')
    # Clap heading to [-pi, pi]
    if 'heading' in raw_attributes:
      raw_attributes['heading'] = utils.clip_heading(raw_attributes['heading'])

    outputs = {
        'cls_outputs': raw_scores,
        'box_outputs': raw_boxes,
        'attribute_outputs': raw_attributes,
    }
    # Cast raw prediction to float32 for loss calculation.
    outputs = tf.nest.map_structure(lambda x: tf.cast(x, tf.float32), outputs)
    if not generate_detections:
      return outputs

    if image_shape is None:
      raise ValueError('Image_shape should not be None for evaluation.')
    if anchor_boxes is None:
      # Generate anchors if needed.
      anchor_boxes = utils.generate_anchors(
          self._min_level,
          self._max_level,
          self._image_size,
          self._anchor_sizes,
      )
      for l in anchor_boxes:
        anchor_boxes[l] = tf.tile(
            tf.expand_dims(anchor_boxes[l], axis=0),
            [tf.shape(image_shape)[0], 1, 1, 1])

    # Generate detected boxes.
    if not self._detection_generator.get_config()['apply_nms']:
      raise ValueError('An NMS algorithm is required for detection generator')
    detections = self._detection_generator(raw_boxes, raw_scores,
                                           anchor_boxes, image_shape,
                                           raw_attributes)
    outputs.update({
        'boxes': detections['detection_boxes'],
        'scores': detections['detection_scores'],
        'classes': detections['detection_classes'],
        'num_detections': detections['num_detections'],
        'attributes': detections['detection_attributes'],
    })
    return outputs

  def call(self,  # pytype: disable=signature-mismatch  # overriding-parameter-count-checks
           pillars: tf.Tensor,
           indices: tf.Tensor,
           image_shape: Optional[tf.Tensor] = None,
           anchor_boxes: Optional[Mapping[str, tf.Tensor]] = None,
           training: bool = None) -> Mapping[str, Any]:
    """Forward pass of the model.

    Notation:
      B: batch size
      H_i: image height at level i
      W_i: image width at level i
      D: number of anchors per location
      C: number of classes to predict
      M: number of detected boxes
      T: attribute size
      P: number of pillars in an example
      N: number of points in a pillar
      D: number of features in a point

    Args:
      pillars: A tensor with shape [B, P, N, D].
      indices: A tensor with shape [B, P, 2].
      image_shape: A tensor with shape [B, 2] representing size of images.
      anchor_boxes: A {level: tensor} dict contains multi level anchor boxes.
        - key: a `str` level.
        - value: a tensor with shape [B, H_i, W_i, 4 * D].
      training: A `bool` indicating whether it's in training mode.

    Returns:
      cls_outputs: A {level: tensor} dict, tensor shape is [B, H_i, W_i, C * D].
      box_outputs: A {level: tensor} dict, tensor shape is [B, H_i, W_i, 4 * D].
      attribute_outputs: A {name: {level: tensor}} dict, tensor shape is
        [B, H_i, W_i, T * D].

      (Below are only for evaluation mode)
      num_detections: A `int` tensor represent number of detected boxes.
      boxes: A tensor with shape [B, M, 4].
      scores: A tensor with shape [B, M].
      classes: A tensor with shape [B, M].
      attributes: A {name: tensor} dict, tensor shape is [B, M, T].

    """
    images = self.featurizer(pillars, indices, training=training)
    features = self.backbone(images)
    features = self.decoder(features)
    raw_scores, raw_boxes, raw_attributes = self.head(features)
    return self.generate_outputs(raw_scores=raw_scores,
                                 raw_boxes=raw_boxes,
                                 raw_attributes=raw_attributes,
                                 image_shape=image_shape,
                                 anchor_boxes=anchor_boxes,
                                 generate_detections=not training)

  @property
  def checkpoint_items(
      self) -> Mapping[str, Union[tf_keras.Model, tf_keras.layers.Layer]]:
    """Returns a dictionary of items to be additionally checkpointed."""
    items = dict(featurizer=self.featurizer,
                 backbone=self.backbone,
                 decoder=self.decoder,
                 head=self.head)
    return items

  @property
  def featurizer(self) -> tf_keras.layers.Layer:
    return self._featurizer

  @property
  def backbone(self) -> tf_keras.Model:
    return self._backbone

  @property
  def decoder(self) -> tf_keras.Model:
    return self._decoder

  @property
  def head(self) -> tf_keras.layers.Layer:
    return self._head

  @property
  def detection_generator(self) -> tf_keras.layers.Layer:
    return self._detection_generator

  def get_config(self) -> Mapping[str, Any]:
    config_dict = {
        'featurizer': self._featurizer,
        'backbone': self._backbone,
        'decoder': self._decoder,
        'head': self._head,
        'detection_generator': self._detection_generator,
        'min_level': self._min_level,
        'max_level': self._max_level,
        'image_size': self._image_size,
        'anchor_sizes': self._anchor_sizes,
    }
    return config_dict

  @classmethod
  def from_config(cls, config: Mapping[str, Any]) -> tf_keras.Model:
    return cls(**config)
