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

"""Build simclr models."""
from typing import Optional
from absl import logging

import tensorflow as tf

layers = tf.keras.layers

PRETRAIN = 'pretrain'
FINETUNE = 'finetune'

PROJECTION_OUTPUT_KEY = 'projection_outputs'
SUPERVISED_OUTPUT_KEY = 'supervised_outputs'


class SimCLRModel(tf.keras.Model):
  """A classification model based on SimCLR framework."""

  def __init__(self,
               backbone: tf.keras.models.Model,
               projection_head: tf.keras.layers.Layer,
               supervised_head: Optional[tf.keras.layers.Layer] = None,
               input_specs=layers.InputSpec(shape=[None, None, None, 3]),
               mode: str = PRETRAIN,
               backbone_trainable: bool = True,
               **kwargs):
    """A classification model based on SimCLR framework.

    Args:
      backbone: a backbone network.
      projection_head: a projection head network.
      supervised_head: a head network for supervised learning, e.g.
        classification head.
      input_specs: `tf.keras.layers.InputSpec` specs of the input tensor.
      mode: `str` indicates mode of training to be executed.
      backbone_trainable: `bool` whether the backbone is trainable or not.
      **kwargs: keyword arguments to be passed.
    """
    super(SimCLRModel, self).__init__(**kwargs)
    self._config_dict = {
        'backbone': backbone,
        'projection_head': projection_head,
        'supervised_head': supervised_head,
        'input_specs': input_specs,
        'mode': mode,
        'backbone_trainable': backbone_trainable,
    }
    self._input_specs = input_specs
    self._backbone = backbone
    self._projection_head = projection_head
    self._supervised_head = supervised_head
    self._mode = mode
    self._backbone_trainable = backbone_trainable

    # Set whether the backbone is trainable
    self._backbone.trainable = backbone_trainable

  def call(self, inputs, training=None, **kwargs):
    model_outputs = {}

    if training and self._mode == PRETRAIN:
      num_transforms = 2
      # Split channels, and optionally apply extra batched augmentation.
      # (bsz, h, w, c*num_transforms) -> [(bsz, h, w, c), ....]
      features_list = tf.split(
          inputs, num_or_size_splits=num_transforms, axis=-1)
      # (num_transforms * bsz, h, w, c)
      features = tf.concat(features_list, 0)
    else:
      num_transforms = 1
      features = inputs

    # Base network forward pass.
    endpoints = self._backbone(
        features, training=training and self._backbone_trainable)
    features = endpoints[max(endpoints.keys())]
    projection_inputs = layers.GlobalAveragePooling2D()(features)

    # Add heads.
    projection_outputs, supervised_inputs = self._projection_head(
        projection_inputs, training)

    if self._supervised_head is not None:
      if self._mode == PRETRAIN:
        logging.info('Ignoring gradient from supervised outputs !')
        # When performing pretraining and supervised_head together, we do not
        # want information from supervised evaluation flowing back into
        # pretraining network. So we put a stop_gradient.
        supervised_outputs = self._supervised_head(
            tf.stop_gradient(supervised_inputs), training)
      else:
        supervised_outputs = self._supervised_head(supervised_inputs, training)
    else:
      supervised_outputs = None

    model_outputs.update({
        PROJECTION_OUTPUT_KEY: projection_outputs,
        SUPERVISED_OUTPUT_KEY: supervised_outputs
    })

    return model_outputs

  @property
  def checkpoint_items(self):
    """Returns a dictionary of items to be additionally checkpointed."""
    if self._supervised_head is not None:
      items = dict(
          backbone=self.backbone,
          projection_head=self.projection_head,
          supervised_head=self.supervised_head)
    else:
      items = dict(backbone=self.backbone, projection_head=self.projection_head)
    return items

  @property
  def backbone(self):
    return self._backbone

  @property
  def projection_head(self):
    return self._projection_head

  @property
  def supervised_head(self):
    return self._supervised_head

  @property
  def mode(self):
    return self._mode

  @mode.setter
  def mode(self, value):
    self._mode = value

  @property
  def backbone_trainable(self):
    return self._backbone_trainable

  @backbone_trainable.setter
  def backbone_trainable(self, value):
    self._backbone_trainable = value
    self._backbone.trainable = value

  def get_config(self):
    return self._config_dict

  @classmethod
  def from_config(cls, config, custom_objects=None):
    return cls(**config)
