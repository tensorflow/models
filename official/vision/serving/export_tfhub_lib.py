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

"""A script to export a TF-Hub SavedModel."""
from typing import List, Optional

# Import libraries

import tensorflow as tf

from official.core import config_definitions as cfg
from official.vision import configs
from official.vision.modeling import factory


def build_model(batch_size: Optional[int],
                input_image_size: List[int],
                params: cfg.ExperimentConfig,
                num_channels: int = 3,
                skip_logits_layer: bool = False) -> tf.keras.Model:
  """Builds a model for TF Hub export.

  Args:
    batch_size: The batch size of input.
    input_image_size: A list of [height, width] specifying the input image size.
    params: The config used to train the model.
    num_channels: The number of input image channels.
    skip_logits_layer: Whether to skip the logits layer for image classification
      model. Default is False.

  Returns:
    A tf.keras.Model instance.

  Raises:
    ValueError: If the task is not supported.
  """
  input_specs = tf.keras.layers.InputSpec(shape=[batch_size] +
                                          input_image_size + [num_channels])
  if isinstance(params.task,
                configs.image_classification.ImageClassificationTask):
    model = factory.build_classification_model(
        input_specs=input_specs,
        model_config=params.task.model,
        l2_regularizer=None,
        skip_logits_layer=skip_logits_layer)
  else:
    raise ValueError('Export module not implemented for {} task.'.format(
        type(params.task)))
  return model


def export_model_to_tfhub(batch_size: Optional[int],
                          input_image_size: List[int],
                          params: cfg.ExperimentConfig,
                          checkpoint_path: str,
                          export_path: str,
                          num_channels: int = 3,
                          skip_logits_layer: bool = False):
  """Export a TF2 model to TF-Hub."""
  model = build_model(batch_size, input_image_size, params, num_channels,
                      skip_logits_layer)
  checkpoint = tf.train.Checkpoint(model=model)
  checkpoint.restore(checkpoint_path).assert_existing_objects_matched()
  model.save(export_path, include_optimizer=False, save_format='tf')
