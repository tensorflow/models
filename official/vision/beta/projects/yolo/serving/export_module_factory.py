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

"""Factory for YOLO export modules."""

from typing import List, Optional

import tensorflow as tf

from official.core import config_definitions as cfg
from official.vision import configs
from official.vision.beta.projects.yolo.configs.yolo import YoloTask
from official.vision.beta.projects.yolo.modeling import factory as yolo_factory
from official.vision.beta.projects.yolo.modeling.backbones import darknet  # pylint: disable=unused-import
from official.vision.beta.projects.yolo.modeling.decoders import yolo_decoder  # pylint: disable=unused-import
from official.vision.beta.projects.yolo.serving import model_fn as yolo_model_fn
from official.vision.dataloaders import classification_input
from official.vision.modeling import factory
from official.vision.serving import export_base_v2 as export_base
from official.vision.serving import export_utils


def create_classification_export_module(
    params: cfg.ExperimentConfig,
    input_type: str,
    batch_size: int,
    input_image_size: List[int],
    num_channels: int = 3) -> export_base.ExportModule:
  """Creates classification export module."""
  input_signature = export_utils.get_image_input_signatures(
      input_type, batch_size, input_image_size, num_channels)
  input_specs = tf.keras.layers.InputSpec(shape=[batch_size] +
                                          input_image_size + [num_channels])

  model = factory.build_classification_model(
      input_specs=input_specs,
      model_config=params.task.model,
      l2_regularizer=None)

  def preprocess_fn(inputs):
    image_tensor = export_utils.parse_image(inputs, input_type,
                                            input_image_size, num_channels)
    # If input_type is `tflite`, do not apply image preprocessing.
    if input_type == 'tflite':
      return image_tensor

    def preprocess_image_fn(inputs):
      return classification_input.Parser.inference_fn(inputs, input_image_size,
                                                      num_channels)

    images = tf.map_fn(
        preprocess_image_fn,
        elems=image_tensor,
        fn_output_signature=tf.TensorSpec(
            shape=input_image_size + [num_channels], dtype=tf.float32))

    return images

  def postprocess_fn(logits):
    probs = tf.nn.softmax(logits)
    return {'logits': logits, 'probs': probs}

  export_module = export_base.ExportModule(
      params,
      model=model,
      input_signature=input_signature,
      preprocessor=preprocess_fn,
      postprocessor=postprocess_fn)
  return export_module


def create_yolo_export_module(
    params: cfg.ExperimentConfig,
    input_type: str,
    batch_size: int,
    input_image_size: List[int],
    num_channels: int = 3) -> export_base.ExportModule:
  """Creates YOLO export module."""
  input_signature = export_utils.get_image_input_signatures(
      input_type, batch_size, input_image_size, num_channels)
  input_specs = tf.keras.layers.InputSpec(shape=[batch_size] +
                                          input_image_size + [num_channels])
  model, _ = yolo_factory.build_yolo(
      input_specs=input_specs,
      model_config=params.task.model,
      l2_regularization=None)

  def preprocess_fn(inputs):
    image_tensor = export_utils.parse_image(inputs, input_type,
                                            input_image_size, num_channels)
    # If input_type is `tflite`, do not apply image preprocessing.
    if input_type == 'tflite':
      return image_tensor

    def preprocess_image_fn(inputs):
      image = tf.cast(inputs, dtype=tf.float32)
      image = image / 255.
      (image, image_info) = yolo_model_fn.letterbox(
          image,
          input_image_size,
          letter_box=params.task.validation_data.parser.letter_box)
      return image, image_info

    images_spec = tf.TensorSpec(shape=input_image_size + [3], dtype=tf.float32)

    image_info_spec = tf.TensorSpec(shape=[4, 2], dtype=tf.float32)

    images, image_info = tf.nest.map_structure(
        tf.identity,
        tf.map_fn(
            preprocess_image_fn,
            elems=image_tensor,
            fn_output_signature=(images_spec, image_info_spec),
            parallel_iterations=32))

    return images, image_info

  def inference_steps(inputs, model):
    images, image_info = inputs
    detection = model(images, training=False)
    detection['bbox'] = yolo_model_fn.undo_info(
        detection['bbox'],
        detection['num_detections'],
        image_info,
        expand=False)

    final_outputs = {
        'detection_boxes': detection['bbox'],
        'detection_scores': detection['confidence'],
        'detection_classes': detection['classes'],
        'num_detections': detection['num_detections']
    }

    return final_outputs

  export_module = export_base.ExportModule(
      params,
      model=model,
      input_signature=input_signature,
      preprocessor=preprocess_fn,
      inference_step=inference_steps)

  return export_module


def get_export_module(params: cfg.ExperimentConfig,
                      input_type: str,
                      batch_size: Optional[int],
                      input_image_size: List[int],
                      num_channels: int = 3) -> export_base.ExportModule:
  """Factory for export modules."""
  if isinstance(params.task,
                configs.image_classification.ImageClassificationTask):
    export_module = create_classification_export_module(params, input_type,
                                                        batch_size,
                                                        input_image_size,
                                                        num_channels)
  elif isinstance(params.task, YoloTask):
    export_module = create_yolo_export_module(params, input_type, batch_size,
                                              input_image_size, num_channels)
  else:
    raise ValueError('Export module not implemented for {} task.'.format(
        type(params.task)))
  return export_module
