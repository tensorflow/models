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

"""Factory for YOLO export modules."""

from typing import Any, Callable, Dict, List, Optional, Text, Union

import tensorflow as tf, tf_keras

from official.core import config_definitions as cfg
from official.core import export_base
from official.projects.yolo.configs import darknet_classification
from official.projects.yolo.configs import yolo
from official.projects.yolo.configs import yolov7
from official.projects.yolo.dataloaders import classification_input
from official.projects.yolo.modeling import factory as yolo_factory
from official.projects.yolo.modeling.backbones import darknet  # pylint: disable=unused-import
from official.projects.yolo.modeling.decoders import yolo_decoder  # pylint: disable=unused-import
from official.projects.yolo.serving import model_fn as yolo_model_fn
from official.vision.modeling import factory
from official.vision.serving import export_utils


class ExportModule(export_base.ExportModule):
  """Base Export Module."""

  def __init__(self,
               params: cfg.ExperimentConfig,
               model: tf_keras.Model,
               input_signature: Union[tf.TensorSpec, Dict[str, tf.TensorSpec]],
               preprocessor: Optional[Callable[..., Any]] = None,
               inference_step: Optional[Callable[..., Any]] = None,
               postprocessor: Optional[Callable[..., Any]] = None,
               eval_postprocessor: Optional[Callable[..., Any]] = None):
    """Initializes a module for export.

    Args:
      params: A dataclass for parameters to the module.
      model: A tf_keras.Model instance to be exported.
      input_signature: tf.TensorSpec, e.g. tf.TensorSpec(shape=[None, 224, 224,
        3], dtype=tf.uint8)
      preprocessor: An optional callable to preprocess the inputs.
      inference_step: An optional callable to forward-pass the model.
      postprocessor: An optional callable to postprocess the model outputs.
      eval_postprocessor: An optional callable to postprocess model outputs used
        for model evaluation.
    """
    super().__init__(
        params,
        model=model,
        preprocessor=preprocessor,
        inference_step=inference_step,
        postprocessor=postprocessor)
    self.eval_postprocessor = eval_postprocessor
    self.input_signature = input_signature

  @tf.function
  def serve(self, inputs: Any) -> Any:
    x = self.preprocessor(inputs=inputs) if self.preprocessor else inputs
    x = self.inference_step(x)
    x = self.postprocessor(x) if self.postprocessor else x
    return x

  @tf.function
  def serve_eval(self, inputs: Any) -> Any:
    x = self.preprocessor(inputs=inputs) if self.preprocessor else inputs
    x = self.inference_step(x)
    x = self.eval_postprocessor(x) if self.eval_postprocessor else x
    return x

  def get_inference_signatures(
      self, function_keys: Dict[Text, Text]):
    """Gets defined function signatures.

    Args:
      function_keys: A dictionary with keys as the function to create signature
        for and values as the signature keys when returns.

    Returns:
      A dictionary with key as signature key and value as concrete functions
        that can be used for tf.saved_model.save.
    """
    signatures = {}
    for _, def_name in function_keys.items():
      if 'eval' in def_name and self.eval_postprocessor:
        signatures[def_name] = self.serve_eval.get_concrete_function(
            self.input_signature)
      else:
        signatures[def_name] = self.serve.get_concrete_function(
            self.input_signature)
    return signatures


def create_classification_export_module(
    params: cfg.ExperimentConfig,
    input_type: str,
    batch_size: int,
    input_image_size: List[int],
    num_channels: int = 3,
    input_name: Optional[str] = None) -> ExportModule:
  """Creates classification export module."""
  input_signature = export_utils.get_image_input_signatures(
      input_type, batch_size, input_image_size, num_channels, input_name)
  input_specs = tf_keras.layers.InputSpec(shape=[batch_size] +
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

  export_module = ExportModule(
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
    num_channels: int = 3,
    input_name: Optional[str] = None) -> ExportModule:
  """Creates YOLO export module."""
  input_signature = export_utils.get_image_input_signatures(
      input_type, batch_size, input_image_size, num_channels, input_name)
  input_specs = tf_keras.layers.InputSpec(shape=[batch_size] +
                                          input_image_size + [num_channels])
  if isinstance(params.task, yolo.YoloTask):
    model, _ = yolo_factory.build_yolo(
        input_specs=input_specs,
        model_config=params.task.model,
        l2_regularization=None)
  elif isinstance(params.task, yolov7.YoloV7Task):
    model = yolo_factory.build_yolov7(
        input_specs=input_specs,
        model_config=params.task.model,
        l2_regularization=None)

  def preprocess_fn(inputs):
    image_tensor = export_utils.parse_image(inputs, input_type,
                                            input_image_size, num_channels)

    def normalize_image_fn(inputs):
      image = tf.cast(inputs, dtype=tf.float32)
      return image / 255.0

    # If input_type is `tflite`, do not apply image preprocessing. Only apply
    # normalization.
    if input_type == 'tflite':
      return normalize_image_fn(image_tensor), None

    def preprocess_image_fn(inputs):
      image = normalize_image_fn(inputs)
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
    detection = model.call(images, training=False)
    if input_type != 'tflite':
      detection['bbox'] = yolo_model_fn.undo_info(
          detection['bbox'],
          detection['num_detections'],
          image_info,
          expand=False,
      )

    final_outputs = {
        'detection_boxes': detection['bbox'],
        'detection_scores': detection['confidence'],
        'detection_classes': detection['classes'],
        'num_detections': detection['num_detections']
    }

    return final_outputs

  export_module = ExportModule(
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
                      num_channels: int = 3,
                      input_name: Optional[str] = None) -> ExportModule:
  """Factory for export modules."""
  if isinstance(params.task,
                darknet_classification.ImageClassificationTask):
    export_module = create_classification_export_module(params, input_type,
                                                        batch_size,
                                                        input_image_size,
                                                        num_channels,
                                                        input_name)
  elif isinstance(params.task, (yolo.YoloTask, yolov7.YoloV7Task)):
    export_module = create_yolo_export_module(params, input_type, batch_size,
                                              input_image_size, num_channels,
                                              input_name)
  else:
    raise ValueError('Export module not implemented for {} task.'.format(
        type(params.task)))
  return export_module
