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

"""Library to facilitate TFLite model conversion."""
import functools
from typing import Iterator, List, Optional

from absl import logging
import tensorflow as tf

from official.core import config_definitions as cfg
from official.vision.beta import configs
from official.vision.beta import tasks


def create_representative_dataset(
    params: cfg.ExperimentConfig) -> tf.data.Dataset:
  """Creates a tf.data.Dataset to load images for representative dataset.

  Args:
    params: An ExperimentConfig.

  Returns:
    A tf.data.Dataset instance.

  Raises:
    ValueError: If task is not supported.
  """
  if isinstance(params.task,
                configs.image_classification.ImageClassificationTask):

    task = tasks.image_classification.ImageClassificationTask(params.task)
  elif isinstance(params.task, configs.retinanet.RetinaNetTask):
    task = tasks.retinanet.RetinaNetTask(params.task)
  elif isinstance(params.task, configs.maskrcnn.MaskRCNNTask):
    task = tasks.maskrcnn.MaskRCNNTask(params.task)
  elif isinstance(params.task,
                  configs.semantic_segmentation.SemanticSegmentationTask):
    task = tasks.semantic_segmentation.SemanticSegmentationTask(params.task)
  else:
    raise ValueError('Task {} not supported.'.format(type(params.task)))
  # Ensure batch size is 1 for TFLite model.
  params.task.train_data.global_batch_size = 1
  params.task.train_data.dtype = 'float32'
  logging.info('Task config: %s', params.task.as_dict())
  return task.build_inputs(params=params.task.train_data)


def representative_dataset(
    params: cfg.ExperimentConfig,
    calibration_steps: int = 2000) -> Iterator[List[tf.Tensor]]:
  """"Creates representative dataset for input calibration.

  Args:
    params: An ExperimentConfig.
    calibration_steps: The steps to do calibration.

  Yields:
    An input image tensor.
  """
  dataset = create_representative_dataset(params=params)
  for image, _ in dataset.take(calibration_steps):
    # Skip images that do not have 3 channels.
    if image.shape[-1] != 3:
      continue
    yield [image]


def convert_tflite_model(saved_model_dir: str,
                         quant_type: Optional[str] = None,
                         params: Optional[cfg.ExperimentConfig] = None,
                         calibration_steps: Optional[int] = 2000) -> bytes:
  """Converts and returns a TFLite model.

  Args:
    saved_model_dir: The directory to the SavedModel.
    quant_type: The post training quantization (PTQ) method. It can be one of
      `default` (dynamic range), `fp16` (float16), `int8` (integer wih float
      fallback), `int8_full` (integer only) and None (no quantization).
    params: An optional ExperimentConfig to load and preprocess input images to
      do calibration for integer quantization.
    calibration_steps: The steps to do calibration.

  Returns:
    A converted TFLite model with optional PTQ.

  Raises:
    ValueError: If `representative_dataset_path` is not present if integer
    quantization is requested.
  """
  converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
  if quant_type:
    if quant_type.startswith('int8'):
      converter.optimizations = [tf.lite.Optimize.DEFAULT]
      converter.representative_dataset = functools.partial(
          representative_dataset,
          params=params,
          calibration_steps=calibration_steps)
      if quant_type == 'int8_full':
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS_INT8
        ]
        converter.inference_input_type = tf.uint8  # or tf.int8
        converter.inference_output_type = tf.uint8  # or tf.int8
    elif quant_type == 'fp16':
      converter.optimizations = [tf.lite.Optimize.DEFAULT]
      converter.target_spec.supported_types = [tf.float16]
    elif quant_type == 'default':
      converter.optimizations = [tf.lite.Optimize.DEFAULT]

  return converter.convert()
