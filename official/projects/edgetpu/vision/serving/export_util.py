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

"""Implements serving with custom post processing."""

import dataclasses
from typing import List, Optional

import tensorflow as tf
import tensorflow_datasets as tfds

from official.core import exp_factory
from official.core import task_factory
from official.modeling.hyperparams import base_config
# pylint: disable=unused-import
from official.projects.edgetpu.vision.configs import mobilenet_edgetpu_config
from official.projects.edgetpu.vision.configs import semantic_segmentation_config
from official.projects.edgetpu.vision.configs import semantic_segmentation_searched_config
from official.projects.edgetpu.vision.modeling import custom_layers
from official.projects.edgetpu.vision.modeling.backbones import mobilenet_edgetpu
from official.projects.edgetpu.vision.tasks import image_classification
from official.projects.edgetpu.vision.tasks import semantic_segmentation as edgetpu_semantic_segmentation
from official.vision.tasks import semantic_segmentation
# pylint: enable=unused-import

MEAN_RGB = [127.5, 127.5, 127.5]
STDDEV_RGB = [127.5, 127.5, 127.5]


@dataclasses.dataclass
class QuantizationConfig(base_config.Config):
  """Configuration for post training quantization.

  Attributes:
    quantize: Whether to quantize model before exporting tflite.
    quantize_less_restrictive: Allows non int8 based intermediate types,
      automatic model output type.
    use_experimental_quantizer: Enables experimental quantizer of
      TFLiteConverter 2.0.
    num_calibration_steps: Number of post-training quantization calibration
      steps to run.
    dataset_name: Name of the dataset to use for quantization calibration.
    dataset_dir: Dataset location.
    dataset_split: The dataset split (train, validation etc.) to use for
      calibration.
  """
  quantize: bool = False
  quantize_less_restrictive: bool = False
  use_experimental_quantizer: bool = True
  dataset_name: Optional[str] = None
  dataset_dir: Optional[str] = None
  dataset_split: Optional[str] = None
  num_calibration_steps: int = 100


@dataclasses.dataclass
class ExportConfig(base_config.Config):
  """Configuration for exporting models as tflite and saved_models.

  Attributes:
    model_name: One of the registered model names.
    output_layer: Layer name to take the output from. Can be used to take the
      output from an intermediate layer.
    ckpt_path: Path of the training checkpoint. If not provided tflite with
      random parameters is exported.
    ckpt_format: Format of the checkpoint. tf_checkpoint is for ckpt files from
      tf.train.Checkpoint.save() method. keras_checkpoint is for ckpt files from
      keras.Model.save_weights() method
    output_dir: Directory to output exported files.
    image_size: Size of the input image. Ideally should be the same as the
      image_size used in training config
    output_layer: Layer name to take the output from. Can be used to take the
      output from an intermediate layer. None means use the original model
      output.
    finalize_method: 'Additional layers to be added to customize serving output
      Supported are (none|(argmax|resize<?>)[,...]).
      - none: do not add extra serving layers.
      - argmax: adds argmax.
      - squeeze: removes dimensions (except batch dim) of size 1 from the shape
        of a tensor.
      - resize<?> (for example resize512): adds resize bilinear|nn to <?> size.
      For example: --finalize_method=resize128,argmax,resize512,squeeze will do
        resize bilinear to 128x128, then argmax then resize nn to 512x512
  """
  quantization_config: QuantizationConfig = QuantizationConfig()
  model_name: Optional[str] = None
  output_layer: Optional[str] = None
  ckpt_path: Optional[str] = None
  ckpt_format: Optional[str] = 'tf_checkpoint'
  output_dir: str = '/tmp/'
  image_size: int = 224
  output_layer: Optional[str] = None
  finalize_method: Optional[List[str]] = None


def finalize_serving(model_output, export_config):
  """Adds extra layers based on the provided configuration."""

  if isinstance(model_output, dict):
    return {
        key: finalize_serving(model_output[key], export_config)
        for key in model_output
    }

  finalize_method = export_config.finalize_method
  output_layer = model_output
  if not finalize_method or finalize_method[0] == 'none':
    return output_layer
  discrete = False
  for i in range(len(finalize_method)):
    if finalize_method[i] == 'argmax':
      discrete = True
      is_argmax_last = (i + 1) == len(finalize_method)
      if is_argmax_last:
        output_layer = tf.argmax(
            output_layer, axis=3, output_type=tf.dtypes.int32)
      else:
        # TODO(tohaspiridonov): add first_match=False when cl/383951533 submited
        output_layer = custom_layers.argmax(
            output_layer, keepdims=True, epsilon=1e-3)
    elif finalize_method[i] == 'squeeze':
      output_layer = tf.squeeze(output_layer, axis=3)
    else:
      resize_params = finalize_method[i].split('resize')
      if len(resize_params) != 2 or resize_params[0]:
        raise ValueError('Cannot finalize with ' + finalize_method[i] + '.')
      resize_to_size = int(resize_params[1])
      if discrete:
        output_layer = tf.image.resize(
            output_layer, [resize_to_size, resize_to_size],
            method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
      else:
        output_layer = tf.image.resize(
            output_layer, [resize_to_size, resize_to_size],
            method=tf.image.ResizeMethod.BILINEAR)
  return output_layer


def preprocess_for_quantization(image_data, image_size, crop_padding=32):
  """Crops to center of image with padding then scales, normalizes image_size.

  Args:
    image_data: A 3D Tensor representing the RGB image data. Image can be of
      arbitrary height and width.
    image_size: image height/width dimension.
    crop_padding: the padding size to use when centering the crop.

  Returns:
    A decoded and cropped image Tensor. Image is normalized to [-1,1].
  """
  shape = tf.shape(image_data)
  image_height = shape[0]
  image_width = shape[1]

  padded_center_crop_size = tf.cast(
      (image_size * 1.0 / (image_size + crop_padding)) *
      tf.cast(tf.minimum(image_height, image_width), tf.float32), tf.int32)

  offset_height = ((image_height - padded_center_crop_size) + 1) // 2
  offset_width = ((image_width - padded_center_crop_size) + 1) // 2

  image = tf.image.crop_to_bounding_box(
      image_data,
      offset_height=offset_height,
      offset_width=offset_width,
      target_height=padded_center_crop_size,
      target_width=padded_center_crop_size)

  image = tf.image.resize([image], [image_size, image_size],
                          method=tf.image.ResizeMethod.BILINEAR)[0]
  image = tf.cast(image, tf.float32)
  image -= tf.constant(MEAN_RGB)
  image /= tf.constant(STDDEV_RGB)
  return image


def representative_dataset_gen(export_config):
  """Gets a python generator of numpy arrays for the given dataset."""
  quantization_config = export_config.quantization_config
  dataset = tfds.builder(
      quantization_config.dataset_name, try_gcs=True)
  dataset.download_and_prepare()
  data = dataset.as_dataset()[quantization_config.dataset_split]
  iterator = data.as_numpy_iterator()
  for _ in range(quantization_config.num_calibration_steps):
    features = next(iterator)
    image = features['image']
    image = preprocess_for_quantization(image, export_config.image_size)
    image = tf.reshape(
        image, [1, export_config.image_size, export_config.image_size, 3])
    yield [image]


def configure_tflite_converter(export_config, converter):
  """Common code for picking up quantization parameters."""
  quantization_config = export_config.quantization_config
  if quantization_config.quantize:
    if (quantization_config.dataset_dir is
        None) and (quantization_config.dataset_name is None):
      raise ValueError(
          'Must provide a representative dataset when quantizing the model.')
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS_INT8
    ]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    if quantization_config.quantize_less_restrictive:
      converter.target_spec.supported_ops += [
          tf.lite.OpsSet.TFLITE_BUILTINS
      ]
      converter.inference_output_type = tf.float32
    def _representative_dataset_gen():
      return representative_dataset_gen(export_config)

    converter.representative_dataset = _representative_dataset_gen


def build_experiment_model(experiment_type):
  """Builds model from experiment type configuration."""
  params = exp_factory.get_exp_config(experiment_type)
  params.validate()
  params.lock()
  task = task_factory.get_task(params.task)
  return task.build_model()
