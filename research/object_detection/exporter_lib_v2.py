# Lint as: python2, python3
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

"""Functions to export object detection inference graph."""
import os
import tensorflow.compat.v2 as tf
from object_detection.builders import model_builder
from object_detection.core import standard_fields as fields
from object_detection.data_decoders import tf_example_decoder
from object_detection.utils import config_util


def _decode_image(encoded_image_string_tensor):
  image_tensor = tf.image.decode_image(encoded_image_string_tensor,
                                       channels=3)
  image_tensor.set_shape((None, None, 3))
  return image_tensor


def _decode_tf_example(tf_example_string_tensor):
  tensor_dict = tf_example_decoder.TfExampleDecoder().decode(
      tf_example_string_tensor)
  image_tensor = tensor_dict[fields.InputDataFields.image]
  return image_tensor


class DetectionInferenceModule(tf.Module):
  """Detection Inference Module."""

  def __init__(self, detection_model):
    """Initializes a module for detection.

    Args:
      detection_model: The detection model to use for inference.
    """
    self._model = detection_model

  def _run_inference_on_images(self, image):
    """Cast image to float and run inference.

    Args:
      image: uint8 Tensor of shape [1, None, None, 3]
    Returns:
      Tensor dictionary holding detections.
    """
    label_id_offset = 1

    image = tf.cast(image, tf.float32)
    image, shapes = self._model.preprocess(image)
    prediction_dict = self._model.predict(image, shapes)
    detections = self._model.postprocess(prediction_dict, shapes)
    classes_field = fields.DetectionResultFields.detection_classes
    detections[classes_field] = (
        tf.cast(detections[classes_field], tf.float32) + label_id_offset)

    for key, val in detections.items():
      detections[key] = tf.cast(val, tf.float32)

    return detections


class DetectionFromImageModule(DetectionInferenceModule):
  """Detection Inference Module for image inputs."""

  @tf.function(
      input_signature=[
          tf.TensorSpec(shape=[1, None, None, 3], dtype=tf.uint8)])
  def __call__(self, input_tensor):
    return self._run_inference_on_images(input_tensor)


class DetectionFromFloatImageModule(DetectionInferenceModule):
  """Detection Inference Module for float image inputs."""

  @tf.function(
      input_signature=[
          tf.TensorSpec(shape=[1, None, None, 3], dtype=tf.float32)])
  def __call__(self, input_tensor):
    return self._run_inference_on_images(input_tensor)


class DetectionFromEncodedImageModule(DetectionInferenceModule):
  """Detection Inference Module for encoded image string inputs."""

  @tf.function(input_signature=[tf.TensorSpec(shape=[1], dtype=tf.string)])
  def __call__(self, input_tensor):
    with tf.device('cpu:0'):
      image = tf.map_fn(
          _decode_image,
          elems=input_tensor,
          dtype=tf.uint8,
          parallel_iterations=32,
          back_prop=False)
    return self._run_inference_on_images(image)


class DetectionFromTFExampleModule(DetectionInferenceModule):
  """Detection Inference Module for TF.Example inputs."""

  @tf.function(input_signature=[tf.TensorSpec(shape=[1], dtype=tf.string)])
  def __call__(self, input_tensor):
    with tf.device('cpu:0'):
      image = tf.map_fn(
          _decode_tf_example,
          elems=input_tensor,
          dtype=tf.uint8,
          parallel_iterations=32,
          back_prop=False)
    return self._run_inference_on_images(image)

DETECTION_MODULE_MAP = {
    'image_tensor': DetectionFromImageModule,
    'encoded_image_string_tensor':
    DetectionFromEncodedImageModule,
    'tf_example': DetectionFromTFExampleModule,
    'float_image_tensor': DetectionFromFloatImageModule
}


def export_inference_graph(input_type,
                           pipeline_config,
                           trained_checkpoint_dir,
                           output_directory):
  """Exports inference graph for the model specified in the pipeline config.

  This function creates `output_directory` if it does not already exist,
  which will hold a copy of the pipeline config with filename `pipeline.config`,
  and two subdirectories named `checkpoint` and `saved_model`
  (containing the exported checkpoint and SavedModel respectively).

  Args:
    input_type: Type of input for the graph. Can be one of ['image_tensor',
      'encoded_image_string_tensor', 'tf_example'].
    pipeline_config: pipeline_pb2.TrainAndEvalPipelineConfig proto.
    trained_checkpoint_dir: Path to the trained checkpoint file.
    output_directory: Path to write outputs.
  Raises:
    ValueError: if input_type is invalid.
  """
  output_checkpoint_directory = os.path.join(output_directory, 'checkpoint')
  output_saved_model_directory = os.path.join(output_directory, 'saved_model')

  detection_model = model_builder.build(pipeline_config.model,
                                        is_training=False)

  ckpt = tf.train.Checkpoint(
      model=detection_model)
  manager = tf.train.CheckpointManager(
      ckpt, trained_checkpoint_dir, max_to_keep=1)
  status = ckpt.restore(manager.latest_checkpoint).expect_partial()

  if input_type not in DETECTION_MODULE_MAP:
    raise ValueError('Unrecognized `input_type`')
  detection_module = DETECTION_MODULE_MAP[input_type](detection_model)
  # Getting the concrete function traces the graph and forces variables to
  # be constructed --- only after this can we save the checkpoint and
  # saved model.
  concrete_function = detection_module.__call__.get_concrete_function()
  status.assert_existing_objects_matched()

  exported_checkpoint_manager = tf.train.CheckpointManager(
      ckpt, output_checkpoint_directory, max_to_keep=1)
  exported_checkpoint_manager.save(checkpoint_number=0)

  tf.saved_model.save(detection_module,
                      output_saved_model_directory,
                      signatures=concrete_function)

  config_util.save_pipeline_config(pipeline_config, output_directory)
