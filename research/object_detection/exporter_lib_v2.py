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
import ast
import os

import tensorflow.compat.v2 as tf
from object_detection.builders import model_builder
from object_detection.core import standard_fields as fields
from object_detection.data_decoders import tf_example_decoder
from object_detection.utils import config_util


INPUT_BUILDER_UTIL_MAP = {
    'model_build': model_builder.build,
}


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


def _combine_side_inputs(side_input_shapes='',
                         side_input_types='',
                         side_input_names=''):
  """Zips the side inputs together.

  Args:
    side_input_shapes: forward-slash-separated list of comma-separated lists
      describing input shapes.
    side_input_types: comma-separated list of the types of the inputs.
    side_input_names: comma-separated list of the names of the inputs.

  Returns:
    a zipped list of side input tuples.
  """
  side_input_shapes = [
      ast.literal_eval('[' + x + ']') for x in side_input_shapes.split('/')
  ]
  side_input_types = eval('[' + side_input_types + ']')  # pylint: disable=eval-used
  side_input_names = side_input_names.split(',')
  return zip(side_input_shapes, side_input_types, side_input_names)


class DetectionInferenceModule(tf.Module):
  """Detection Inference Module."""

  def __init__(self, detection_model,
               use_side_inputs=False,
               zipped_side_inputs=None):
    """Initializes a module for detection.

    Args:
      detection_model: the detection model to use for inference.
      use_side_inputs: whether to use side inputs.
      zipped_side_inputs: the zipped side inputs.
    """
    self._model = detection_model

  def _get_side_input_signature(self, zipped_side_inputs):
    sig = []
    side_input_names = []
    for info in zipped_side_inputs:
      sig.append(tf.TensorSpec(shape=info[0],
                               dtype=info[1],
                               name=info[2]))
      side_input_names.append(info[2])
    return sig

  def _get_side_names_from_zip(self, zipped_side_inputs):
    return [side[2] for side in zipped_side_inputs]

  def _run_inference_on_images(self, image, **kwargs):
    """Cast image to float and run inference.

    Args:
      image: uint8 Tensor of shape [1, None, None, 3].
      **kwargs: additional keyword arguments.

    Returns:
      Tensor dictionary holding detections.
    """
    label_id_offset = 1

    image = tf.cast(image, tf.float32)
    image, shapes = self._model.preprocess(image)
    prediction_dict = self._model.predict(image, shapes, **kwargs)
    detections = self._model.postprocess(prediction_dict, shapes)
    classes_field = fields.DetectionResultFields.detection_classes
    detections[classes_field] = (
        tf.cast(detections[classes_field], tf.float32) + label_id_offset)

    for key, val in detections.items():
      detections[key] = tf.cast(val, tf.float32)

    return detections


class DetectionFromImageModule(DetectionInferenceModule):
  """Detection Inference Module for image inputs."""

  def __init__(self, detection_model,
               use_side_inputs=False,
               zipped_side_inputs=None):
    """Initializes a module for detection.

    Args:
      detection_model: the detection model to use for inference.
      use_side_inputs: whether to use side inputs.
      zipped_side_inputs: the zipped side inputs.
    """
    if zipped_side_inputs is None:
      zipped_side_inputs = []
    sig = [tf.TensorSpec(shape=[1, None, None, 3],
                         dtype=tf.uint8,
                         name='input_tensor')]
    if use_side_inputs:
      sig.extend(self._get_side_input_signature(zipped_side_inputs))
    self._side_input_names = self._get_side_names_from_zip(zipped_side_inputs)

    def call_func(input_tensor, *side_inputs):
      kwargs = dict(zip(self._side_input_names, side_inputs))
      return self._run_inference_on_images(input_tensor, **kwargs)

    self.__call__ = tf.function(call_func, input_signature=sig)

    # TODO(kaushikshiv): Check if omitting the signature also works.
    super(DetectionFromImageModule, self).__init__(detection_model,
                                                   use_side_inputs,
                                                   zipped_side_inputs)


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
                           output_directory,
                           use_side_inputs=False,
                           side_input_shapes='',
                           side_input_types='',
                           side_input_names=''):
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
    use_side_inputs: boolean that determines whether side inputs should be
      included in the input signature.
    side_input_shapes: forward-slash-separated list of comma-separated lists
        describing input shapes.
    side_input_types: comma-separated list of the types of the inputs.
    side_input_names: comma-separated list of the names of the inputs.
  Raises:
    ValueError: if input_type is invalid.
  """
  output_checkpoint_directory = os.path.join(output_directory, 'checkpoint')
  output_saved_model_directory = os.path.join(output_directory, 'saved_model')

  detection_model = INPUT_BUILDER_UTIL_MAP['model_build'](
      pipeline_config.model, is_training=False)

  ckpt = tf.train.Checkpoint(
      model=detection_model)
  manager = tf.train.CheckpointManager(
      ckpt, trained_checkpoint_dir, max_to_keep=1)
  status = ckpt.restore(manager.latest_checkpoint).expect_partial()

  if input_type not in DETECTION_MODULE_MAP:
    raise ValueError('Unrecognized `input_type`')
  if use_side_inputs and input_type != 'image_tensor':
    raise ValueError('Side inputs supported for image_tensor input type only.')

  zipped_side_inputs = []
  if use_side_inputs:
    zipped_side_inputs = _combine_side_inputs(side_input_shapes,
                                              side_input_types,
                                              side_input_names)

  detection_module = DETECTION_MODULE_MAP[input_type](detection_model,
                                                      use_side_inputs,
                                                      list(zipped_side_inputs))
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
