# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
"""Python library for ssd model, tailored for TPU inference."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.compat.v1 as tf

# pylint: disable=g-import-not-at-top
# Checking TF version, because this module relies on TPUPartitionedCall
# in tensorflow.python.tpu, which is not available until TF r1.14.
major, minor, _ = tf.__version__.split('.')  # pylint: disable=protected-access
if int(major) < 1 or (int(major == 1) and int(minor) < 14):
  raise RuntimeError(
      'TensorFlow version >= 1.14 is required. Found ({}).'.format(
          tf.__version__))  # pylint: disable=protected-access

from tensorflow.python.framework import function
from tensorflow.python.tpu import functional as tpu_functional
from tensorflow.python.tpu import tpu
from tensorflow.python.tpu.bfloat16 import bfloat16_scope
from tensorflow.python.tpu.ops import tpu_ops
from object_detection import exporter
from object_detection.builders import model_builder
from object_detection.tpu_exporters import utils

ANCHORS = 'anchors'
BOX_ENCODINGS = 'box_encodings'
CLASS_PREDICTIONS_WITH_BACKGROUND = 'class_predictions_with_background'


def get_prediction_tensor_shapes(pipeline_config):
  """Gets static shapes of tensors by building the graph on CPU.

  This function builds the graph on CPU and obtain static shapes of output
  tensors from TPUPartitionedCall. Shapes information are later used for setting
  shapes of tensors when TPU graphs are built. This is necessary because tensors
  coming out of TPUPartitionedCall lose their shape information, which are
  needed for a lot of CPU operations later.
  Args:
    pipeline_config: A TrainEvalPipelineConfig proto.

  Returns:
    A python dict of tensors' names and their shapes.
  """
  detection_model = model_builder.build(
      pipeline_config.model, is_training=False)
  _, input_tensors = exporter.input_placeholder_fn_map['image_tensor']()
  inputs = tf.cast(input_tensors, dtype=tf.float32)
  preprocessed_inputs, true_image_shapes = detection_model.preprocess(inputs)
  prediction_dict = detection_model.predict(preprocessed_inputs,
                                            true_image_shapes)

  return {
      BOX_ENCODINGS:
          prediction_dict[BOX_ENCODINGS].shape.as_list(),
      CLASS_PREDICTIONS_WITH_BACKGROUND:
          prediction_dict[CLASS_PREDICTIONS_WITH_BACKGROUND].shape.as_list(),
      ANCHORS:
          prediction_dict[ANCHORS].shape.as_list(),
  }


def recover_shape(preprocessed_inputs, prediction_outputs, shapes_info):
  """Recovers shape from TPUPartitionedCall.

  Args:
    preprocessed_inputs: 4D tensor, shaped (batch, channels, height, width)
    prediction_outputs: Python list of tensors, in the following order -
      box_encodings - 3D tensor, shaped (code_size, batch, num_anchors);
      class_predictions_with_background - 3D tensor, shaped (num_classes + 1,
      batch, num_anchors); anchors - 2D tensor, shaped (4, num_anchors)
    shapes_info: Python dict of tensor shapes as lists.

  Returns:
    preprocessed_inputs: 4D tensor, shaped (batch, height, width, channels)
    box_encodings: 3D tensor, shaped (batch, num_anchors, code_size)
    class_predictions_with_background: 3D tensor,
        shaped (batch, num_anchors, num_classes + 1)
    anchors: 2D tensor, shaped (num_anchors, 4)
  """
  # Dimshuffle: (b, c, h, w) -> (b, h, w, c)
  preprocessed_inputs = tf.transpose(preprocessed_inputs, perm=[0, 2, 3, 1])

  box_encodings = tf.transpose(prediction_outputs[0], perm=[1, 2, 0])
  # [None, None, detection_model._box_coder.code_size]
  box_encodings.set_shape(shapes_info[BOX_ENCODINGS])

  class_predictions_with_background = tf.transpose(
      prediction_outputs[1], perm=[1, 2, 0])
  # [None, None, num_classes + 1]
  class_predictions_with_background.set_shape(
      shapes_info[CLASS_PREDICTIONS_WITH_BACKGROUND])

  anchors = tf.transpose(prediction_outputs[2], perm=[1, 0])
  # [None, 4]
  anchors.set_shape(shapes_info[ANCHORS])

  return (preprocessed_inputs, box_encodings, class_predictions_with_background,
          anchors)


def build_graph(pipeline_config,
                shapes_info,
                input_type='encoded_image_string_tensor',
                use_bfloat16=False):
  """Builds TPU serving graph of ssd to be exported.

  Args:
    pipeline_config: A TrainEvalPipelineConfig proto.
    shapes_info: A python dict of tensors' names and their shapes, returned by
      `get_prediction_tensor_shapes()`.
    input_type: One of
                'encoded_image_string_tensor': a 1d tensor with dtype=tf.string
                'image_tensor': a 4d tensor with dtype=tf.uint8
                'tf_example': a 1d tensor with dtype=tf.string
    use_bfloat16: If true, use tf.bfloat16 on TPU.

  Returns:
    placeholder_tensor: A placeholder tensor, type determined by `input_type`.
    result_tensor_dict: A python dict of tensors' names and tensors.
  """

  detection_model = model_builder.build(
      pipeline_config.model, is_training=False)

  placeholder_tensor, input_tensors = \
      exporter.input_placeholder_fn_map[input_type]()

  inputs = tf.cast(input_tensors, dtype=tf.float32)
  preprocessed_inputs, true_image_shapes = detection_model.preprocess(inputs)

  # Dimshuffle: (b, h, w, c) -> (b, c, h, w)
  # This is to avoid extra padding due to TPU memory layout:
  # We swap larger dimensions in and smaller dimensions out, so that small
  # dimensions don't get padded tens / hundreds times of its own size.
  # This trick is applied to other similar tensors below.
  preprocessed_inputs = tf.transpose(preprocessed_inputs, perm=[0, 3, 1, 2])
  if use_bfloat16:
    preprocessed_inputs = tf.cast(preprocessed_inputs, dtype=tf.bfloat16)

  def predict_tpu_subgraph(preprocessed_inputs, true_image_shapes):
    """Wraps over the CPU version of `predict()`.

    This builds a same graph as the original `predict()`, manipulates
    result tensors' dimensions to be memory efficient on TPU, and
    returns them as list of tensors.

    Args:
      preprocessed_inputs: A 4D tensor of shape (batch, channels, height, width)
      true_image_shapes: True image shapes tensor.

    Returns:
      A Python list of tensors:
        box_encodings: 3D tensor of shape (code_size, batch_size, num_anchors)
        class_predictions_with_background: 3D tensor,
            shape (num_classes + 1, batch_size, num_anchors)
        anchors: 2D tensor of shape (4, num_anchors)
    """
    # Dimshuffle: (b, c, h, w) -> (b, h, w, c)
    preprocessed_inputs = tf.transpose(preprocessed_inputs, perm=[0, 2, 3, 1])
    if use_bfloat16:
      with bfloat16_scope():
        prediction_dict = detection_model.predict(preprocessed_inputs,
                                                  true_image_shapes)
    else:
      prediction_dict = detection_model.predict(preprocessed_inputs,
                                                true_image_shapes)

    # Dimshuffle: (batch, anchors, depth) -> (depth, batch, anchors)
    return [
        tf.transpose(prediction_dict[BOX_ENCODINGS], perm=[2, 0, 1]),
        tf.transpose(
            prediction_dict[CLASS_PREDICTIONS_WITH_BACKGROUND], perm=[2, 0, 1]),
        tf.transpose(prediction_dict[ANCHORS], perm=[1, 0]),
    ]

  @function.Defun(capture_resource_var_by_value=False)
  def predict_tpu():
    return tpu.rewrite(predict_tpu_subgraph,
                       [preprocessed_inputs, true_image_shapes])

  prediction_outputs = tpu_functional.TPUPartitionedCall(
      args=predict_tpu.captured_inputs,
      device_ordinal=tpu_ops.tpu_ordinal_selector(),
      Tout=[o.type for o in predict_tpu.definition.signature.output_arg],
      f=predict_tpu)

  (preprocessed_inputs, box_encodings, class_predictions_with_background,
   anchors) = recover_shape(preprocessed_inputs, prediction_outputs,
                            shapes_info)

  output_tensors = {
      'preprocessed_inputs': preprocessed_inputs,
      BOX_ENCODINGS: box_encodings,
      CLASS_PREDICTIONS_WITH_BACKGROUND: class_predictions_with_background,
      ANCHORS: anchors,
  }

  if use_bfloat16:
    output_tensors = utils.bfloat16_to_float32_nested(output_tensors)

  postprocessed_tensors = detection_model.postprocess(output_tensors,
                                                      true_image_shapes)
  result_tensor_dict = exporter.add_output_tensor_nodes(postprocessed_tensors,
                                                        'inference_op')

  return placeholder_tensor, result_tensor_dict
