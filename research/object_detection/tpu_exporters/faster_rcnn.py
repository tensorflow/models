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
"""Python library for faster_rcnn model, tailored for TPU inference."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# pylint: disable=protected-access
import tensorflow as tf

# pylint: disable=g-import-not-at-top
# Checking TF version, because this module relies on TPUPartitionedCall
# in tensorflow.python.tpu, which is not available until TF r1.14.
major, minor, _ = tf.__version__.split('.')  # pylint: disable=protected-access
if int(major) < 1 or (int(major == 1) and int(minor) < 14):
  raise RuntimeError(
      'TensorFlow version >= 1.14 is required. Found ({}).'.format(
          tf.__version__))

from tensorflow.python.framework import function
from tensorflow.python.tpu import functional as tpu_functional
from tensorflow.python.tpu.ops import tpu_ops
from object_detection import exporter
from object_detection.builders import model_builder
from object_detection.tpu_exporters import utils

ANCHORS = 'anchors'
BOX_CLASSIFIER_FEATURES = 'box_classifier_features'
BOX_ENCODINGS = 'box_encodings'
CLASS_PREDICTIONS_WITH_BACKGROUND = 'class_predictions_with_background'
IMAGE_SHAPE = 'image_shape'
NUM_PROPOSALS = 'num_proposals'
PROPOSAL_BOXES = 'proposal_boxes'
PROPOSAL_BOXES_NORMALIZED = 'proposal_boxes_normalized'
REFINED_BOX_ENCODINGS = 'refined_box_encodings'
RPN_BOX_ENCODINGS = 'rpn_box_encodings'
RPN_BOX_PREDICTOR_FEATURES = 'rpn_box_predictor_features'
RPN_FEATURES_TO_CROP = 'rpn_features_to_crop'
RPN_OBJECTNESS_PREDICTIONS_WITH_BACKGROUND = \
    'rpn_objectness_predictions_with_background'

INPUT_BUILDER_UTIL_MAP = {
    'model_build': model_builder.build,
}


def modify_config(pipeline_config):
  """Modifies pipeline config to build the correct graph for TPU."""
  # faster_rcnn.use_static_shapes and faster_rcnn.use_static_shapes_for_eval
  # are set to True in order for detection_model.use_static_shapes to be True.
  # We need to set this so that clip_to_window in _predict_first_stage
  # can work on TPU. However as a side-effect, the flag forces the use of
  # padded version of NMS.
  pipeline_config.model.faster_rcnn.use_static_shapes = True
  pipeline_config.model.faster_rcnn.use_static_shapes_for_eval = True
  pipeline_config.model.faster_rcnn.use_matmul_crop_and_resize = True
  pipeline_config.model.faster_rcnn.clip_anchors_to_image = True
  return pipeline_config


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
  pipeline_config = modify_config(pipeline_config)
  detection_model = INPUT_BUILDER_UTIL_MAP['model_build'](
      pipeline_config.model, is_training=False)

  _, input_tensors = exporter.input_placeholder_fn_map['image_tensor']()

  inputs = tf.cast(input_tensors, dtype=tf.float32)
  preprocessed_inputs, true_image_shapes = detection_model.preprocess(inputs)

  prediction_dict = detection_model.predict(preprocessed_inputs,
                                            true_image_shapes)

  shapes_info = {k: v.shape.as_list() for k, v in prediction_dict.items()}
  return shapes_info


def build_graph(pipeline_config,
                shapes_info,
                input_type='encoded_image_string_tensor',
                use_bfloat16=True):
  """Builds serving graph of faster_rcnn to be exported.

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
  pipeline_config = modify_config(pipeline_config)
  detection_model = INPUT_BUILDER_UTIL_MAP['model_build'](
      pipeline_config.model, is_training=False)

  placeholder_tensor, input_tensors = \
      exporter.input_placeholder_fn_map[input_type]()

  # CPU pre-processing
  inputs = tf.cast(input_tensors, dtype=tf.float32)
  preprocessed_inputs, true_image_shapes = detection_model.preprocess(inputs)

  # Dimshuffle: [b, h, w, c] -> [b, c, h, w]
  preprocessed_inputs = tf.transpose(preprocessed_inputs, perm=[0, 3, 1, 2])
  if use_bfloat16:
    preprocessed_inputs = tf.cast(preprocessed_inputs, dtype=tf.bfloat16)

  # TPU feature extraction
  def tpu_subgraph_predict_fn(preprocessed_inputs, true_image_shapes):
    """Defines the first part of graph on TPU."""
    # [b, c, h, w] -> [b, h, w, c]
    preprocessed_inputs = tf.transpose(preprocessed_inputs, perm=[0, 2, 3, 1])

    prediction_dict = detection_model.predict(preprocessed_inputs,
                                              true_image_shapes)

    return (
        # [batch, anchor, depth] -> [depth, batch, anchor]
        tf.transpose(prediction_dict[RPN_BOX_ENCODINGS], perm=[2, 0, 1]),
        # [batch, anchor, depth] -> [depth, batch, anchor]
        tf.transpose(
            prediction_dict[RPN_OBJECTNESS_PREDICTIONS_WITH_BACKGROUND],
            perm=[2, 0, 1]),
        # [anchors, depth]
        tf.transpose(prediction_dict[ANCHORS], perm=[1, 0]),
        # [num_proposals, num_classes, code_size]
        prediction_dict[REFINED_BOX_ENCODINGS],
        prediction_dict[CLASS_PREDICTIONS_WITH_BACKGROUND],
        prediction_dict[NUM_PROPOSALS],
        prediction_dict[PROPOSAL_BOXES])

  @function.Defun(capture_resource_var_by_value=False)
  def tpu_subgraph_predict():
    if use_bfloat16:
      with tf.contrib.tpu.bfloat16_scope():
        return tf.contrib.tpu.rewrite(tpu_subgraph_predict_fn,
                                      [preprocessed_inputs, true_image_shapes])
    else:
      return tf.contrib.tpu.rewrite(tpu_subgraph_predict_fn,
                                    [preprocessed_inputs, true_image_shapes])

  (rpn_box_encodings, rpn_objectness_predictions_with_background, anchors,
   refined_box_encodings, class_predictions_with_background, num_proposals,
   proposal_boxes) = tpu_functional.TPUPartitionedCall(
       args=tpu_subgraph_predict.captured_inputs,
       device_ordinal=tpu_ops.tpu_ordinal_selector(),
       Tout=[
           o.type for o in tpu_subgraph_predict.definition.signature.output_arg
       ],
       f=tpu_subgraph_predict)

  prediction_dict = {
      RPN_BOX_ENCODINGS:
          tf.transpose(rpn_box_encodings, perm=[1, 2, 0]),
      RPN_OBJECTNESS_PREDICTIONS_WITH_BACKGROUND:
          tf.transpose(
              rpn_objectness_predictions_with_background, perm=[1, 2, 0]),
      ANCHORS:
          tf.transpose(anchors, perm=[1, 0]),
      REFINED_BOX_ENCODINGS:
          refined_box_encodings,
      CLASS_PREDICTIONS_WITH_BACKGROUND:
          class_predictions_with_background,
      NUM_PROPOSALS:
          num_proposals,
      PROPOSAL_BOXES:
          proposal_boxes
  }

  for k in prediction_dict:
    prediction_dict[k].set_shape(shapes_info[k])

  if use_bfloat16:
    prediction_dict = utils.bfloat16_to_float32_nested(prediction_dict)

  # CPU post-processing (NMS)
  postprocessed_tensors = detection_model.postprocess(prediction_dict,
                                                      true_image_shapes)
  result_tensor_dict = exporter.add_output_tensor_nodes(postprocessed_tensors,
                                                        'inference_op')

  return placeholder_tensor, result_tensor_dict
