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
"""Python library for exporting SavedModel, tailored for TPU inference."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.compat.v1 as tf

from google.protobuf import text_format
# pylint: disable=g-direct-tensorflow-import
from tensorflow.python.saved_model import loader
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.tpu import tpu
# pylint: enable=g-direct-tensorflow-import
from object_detection.protos import pipeline_pb2
from object_detection.tpu_exporters import faster_rcnn
from object_detection.tpu_exporters import ssd

model_map = {
    'faster_rcnn': faster_rcnn,
    'ssd': ssd,
}


def parse_pipeline_config(pipeline_config_file):
  """Returns pipeline config and meta architecture name."""
  with tf.gfile.GFile(pipeline_config_file, 'r') as config_file:
    config_str = config_file.read()
  pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()
  text_format.Merge(config_str, pipeline_config)
  meta_arch = pipeline_config.model.WhichOneof('model')

  return pipeline_config, meta_arch


def export(pipeline_config_file,
           ckpt_path,
           export_dir,
           input_placeholder_name='placeholder_tensor',
           input_type='encoded_image_string_tensor',
           use_bfloat16=False):
  """Exports as SavedModel.

  Args:
    pipeline_config_file: Pipeline config file name.
    ckpt_path: Training checkpoint path.
    export_dir: Directory to export SavedModel.
    input_placeholder_name: input placeholder's name in SavedModel signature.
    input_type: One of
                'encoded_image_string_tensor': a 1d tensor with dtype=tf.string
                'image_tensor': a 4d tensor with dtype=tf.uint8
                'tf_example': a 1d tensor with dtype=tf.string
    use_bfloat16: If true, use tf.bfloat16 on TPU.
  """
  pipeline_config, meta_arch = parse_pipeline_config(pipeline_config_file)

  shapes_info = model_map[meta_arch].get_prediction_tensor_shapes(
      pipeline_config)

  with tf.Graph().as_default(), tf.Session() as sess:
    placeholder_tensor, result_tensor_dict = model_map[meta_arch].build_graph(
        pipeline_config, shapes_info, input_type, use_bfloat16)

    saver = tf.train.Saver()
    init_op = tf.global_variables_initializer()

    sess.run(init_op)
    if ckpt_path is not None:
      saver.restore(sess, ckpt_path)

    # export saved model
    builder = tf.saved_model.builder.SavedModelBuilder(export_dir)
    tensor_info_inputs = {
        input_placeholder_name:
            tf.saved_model.utils.build_tensor_info(placeholder_tensor)
    }
    tensor_info_outputs = {
        k: tf.saved_model.utils.build_tensor_info(v)
        for k, v in result_tensor_dict.items()
    }
    detection_signature = (
        tf.saved_model.signature_def_utils.build_signature_def(
            inputs=tensor_info_inputs,
            outputs=tensor_info_outputs,
            method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME))

    tf.logging.info('Inputs:\n{}\nOutputs:{}\nPredict method name:{}'.format(
        tensor_info_inputs, tensor_info_outputs,
        tf.saved_model.signature_constants.PREDICT_METHOD_NAME))
    # Graph for TPU.
    builder.add_meta_graph_and_variables(
        sess, [
            tf.saved_model.tag_constants.SERVING,
            tf.saved_model.tag_constants.TPU
        ],
        signature_def_map={
            tf.saved_model.signature_constants
            .DEFAULT_SERVING_SIGNATURE_DEF_KEY:
                detection_signature,
        },
        strip_default_attrs=True)
    # Graph for CPU, this is for passing infra validation.
    builder.add_meta_graph(
        [tf.saved_model.tag_constants.SERVING],
        signature_def_map={
            tf.saved_model.signature_constants
            .DEFAULT_SERVING_SIGNATURE_DEF_KEY:
                detection_signature,
        },
        strip_default_attrs=True)
    builder.save(as_text=False)
    tf.logging.info('Model saved to {}'.format(export_dir))


def run_inference(inputs,
                  pipeline_config_file,
                  ckpt_path,
                  input_type='encoded_image_string_tensor',
                  use_bfloat16=False,
                  repeat=1):
  """Runs inference on TPU.

  Args:
    inputs: Input image with the same type as `input_type`
    pipeline_config_file: Pipeline config file name.
    ckpt_path: Training checkpoint path.
    input_type: One of
                'encoded_image_string_tensor': a 1d tensor with dtype=tf.string
                'image_tensor': a 4d tensor with dtype=tf.uint8
                'tf_example': a 1d tensor with dtype=tf.string
    use_bfloat16: If true, use tf.bfloat16 on TPU.
    repeat: Number of times to repeat running the provided input for profiling.

  Returns:
    A dict of resulting tensors.
  """

  pipeline_config, meta_arch = parse_pipeline_config(pipeline_config_file)

  shapes_info = model_map[meta_arch].get_prediction_tensor_shapes(
      pipeline_config)

  with tf.Graph().as_default(), tf.Session() as sess:
    placeholder_tensor, result_tensor_dict = model_map[meta_arch].build_graph(
        pipeline_config, shapes_info, input_type, use_bfloat16)

    saver = tf.train.Saver()
    init_op = tf.global_variables_initializer()

    sess.run(tpu.initialize_system())

    sess.run(init_op)
    if ckpt_path is not None:
      saver.restore(sess, ckpt_path)

    for _ in range(repeat):
      tensor_dict_out = sess.run(
          result_tensor_dict, feed_dict={placeholder_tensor: [inputs]})

    sess.run(tpu.shutdown_system())

    return tensor_dict_out


def run_inference_from_saved_model(inputs,
                                   saved_model_dir,
                                   input_placeholder_name='placeholder_tensor',
                                   repeat=1):
  """Loads saved model and run inference on TPU.

  Args:
    inputs: Input image with the same type as `input_type`
    saved_model_dir: The directory SavedModel being exported to.
    input_placeholder_name: input placeholder's name in SavedModel signature.
    repeat: Number of times to repeat running the provided input for profiling.

  Returns:
    A dict of resulting tensors.
  """
  with tf.Graph().as_default(), tf.Session() as sess:
    meta_graph = loader.load(sess, [tag_constants.SERVING, tag_constants.TPU],
                             saved_model_dir)

    sess.run(tpu.initialize_system())

    key_prediction = signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY

    tensor_name_input = (
        meta_graph.signature_def[key_prediction].inputs[input_placeholder_name]
        .name)
    tensor_name_output = {
        k: v.name
        for k, v in (meta_graph.signature_def[key_prediction].outputs.items())
    }

    for _ in range(repeat):
      tensor_dict_out = sess.run(
          tensor_name_output, feed_dict={tensor_name_input: [inputs]})

    sess.run(tpu.shutdown_system())

    return tensor_dict_out
