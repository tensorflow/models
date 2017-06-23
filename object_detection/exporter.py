# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
import logging
import os
import tensorflow as tf
from tensorflow.python import pywrap_tensorflow
from tensorflow.python.client import session
from tensorflow.python.framework import graph_util
from tensorflow.python.framework import importer
from tensorflow.python.platform import gfile
from tensorflow.python.training import saver as saver_lib
from object_detection.builders import model_builder
from object_detection.core import standard_fields as fields
from object_detection.data_decoders import tf_example_decoder

slim = tf.contrib.slim


# TODO: Replace with freeze_graph.freeze_graph_with_def_protos when newer
# version of Tensorflow becomes more common.
def freeze_graph_with_def_protos(
    input_graph_def,
    input_saver_def,
    input_checkpoint,
    output_node_names,
    restore_op_name,
    filename_tensor_name,
    output_graph,
    clear_devices,
    initializer_nodes,
    variable_names_blacklist=''):
  """Converts all variables in a graph and checkpoint into constants."""
  del restore_op_name, filename_tensor_name  # Unused by updated loading code.

  # 'input_checkpoint' may be a prefix if we're using Saver V2 format
  if not saver_lib.checkpoint_exists(input_checkpoint):
    logging.info('Input checkpoint "' + input_checkpoint + '" does not exist!')
    return -1

  if not output_node_names:
    logging.info('You must supply the name of a node to --output_node_names.')
    return -1

  # Remove all the explicit device specifications for this node. This helps to
  # make the graph more portable.
  if clear_devices:
    for node in input_graph_def.node:
      node.device = ''

  _ = importer.import_graph_def(input_graph_def, name='')

  with session.Session() as sess:
    if input_saver_def:
      saver = saver_lib.Saver(saver_def=input_saver_def)
      saver.restore(sess, input_checkpoint)
    else:
      var_list = {}
      reader = pywrap_tensorflow.NewCheckpointReader(input_checkpoint)
      var_to_shape_map = reader.get_variable_to_shape_map()
      for key in var_to_shape_map:
        try:
          tensor = sess.graph.get_tensor_by_name(key + ':0')
        except KeyError:
          # This tensor doesn't exist in the graph (for example it's
          # 'global_step' or a similar housekeeping element) so skip it.
          continue
        var_list[key] = tensor
      saver = saver_lib.Saver(var_list=var_list)
      saver.restore(sess, input_checkpoint)
      if initializer_nodes:
        sess.run(initializer_nodes)

    variable_names_blacklist = (variable_names_blacklist.split(',') if
                                variable_names_blacklist else None)
    output_graph_def = graph_util.convert_variables_to_constants(
        sess,
        input_graph_def,
        output_node_names.split(','),
        variable_names_blacklist=variable_names_blacklist)

  with gfile.GFile(output_graph, 'wb') as f:
    f.write(output_graph_def.SerializeToString())
  logging.info('%d ops in the final graph.', len(output_graph_def.node))


# TODO: Support batch tf example inputs.
def _tf_example_input_placeholder():
  tf_example_placeholder = tf.placeholder(
      tf.string, shape=[], name='tf_example')
  tensor_dict = tf_example_decoder.TfExampleDecoder().Decode(
      tf_example_placeholder)
  image = tensor_dict[fields.InputDataFields.image]
  return tf.expand_dims(image, axis=0)


def _image_tensor_input_placeholder():
  return tf.placeholder(dtype=tf.uint8,
                        shape=(1, None, None, 3),
                        name='image_tensor')

input_placeholder_fn_map = {
    'tf_example': _tf_example_input_placeholder,
    'image_tensor': _image_tensor_input_placeholder
}


def _add_output_tensor_nodes(postprocessed_tensors):
  """Adds output nodes for detection boxes and scores.

  Adds the following nodes for output tensors -
    * num_detections: float32 tensor of shape [batch_size].
    * detection_boxes: float32 tensor of shape [batch_size, num_boxes, 4]
      containing detected boxes.
    * detection_scores: float32 tensor of shape [batch_size, num_boxes]
      containing scores for the detected boxes.
    * detection_classes: float32 tensor of shape [batch_size, num_boxes]
      containing class predictions for the detected boxes.

  Args:
    postprocessed_tensors: a dictionary containing the following fields
      'detection_boxes': [batch, max_detections, 4]
      'detection_scores': [batch, max_detections]
      'detection_classes': [batch, max_detections]
      'num_detections': [batch]
  """
  label_id_offset = 1
  boxes = postprocessed_tensors.get('detection_boxes')
  scores = postprocessed_tensors.get('detection_scores')
  classes = postprocessed_tensors.get('detection_classes') + label_id_offset
  num_detections = postprocessed_tensors.get('num_detections')
  tf.identity(boxes, name='detection_boxes')
  tf.identity(scores, name='detection_scores')
  tf.identity(classes, name='detection_classes')
  tf.identity(num_detections, name='num_detections')


def _write_inference_graph(inference_graph_path,
                           checkpoint_path=None,
                           use_moving_averages=False,
                           output_node_names=(
                               'num_detections,detection_scores,'
                               'detection_boxes,detection_classes')):
  """Writes inference graph to disk with the option to bake in weights.

  If checkpoint_path is not None bakes the weights into the graph thereby
  eliminating the need of checkpoint files during inference. If the model
  was trained with moving averages, setting use_moving_averages to true
  restores the moving averages, otherwise the original set of variables
  is restored.

  Args:
    inference_graph_path: Path to write inference graph.
    checkpoint_path: Optional path to the checkpoint file.
    use_moving_averages: Whether to export the original or the moving averages
      of the trainable variables from the checkpoint.
    output_node_names: Output tensor names, defaults are: num_detections,
      detection_scores, detection_boxes, detection_classes.
  """
  inference_graph_def = tf.get_default_graph().as_graph_def()
  if checkpoint_path:
    saver = None
    if use_moving_averages:
      variable_averages = tf.train.ExponentialMovingAverage(0.0)
      variables_to_restore = variable_averages.variables_to_restore()
      saver = tf.train.Saver(variables_to_restore)
    else:
      saver = tf.train.Saver()
    freeze_graph_with_def_protos(
        input_graph_def=inference_graph_def,
        input_saver_def=saver.as_saver_def(),
        input_checkpoint=checkpoint_path,
        output_node_names=output_node_names,
        restore_op_name='save/restore_all',
        filename_tensor_name='save/Const:0',
        output_graph=inference_graph_path,
        clear_devices=True,
        initializer_nodes='')
    return
  tf.train.write_graph(inference_graph_def,
                       os.path.dirname(inference_graph_path),
                       os.path.basename(inference_graph_path),
                       as_text=False)


def _export_inference_graph(input_type,
                            detection_model,
                            use_moving_averages,
                            checkpoint_path,
                            inference_graph_path):
  if input_type not in input_placeholder_fn_map:
    raise ValueError('Unknown input type: {}'.format(input_type))
  inputs = tf.to_float(input_placeholder_fn_map[input_type]())
  preprocessed_inputs = detection_model.preprocess(inputs)
  output_tensors = detection_model.predict(preprocessed_inputs)
  postprocessed_tensors = detection_model.postprocess(output_tensors)
  _add_output_tensor_nodes(postprocessed_tensors)
  _write_inference_graph(inference_graph_path, checkpoint_path,
                         use_moving_averages)


def export_inference_graph(input_type, pipeline_config, checkpoint_path,
                           inference_graph_path):
  """Exports inference graph for the model specified in the pipeline config.

  Args:
    input_type: Type of input for the graph. Can be one of [`image_tensor`,
      `tf_example`].
    pipeline_config: pipeline_pb2.TrainAndEvalPipelineConfig proto.
    checkpoint_path: Path to the checkpoint file to freeze.
    inference_graph_path: Path to write inference graph to.
  """
  detection_model = model_builder.build(pipeline_config.model,
                                        is_training=False)
  _export_inference_graph(input_type, detection_model,
                          pipeline_config.eval_config.use_moving_averages,
                          checkpoint_path, inference_graph_path)
