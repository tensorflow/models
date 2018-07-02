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
import tempfile
import tensorflow as tf
from tensorflow.core.protobuf import saver_pb2
from tensorflow.python import pywrap_tensorflow
from tensorflow.python.client import session
from tensorflow.python.framework import graph_util
from tensorflow.python.platform import gfile
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.training import saver as saver_lib
from object_detection.builders import model_builder
from object_detection.core import standard_fields as fields
from object_detection.data_decoders import tf_example_decoder
from object_detection.utils import config_util

slim = tf.contrib.slim


# TODO(derekjchow): Replace with freeze_graph.freeze_graph_with_def_protos when
# newer version of Tensorflow becomes more common.
def freeze_graph_with_def_protos(
    input_graph_def,
    input_saver_def,
    input_checkpoint,
    output_node_names,
    restore_op_name,
    filename_tensor_name,
    clear_devices,
    initializer_nodes,
    variable_names_blacklist=''):
  """Converts all variables in a graph and checkpoint into constants."""
  del restore_op_name, filename_tensor_name  # Unused by updated loading code.

  # 'input_checkpoint' may be a prefix if we're using Saver V2 format
  if not saver_lib.checkpoint_exists(input_checkpoint):
    raise ValueError(
        'Input checkpoint "' + input_checkpoint + '" does not exist!')

  if not output_node_names:
    raise ValueError(
        'You must supply the name of a node to --output_node_names.')

  # Remove all the explicit device specifications for this node. This helps to
  # make the graph more portable.
  if clear_devices:
    for node in input_graph_def.node:
      node.device = ''

  with tf.Graph().as_default():
    tf.import_graph_def(input_graph_def, name='')
    config = tf.ConfigProto(graph_options=tf.GraphOptions())
    with session.Session(config=config) as sess:
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

  return output_graph_def


def replace_variable_values_with_moving_averages(graph,
                                                 current_checkpoint_file,
                                                 new_checkpoint_file):
  """Replaces variable values in the checkpoint with their moving averages.

  If the current checkpoint has shadow variables maintaining moving averages of
  the variables defined in the graph, this function generates a new checkpoint
  where the variables contain the values of their moving averages.

  Args:
    graph: a tf.Graph object.
    current_checkpoint_file: a checkpoint containing both original variables and
      their moving averages.
    new_checkpoint_file: file path to write a new checkpoint.
  """
  with graph.as_default():
    variable_averages = tf.train.ExponentialMovingAverage(0.0)
    ema_variables_to_restore = variable_averages.variables_to_restore()
    with tf.Session() as sess:
      read_saver = tf.train.Saver(ema_variables_to_restore)
      read_saver.restore(sess, current_checkpoint_file)
      write_saver = tf.train.Saver()
      write_saver.save(sess, new_checkpoint_file)


def _image_tensor_input_placeholder(input_shape=None):
  """Returns input placeholder and a 4-D uint8 image tensor."""
  if input_shape is None:
    input_shape = (None, None, None, 3)
  input_tensor = tf.placeholder(
      dtype=tf.uint8, shape=input_shape, name='image_tensor')
  return input_tensor, input_tensor


def _tf_example_input_placeholder():
  """Returns input that accepts a batch of strings with tf examples.

  Returns:
    a tuple of input placeholder and the output decoded images.
  """
  batch_tf_example_placeholder = tf.placeholder(
      tf.string, shape=[None], name='tf_example')
  def decode(tf_example_string_tensor):
    tensor_dict = tf_example_decoder.TfExampleDecoder().decode(
        tf_example_string_tensor)
    image_tensor = tensor_dict[fields.InputDataFields.image]
    return image_tensor
  return (batch_tf_example_placeholder,
          tf.map_fn(decode,
                    elems=batch_tf_example_placeholder,
                    dtype=tf.uint8,
                    parallel_iterations=32,
                    back_prop=False))


def _encoded_image_string_tensor_input_placeholder():
  """Returns input that accepts a batch of PNG or JPEG strings.

  Returns:
    a tuple of input placeholder and the output decoded images.
  """
  batch_image_str_placeholder = tf.placeholder(
      dtype=tf.string,
      shape=[None],
      name='encoded_image_string_tensor')
  def decode(encoded_image_string_tensor):
    image_tensor = tf.image.decode_image(encoded_image_string_tensor,
                                         channels=3)
    image_tensor.set_shape((None, None, 3))
    return image_tensor
  return (batch_image_str_placeholder,
          tf.map_fn(
              decode,
              elems=batch_image_str_placeholder,
              dtype=tf.uint8,
              parallel_iterations=32,
              back_prop=False))


input_placeholder_fn_map = {
    'image_tensor': _image_tensor_input_placeholder,
    'encoded_image_string_tensor':
    _encoded_image_string_tensor_input_placeholder,
    'tf_example': _tf_example_input_placeholder,
}


def _add_output_tensor_nodes(postprocessed_tensors,
                             output_collection_name='inference_op'):
  """Adds output nodes for detection boxes and scores.

  Adds the following nodes for output tensors -
    * num_detections: float32 tensor of shape [batch_size].
    * detection_boxes: float32 tensor of shape [batch_size, num_boxes, 4]
      containing detected boxes.
    * detection_scores: float32 tensor of shape [batch_size, num_boxes]
      containing scores for the detected boxes.
    * detection_classes: float32 tensor of shape [batch_size, num_boxes]
      containing class predictions for the detected boxes.
    * detection_keypoints: (Optional) float32 tensor of shape
      [batch_size, num_boxes, num_keypoints, 2] containing keypoints for each
      detection box.
    * detection_masks: (Optional) float32 tensor of shape
      [batch_size, num_boxes, mask_height, mask_width] containing masks for each
      detection box.

  Args:
    postprocessed_tensors: a dictionary containing the following fields
      'detection_boxes': [batch, max_detections, 4]
      'detection_scores': [batch, max_detections]
      'detection_classes': [batch, max_detections]
      'detection_masks': [batch, max_detections, mask_height, mask_width]
        (optional).
      'num_detections': [batch]
    output_collection_name: Name of collection to add output tensors to.

  Returns:
    A tensor dict containing the added output tensor nodes.
  """
  detection_fields = fields.DetectionResultFields
  label_id_offset = 1
  boxes = postprocessed_tensors.get(detection_fields.detection_boxes)
  scores = postprocessed_tensors.get(detection_fields.detection_scores)
  classes = postprocessed_tensors.get(
      detection_fields.detection_classes) + label_id_offset
  keypoints = postprocessed_tensors.get(detection_fields.detection_keypoints)
  masks = postprocessed_tensors.get(detection_fields.detection_masks)
  num_detections = postprocessed_tensors.get(detection_fields.num_detections)
  outputs = {}
  outputs[detection_fields.detection_boxes] = tf.identity(
      boxes, name=detection_fields.detection_boxes)
  outputs[detection_fields.detection_scores] = tf.identity(
      scores, name=detection_fields.detection_scores)
  outputs[detection_fields.detection_classes] = tf.identity(
      classes, name=detection_fields.detection_classes)
  outputs[detection_fields.num_detections] = tf.identity(
      num_detections, name=detection_fields.num_detections)
  if keypoints is not None:
    outputs[detection_fields.detection_keypoints] = tf.identity(
        keypoints, name=detection_fields.detection_keypoints)
  if masks is not None:
    outputs[detection_fields.detection_masks] = tf.identity(
        masks, name=detection_fields.detection_masks)
  for output_key in outputs:
    tf.add_to_collection(output_collection_name, outputs[output_key])

  return outputs


def write_frozen_graph(frozen_graph_path, frozen_graph_def):
  """Writes frozen graph to disk.

  Args:
    frozen_graph_path: Path to write inference graph.
    frozen_graph_def: tf.GraphDef holding frozen graph.
  """
  with gfile.GFile(frozen_graph_path, 'wb') as f:
    f.write(frozen_graph_def.SerializeToString())
  logging.info('%d ops in the final graph.', len(frozen_graph_def.node))


def write_saved_model(saved_model_path,
                      frozen_graph_def,
                      inputs,
                      outputs):
  """Writes SavedModel to disk.

  If checkpoint_path is not None bakes the weights into the graph thereby
  eliminating the need of checkpoint files during inference. If the model
  was trained with moving averages, setting use_moving_averages to true
  restores the moving averages, otherwise the original set of variables
  is restored.

  Args:
    saved_model_path: Path to write SavedModel.
    frozen_graph_def: tf.GraphDef holding frozen graph.
    inputs: The input placeholder tensor.
    outputs: A tensor dictionary containing the outputs of a DetectionModel.
  """
  with tf.Graph().as_default():
    with session.Session() as sess:

      tf.import_graph_def(frozen_graph_def, name='')

      builder = tf.saved_model.builder.SavedModelBuilder(saved_model_path)

      tensor_info_inputs = {
          'inputs': tf.saved_model.utils.build_tensor_info(inputs)}
      tensor_info_outputs = {}
      for k, v in outputs.items():
        tensor_info_outputs[k] = tf.saved_model.utils.build_tensor_info(v)

      detection_signature = (
          tf.saved_model.signature_def_utils.build_signature_def(
              inputs=tensor_info_inputs,
              outputs=tensor_info_outputs,
              method_name=signature_constants.PREDICT_METHOD_NAME))

      builder.add_meta_graph_and_variables(
          sess, [tf.saved_model.tag_constants.SERVING],
          signature_def_map={
              signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
                  detection_signature,
          },
      )
      builder.save()


def write_graph_and_checkpoint(inference_graph_def,
                               model_path,
                               input_saver_def,
                               trained_checkpoint_prefix):
  """Writes the graph and the checkpoint into disk."""
  for node in inference_graph_def.node:
    node.device = ''
  with tf.Graph().as_default():
    tf.import_graph_def(inference_graph_def, name='')
    with session.Session() as sess:
      saver = saver_lib.Saver(saver_def=input_saver_def,
                              save_relative_paths=True)
      saver.restore(sess, trained_checkpoint_prefix)
      saver.save(sess, model_path)


def _get_outputs_from_inputs(input_tensors, detection_model,
                             output_collection_name):
  inputs = tf.to_float(input_tensors)
  preprocessed_inputs, true_image_shapes = detection_model.preprocess(inputs)
  output_tensors = detection_model.predict(
      preprocessed_inputs, true_image_shapes)
  postprocessed_tensors = detection_model.postprocess(
      output_tensors, true_image_shapes)
  return _add_output_tensor_nodes(postprocessed_tensors,
                                  output_collection_name)


def _build_detection_graph(input_type, detection_model, input_shape,
                           output_collection_name, graph_hook_fn):
  """Build the detection graph."""
  if input_type not in input_placeholder_fn_map:
    raise ValueError('Unknown input type: {}'.format(input_type))
  placeholder_args = {}
  if input_shape is not None:
    if input_type != 'image_tensor':
      raise ValueError('Can only specify input shape for `image_tensor` '
                       'inputs.')
    placeholder_args['input_shape'] = input_shape
  placeholder_tensor, input_tensors = input_placeholder_fn_map[input_type](
      **placeholder_args)
  outputs = _get_outputs_from_inputs(
      input_tensors=input_tensors,
      detection_model=detection_model,
      output_collection_name=output_collection_name)

  # Add global step to the graph.
  slim.get_or_create_global_step()

  if graph_hook_fn: graph_hook_fn()

  return outputs, placeholder_tensor


def _export_inference_graph(input_type,
                            detection_model,
                            use_moving_averages,
                            trained_checkpoint_prefix,
                            output_directory,
                            additional_output_tensor_names=None,
                            input_shape=None,
                            output_collection_name='inference_op',
                            graph_hook_fn=None,
                            write_inference_graph=False):
  """Export helper."""
  tf.gfile.MakeDirs(output_directory)
  frozen_graph_path = os.path.join(output_directory,
                                   'frozen_inference_graph.pb')
  saved_model_path = os.path.join(output_directory, 'saved_model')
  model_path = os.path.join(output_directory, 'model.ckpt')

  outputs, placeholder_tensor = _build_detection_graph(
      input_type=input_type,
      detection_model=detection_model,
      input_shape=input_shape,
      output_collection_name=output_collection_name,
      graph_hook_fn=graph_hook_fn)

  saver_kwargs = {}
  if use_moving_averages:
    # This check is to be compatible with both version of SaverDef.
    if os.path.isfile(trained_checkpoint_prefix):
      saver_kwargs['write_version'] = saver_pb2.SaverDef.V1
      temp_checkpoint_prefix = tempfile.NamedTemporaryFile().name
    else:
      temp_checkpoint_prefix = tempfile.mkdtemp()
    replace_variable_values_with_moving_averages(
        tf.get_default_graph(), trained_checkpoint_prefix,
        temp_checkpoint_prefix)
    checkpoint_to_use = temp_checkpoint_prefix
  else:
    checkpoint_to_use = trained_checkpoint_prefix

  saver = tf.train.Saver(**saver_kwargs)
  input_saver_def = saver.as_saver_def()

  write_graph_and_checkpoint(
      inference_graph_def=tf.get_default_graph().as_graph_def(),
      model_path=model_path,
      input_saver_def=input_saver_def,
      trained_checkpoint_prefix=checkpoint_to_use)
  if write_inference_graph:
    inference_graph_def = tf.get_default_graph().as_graph_def()
    inference_graph_path = os.path.join(output_directory,
                                        'inference_graph.pbtxt')
    for node in inference_graph_def.node:
      node.device = ''
    with gfile.GFile(inference_graph_path, 'wb') as f:
      f.write(str(inference_graph_def))

  if additional_output_tensor_names is not None:
    output_node_names = ','.join(outputs.keys()+additional_output_tensor_names)
  else:
    output_node_names = ','.join(outputs.keys())

  frozen_graph_def = freeze_graph_with_def_protos(
      input_graph_def=tf.get_default_graph().as_graph_def(),
      input_saver_def=input_saver_def,
      input_checkpoint=checkpoint_to_use,
      output_node_names=output_node_names,
      restore_op_name='save/restore_all',
      filename_tensor_name='save/Const:0',
      clear_devices=True,
      initializer_nodes='')
  write_frozen_graph(frozen_graph_path, frozen_graph_def)
  write_saved_model(saved_model_path, frozen_graph_def,
                    placeholder_tensor, outputs)


def export_inference_graph(input_type,
                           pipeline_config,
                           trained_checkpoint_prefix,
                           output_directory,
                           input_shape=None,
                           output_collection_name='inference_op',
                           additional_output_tensor_names=None,
                           write_inference_graph=False):
  """Exports inference graph for the model specified in the pipeline config.

  Args:
    input_type: Type of input for the graph. Can be one of ['image_tensor',
      'encoded_image_string_tensor', 'tf_example'].
    pipeline_config: pipeline_pb2.TrainAndEvalPipelineConfig proto.
    trained_checkpoint_prefix: Path to the trained checkpoint file.
    output_directory: Path to write outputs.
    input_shape: Sets a fixed shape for an `image_tensor` input. If not
      specified, will default to [None, None, None, 3].
    output_collection_name: Name of collection to add output tensors to.
      If None, does not add output tensors to a collection.
    additional_output_tensor_names: list of additional output
      tensors to include in the frozen graph.
    write_inference_graph: If true, writes inference graph to disk.
  """
  detection_model = model_builder.build(pipeline_config.model,
                                        is_training=False)
  _export_inference_graph(
      input_type,
      detection_model,
      pipeline_config.eval_config.use_moving_averages,
      trained_checkpoint_prefix,
      output_directory,
      additional_output_tensor_names,
      input_shape,
      output_collection_name,
      graph_hook_fn=None,
      write_inference_graph=write_inference_graph)
  pipeline_config.eval_config.use_moving_averages = False
  config_util.save_pipeline_config(pipeline_config, output_directory)
