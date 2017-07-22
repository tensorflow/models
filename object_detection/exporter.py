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
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.training import saver as saver_lib
from object_detection.builders import model_builder
from object_detection.core import standard_fields as fields
from object_detection.data_decoders import tf_example_decoder

slim = tf.contrib.slim


# TODO: Replace with freeze_graph.freeze_graph_with_def_protos when
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

  return output_graph_def


def get_frozen_graph_def(inference_graph_def, use_moving_averages,
                         input_checkpoint, output_node_names):
  """Freezes all variables in a graph definition."""
  saver = None
  if use_moving_averages:
    variable_averages = tf.train.ExponentialMovingAverage(0.0)
    variables_to_restore = variable_averages.variables_to_restore()
    saver = tf.train.Saver(variables_to_restore)
  else:
    saver = tf.train.Saver()

  frozen_graph_def = freeze_graph_with_def_protos(
      input_graph_def=inference_graph_def,
      input_saver_def=saver.as_saver_def(),
      input_checkpoint=input_checkpoint,
      output_node_names=output_node_names,
      restore_op_name='save/restore_all',
      filename_tensor_name='save/Const:0',
      clear_devices=True,
      initializer_nodes='')
  return frozen_graph_def


# TODO: Support batch tf example inputs.
def _tf_example_input_placeholder():
  tf_example_placeholder = tf.placeholder(
      tf.string, shape=[], name='tf_example')
  tensor_dict = tf_example_decoder.TfExampleDecoder().decode(
      tf_example_placeholder)
  image = tensor_dict[fields.InputDataFields.image]
  return tf.expand_dims(image, axis=0)


def _image_tensor_input_placeholder():
  return tf.placeholder(dtype=tf.uint8,
                        shape=(1, None, None, 3),
                        name='image_tensor')


def _encoded_image_string_tensor_input_placeholder():
  image_str = tf.placeholder(dtype=tf.string,
                             shape=[],
                             name='encoded_image_string_tensor')
  image_tensor = tf.image.decode_image(image_str, channels=3)
  image_tensor.set_shape((None, None, 3))
  return tf.expand_dims(image_tensor, axis=0)


input_placeholder_fn_map = {
    'image_tensor': _image_tensor_input_placeholder,
    'encoded_image_string_tensor':
    _encoded_image_string_tensor_input_placeholder,
    'tf_example': _tf_example_input_placeholder,
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

  Returns:
    A tensor dict containing the added output tensor nodes.
  """
  label_id_offset = 1
  boxes = postprocessed_tensors.get('detection_boxes')
  scores = postprocessed_tensors.get('detection_scores')
  classes = postprocessed_tensors.get('detection_classes') + label_id_offset
  masks = postprocessed_tensors.get('detection_masks')
  num_detections = postprocessed_tensors.get('num_detections')
  outputs = {}
  outputs['detection_boxes'] = tf.identity(boxes, name='detection_boxes')
  outputs['detection_scores'] = tf.identity(scores, name='detection_scores')
  outputs['detection_classes'] = tf.identity(classes, name='detection_classes')
  outputs['num_detections'] = tf.identity(num_detections, name='num_detections')
  if masks is not None:
    outputs['detection_masks'] = tf.identity(masks, name='detection_masks')
  return outputs


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
    output_graph_def = get_frozen_graph_def(
        inference_graph_def=inference_graph_def,
        use_moving_averages=use_moving_averages,
        input_checkpoint=checkpoint_path,
        output_node_names=output_node_names,
    )

    with gfile.GFile(inference_graph_path, 'wb') as f:
      f.write(output_graph_def.SerializeToString())
    logging.info('%d ops in the final graph.', len(output_graph_def.node))

    return
  tf.train.write_graph(inference_graph_def,
                       os.path.dirname(inference_graph_path),
                       os.path.basename(inference_graph_path),
                       as_text=False)


def _write_saved_model(inference_graph_path, inputs, outputs,
                       checkpoint_path=None, use_moving_averages=False):
  """Writes SavedModel to disk.

  If checkpoint_path is not None bakes the weights into the graph thereby
  eliminating the need of checkpoint files during inference. If the model
  was trained with moving averages, setting use_moving_averages to true
  restores the moving averages, otherwise the original set of variables
  is restored.

  Args:
    inference_graph_path: Path to write inference graph.
    inputs: The input image tensor to use for detection.
    outputs: A tensor dictionary containing the outputs of a DetectionModel.
    checkpoint_path: Optional path to the checkpoint file.
    use_moving_averages: Whether to export the original or the moving averages
      of the trainable variables from the checkpoint.
  """
  inference_graph_def = tf.get_default_graph().as_graph_def()
  checkpoint_graph_def = None
  if checkpoint_path:
    output_node_names = ','.join(outputs.keys())
    checkpoint_graph_def = get_frozen_graph_def(
        inference_graph_def=inference_graph_def,
        use_moving_averages=use_moving_averages,
        input_checkpoint=checkpoint_path,
        output_node_names=output_node_names
    )

  with tf.Graph().as_default():
    with session.Session() as sess:

      tf.import_graph_def(checkpoint_graph_def)

      builder = tf.saved_model.builder.SavedModelBuilder(inference_graph_path)

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


def _export_inference_graph(input_type,
                            detection_model,
                            use_moving_averages,
                            checkpoint_path,
                            inference_graph_path,
                            export_as_saved_model=False):
  """Export helper."""
  if input_type not in input_placeholder_fn_map:
    raise ValueError('Unknown input type: {}'.format(input_type))
  inputs = tf.to_float(input_placeholder_fn_map[input_type]())
  preprocessed_inputs = detection_model.preprocess(inputs)
  output_tensors = detection_model.predict(preprocessed_inputs)
  postprocessed_tensors = detection_model.postprocess(output_tensors)
  outputs = _add_output_tensor_nodes(postprocessed_tensors)
  out_node_names = list(outputs.keys())
  if export_as_saved_model:
    _write_saved_model(inference_graph_path, inputs, outputs, checkpoint_path,
                       use_moving_averages)
  else:
    _write_inference_graph(inference_graph_path, checkpoint_path,
                           use_moving_averages,
                           output_node_names=','.join(out_node_names))


def export_inference_graph(input_type, pipeline_config, checkpoint_path,
                           inference_graph_path, export_as_saved_model=False):
  """Exports inference graph for the model specified in the pipeline config.

  Args:
    input_type: Type of input for the graph. Can be one of [`image_tensor`,
      `tf_example`].
    pipeline_config: pipeline_pb2.TrainAndEvalPipelineConfig proto.
    checkpoint_path: Path to the checkpoint file to freeze.
    inference_graph_path: Path to write inference graph to.
    export_as_saved_model: If the model should be exported as a SavedModel. If
                           false, it is saved as an inference graph.
  """
  detection_model = model_builder.build(pipeline_config.model,
                                        is_training=False)
  _export_inference_graph(input_type, detection_model,
                          pipeline_config.eval_config.use_moving_averages,
                          checkpoint_path, inference_graph_path,
                          export_as_saved_model)
