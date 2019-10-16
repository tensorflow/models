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
import os
import tempfile
import tensorflow as tf
from tensorflow.contrib.quantize.python import graph_matcher
from tensorflow.core.protobuf import saver_pb2
from tensorflow.python.tools import freeze_graph  # pylint: disable=g-direct-tensorflow-import
from object_detection.builders import graph_rewriter_builder
from object_detection.builders import model_builder
from object_detection.core import standard_fields as fields
from object_detection.data_decoders import tf_example_decoder
from object_detection.utils import config_util
from object_detection.utils import shape_utils

slim = tf.contrib.slim

freeze_graph_with_def_protos = freeze_graph.freeze_graph_with_def_protos


def rewrite_nn_resize_op(is_quantized=False):
  """Replaces a custom nearest-neighbor resize op with the Tensorflow version.

  Some graphs use this custom version for TPU-compatibility.

  Args:
    is_quantized: True if the default graph is quantized.
  """
  def remove_nn():
    """Remove nearest neighbor upsampling structure and replace with TF op."""
    input_pattern = graph_matcher.OpTypePattern(
        'FakeQuantWithMinMaxVars' if is_quantized else '*')
    stack_1_pattern = graph_matcher.OpTypePattern(
        'Pack', inputs=[input_pattern, input_pattern], ordered_inputs=False)
    stack_2_pattern = graph_matcher.OpTypePattern(
        'Pack', inputs=[stack_1_pattern, stack_1_pattern], ordered_inputs=False)
    reshape_pattern = graph_matcher.OpTypePattern(
        'Reshape', inputs=[stack_2_pattern, 'Const'], ordered_inputs=False)
    consumer_pattern = graph_matcher.OpTypePattern(
        'Add|AddV2|Max|Mul', inputs=[reshape_pattern, '*'],
        ordered_inputs=False)

    match_counter = 0
    matcher = graph_matcher.GraphMatcher(consumer_pattern)
    for match in matcher.match_graph(tf.get_default_graph()):
      match_counter += 1
      projection_op = match.get_op(input_pattern)
      reshape_op = match.get_op(reshape_pattern)
      consumer_op = match.get_op(consumer_pattern)
      nn_resize = tf.image.resize_nearest_neighbor(
          projection_op.outputs[0],
          reshape_op.outputs[0].shape.dims[1:3],
          align_corners=False,
          name=os.path.split(reshape_op.name)[0] + '/resize_nearest_neighbor')

      for index, op_input in enumerate(consumer_op.inputs):
        if op_input == reshape_op.outputs[0]:
          consumer_op._update_input(index, nn_resize)  # pylint: disable=protected-access
          break

    tf.logging.info('Found and fixed {} matches'.format(match_counter))
    return match_counter

  # Applying twice because both inputs to Add could be NN pattern
  total_removals = 0
  while remove_nn():
    total_removals += 1
    # This number is chosen based on the nas-fpn architecture.
    if total_removals > 4:
      raise ValueError('Graph removal encountered a infinite loop.')


def replace_variable_values_with_moving_averages(graph,
                                                 current_checkpoint_file,
                                                 new_checkpoint_file,
                                                 no_ema_collection=None):
  """Replaces variable values in the checkpoint with their moving averages.

  If the current checkpoint has shadow variables maintaining moving averages of
  the variables defined in the graph, this function generates a new checkpoint
  where the variables contain the values of their moving averages.

  Args:
    graph: a tf.Graph object.
    current_checkpoint_file: a checkpoint containing both original variables and
      their moving averages.
    new_checkpoint_file: file path to write a new checkpoint.
    no_ema_collection: A list of namescope substrings to match the variables
      to eliminate EMA.
  """
  with graph.as_default():
    variable_averages = tf.train.ExponentialMovingAverage(0.0)
    ema_variables_to_restore = variable_averages.variables_to_restore()
    ema_variables_to_restore = config_util.remove_unecessary_ema(
        ema_variables_to_restore, no_ema_collection)
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
          shape_utils.static_or_dynamic_map_fn(
              decode,
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


def add_output_tensor_nodes(postprocessed_tensors,
                            output_collection_name='inference_op'):
  """Adds output nodes for detection boxes and scores.

  Adds the following nodes for output tensors -
    * num_detections: float32 tensor of shape [batch_size].
    * detection_boxes: float32 tensor of shape [batch_size, num_boxes, 4]
      containing detected boxes.
    * detection_scores: float32 tensor of shape [batch_size, num_boxes]
      containing scores for the detected boxes.
    * detection_multiclass_scores: (Optional) float32 tensor of shape
      [batch_size, num_boxes, num_classes_with_background] for containing class
      score distribution for detected boxes including background if any.
    * detection_features: (Optional) float32 tensor of shape
      [batch, num_boxes, roi_height, roi_width, depth]
      containing classifier features
      for each detected box
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
      'detection_multiclass_scores': [batch, max_detections,
        num_classes_with_background]
      'detection_features': [batch, num_boxes, roi_height, roi_width, depth]
      'detection_classes': [batch, max_detections]
      'detection_masks': [batch, max_detections, mask_height, mask_width]
        (optional).
      'detection_keypoints': [batch, max_detections, num_keypoints, 2]
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
  multiclass_scores = postprocessed_tensors.get(
      detection_fields.detection_multiclass_scores)
  box_classifier_features = postprocessed_tensors.get(
      detection_fields.detection_features)
  raw_boxes = postprocessed_tensors.get(detection_fields.raw_detection_boxes)
  raw_scores = postprocessed_tensors.get(detection_fields.raw_detection_scores)
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
  if multiclass_scores is not None:
    outputs[detection_fields.detection_multiclass_scores] = tf.identity(
        multiclass_scores, name=detection_fields.detection_multiclass_scores)
  if box_classifier_features is not None:
    outputs[detection_fields.detection_features] = tf.identity(
        box_classifier_features,
        name=detection_fields.detection_features)
  outputs[detection_fields.detection_classes] = tf.identity(
      classes, name=detection_fields.detection_classes)
  outputs[detection_fields.num_detections] = tf.identity(
      num_detections, name=detection_fields.num_detections)
  if raw_boxes is not None:
    outputs[detection_fields.raw_detection_boxes] = tf.identity(
        raw_boxes, name=detection_fields.raw_detection_boxes)
  if raw_scores is not None:
    outputs[detection_fields.raw_detection_scores] = tf.identity(
        raw_scores, name=detection_fields.raw_detection_scores)
  if keypoints is not None:
    outputs[detection_fields.detection_keypoints] = tf.identity(
        keypoints, name=detection_fields.detection_keypoints)
  if masks is not None:
    outputs[detection_fields.detection_masks] = tf.identity(
        masks, name=detection_fields.detection_masks)
  for output_key in outputs:
    tf.add_to_collection(output_collection_name, outputs[output_key])

  return outputs


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
    with tf.Session() as sess:

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
              method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME
          ))

      builder.add_meta_graph_and_variables(
          sess,
          [tf.saved_model.tag_constants.SERVING],
          signature_def_map={
              tf.saved_model.signature_constants
              .DEFAULT_SERVING_SIGNATURE_DEF_KEY:
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
    with tf.Session() as sess:
      saver = tf.train.Saver(
          saver_def=input_saver_def, save_relative_paths=True)
      saver.restore(sess, trained_checkpoint_prefix)
      saver.save(sess, model_path)


def _get_outputs_from_inputs(input_tensors, detection_model,
                             output_collection_name):
  inputs = tf.cast(input_tensors, dtype=tf.float32)
  preprocessed_inputs, true_image_shapes = detection_model.preprocess(inputs)
  output_tensors = detection_model.predict(
      preprocessed_inputs, true_image_shapes)
  postprocessed_tensors = detection_model.postprocess(
      output_tensors, true_image_shapes)
  return add_output_tensor_nodes(postprocessed_tensors,
                                 output_collection_name)


def build_detection_graph(input_type, detection_model, input_shape,
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
                            write_inference_graph=False,
                            temp_checkpoint_prefix=''):
  """Export helper."""
  tf.gfile.MakeDirs(output_directory)
  frozen_graph_path = os.path.join(output_directory,
                                   'frozen_inference_graph.pb')
  saved_model_path = os.path.join(output_directory, 'saved_model')
  model_path = os.path.join(output_directory, 'model.ckpt')

  outputs, placeholder_tensor = build_detection_graph(
      input_type=input_type,
      detection_model=detection_model,
      input_shape=input_shape,
      output_collection_name=output_collection_name,
      graph_hook_fn=graph_hook_fn)

  profile_inference_graph(tf.get_default_graph())
  saver_kwargs = {}
  if use_moving_averages:
    if not temp_checkpoint_prefix:
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
    with tf.gfile.GFile(inference_graph_path, 'wb') as f:
      f.write(str(inference_graph_def))

  if additional_output_tensor_names is not None:
    output_node_names = ','.join(outputs.keys()+additional_output_tensor_names)
  else:
    output_node_names = ','.join(outputs.keys())

  frozen_graph_def = freeze_graph.freeze_graph_with_def_protos(
      input_graph_def=tf.get_default_graph().as_graph_def(),
      input_saver_def=input_saver_def,
      input_checkpoint=checkpoint_to_use,
      output_node_names=output_node_names,
      restore_op_name='save/restore_all',
      filename_tensor_name='save/Const:0',
      output_graph=frozen_graph_path,
      clear_devices=True,
      initializer_nodes='')

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
  graph_rewriter_fn = None
  if pipeline_config.HasField('graph_rewriter'):
    graph_rewriter_config = pipeline_config.graph_rewriter
    graph_rewriter_fn = graph_rewriter_builder.build(graph_rewriter_config,
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
      graph_hook_fn=graph_rewriter_fn,
      write_inference_graph=write_inference_graph)
  pipeline_config.eval_config.use_moving_averages = False
  config_util.save_pipeline_config(pipeline_config, output_directory)


def profile_inference_graph(graph):
  """Profiles the inference graph.

  Prints model parameters and computation FLOPs given an inference graph.
  BatchNorms are excluded from the parameter count due to the fact that
  BatchNorms are usually folded. BatchNorm, Initializer, Regularizer
  and BiasAdd are not considered in FLOP count.

  Args:
    graph: the inference graph.
  """
  tfprof_vars_option = (
      tf.contrib.tfprof.model_analyzer.TRAINABLE_VARS_PARAMS_STAT_OPTIONS)
  tfprof_flops_option = tf.contrib.tfprof.model_analyzer.FLOAT_OPS_OPTIONS

  # Batchnorm is usually folded during inference.
  tfprof_vars_option['trim_name_regexes'] = ['.*BatchNorm.*']
  # Initializer and Regularizer are only used in training.
  tfprof_flops_option['trim_name_regexes'] = [
      '.*BatchNorm.*', '.*Initializer.*', '.*Regularizer.*', '.*BiasAdd.*'
  ]

  tf.contrib.tfprof.model_analyzer.print_model_analysis(
      graph,
      tfprof_options=tfprof_vars_option)

  tf.contrib.tfprof.model_analyzer.print_model_analysis(
      graph,
      tfprof_options=tfprof_flops_option)
