# Lint as: python2, python3
# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
"""Exports an SSD detection model to use with tf-lite.

See export_tflite_ssd_graph.py for usage.
"""
import os
import tempfile
import numpy as np
import tensorflow as tf
from tensorflow.core.framework import attr_value_pb2
from tensorflow.core.framework import types_pb2
from tensorflow.core.protobuf import saver_pb2
from tensorflow.tools.graph_transforms import TransformGraph
from object_detection import exporter
from object_detection.builders import graph_rewriter_builder
from object_detection.builders import model_builder
from object_detection.builders import post_processing_builder
from object_detection.core import box_list

_DEFAULT_NUM_CHANNELS = 3
_DEFAULT_NUM_COORD_BOX = 4


def get_const_center_size_encoded_anchors(anchors):
  """Exports center-size encoded anchors as a constant tensor.

  Args:
    anchors: a float32 tensor of shape [num_anchors, 4] containing the anchor
      boxes

  Returns:
    encoded_anchors: a float32 constant tensor of shape [num_anchors, 4]
    containing the anchor boxes.
  """
  anchor_boxlist = box_list.BoxList(anchors)
  y, x, h, w = anchor_boxlist.get_center_coordinates_and_sizes()
  num_anchors = y.get_shape().as_list()

  with tf.Session() as sess:
    y_out, x_out, h_out, w_out = sess.run([y, x, h, w])
  encoded_anchors = tf.constant(
      np.transpose(np.stack((y_out, x_out, h_out, w_out))),
      dtype=tf.float32,
      shape=[num_anchors[0], _DEFAULT_NUM_COORD_BOX],
      name='anchors')
  return encoded_anchors


def append_postprocessing_op(frozen_graph_def,
                             max_detections,
                             max_classes_per_detection,
                             nms_score_threshold,
                             nms_iou_threshold,
                             num_classes,
                             scale_values,
                             detections_per_class=100,
                             use_regular_nms=False,
                             additional_output_tensors=()):
  """Appends postprocessing custom op.

  Args:
    frozen_graph_def: Frozen GraphDef for SSD model after freezing the
      checkpoint
    max_detections: Maximum number of detections (boxes) to show
    max_classes_per_detection: Number of classes to display per detection
    nms_score_threshold: Score threshold used in Non-maximal suppression in
      post-processing
    nms_iou_threshold: Intersection-over-union threshold used in Non-maximal
      suppression in post-processing
    num_classes: number of classes in SSD detector
    scale_values: scale values is a dict with following key-value pairs
      {y_scale: 10, x_scale: 10, h_scale: 5, w_scale: 5} that are used in decode
        centersize boxes
    detections_per_class: In regular NonMaxSuppression, number of anchors used
      for NonMaxSuppression per class
    use_regular_nms: Flag to set postprocessing op to use Regular NMS instead of
      Fast NMS.
    additional_output_tensors: Array of additional tensor names to output.
      Tensors are appended after postprocessing output.

  Returns:
    transformed_graph_def: Frozen GraphDef with postprocessing custom op
    appended
    TFLite_Detection_PostProcess custom op node has four outputs:
    detection_boxes: a float32 tensor of shape [1, num_boxes, 4] with box
    locations
    detection_classes: a float32 tensor of shape [1, num_boxes]
    with class indices
    detection_scores: a float32 tensor of shape [1, num_boxes]
    with class scores
    num_boxes: a float32 tensor of size 1 containing the number of detected
    boxes
  """
  new_output = frozen_graph_def.node.add()
  new_output.op = 'TFLite_Detection_PostProcess'
  new_output.name = 'TFLite_Detection_PostProcess'
  new_output.attr['_output_quantized'].CopyFrom(
      attr_value_pb2.AttrValue(b=True))
  new_output.attr['_output_types'].list.type.extend([
      types_pb2.DT_FLOAT, types_pb2.DT_FLOAT, types_pb2.DT_FLOAT,
      types_pb2.DT_FLOAT
  ])
  new_output.attr['_support_output_type_float_in_quantized_op'].CopyFrom(
      attr_value_pb2.AttrValue(b=True))
  new_output.attr['max_detections'].CopyFrom(
      attr_value_pb2.AttrValue(i=max_detections))
  new_output.attr['max_classes_per_detection'].CopyFrom(
      attr_value_pb2.AttrValue(i=max_classes_per_detection))
  new_output.attr['nms_score_threshold'].CopyFrom(
      attr_value_pb2.AttrValue(f=nms_score_threshold.pop()))
  new_output.attr['nms_iou_threshold'].CopyFrom(
      attr_value_pb2.AttrValue(f=nms_iou_threshold.pop()))
  new_output.attr['num_classes'].CopyFrom(
      attr_value_pb2.AttrValue(i=num_classes))

  new_output.attr['y_scale'].CopyFrom(
      attr_value_pb2.AttrValue(f=scale_values['y_scale'].pop()))
  new_output.attr['x_scale'].CopyFrom(
      attr_value_pb2.AttrValue(f=scale_values['x_scale'].pop()))
  new_output.attr['h_scale'].CopyFrom(
      attr_value_pb2.AttrValue(f=scale_values['h_scale'].pop()))
  new_output.attr['w_scale'].CopyFrom(
      attr_value_pb2.AttrValue(f=scale_values['w_scale'].pop()))
  new_output.attr['detections_per_class'].CopyFrom(
      attr_value_pb2.AttrValue(i=detections_per_class))
  new_output.attr['use_regular_nms'].CopyFrom(
      attr_value_pb2.AttrValue(b=use_regular_nms))

  new_output.input.extend(
      ['raw_outputs/box_encodings', 'raw_outputs/class_predictions', 'anchors'])
  # Transform the graph to append new postprocessing op
  input_names = []
  output_names = ['TFLite_Detection_PostProcess'
                 ] + list(additional_output_tensors)
  transforms = ['strip_unused_nodes']
  transformed_graph_def = TransformGraph(frozen_graph_def, input_names,
                                         output_names, transforms)
  return transformed_graph_def


def export_tflite_graph(pipeline_config,
                        trained_checkpoint_prefix,
                        output_dir,
                        add_postprocessing_op,
                        max_detections,
                        max_classes_per_detection,
                        detections_per_class=100,
                        use_regular_nms=False,
                        binary_graph_name='tflite_graph.pb',
                        txt_graph_name='tflite_graph.pbtxt',
                        additional_output_tensors=()):
  """Exports a tflite compatible graph and anchors for ssd detection model.

  Anchors are written to a tensor and tflite compatible graph
  is written to output_dir/tflite_graph.pb.

  Args:
    pipeline_config: a pipeline.proto object containing the configuration for
      SSD model to export.
    trained_checkpoint_prefix: a file prefix for the checkpoint containing the
      trained parameters of the SSD model.
    output_dir: A directory to write the tflite graph and anchor file to.
    add_postprocessing_op: If add_postprocessing_op is true: frozen graph adds a
      TFLite_Detection_PostProcess custom op
    max_detections: Maximum number of detections (boxes) to show
    max_classes_per_detection: Number of classes to display per detection
    detections_per_class: In regular NonMaxSuppression, number of anchors used
      for NonMaxSuppression per class
    use_regular_nms: Flag to set postprocessing op to use Regular NMS instead of
      Fast NMS.
    binary_graph_name: Name of the exported graph file in binary format.
    txt_graph_name: Name of the exported graph file in text format.
    additional_output_tensors: Array of additional tensor names to output.
      Additional tensors are appended to the end of output tensor list.

  Raises:
    ValueError: if the pipeline config contains models other than ssd or uses an
      fixed_shape_resizer and provides a shape as well.
  """
  tf.gfile.MakeDirs(output_dir)
  if pipeline_config.model.WhichOneof('model') != 'ssd':
    raise ValueError('Only ssd models are supported in tflite. '
                     'Found {} in config'.format(
                         pipeline_config.model.WhichOneof('model')))

  num_classes = pipeline_config.model.ssd.num_classes
  nms_score_threshold = {
      pipeline_config.model.ssd.post_processing.batch_non_max_suppression
      .score_threshold
  }
  nms_iou_threshold = {
      pipeline_config.model.ssd.post_processing.batch_non_max_suppression
      .iou_threshold
  }
  scale_values = {}
  scale_values['y_scale'] = {
      pipeline_config.model.ssd.box_coder.faster_rcnn_box_coder.y_scale
  }
  scale_values['x_scale'] = {
      pipeline_config.model.ssd.box_coder.faster_rcnn_box_coder.x_scale
  }
  scale_values['h_scale'] = {
      pipeline_config.model.ssd.box_coder.faster_rcnn_box_coder.height_scale
  }
  scale_values['w_scale'] = {
      pipeline_config.model.ssd.box_coder.faster_rcnn_box_coder.width_scale
  }

  image_resizer_config = pipeline_config.model.ssd.image_resizer
  image_resizer = image_resizer_config.WhichOneof('image_resizer_oneof')
  num_channels = _DEFAULT_NUM_CHANNELS
  if image_resizer == 'fixed_shape_resizer':
    height = image_resizer_config.fixed_shape_resizer.height
    width = image_resizer_config.fixed_shape_resizer.width
    if image_resizer_config.fixed_shape_resizer.convert_to_grayscale:
      num_channels = 1
    shape = [1, height, width, num_channels]
  else:
    raise ValueError(
        'Only fixed_shape_resizer'
        'is supported with tflite. Found {}'.format(
            image_resizer_config.WhichOneof('image_resizer_oneof')))

  image = tf.placeholder(
      tf.float32, shape=shape, name='normalized_input_image_tensor')

  detection_model = model_builder.build(
      pipeline_config.model, is_training=False)
  predicted_tensors = detection_model.predict(image, true_image_shapes=None)
  # The score conversion occurs before the post-processing custom op
  _, score_conversion_fn = post_processing_builder.build(
      pipeline_config.model.ssd.post_processing)
  class_predictions = score_conversion_fn(
      predicted_tensors['class_predictions_with_background'])

  with tf.name_scope('raw_outputs'):
    # 'raw_outputs/box_encodings': a float32 tensor of shape [1, num_anchors, 4]
    #  containing the encoded box predictions. Note that these are raw
    #  predictions and no Non-Max suppression is applied on them and
    #  no decode center size boxes is applied to them.
    tf.identity(predicted_tensors['box_encodings'], name='box_encodings')
    # 'raw_outputs/class_predictions': a float32 tensor of shape
    #  [1, num_anchors, num_classes] containing the class scores for each anchor
    #  after applying score conversion.
    tf.identity(class_predictions, name='class_predictions')
  # 'anchors': a float32 tensor of shape
  #   [4, num_anchors] containing the anchors as a constant node.
  tf.identity(
      get_const_center_size_encoded_anchors(predicted_tensors['anchors']),
      name='anchors')

  # Add global step to the graph, so we know the training step number when we
  # evaluate the model.
  tf.train.get_or_create_global_step()

  # graph rewriter
  is_quantized = pipeline_config.HasField('graph_rewriter')
  if is_quantized:
    graph_rewriter_config = pipeline_config.graph_rewriter
    graph_rewriter_fn = graph_rewriter_builder.build(
        graph_rewriter_config, is_training=False)
    graph_rewriter_fn()

  if pipeline_config.model.ssd.feature_extractor.HasField('fpn'):
    exporter.rewrite_nn_resize_op(is_quantized)

  # freeze the graph
  saver_kwargs = {}
  if pipeline_config.eval_config.use_moving_averages:
    saver_kwargs['write_version'] = saver_pb2.SaverDef.V1
    moving_average_checkpoint = tempfile.NamedTemporaryFile()
    exporter.replace_variable_values_with_moving_averages(
        tf.get_default_graph(), trained_checkpoint_prefix,
        moving_average_checkpoint.name)
    checkpoint_to_use = moving_average_checkpoint.name
  else:
    checkpoint_to_use = trained_checkpoint_prefix

  saver = tf.train.Saver(**saver_kwargs)
  input_saver_def = saver.as_saver_def()
  frozen_graph_def = exporter.freeze_graph_with_def_protos(
      input_graph_def=tf.get_default_graph().as_graph_def(),
      input_saver_def=input_saver_def,
      input_checkpoint=checkpoint_to_use,
      output_node_names=','.join([
          'raw_outputs/box_encodings', 'raw_outputs/class_predictions',
          'anchors'
      ] + list(additional_output_tensors)),
      restore_op_name='save/restore_all',
      filename_tensor_name='save/Const:0',
      clear_devices=True,
      output_graph='',
      initializer_nodes='')

  # Add new operation to do post processing in a custom op (TF Lite only)
  if add_postprocessing_op:
    transformed_graph_def = append_postprocessing_op(
        frozen_graph_def,
        max_detections,
        max_classes_per_detection,
        nms_score_threshold,
        nms_iou_threshold,
        num_classes,
        scale_values,
        detections_per_class,
        use_regular_nms,
        additional_output_tensors=additional_output_tensors)
  else:
    # Return frozen without adding post-processing custom op
    transformed_graph_def = frozen_graph_def

  binary_graph = os.path.join(output_dir, binary_graph_name)
  with tf.gfile.GFile(binary_graph, 'wb') as f:
    f.write(transformed_graph_def.SerializeToString())
  txt_graph = os.path.join(output_dir, txt_graph_name)
  with tf.gfile.GFile(txt_graph, 'w') as f:
    f.write(str(transformed_graph_def))
