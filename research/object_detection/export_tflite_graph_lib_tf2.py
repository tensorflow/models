# Lint as: python3
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
"""Library to export TFLite-compatible SavedModel from TF2 detection models."""
import os
import numpy as np
import tensorflow.compat.v1 as tf1
import tensorflow.compat.v2 as tf

from object_detection.builders import model_builder
from object_detection.builders import post_processing_builder
from object_detection.core import box_list
from object_detection.core import standard_fields as fields

_DEFAULT_NUM_CHANNELS = 3
_DEFAULT_NUM_COORD_BOX = 4
_MAX_CLASSES_PER_DETECTION = 1
_DETECTION_POSTPROCESS_FUNC = 'TFLite_Detection_PostProcess'


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

  with tf1.Session() as sess:
    y_out, x_out, h_out, w_out = sess.run([y, x, h, w])
  encoded_anchors = tf1.constant(
      np.transpose(np.stack((y_out, x_out, h_out, w_out))),
      dtype=tf1.float32,
      shape=[num_anchors[0], _DEFAULT_NUM_COORD_BOX],
      name='anchors')
  return num_anchors[0], encoded_anchors


class SSDModule(tf.Module):
  """Inference Module for TFLite-friendly SSD models."""

  def __init__(self, pipeline_config, detection_model, max_detections,
               use_regular_nms):
    """Initialization.

    Args:
      pipeline_config: The original pipeline_pb2.TrainEvalPipelineConfig
      detection_model: The detection model to use for inference.
      max_detections: Max detections desired from the TFLite model.
      use_regular_nms: If True, TFLite model uses the (slower) multi-class NMS.
    """
    self._process_config(pipeline_config)
    self._pipeline_config = pipeline_config
    self._model = detection_model
    self._max_detections = max_detections
    self._use_regular_nms = use_regular_nms

  def _process_config(self, pipeline_config):
    self._num_classes = pipeline_config.model.ssd.num_classes
    self._nms_score_threshold = pipeline_config.model.ssd.post_processing.batch_non_max_suppression.score_threshold
    self._nms_iou_threshold = pipeline_config.model.ssd.post_processing.batch_non_max_suppression.iou_threshold
    self._scale_values = {}
    self._scale_values[
        'y_scale'] = pipeline_config.model.ssd.box_coder.faster_rcnn_box_coder.y_scale
    self._scale_values[
        'x_scale'] = pipeline_config.model.ssd.box_coder.faster_rcnn_box_coder.x_scale
    self._scale_values[
        'h_scale'] = pipeline_config.model.ssd.box_coder.faster_rcnn_box_coder.height_scale
    self._scale_values[
        'w_scale'] = pipeline_config.model.ssd.box_coder.faster_rcnn_box_coder.width_scale

    image_resizer_config = pipeline_config.model.ssd.image_resizer
    image_resizer = image_resizer_config.WhichOneof('image_resizer_oneof')
    self._num_channels = _DEFAULT_NUM_CHANNELS

    if image_resizer == 'fixed_shape_resizer':
      self._height = image_resizer_config.fixed_shape_resizer.height
      self._width = image_resizer_config.fixed_shape_resizer.width
      if image_resizer_config.fixed_shape_resizer.convert_to_grayscale:
        self._num_channels = 1
    else:
      raise ValueError(
          'Only fixed_shape_resizer'
          'is supported with tflite. Found {}'.format(
              image_resizer_config.WhichOneof('image_resizer_oneof')))

  def input_shape(self):
    """Returns shape of TFLite model input."""
    return [1, self._height, self._width, self._num_channels]

  def postprocess_implements_signature(self):
    """Returns tf.implements signature for MLIR legalization of TFLite NMS."""
    implements_signature = [
        'name: "%s"' % _DETECTION_POSTPROCESS_FUNC,
        'attr { key: "max_detections" value { i: %d } }' % self._max_detections,
        'attr { key: "max_classes_per_detection" value { i: %d } }' %
        _MAX_CLASSES_PER_DETECTION,
        'attr { key: "use_regular_nms" value { b: %s } }' %
        str(self._use_regular_nms).lower(),
        'attr { key: "nms_score_threshold" value { f: %f } }' %
        self._nms_score_threshold,
        'attr { key: "nms_iou_threshold" value { f: %f } }' %
        self._nms_iou_threshold,
        'attr { key: "y_scale" value { f: %f } }' %
        self._scale_values['y_scale'],
        'attr { key: "x_scale" value { f: %f } }' %
        self._scale_values['x_scale'],
        'attr { key: "h_scale" value { f: %f } }' %
        self._scale_values['h_scale'],
        'attr { key: "w_scale" value { f: %f } }' %
        self._scale_values['w_scale'],
        'attr { key: "num_classes" value { i: %d } }' % self._num_classes
    ]
    implements_signature = ' '.join(implements_signature)
    return implements_signature

  def _get_postprocess_fn(self, num_anchors, num_classes):
    # There is no TF equivalent for TFLite's custom post-processing op.
    # So we add an 'empty' composite function here, that is legalized to the
    # custom op with MLIR.
    @tf.function(
        experimental_implements=self.postprocess_implements_signature())
    # pylint: disable=g-unused-argument,unused-argument
    def dummy_post_processing(box_encodings, class_predictions, anchors):
      boxes = tf.constant(0.0, dtype=tf.float32, name='boxes')
      scores = tf.constant(0.0, dtype=tf.float32, name='scores')
      classes = tf.constant(0.0, dtype=tf.float32, name='classes')
      num_detections = tf.constant(0.0, dtype=tf.float32, name='num_detections')
      return boxes, classes, scores, num_detections

    return dummy_post_processing

  @tf.function
  def inference_fn(self, image):
    """Encapsulates SSD inference for TFLite conversion.

    NOTE: The Args & Returns sections below indicate the TFLite model signature,
    and not what the TF graph does (since the latter does not include the custom
    NMS op used by TFLite)

    Args:
      image: a float32 tensor of shape [num_anchors, 4] containing the anchor
        boxes

    Returns:
      num_detections: a float32 scalar denoting number of total detections.
      classes: a float32 tensor denoting class ID for each detection.
      scores: a float32 tensor denoting score for each detection.
      boxes: a float32 tensor denoting coordinates of each detected box.
    """
    predicted_tensors = self._model.predict(image, true_image_shapes=None)
    # The score conversion occurs before the post-processing custom op
    _, score_conversion_fn = post_processing_builder.build(
        self._pipeline_config.model.ssd.post_processing)
    class_predictions = score_conversion_fn(
        predicted_tensors['class_predictions_with_background'])

    with tf.name_scope('raw_outputs'):
      # 'raw_outputs/box_encodings': a float32 tensor of shape
      #   [1, num_anchors, 4] containing the encoded box predictions. Note that
      #   these are raw predictions and no Non-Max suppression is applied on
      #   them and no decode center size boxes is applied to them.
      box_encodings = tf.identity(
          predicted_tensors['box_encodings'], name='box_encodings')
      # 'raw_outputs/class_predictions': a float32 tensor of shape
      #   [1, num_anchors, num_classes] containing the class scores for each
      #   anchor after applying score conversion.
      class_predictions = tf.identity(
          class_predictions, name='class_predictions')
    # 'anchors': a float32 tensor of shape
    #   [4, num_anchors] containing the anchors as a constant node.
    num_anchors, anchors = get_const_center_size_encoded_anchors(
        predicted_tensors['anchors'])
    anchors = tf.identity(anchors, name='anchors')

    # tf.function@ seems to reverse order of inputs, so reverse them here.
    return self._get_postprocess_fn(num_anchors,
                                    self._num_classes)(box_encodings,
                                                       class_predictions,
                                                       anchors)[::-1]


class CenterNetModule(tf.Module):
  """Inference Module for TFLite-friendly CenterNet models.

  The exported CenterNet model includes the preprocessing and postprocessing
  logics so the caller should pass in the raw image pixel values. It supports
  both object detection and keypoint estimation task.
  """

  def __init__(self, pipeline_config, max_detections, include_keypoints,
               label_map_path=''):
    """Initialization.

    Args:
      pipeline_config: The original pipeline_pb2.TrainEvalPipelineConfig
      max_detections: Max detections desired from the TFLite model.
      include_keypoints: If set true, the output dictionary will include the
        keypoint coordinates and keypoint confidence scores.
      label_map_path: Path to the label map which is used by CenterNet keypoint
        estimation task. If provided, the label_map_path in the configuration
        will be replaced by this one.
    """
    self._max_detections = max_detections
    self._include_keypoints = include_keypoints
    self._process_config(pipeline_config)
    if include_keypoints and label_map_path:
      pipeline_config.model.center_net.keypoint_label_map_path = label_map_path
    self._pipeline_config = pipeline_config
    self._model = model_builder.build(
        self._pipeline_config.model, is_training=False)

  def get_model(self):
    return self._model

  def _process_config(self, pipeline_config):
    self._num_classes = pipeline_config.model.center_net.num_classes

    center_net_config = pipeline_config.model.center_net
    image_resizer_config = center_net_config.image_resizer
    image_resizer = image_resizer_config.WhichOneof('image_resizer_oneof')
    self._num_channels = _DEFAULT_NUM_CHANNELS

    if image_resizer == 'fixed_shape_resizer':
      self._height = image_resizer_config.fixed_shape_resizer.height
      self._width = image_resizer_config.fixed_shape_resizer.width
      if image_resizer_config.fixed_shape_resizer.convert_to_grayscale:
        self._num_channels = 1
    else:
      raise ValueError(
          'Only fixed_shape_resizer'
          'is supported with tflite. Found {}'.format(image_resizer))

    center_net_config.object_center_params.max_box_predictions = (
        self._max_detections)

    if not self._include_keypoints:
      del center_net_config.keypoint_estimation_task[:]

  def input_shape(self):
    """Returns shape of TFLite model input."""
    return [1, self._height, self._width, self._num_channels]

  @tf.function
  def inference_fn(self, image):
    """Encapsulates CenterNet inference for TFLite conversion.

    Args:
      image: a float32 tensor of shape [1, image_height, image_width, channel]
        denoting the image pixel values.

    Returns:
      A dictionary of predicted tensors:
        classes: a float32 tensor with shape [1, max_detections] denoting class
          ID for each detection.
        scores: a float32 tensor with shape [1, max_detections] denoting score
          for each detection.
        boxes: a float32 tensor with shape [1, max_detections, 4] denoting
          coordinates of each detected box.
        keypoints: a float32 with shape [1, max_detections, num_keypoints, 2]
          denoting the predicted keypoint coordinates (normalized in between
          0-1). Note that [:, :, :, 0] represents the y coordinates and
          [:, :, :, 1] represents the x coordinates.
        keypoint_scores: a float32 with shape [1, max_detections, num_keypoints]
          denoting keypoint confidence scores.
    """
    image = tf.cast(image, tf.float32)
    image, shapes = self._model.preprocess(image)
    prediction_dict = self._model.predict(image, None)
    detections = self._model.postprocess(
        prediction_dict, true_image_shapes=shapes)

    field_names = fields.DetectionResultFields
    classes_field = field_names.detection_classes
    classes = tf.cast(detections[classes_field], tf.float32)
    num_detections = tf.cast(detections[field_names.num_detections], tf.float32)

    if self._include_keypoints:
      model_outputs = (detections[field_names.detection_boxes], classes,
                       detections[field_names.detection_scores], num_detections,
                       detections[field_names.detection_keypoints],
                       detections[field_names.detection_keypoint_scores])
    else:
      model_outputs = (detections[field_names.detection_boxes], classes,
                       detections[field_names.detection_scores], num_detections)

    # tf.function@ seems to reverse order of inputs, so reverse them here.
    return model_outputs[::-1]


def export_tflite_model(pipeline_config, trained_checkpoint_dir,
                        output_directory, max_detections, use_regular_nms,
                        include_keypoints=False, label_map_path=''):
  """Exports inference SavedModel for TFLite conversion.

  NOTE: Only supports SSD meta-architectures for now, and the output model will
  have static-shaped, single-batch input.

  This function creates `output_directory` if it does not already exist,
  which will hold the intermediate SavedModel that can be used with the TFLite
  converter.

  Args:
    pipeline_config: pipeline_pb2.TrainAndEvalPipelineConfig proto.
    trained_checkpoint_dir: Path to the trained checkpoint file.
    output_directory: Path to write outputs.
    max_detections: Max detections desired from the TFLite model.
    use_regular_nms: If True, TFLite model uses the (slower) multi-class NMS.
      Note that this argument is only used by the SSD model.
    include_keypoints: Decides whether to also output the keypoint predictions.
      Note that this argument is only used by the CenterNet model.
    label_map_path: Path to the label map which is used by CenterNet keypoint
      estimation task. If provided, the label_map_path in the configuration will
      be replaced by this one.

  Raises:
    ValueError: if pipeline is invalid.
  """
  output_saved_model_directory = os.path.join(output_directory, 'saved_model')

  # Build the underlying model using pipeline config.
  # TODO(b/162842801): Add support for other architectures.
  if pipeline_config.model.WhichOneof('model') == 'ssd':
    detection_model = model_builder.build(
        pipeline_config.model, is_training=False)
    ckpt = tf.train.Checkpoint(model=detection_model)
    # The module helps build a TF SavedModel appropriate for TFLite conversion.
    detection_module = SSDModule(pipeline_config, detection_model,
                                 max_detections, use_regular_nms)
  elif pipeline_config.model.WhichOneof('model') == 'center_net':
    detection_module = CenterNetModule(
        pipeline_config, max_detections, include_keypoints,
        label_map_path=label_map_path)
    ckpt = tf.train.Checkpoint(model=detection_module.get_model())
  else:
    raise ValueError('Only ssd or center_net models are supported in tflite. '
                     'Found {} in config'.format(
                         pipeline_config.model.WhichOneof('model')))

  manager = tf.train.CheckpointManager(
      ckpt, trained_checkpoint_dir, max_to_keep=1)
  status = ckpt.restore(manager.latest_checkpoint).expect_partial()

  # Getting the concrete function traces the graph and forces variables to
  # be constructed; only after this can we save the saved model.
  status.assert_existing_objects_matched()
  concrete_function = detection_module.inference_fn.get_concrete_function(
      tf.TensorSpec(
          shape=detection_module.input_shape(), dtype=tf.float32, name='input'))
  status.assert_existing_objects_matched()

  # Export SavedModel.
  tf.saved_model.save(
      detection_module,
      output_saved_model_directory,
      signatures=concrete_function)
