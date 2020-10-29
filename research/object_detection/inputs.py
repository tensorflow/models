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
"""Model input function for tf-learn object detection model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools

import tensorflow.compat.v1 as tf
from object_detection.builders import dataset_builder
from object_detection.builders import image_resizer_builder
from object_detection.builders import model_builder
from object_detection.builders import preprocessor_builder
from object_detection.core import box_list
from object_detection.core import box_list_ops
from object_detection.core import densepose_ops
from object_detection.core import keypoint_ops
from object_detection.core import preprocessor
from object_detection.core import standard_fields as fields
from object_detection.data_decoders import tf_example_decoder
from object_detection.protos import eval_pb2
from object_detection.protos import image_resizer_pb2
from object_detection.protos import input_reader_pb2
from object_detection.protos import model_pb2
from object_detection.protos import train_pb2
from object_detection.utils import config_util
from object_detection.utils import ops as util_ops
from object_detection.utils import shape_utils

HASH_KEY = 'hash'
HASH_BINS = 1 << 31
SERVING_FED_EXAMPLE_KEY = 'serialized_example'
_LABEL_OFFSET = 1

# A map of names to methods that help build the input pipeline.
INPUT_BUILDER_UTIL_MAP = {
    'dataset_build': dataset_builder.build,
    'model_build': model_builder.build,
}


def _multiclass_scores_or_one_hot_labels(multiclass_scores,
                                         groundtruth_boxes,
                                         groundtruth_classes, num_classes):
  """Returns one-hot encoding of classes when multiclass_scores is empty."""
  # Replace groundtruth_classes tensor with multiclass_scores tensor when its
  # non-empty. If multiclass_scores is empty fall back on groundtruth_classes
  # tensor.
  def true_fn():
    return tf.reshape(multiclass_scores,
                      [tf.shape(groundtruth_boxes)[0], num_classes])
  def false_fn():
    return tf.one_hot(groundtruth_classes, num_classes)
  return tf.cond(tf.size(multiclass_scores) > 0, true_fn, false_fn)


def _convert_labeled_classes_to_k_hot(groundtruth_labeled_classes, num_classes):
  """Returns k-hot encoding of the labeled classes."""

  # If the input labeled_classes is empty, it assumes all classes are
  # exhaustively labeled, thus returning an all-one encoding.
  def true_fn():
    return tf.sparse_to_dense(
        groundtruth_labeled_classes - _LABEL_OFFSET, [num_classes],
        tf.constant(1, dtype=tf.float32),
        validate_indices=False)

  def false_fn():
    return tf.ones(num_classes, dtype=tf.float32)

  return tf.cond(tf.size(groundtruth_labeled_classes) > 0, true_fn, false_fn)


def _remove_unrecognized_classes(class_ids, unrecognized_label):
  """Returns class ids with unrecognized classes filtered out."""

  recognized_indices = tf.squeeze(
      tf.where(tf.greater(class_ids, unrecognized_label)), -1)
  return tf.gather(class_ids, recognized_indices)


def assert_or_prune_invalid_boxes(boxes):
  """Makes sure boxes have valid sizes (ymax >= ymin, xmax >= xmin).

  When the hardware supports assertions, the function raises an error when
  boxes have an invalid size. If assertions are not supported (e.g. on TPU),
  boxes with invalid sizes are filtered out.

  Args:
    boxes: float tensor of shape [num_boxes, 4]

  Returns:
    boxes: float tensor of shape [num_valid_boxes, 4] with invalid boxes
      filtered out.

  Raises:
    tf.errors.InvalidArgumentError: When we detect boxes with invalid size.
      This is not supported on TPUs.
  """

  ymin, xmin, ymax, xmax = tf.split(
      boxes, num_or_size_splits=4, axis=1)

  height_check = tf.Assert(tf.reduce_all(ymax >= ymin), [ymin, ymax])
  width_check = tf.Assert(tf.reduce_all(xmax >= xmin), [xmin, xmax])

  with tf.control_dependencies([height_check, width_check]):
    boxes_tensor = tf.concat([ymin, xmin, ymax, xmax], axis=1)
    boxlist = box_list.BoxList(boxes_tensor)
    # TODO(b/149221748) Remove pruning when XLA supports assertions.
    boxlist = box_list_ops.prune_small_boxes(boxlist, 0)

  return boxlist.get()


def transform_input_data(tensor_dict,
                         model_preprocess_fn,
                         image_resizer_fn,
                         num_classes,
                         data_augmentation_fn=None,
                         merge_multiple_boxes=False,
                         retain_original_image=False,
                         use_multiclass_scores=False,
                         use_bfloat16=False,
                         retain_original_image_additional_channels=False,
                         keypoint_type_weight=None):
  """A single function that is responsible for all input data transformations.

  Data transformation functions are applied in the following order.
  1. If key fields.InputDataFields.image_additional_channels is present in
     tensor_dict, the additional channels will be merged into
     fields.InputDataFields.image.
  2. data_augmentation_fn (optional): applied on tensor_dict.
  3. model_preprocess_fn: applied only on image tensor in tensor_dict.
  4. keypoint_type_weight (optional): If groundtruth keypoints are in
     the tensor dictionary, per-keypoint weights are produced. These weights are
     initialized by `keypoint_type_weight` (or ones if left None).
     Then, for all keypoints that are not visible, the weights are set to 0 (to
     avoid penalizing the model in a loss function).
  5. image_resizer_fn: applied on original image and instance mask tensor in
     tensor_dict.
  6. one_hot_encoding: applied to classes tensor in tensor_dict.
  7. merge_multiple_boxes (optional): when groundtruth boxes are exactly the
     same they can be merged into a single box with an associated k-hot class
     label.

  Args:
    tensor_dict: dictionary containing input tensors keyed by
      fields.InputDataFields.
    model_preprocess_fn: model's preprocess function to apply on image tensor.
      This function must take in a 4-D float tensor and return a 4-D preprocess
      float tensor and a tensor containing the true image shape.
    image_resizer_fn: image resizer function to apply on groundtruth instance
      `masks. This function must take a 3-D float tensor of an image and a 3-D
      tensor of instance masks and return a resized version of these along with
      the true shapes.
    num_classes: number of max classes to one-hot (or k-hot) encode the class
      labels.
    data_augmentation_fn: (optional) data augmentation function to apply on
      input `tensor_dict`.
    merge_multiple_boxes: (optional) whether to merge multiple groundtruth boxes
      and classes for a given image if the boxes are exactly the same.
    retain_original_image: (optional) whether to retain original image in the
      output dictionary.
    use_multiclass_scores: whether to use multiclass scores as class targets
      instead of one-hot encoding of `groundtruth_classes`. When
      this is True and multiclass_scores is empty, one-hot encoding of
      `groundtruth_classes` is used as a fallback.
    use_bfloat16: (optional) a bool, whether to use bfloat16 in training.
    retain_original_image_additional_channels: (optional) Whether to retain
      original image additional channels in the output dictionary.
    keypoint_type_weight: A list (of length num_keypoints) containing
      groundtruth loss weights to use for each keypoint. If None, will use a
      weight of 1.

  Returns:
    A dictionary keyed by fields.InputDataFields containing the tensors obtained
    after applying all the transformations.

  Raises:
    KeyError: If both groundtruth_labeled_classes and groundtruth_image_classes
      are provided by the decoder in tensor_dict since both fields are
      considered to contain the same information.
  """
  out_tensor_dict = tensor_dict.copy()

  input_fields = fields.InputDataFields
  labeled_classes_field = input_fields.groundtruth_labeled_classes
  image_classes_field = input_fields.groundtruth_image_classes
  verified_neg_classes_field = input_fields.groundtruth_verified_neg_classes
  not_exhaustive_field = input_fields.groundtruth_not_exhaustive_classes

  if (labeled_classes_field in out_tensor_dict and
      image_classes_field in out_tensor_dict):
    raise KeyError('groundtruth_labeled_classes and groundtruth_image_classes'
                   'are provided by the decoder, but only one should be set.')

  for field in [labeled_classes_field,
                image_classes_field,
                verified_neg_classes_field,
                not_exhaustive_field]:
    if field in out_tensor_dict:
      out_tensor_dict[field] = _remove_unrecognized_classes(
          out_tensor_dict[field], unrecognized_label=-1)
      out_tensor_dict[field] = _convert_labeled_classes_to_k_hot(
          out_tensor_dict[field], num_classes)

  if input_fields.multiclass_scores in out_tensor_dict:
    out_tensor_dict[
        input_fields
        .multiclass_scores] = _multiclass_scores_or_one_hot_labels(
            out_tensor_dict[input_fields.multiclass_scores],
            out_tensor_dict[input_fields.groundtruth_boxes],
            out_tensor_dict[input_fields.groundtruth_classes],
            num_classes)

  if input_fields.groundtruth_boxes in out_tensor_dict:
    out_tensor_dict = util_ops.filter_groundtruth_with_nan_box_coordinates(
        out_tensor_dict)
    out_tensor_dict = util_ops.filter_unrecognized_classes(out_tensor_dict)

  if retain_original_image:
    out_tensor_dict[input_fields.original_image] = tf.cast(
        image_resizer_fn(out_tensor_dict[input_fields.image],
                         None)[0], tf.uint8)

  if input_fields.image_additional_channels in out_tensor_dict:
    channels = out_tensor_dict[input_fields.image_additional_channels]
    out_tensor_dict[input_fields.image] = tf.concat(
        [out_tensor_dict[input_fields.image], channels], axis=2)
    if retain_original_image_additional_channels:
      out_tensor_dict[
          input_fields.image_additional_channels] = tf.cast(
              image_resizer_fn(channels, None)[0], tf.uint8)

  # Apply data augmentation ops.
  if data_augmentation_fn is not None:
    out_tensor_dict = data_augmentation_fn(out_tensor_dict)

  # Apply model preprocessing ops and resize instance masks.
  image = out_tensor_dict[input_fields.image]
  preprocessed_resized_image, true_image_shape = model_preprocess_fn(
      tf.expand_dims(tf.cast(image, dtype=tf.float32), axis=0))

  preprocessed_shape = tf.shape(preprocessed_resized_image)
  new_height, new_width = preprocessed_shape[1], preprocessed_shape[2]

  im_box = tf.stack([
      0.0, 0.0,
      tf.to_float(new_height) / tf.to_float(true_image_shape[0, 0]),
      tf.to_float(new_width) / tf.to_float(true_image_shape[0, 1])
  ])

  if input_fields.groundtruth_boxes in tensor_dict:
    bboxes = out_tensor_dict[input_fields.groundtruth_boxes]
    boxlist = box_list.BoxList(bboxes)
    realigned_bboxes = box_list_ops.change_coordinate_frame(boxlist, im_box)

    realigned_boxes_tensor = realigned_bboxes.get()
    valid_boxes_tensor = assert_or_prune_invalid_boxes(realigned_boxes_tensor)
    out_tensor_dict[
        input_fields.groundtruth_boxes] = valid_boxes_tensor

  if input_fields.groundtruth_keypoints in tensor_dict:
    keypoints = out_tensor_dict[input_fields.groundtruth_keypoints]
    realigned_keypoints = keypoint_ops.change_coordinate_frame(keypoints,
                                                               im_box)
    out_tensor_dict[
        input_fields.groundtruth_keypoints] = realigned_keypoints
    flds_gt_kpt = input_fields.groundtruth_keypoints
    flds_gt_kpt_vis = input_fields.groundtruth_keypoint_visibilities
    flds_gt_kpt_weights = input_fields.groundtruth_keypoint_weights
    if flds_gt_kpt_vis not in out_tensor_dict:
      out_tensor_dict[flds_gt_kpt_vis] = tf.ones_like(
          out_tensor_dict[flds_gt_kpt][:, :, 0],
          dtype=tf.bool)
    out_tensor_dict[flds_gt_kpt_weights] = (
        keypoint_ops.keypoint_weights_from_visibilities(
            out_tensor_dict[flds_gt_kpt_vis],
            keypoint_type_weight))

  dp_surface_coords_fld = input_fields.groundtruth_dp_surface_coords
  if dp_surface_coords_fld in tensor_dict:
    dp_surface_coords = out_tensor_dict[dp_surface_coords_fld]
    realigned_dp_surface_coords = densepose_ops.change_coordinate_frame(
        dp_surface_coords, im_box)
    out_tensor_dict[dp_surface_coords_fld] = realigned_dp_surface_coords

  if use_bfloat16:
    preprocessed_resized_image = tf.cast(
        preprocessed_resized_image, tf.bfloat16)
    if input_fields.context_features in out_tensor_dict:
      out_tensor_dict[input_fields.context_features] = tf.cast(
          out_tensor_dict[input_fields.context_features], tf.bfloat16)
  out_tensor_dict[input_fields.image] = tf.squeeze(
      preprocessed_resized_image, axis=0)
  out_tensor_dict[input_fields.true_image_shape] = tf.squeeze(
      true_image_shape, axis=0)
  if input_fields.groundtruth_instance_masks in out_tensor_dict:
    masks = out_tensor_dict[input_fields.groundtruth_instance_masks]
    _, resized_masks, _ = image_resizer_fn(image, masks)
    if use_bfloat16:
      resized_masks = tf.cast(resized_masks, tf.bfloat16)
    out_tensor_dict[
        input_fields.groundtruth_instance_masks] = resized_masks

  zero_indexed_groundtruth_classes = out_tensor_dict[
      input_fields.groundtruth_classes] - _LABEL_OFFSET
  if use_multiclass_scores:
    out_tensor_dict[
        input_fields.groundtruth_classes] = out_tensor_dict[
            input_fields.multiclass_scores]
  else:
    out_tensor_dict[input_fields.groundtruth_classes] = tf.one_hot(
        zero_indexed_groundtruth_classes, num_classes)
  out_tensor_dict.pop(input_fields.multiclass_scores, None)

  if input_fields.groundtruth_confidences in out_tensor_dict:
    groundtruth_confidences = out_tensor_dict[
        input_fields.groundtruth_confidences]
    # Map the confidences to the one-hot encoding of classes
    out_tensor_dict[input_fields.groundtruth_confidences] = (
        tf.reshape(groundtruth_confidences, [-1, 1]) *
        out_tensor_dict[input_fields.groundtruth_classes])
  else:
    groundtruth_confidences = tf.ones_like(
        zero_indexed_groundtruth_classes, dtype=tf.float32)
    out_tensor_dict[input_fields.groundtruth_confidences] = (
        out_tensor_dict[input_fields.groundtruth_classes])

  if merge_multiple_boxes:
    merged_boxes, merged_classes, merged_confidences, _ = (
        util_ops.merge_boxes_with_multiple_labels(
            out_tensor_dict[input_fields.groundtruth_boxes],
            zero_indexed_groundtruth_classes,
            groundtruth_confidences,
            num_classes))
    merged_classes = tf.cast(merged_classes, tf.float32)
    out_tensor_dict[input_fields.groundtruth_boxes] = merged_boxes
    out_tensor_dict[input_fields.groundtruth_classes] = merged_classes
    out_tensor_dict[input_fields.groundtruth_confidences] = (
        merged_confidences)
  if input_fields.groundtruth_boxes in out_tensor_dict:
    out_tensor_dict[input_fields.num_groundtruth_boxes] = tf.shape(
        out_tensor_dict[input_fields.groundtruth_boxes])[0]

  return out_tensor_dict


def pad_input_data_to_static_shapes(tensor_dict,
                                    max_num_boxes,
                                    num_classes,
                                    spatial_image_shape=None,
                                    max_num_context_features=None,
                                    context_feature_length=None,
                                    max_dp_points=336):
  """Pads input tensors to static shapes.

  In case num_additional_channels > 0, we assume that the additional channels
  have already been concatenated to the base image.

  Args:
    tensor_dict: Tensor dictionary of input data
    max_num_boxes: Max number of groundtruth boxes needed to compute shapes for
      padding.
    num_classes: Number of classes in the dataset needed to compute shapes for
      padding.
    spatial_image_shape: A list of two integers of the form [height, width]
      containing expected spatial shape of the image.
    max_num_context_features (optional): The maximum number of context
      features needed to compute shapes padding.
    context_feature_length (optional): The length of the context feature.
    max_dp_points (optional): The maximum number of DensePose sampled points per
      instance. The default (336) is selected since the original DensePose paper
      (https://arxiv.org/pdf/1802.00434.pdf) indicates that the maximum number
      of samples per part is 14, and therefore 24 * 14 = 336 is the maximum
      sampler per instance.

  Returns:
    A dictionary keyed by fields.InputDataFields containing padding shapes for
    tensors in the dataset.

  Raises:
    ValueError: If groundtruth classes is neither rank 1 nor rank 2, or if we
      detect that additional channels have not been concatenated yet, or if
      max_num_context_features is not specified and context_features is in the
      tensor dict.
  """

  if not spatial_image_shape or spatial_image_shape == [-1, -1]:
    height, width = None, None
  else:
    height, width = spatial_image_shape  # pylint: disable=unpacking-non-sequence

  input_fields = fields.InputDataFields
  num_additional_channels = 0
  if input_fields.image_additional_channels in tensor_dict:
    num_additional_channels = shape_utils.get_dim_as_int(tensor_dict[
        input_fields.image_additional_channels].shape[2])

  # We assume that if num_additional_channels > 0, then it has already been
  # concatenated to the base image (but not the ground truth).
  num_channels = 3
  if input_fields.image in tensor_dict:
    num_channels = shape_utils.get_dim_as_int(
        tensor_dict[input_fields.image].shape[2])

  if num_additional_channels:
    if num_additional_channels >= num_channels:
      raise ValueError(
          'Image must be already concatenated with additional channels.')

    if (input_fields.original_image in tensor_dict and
        shape_utils.get_dim_as_int(
            tensor_dict[input_fields.original_image].shape[2]) ==
        num_channels):
      raise ValueError(
          'Image must be already concatenated with additional channels.')

  if input_fields.context_features in tensor_dict and (
      max_num_context_features is None):
    raise ValueError('max_num_context_features must be specified in the model '
                     'config if include_context is specified in the input '
                     'config')

  padding_shapes = {
      input_fields.image: [height, width, num_channels],
      input_fields.original_image_spatial_shape: [2],
      input_fields.image_additional_channels: [
          height, width, num_additional_channels
      ],
      input_fields.source_id: [],
      input_fields.filename: [],
      input_fields.key: [],
      input_fields.groundtruth_difficult: [max_num_boxes],
      input_fields.groundtruth_boxes: [max_num_boxes, 4],
      input_fields.groundtruth_classes: [max_num_boxes, num_classes],
      input_fields.groundtruth_instance_masks: [
          max_num_boxes, height, width
      ],
      input_fields.groundtruth_is_crowd: [max_num_boxes],
      input_fields.groundtruth_group_of: [max_num_boxes],
      input_fields.groundtruth_area: [max_num_boxes],
      input_fields.groundtruth_weights: [max_num_boxes],
      input_fields.groundtruth_confidences: [
          max_num_boxes, num_classes
      ],
      input_fields.num_groundtruth_boxes: [],
      input_fields.groundtruth_label_types: [max_num_boxes],
      input_fields.groundtruth_label_weights: [max_num_boxes],
      input_fields.true_image_shape: [3],
      input_fields.groundtruth_image_classes: [num_classes],
      input_fields.groundtruth_image_confidences: [num_classes],
      input_fields.groundtruth_labeled_classes: [num_classes],
  }

  if input_fields.original_image in tensor_dict:
    padding_shapes[input_fields.original_image] = [
        height, width,
        shape_utils.get_dim_as_int(tensor_dict[input_fields.
                                               original_image].shape[2])
    ]
  if input_fields.groundtruth_keypoints in tensor_dict:
    tensor_shape = (
        tensor_dict[input_fields.groundtruth_keypoints].shape)
    padding_shape = [max_num_boxes,
                     shape_utils.get_dim_as_int(tensor_shape[1]),
                     shape_utils.get_dim_as_int(tensor_shape[2])]
    padding_shapes[input_fields.groundtruth_keypoints] = padding_shape
  if input_fields.groundtruth_keypoint_visibilities in tensor_dict:
    tensor_shape = tensor_dict[input_fields.
                               groundtruth_keypoint_visibilities].shape
    padding_shape = [max_num_boxes, shape_utils.get_dim_as_int(tensor_shape[1])]
    padding_shapes[input_fields.
                   groundtruth_keypoint_visibilities] = padding_shape

  if input_fields.groundtruth_keypoint_weights in tensor_dict:
    tensor_shape = (
        tensor_dict[input_fields.groundtruth_keypoint_weights].shape)
    padding_shape = [max_num_boxes, shape_utils.get_dim_as_int(tensor_shape[1])]
    padding_shapes[input_fields.
                   groundtruth_keypoint_weights] = padding_shape
  if input_fields.groundtruth_dp_num_points in tensor_dict:
    padding_shapes[
        input_fields.groundtruth_dp_num_points] = [max_num_boxes]
    padding_shapes[
        input_fields.groundtruth_dp_part_ids] = [
            max_num_boxes, max_dp_points]
    padding_shapes[
        input_fields.groundtruth_dp_surface_coords] = [
            max_num_boxes, max_dp_points, 4]
  if input_fields.groundtruth_track_ids in tensor_dict:
    padding_shapes[
        input_fields.groundtruth_track_ids] = [max_num_boxes]

  if input_fields.groundtruth_verified_neg_classes in tensor_dict:
    padding_shapes[
        input_fields.groundtruth_verified_neg_classes] = [num_classes]
  if input_fields.groundtruth_not_exhaustive_classes in tensor_dict:
    padding_shapes[
        input_fields.groundtruth_not_exhaustive_classes] = [num_classes]

  # Prepare for ContextRCNN related fields.
  if input_fields.context_features in tensor_dict:
    padding_shape = [max_num_context_features, context_feature_length]
    padding_shapes[input_fields.context_features] = padding_shape

    tensor_shape = tf.shape(
        tensor_dict[input_fields.context_features])
    tensor_dict[input_fields.valid_context_size] = tensor_shape[0]
    padding_shapes[input_fields.valid_context_size] = []
  if input_fields.context_feature_length in tensor_dict:
    padding_shapes[input_fields.context_feature_length] = []

  if input_fields.is_annotated in tensor_dict:
    padding_shapes[input_fields.is_annotated] = []

  padded_tensor_dict = {}
  for tensor_name in tensor_dict:
    padded_tensor_dict[tensor_name] = shape_utils.pad_or_clip_nd(
        tensor_dict[tensor_name], padding_shapes[tensor_name])

  # Make sure that the number of groundtruth boxes now reflects the
  # padded/clipped tensors.
  if input_fields.num_groundtruth_boxes in padded_tensor_dict:
    padded_tensor_dict[input_fields.num_groundtruth_boxes] = (
        tf.minimum(
            padded_tensor_dict[input_fields.num_groundtruth_boxes],
            max_num_boxes))
  return padded_tensor_dict


def augment_input_data(tensor_dict, data_augmentation_options):
  """Applies data augmentation ops to input tensors.

  Args:
    tensor_dict: A dictionary of input tensors keyed by fields.InputDataFields.
    data_augmentation_options: A list of tuples, where each tuple contains a
      function and a dictionary that contains arguments and their values.
      Usually, this is the output of core/preprocessor.build.

  Returns:
    A dictionary of tensors obtained by applying data augmentation ops to the
    input tensor dictionary.
  """
  tensor_dict[fields.InputDataFields.image] = tf.expand_dims(
      tf.cast(tensor_dict[fields.InputDataFields.image], dtype=tf.float32), 0)

  include_instance_masks = (fields.InputDataFields.groundtruth_instance_masks
                            in tensor_dict)
  include_keypoints = (fields.InputDataFields.groundtruth_keypoints
                       in tensor_dict)
  include_keypoint_visibilities = (
      fields.InputDataFields.groundtruth_keypoint_visibilities in tensor_dict)
  include_label_weights = (fields.InputDataFields.groundtruth_weights
                           in tensor_dict)
  include_label_confidences = (fields.InputDataFields.groundtruth_confidences
                               in tensor_dict)
  include_multiclass_scores = (fields.InputDataFields.multiclass_scores in
                               tensor_dict)
  dense_pose_fields = [fields.InputDataFields.groundtruth_dp_num_points,
                       fields.InputDataFields.groundtruth_dp_part_ids,
                       fields.InputDataFields.groundtruth_dp_surface_coords]
  include_dense_pose = all(field in tensor_dict for field in dense_pose_fields)
  tensor_dict = preprocessor.preprocess(
      tensor_dict, data_augmentation_options,
      func_arg_map=preprocessor.get_default_func_arg_map(
          include_label_weights=include_label_weights,
          include_label_confidences=include_label_confidences,
          include_multiclass_scores=include_multiclass_scores,
          include_instance_masks=include_instance_masks,
          include_keypoints=include_keypoints,
          include_keypoint_visibilities=include_keypoint_visibilities,
          include_dense_pose=include_dense_pose))
  tensor_dict[fields.InputDataFields.image] = tf.squeeze(
      tensor_dict[fields.InputDataFields.image], axis=0)
  return tensor_dict


def _get_labels_dict(input_dict):
  """Extracts labels dict from input dict."""
  required_label_keys = [
      fields.InputDataFields.num_groundtruth_boxes,
      fields.InputDataFields.groundtruth_boxes,
      fields.InputDataFields.groundtruth_classes,
      fields.InputDataFields.groundtruth_weights,
  ]
  labels_dict = {}
  for key in required_label_keys:
    labels_dict[key] = input_dict[key]

  optional_label_keys = [
      fields.InputDataFields.groundtruth_confidences,
      fields.InputDataFields.groundtruth_labeled_classes,
      fields.InputDataFields.groundtruth_keypoints,
      fields.InputDataFields.groundtruth_instance_masks,
      fields.InputDataFields.groundtruth_area,
      fields.InputDataFields.groundtruth_is_crowd,
      fields.InputDataFields.groundtruth_group_of,
      fields.InputDataFields.groundtruth_difficult,
      fields.InputDataFields.groundtruth_keypoint_visibilities,
      fields.InputDataFields.groundtruth_keypoint_weights,
      fields.InputDataFields.groundtruth_dp_num_points,
      fields.InputDataFields.groundtruth_dp_part_ids,
      fields.InputDataFields.groundtruth_dp_surface_coords,
      fields.InputDataFields.groundtruth_track_ids,
      fields.InputDataFields.groundtruth_verified_neg_classes,
      fields.InputDataFields.groundtruth_not_exhaustive_classes
  ]

  for key in optional_label_keys:
    if key in input_dict:
      labels_dict[key] = input_dict[key]
  if fields.InputDataFields.groundtruth_difficult in labels_dict:
    labels_dict[fields.InputDataFields.groundtruth_difficult] = tf.cast(
        labels_dict[fields.InputDataFields.groundtruth_difficult], tf.int32)
  return labels_dict


def _replace_empty_string_with_random_number(string_tensor):
  """Returns string unchanged if non-empty, and random string tensor otherwise.

  The random string is an integer 0 and 2**63 - 1, casted as string.


  Args:
    string_tensor: A tf.tensor of dtype string.

  Returns:
    out_string: A tf.tensor of dtype string. If string_tensor contains the empty
      string, out_string will contain a random integer casted to a string.
      Otherwise string_tensor is returned unchanged.

  """

  empty_string = tf.constant('', dtype=tf.string, name='EmptyString')

  random_source_id = tf.as_string(
      tf.random_uniform(shape=[], maxval=2**63 - 1, dtype=tf.int64))

  out_string = tf.cond(
      tf.equal(string_tensor, empty_string),
      true_fn=lambda: random_source_id,
      false_fn=lambda: string_tensor)

  return out_string


def _get_features_dict(input_dict, include_source_id=False):
  """Extracts features dict from input dict."""

  source_id = _replace_empty_string_with_random_number(
      input_dict[fields.InputDataFields.source_id])

  hash_from_source_id = tf.string_to_hash_bucket_fast(source_id, HASH_BINS)
  features = {
      fields.InputDataFields.image:
          input_dict[fields.InputDataFields.image],
      HASH_KEY: tf.cast(hash_from_source_id, tf.int32),
      fields.InputDataFields.true_image_shape:
          input_dict[fields.InputDataFields.true_image_shape],
      fields.InputDataFields.original_image_spatial_shape:
          input_dict[fields.InputDataFields.original_image_spatial_shape]
  }
  if include_source_id:
    features[fields.InputDataFields.source_id] = source_id
  if fields.InputDataFields.original_image in input_dict:
    features[fields.InputDataFields.original_image] = input_dict[
        fields.InputDataFields.original_image]
  if fields.InputDataFields.image_additional_channels in input_dict:
    features[fields.InputDataFields.image_additional_channels] = input_dict[
        fields.InputDataFields.image_additional_channels]
  if fields.InputDataFields.context_features in input_dict:
    features[fields.InputDataFields.context_features] = input_dict[
        fields.InputDataFields.context_features]
  if fields.InputDataFields.valid_context_size in input_dict:
    features[fields.InputDataFields.valid_context_size] = input_dict[
        fields.InputDataFields.valid_context_size]
  return features


def create_train_input_fn(train_config, train_input_config,
                          model_config):
  """Creates a train `input` function for `Estimator`.

  Args:
    train_config: A train_pb2.TrainConfig.
    train_input_config: An input_reader_pb2.InputReader.
    model_config: A model_pb2.DetectionModel.

  Returns:
    `input_fn` for `Estimator` in TRAIN mode.
  """

  def _train_input_fn(params=None):
    return train_input(train_config, train_input_config, model_config,
                       params=params)

  return _train_input_fn


def train_input(train_config, train_input_config,
                model_config, model=None, params=None, input_context=None):
  """Returns `features` and `labels` tensor dictionaries for training.

  Args:
    train_config: A train_pb2.TrainConfig.
    train_input_config: An input_reader_pb2.InputReader.
    model_config: A model_pb2.DetectionModel.
    model: A pre-constructed Detection Model.
      If None, one will be created from the config.
    params: Parameter dictionary passed from the estimator.
    input_context: optional, A tf.distribute.InputContext object used to
      shard filenames and compute per-replica batch_size when this function
      is being called per-replica.

  Returns:
    A tf.data.Dataset that holds (features, labels) tuple.

    features: Dictionary of feature tensors.
      features[fields.InputDataFields.image] is a [batch_size, H, W, C]
        float32 tensor with preprocessed images.
      features[HASH_KEY] is a [batch_size] int32 tensor representing unique
        identifiers for the images.
      features[fields.InputDataFields.true_image_shape] is a [batch_size, 3]
        int32 tensor representing the true image shapes, as preprocessed
        images could be padded.
      features[fields.InputDataFields.original_image] (optional) is a
        [batch_size, H, W, C] float32 tensor with original images.
    labels: Dictionary of groundtruth tensors.
      labels[fields.InputDataFields.num_groundtruth_boxes] is a [batch_size]
        int32 tensor indicating the number of groundtruth boxes.
      labels[fields.InputDataFields.groundtruth_boxes] is a
        [batch_size, num_boxes, 4] float32 tensor containing the corners of
        the groundtruth boxes.
      labels[fields.InputDataFields.groundtruth_classes] is a
        [batch_size, num_boxes, num_classes] float32 one-hot tensor of
        classes.
      labels[fields.InputDataFields.groundtruth_weights] is a
        [batch_size, num_boxes] float32 tensor containing groundtruth weights
        for the boxes.
      -- Optional --
      labels[fields.InputDataFields.groundtruth_instance_masks] is a
        [batch_size, num_boxes, H, W] float32 tensor containing only binary
        values, which represent instance masks for objects.
      labels[fields.InputDataFields.groundtruth_keypoints] is a
        [batch_size, num_boxes, num_keypoints, 2] float32 tensor containing
        keypoints for each box.
      labels[fields.InputDataFields.groundtruth_weights] is a
        [batch_size, num_boxes, num_keypoints] float32 tensor containing
        groundtruth weights for the keypoints.
      labels[fields.InputDataFields.groundtruth_visibilities] is a
        [batch_size, num_boxes, num_keypoints] bool tensor containing
        groundtruth visibilities for each keypoint.
      labels[fields.InputDataFields.groundtruth_labeled_classes] is a
        [batch_size, num_classes] float32 k-hot tensor of classes.
      labels[fields.InputDataFields.groundtruth_dp_num_points] is a
        [batch_size, num_boxes] int32 tensor with the number of sampled
        DensePose points per object.
      labels[fields.InputDataFields.groundtruth_dp_part_ids] is a
        [batch_size, num_boxes, max_sampled_points] int32 tensor with the
        DensePose part ids (0-indexed) per object.
      labels[fields.InputDataFields.groundtruth_dp_surface_coords] is a
        [batch_size, num_boxes, max_sampled_points, 4] float32 tensor with the
        DensePose surface coordinates. The format is (y, x, v, u), where (y, x)
        are normalized image coordinates and (v, u) are normalized surface part
        coordinates.
      labels[fields.InputDataFields.groundtruth_track_ids] is a
        [batch_size, num_boxes] int32 tensor with the track ID for each object.

  Raises:
    TypeError: if the `train_config`, `train_input_config` or `model_config`
      are not of the correct type.
  """
  if not isinstance(train_config, train_pb2.TrainConfig):
    raise TypeError('For training mode, the `train_config` must be a '
                    'train_pb2.TrainConfig.')
  if not isinstance(train_input_config, input_reader_pb2.InputReader):
    raise TypeError('The `train_input_config` must be a '
                    'input_reader_pb2.InputReader.')
  if not isinstance(model_config, model_pb2.DetectionModel):
    raise TypeError('The `model_config` must be a '
                    'model_pb2.DetectionModel.')

  if model is None:
    model_preprocess_fn = INPUT_BUILDER_UTIL_MAP['model_build'](
        model_config, is_training=True).preprocess
  else:
    model_preprocess_fn = model.preprocess

  num_classes = config_util.get_number_of_classes(model_config)

  def transform_and_pad_input_data_fn(tensor_dict):
    """Combines transform and pad operation."""
    data_augmentation_options = [
        preprocessor_builder.build(step)
        for step in train_config.data_augmentation_options
    ]
    data_augmentation_fn = functools.partial(
        augment_input_data,
        data_augmentation_options=data_augmentation_options)

    image_resizer_config = config_util.get_image_resizer_config(model_config)
    image_resizer_fn = image_resizer_builder.build(image_resizer_config)
    keypoint_type_weight = train_input_config.keypoint_type_weight or None
    transform_data_fn = functools.partial(
        transform_input_data, model_preprocess_fn=model_preprocess_fn,
        image_resizer_fn=image_resizer_fn,
        num_classes=num_classes,
        data_augmentation_fn=data_augmentation_fn,
        merge_multiple_boxes=train_config.merge_multiple_label_boxes,
        retain_original_image=train_config.retain_original_images,
        use_multiclass_scores=train_config.use_multiclass_scores,
        use_bfloat16=train_config.use_bfloat16,
        keypoint_type_weight=keypoint_type_weight)

    tensor_dict = pad_input_data_to_static_shapes(
        tensor_dict=transform_data_fn(tensor_dict),
        max_num_boxes=train_input_config.max_number_of_boxes,
        num_classes=num_classes,
        spatial_image_shape=config_util.get_spatial_image_size(
            image_resizer_config),
        max_num_context_features=config_util.get_max_num_context_features(
            model_config),
        context_feature_length=config_util.get_context_feature_length(
            model_config))
    include_source_id = train_input_config.include_source_id
    return (_get_features_dict(tensor_dict, include_source_id),
            _get_labels_dict(tensor_dict))
  reduce_to_frame_fn = get_reduce_to_frame_fn(train_input_config, True)

  dataset = INPUT_BUILDER_UTIL_MAP['dataset_build'](
      train_input_config,
      transform_input_data_fn=transform_and_pad_input_data_fn,
      batch_size=params['batch_size'] if params else train_config.batch_size,
      input_context=input_context,
      reduce_to_frame_fn=reduce_to_frame_fn)
  return dataset


def create_eval_input_fn(eval_config, eval_input_config, model_config):
  """Creates an eval `input` function for `Estimator`.

  Args:
    eval_config: An eval_pb2.EvalConfig.
    eval_input_config: An input_reader_pb2.InputReader.
    model_config: A model_pb2.DetectionModel.

  Returns:
    `input_fn` for `Estimator` in EVAL mode.
  """

  def _eval_input_fn(params=None):
    return eval_input(eval_config, eval_input_config, model_config,
                      params=params)

  return _eval_input_fn


def eval_input(eval_config, eval_input_config, model_config,
               model=None, params=None):
  """Returns `features` and `labels` tensor dictionaries for evaluation.

  Args:
    eval_config: An eval_pb2.EvalConfig.
    eval_input_config: An input_reader_pb2.InputReader.
    model_config: A model_pb2.DetectionModel.
    model: A pre-constructed Detection Model.
      If None, one will be created from the config.
    params: Parameter dictionary passed from the estimator.

  Returns:
    A tf.data.Dataset that holds (features, labels) tuple.

    features: Dictionary of feature tensors.
      features[fields.InputDataFields.image] is a [1, H, W, C] float32 tensor
        with preprocessed images.
      features[HASH_KEY] is a [1] int32 tensor representing unique
        identifiers for the images.
      features[fields.InputDataFields.true_image_shape] is a [1, 3]
        int32 tensor representing the true image shapes, as preprocessed
        images could be padded.
      features[fields.InputDataFields.original_image] is a [1, H', W', C]
        float32 tensor with the original image.
    labels: Dictionary of groundtruth tensors.
      labels[fields.InputDataFields.groundtruth_boxes] is a [1, num_boxes, 4]
        float32 tensor containing the corners of the groundtruth boxes.
      labels[fields.InputDataFields.groundtruth_classes] is a
        [num_boxes, num_classes] float32 one-hot tensor of classes.
      labels[fields.InputDataFields.groundtruth_area] is a [1, num_boxes]
        float32 tensor containing object areas.
      labels[fields.InputDataFields.groundtruth_is_crowd] is a [1, num_boxes]
        bool tensor indicating if the boxes enclose a crowd.
      labels[fields.InputDataFields.groundtruth_difficult] is a [1, num_boxes]
        int32 tensor indicating if the boxes represent difficult instances.
      -- Optional --
      labels[fields.InputDataFields.groundtruth_instance_masks] is a
        [1, num_boxes, H, W] float32 tensor containing only binary values,
        which represent instance masks for objects.
      labels[fields.InputDataFields.groundtruth_weights] is a
        [batch_size, num_boxes, num_keypoints] float32 tensor containing
        groundtruth weights for the keypoints.
      labels[fields.InputDataFields.groundtruth_visibilities] is a
        [batch_size, num_boxes, num_keypoints] bool tensor containing
        groundtruth visibilities for each keypoint.
      labels[fields.InputDataFields.groundtruth_group_of] is a [1, num_boxes]
        bool tensor indicating if the box covers more than 5 instances of the
        same class which heavily occlude each other.
      labels[fields.InputDataFields.groundtruth_labeled_classes] is a
        [num_boxes, num_classes] float32 k-hot tensor of classes.
      labels[fields.InputDataFields.groundtruth_dp_num_points] is a
        [batch_size, num_boxes] int32 tensor with the number of sampled
        DensePose points per object.
      labels[fields.InputDataFields.groundtruth_dp_part_ids] is a
        [batch_size, num_boxes, max_sampled_points] int32 tensor with the
        DensePose part ids (0-indexed) per object.
      labels[fields.InputDataFields.groundtruth_dp_surface_coords] is a
        [batch_size, num_boxes, max_sampled_points, 4] float32 tensor with the
        DensePose surface coordinates. The format is (y, x, v, u), where (y, x)
        are normalized image coordinates and (v, u) are normalized surface part
        coordinates.
      labels[fields.InputDataFields.groundtruth_track_ids] is a
        [batch_size, num_boxes] int32 tensor with the track ID for each object.

  Raises:
    TypeError: if the `eval_config`, `eval_input_config` or `model_config`
      are not of the correct type.
  """
  params = params or {}
  if not isinstance(eval_config, eval_pb2.EvalConfig):
    raise TypeError('For eval mode, the `eval_config` must be a '
                    'train_pb2.EvalConfig.')
  if not isinstance(eval_input_config, input_reader_pb2.InputReader):
    raise TypeError('The `eval_input_config` must be a '
                    'input_reader_pb2.InputReader.')
  if not isinstance(model_config, model_pb2.DetectionModel):
    raise TypeError('The `model_config` must be a '
                    'model_pb2.DetectionModel.')

  if eval_config.force_no_resize:
    arch = model_config.WhichOneof('model')
    arch_config = getattr(model_config, arch)
    image_resizer_proto = image_resizer_pb2.ImageResizer()
    image_resizer_proto.identity_resizer.CopyFrom(
        image_resizer_pb2.IdentityResizer())
    arch_config.image_resizer.CopyFrom(image_resizer_proto)

  if model is None:
    model_preprocess_fn = INPUT_BUILDER_UTIL_MAP['model_build'](
        model_config, is_training=False).preprocess
  else:
    model_preprocess_fn = model.preprocess

  def transform_and_pad_input_data_fn(tensor_dict):
    """Combines transform and pad operation."""
    num_classes = config_util.get_number_of_classes(model_config)

    image_resizer_config = config_util.get_image_resizer_config(model_config)
    image_resizer_fn = image_resizer_builder.build(image_resizer_config)
    keypoint_type_weight = eval_input_config.keypoint_type_weight or None

    transform_data_fn = functools.partial(
        transform_input_data, model_preprocess_fn=model_preprocess_fn,
        image_resizer_fn=image_resizer_fn,
        num_classes=num_classes,
        data_augmentation_fn=None,
        retain_original_image=eval_config.retain_original_images,
        retain_original_image_additional_channels=
        eval_config.retain_original_image_additional_channels,
        keypoint_type_weight=keypoint_type_weight)
    tensor_dict = pad_input_data_to_static_shapes(
        tensor_dict=transform_data_fn(tensor_dict),
        max_num_boxes=eval_input_config.max_number_of_boxes,
        num_classes=config_util.get_number_of_classes(model_config),
        spatial_image_shape=config_util.get_spatial_image_size(
            image_resizer_config),
        max_num_context_features=config_util.get_max_num_context_features(
            model_config),
        context_feature_length=config_util.get_context_feature_length(
            model_config))
    include_source_id = eval_input_config.include_source_id
    return (_get_features_dict(tensor_dict, include_source_id),
            _get_labels_dict(tensor_dict))

  reduce_to_frame_fn = get_reduce_to_frame_fn(eval_input_config, False)

  dataset = INPUT_BUILDER_UTIL_MAP['dataset_build'](
      eval_input_config,
      batch_size=params['batch_size'] if params else eval_config.batch_size,
      transform_input_data_fn=transform_and_pad_input_data_fn,
      reduce_to_frame_fn=reduce_to_frame_fn)
  return dataset


def create_predict_input_fn(model_config, predict_input_config):
  """Creates a predict `input` function for `Estimator`.

  Args:
    model_config: A model_pb2.DetectionModel.
    predict_input_config: An input_reader_pb2.InputReader.

  Returns:
    `input_fn` for `Estimator` in PREDICT mode.
  """

  def _predict_input_fn(params=None):
    """Decodes serialized tf.Examples and returns `ServingInputReceiver`.

    Args:
      params: Parameter dictionary passed from the estimator.

    Returns:
      `ServingInputReceiver`.
    """
    del params
    example = tf.placeholder(dtype=tf.string, shape=[], name='tf_example')

    num_classes = config_util.get_number_of_classes(model_config)
    model_preprocess_fn = INPUT_BUILDER_UTIL_MAP['model_build'](
        model_config, is_training=False).preprocess

    image_resizer_config = config_util.get_image_resizer_config(model_config)
    image_resizer_fn = image_resizer_builder.build(image_resizer_config)

    transform_fn = functools.partial(
        transform_input_data, model_preprocess_fn=model_preprocess_fn,
        image_resizer_fn=image_resizer_fn,
        num_classes=num_classes,
        data_augmentation_fn=None)

    decoder = tf_example_decoder.TfExampleDecoder(
        load_instance_masks=False,
        num_additional_channels=predict_input_config.num_additional_channels)
    input_dict = transform_fn(decoder.decode(example))
    images = tf.cast(input_dict[fields.InputDataFields.image], dtype=tf.float32)
    images = tf.expand_dims(images, axis=0)
    true_image_shape = tf.expand_dims(
        input_dict[fields.InputDataFields.true_image_shape], axis=0)

    return tf.estimator.export.ServingInputReceiver(
        features={
            fields.InputDataFields.image: images,
            fields.InputDataFields.true_image_shape: true_image_shape},
        receiver_tensors={SERVING_FED_EXAMPLE_KEY: example})

  return _predict_input_fn


def get_reduce_to_frame_fn(input_reader_config, is_training):
  """Returns a function reducing sequence tensors to single frame tensors.

  If the input type is not TF_SEQUENCE_EXAMPLE, the tensors are passed through
  this function unchanged. Otherwise, when in training mode, a single frame is
  selected at random from the sequence example, and the tensors for that frame
  are converted to single frame tensors, with all associated context features.
  In evaluation mode all frames are converted to single frame tensors with
  copied context tensors. After the sequence example tensors are converted into
  one or many single frame tensors, the images from each frame are decoded.

  Args:
    input_reader_config: An input_reader_pb2.InputReader.
    is_training: Whether we are in training mode.

  Returns:
    `reduce_to_frame_fn` for the dataset builder
  """
  if input_reader_config.input_type != (
      input_reader_pb2.InputType.Value('TF_SEQUENCE_EXAMPLE')):
    return lambda dataset, dataset_map_fn, batch_size, config: dataset
  else:
    def reduce_to_frame(dataset, dataset_map_fn, batch_size,
                        input_reader_config):
      """Returns a function reducing sequence tensors to single frame tensors.

      Args:
        dataset: A tf dataset containing sequence tensors.
        dataset_map_fn: A function that handles whether to
          map_with_legacy_function for this dataset
        batch_size: used if map_with_legacy_function is true to determine
          num_parallel_calls
        input_reader_config: used if map_with_legacy_function is true to
          determine num_parallel_calls

      Returns:
        A tf dataset containing single frame tensors.
      """
      if is_training:
        def get_single_frame(tensor_dict):
          """Returns a random frame from a sequence.

          Picks a random frame and returns slices of sequence tensors
          corresponding to the random frame. Returns non-sequence tensors
          unchanged.

          Args:
            tensor_dict: A dictionary containing sequence tensors.

          Returns:
            Tensors for a single random frame within the sequence.
          """
          num_frames = tf.cast(
              tf.shape(tensor_dict[fields.InputDataFields.source_id])[0],
              dtype=tf.int32)
          if input_reader_config.frame_index == -1:
            frame_index = tf.random.uniform((), minval=0, maxval=num_frames,
                                            dtype=tf.int32)
          else:
            frame_index = tf.constant(input_reader_config.frame_index,
                                      dtype=tf.int32)
          out_tensor_dict = {}
          for key in tensor_dict:
            if key in fields.SEQUENCE_FIELDS:
              # Slice random frame from sequence tensors
              out_tensor_dict[key] = tensor_dict[key][frame_index]
            else:
              # Copy all context tensors.
              out_tensor_dict[key] = tensor_dict[key]
          return out_tensor_dict
        dataset = dataset_map_fn(dataset, get_single_frame, batch_size,
                                 input_reader_config)
      else:
        dataset = dataset_map_fn(dataset, util_ops.tile_context_tensors,
                                 batch_size, input_reader_config)
        dataset = dataset.unbatch()
      # Decode frame here as SequenceExample tensors contain encoded images.
      dataset = dataset_map_fn(dataset, util_ops.decode_image, batch_size,
                               input_reader_config)
      return dataset
    return reduce_to_frame
