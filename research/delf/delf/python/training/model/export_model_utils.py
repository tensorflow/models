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
"""Helper functions for DELF model exporting."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from delf import feature_extractor
from delf.python.training.datasets import googlelandmarks as gld
from object_detection.core import box_list
from object_detection.core import box_list_ops


def ExtractLocalFeatures(image, image_scales, max_feature_num, abs_thres, iou,
                         attention_model_fn, stride_factor):
  """Extract local features for input image.

  Args:
    image: image tensor of type tf.uint8 with shape [h, w, channels].
    image_scales: 1D float tensor which contains float scales used for image
      pyramid construction.
    max_feature_num: int tensor denotes the maximum selected feature points.
    abs_thres: float tensor denotes the score threshold for feature selection.
    iou: float scalar denotes the iou threshold for NMS.
    attention_model_fn: model function. Follows the signature:
      * Args:
        * `images`: Image tensor which is re-scaled.
      * Returns:
        * `attention_prob`: attention map after the non-linearity.
        * `feature_map`: feature map after ResNet convolution.
    stride_factor: integer accounting for striding after block3.

  Returns:
    boxes: [N, 4] float tensor which denotes the selected receptive box. N is
      the number of final feature points which pass through keypoint selection
      and NMS steps.
    features: [N, depth] float tensor.
    feature_scales: [N] float tensor. It is the inverse of the input image
      scales such that larger image scales correspond to larger image regions,
      which is compatible with keypoints detected with other techniques, for
      example Congas.
    scores: [N, 1] float tensor denotes the attention score.

  """
  original_image_shape_float = tf.gather(
      tf.dtypes.cast(tf.shape(image), tf.float32), [0, 1])

  image_tensor = gld.NormalizeImages(
      image, pixel_value_offset=128.0, pixel_value_scale=128.0)
  image_tensor = tf.expand_dims(image_tensor, 0, name='image/expand_dims')

  # Hard code the feature depth and receptive field parameters for now.
  rf, stride, padding = [291.0, 16.0 * stride_factor, 145.0]
  feature_depth = 1024

  def _ProcessSingleScale(scale_index, boxes, features, scales, scores):
    """Resizes the image and run feature extraction and keypoint selection.

       This function will be passed into tf.while_loop() and be called
       repeatedly. The input boxes are collected from the previous iteration
       [0: scale_index -1]. We get the current scale by
       image_scales[scale_index], and run resize image, feature extraction and
       keypoint selection. Then we will get a new set of selected_boxes for
       current scale. In the end, we concat the previous boxes with current
       selected_boxes as the output.
    Args:
      scale_index: A valid index in the image_scales.
      boxes: Box tensor with the shape of [N, 4].
      features: Feature tensor with the shape of [N, depth].
      scales: Scale tensor with the shape of [N].
      scores: Attention score tensor with the shape of [N].

    Returns:
      scale_index: The next scale index for processing.
      boxes: Concatenated box tensor with the shape of [K, 4]. K >= N.
      features: Concatenated feature tensor with the shape of [K, depth].
      scales: Concatenated scale tensor with the shape of [K].
      scores: Concatenated score tensor with the shape of [K].
    """
    scale = tf.gather(image_scales, scale_index)
    new_image_size = tf.dtypes.cast(
        tf.round(original_image_shape_float * scale), tf.int32)
    resized_image = tf.image.resize(image_tensor, new_image_size)

    attention_prob, feature_map = attention_model_fn(resized_image)
    attention_prob = tf.squeeze(attention_prob, axis=[0])
    feature_map = tf.squeeze(feature_map, axis=[0])

    rf_boxes = feature_extractor.CalculateReceptiveBoxes(
        tf.shape(feature_map)[0],
        tf.shape(feature_map)[1], rf, stride, padding)

    # Re-project back to the original image space.
    rf_boxes = tf.divide(rf_boxes, scale)
    attention_prob = tf.reshape(attention_prob, [-1])
    feature_map = tf.reshape(feature_map, [-1, feature_depth])

    # Use attention score to select feature vectors.
    indices = tf.reshape(tf.where(attention_prob >= abs_thres), [-1])
    selected_boxes = tf.gather(rf_boxes, indices)
    selected_features = tf.gather(feature_map, indices)
    selected_scores = tf.gather(attention_prob, indices)
    selected_scales = tf.ones_like(selected_scores, tf.float32) / scale

    # Concat with the previous result from different scales.
    boxes = tf.concat([boxes, selected_boxes], 0)
    features = tf.concat([features, selected_features], 0)
    scales = tf.concat([scales, selected_scales], 0)
    scores = tf.concat([scores, selected_scores], 0)

    return scale_index + 1, boxes, features, scales, scores

  output_boxes = tf.zeros([0, 4], dtype=tf.float32)
  output_features = tf.zeros([0, feature_depth], dtype=tf.float32)
  output_scales = tf.zeros([0], dtype=tf.float32)
  output_scores = tf.zeros([0], dtype=tf.float32)

  # Process the first scale separately, the following scales will reuse the
  # graph variables.
  (_, output_boxes, output_features, output_scales,
   output_scores) = _ProcessSingleScale(0, output_boxes, output_features,
                                        output_scales, output_scores)

  i = tf.constant(1, dtype=tf.int32)
  num_scales = tf.shape(image_scales)[0]
  keep_going = lambda j, b, f, scales, scores: tf.less(j, num_scales)

  (_, output_boxes, output_features, output_scales,
   output_scores) = tf.nest.map_structure(
       tf.stop_gradient,
       tf.while_loop(
           cond=keep_going,
           body=_ProcessSingleScale,
           loop_vars=[
               i, output_boxes, output_features, output_scales, output_scores
           ],
           shape_invariants=[
               i.get_shape(),
               tf.TensorShape([None, 4]),
               tf.TensorShape([None, feature_depth]),
               tf.TensorShape([None]),
               tf.TensorShape([None])
           ]))

  feature_boxes = box_list.BoxList(output_boxes)
  feature_boxes.add_field('features', output_features)
  feature_boxes.add_field('scales', output_scales)
  feature_boxes.add_field('scores', output_scores)

  nms_max_boxes = tf.minimum(max_feature_num, feature_boxes.num_boxes())
  final_boxes = box_list_ops.non_max_suppression(feature_boxes, iou,
                                                 nms_max_boxes)

  return final_boxes.get(), final_boxes.get_field(
      'features'), final_boxes.get_field('scales'), tf.expand_dims(
          final_boxes.get_field('scores'), 1)


@tf.function
def ExtractGlobalFeatures(image,
                          image_scales,
                          global_scales_ind,
                          model_fn,
                          multi_scale_pool_type='None',
                          normalize_global_descriptor=False):
  """Extract global features for input image.

  Args:
    image: image tensor of type tf.uint8 with shape [h, w, channels].
    image_scales: 1D float tensor which contains float scales used for image
      pyramid construction.
    global_scales_ind: Feature extraction happens only for a subset of
      `image_scales`, those with corresponding indices from this tensor.
    model_fn: model function. Follows the signature:
      * Args:
        * `images`: Image tensor which is re-scaled.
      * Returns:
        * `global_descriptors`: Global descriptors for input images.
    multi_scale_pool_type: If set, the global descriptor of each scale is pooled
      and a 1D global descriptor is returned.
    normalize_global_descriptor: If True, output global descriptors are
      L2-normalized.

  Returns:
    global_descriptors: If `multi_scale_pool_type` is 'None', returns a [S, D]
      float tensor. S is the number of scales, and D the global descriptor
      dimensionality. Each D-dimensional entry is a global descriptor, which may
      be L2-normalized depending on `normalize_global_descriptor`. If
      `multi_scale_pool_type` is not 'None', returns a [D] float tensor with the
      pooled global descriptor.

  """
  original_image_shape_float = tf.gather(
      tf.dtypes.cast(tf.shape(image), tf.float32), [0, 1])
  image_tensor = gld.NormalizeImages(
      image, pixel_value_offset=128.0, pixel_value_scale=128.0)
  image_tensor = tf.expand_dims(image_tensor, 0, name='image/expand_dims')

  def _ResizeAndExtract(scale_index):
    """Helper function to resize image then extract global feature.

    Args:
      scale_index: A valid index in image_scales.

    Returns:
      global_descriptor: [1,D] tensor denoting the extracted global descriptor.
    """
    scale = tf.gather(image_scales, scale_index)
    new_image_size = tf.dtypes.cast(
        tf.round(original_image_shape_float * scale), tf.int32)
    resized_image = tf.image.resize(image_tensor, new_image_size)
    global_descriptor = model_fn(resized_image)
    return global_descriptor

  # First loop to find initial scale to be used.
  num_scales = tf.shape(image_scales)[0]
  initial_scale_index = tf.constant(-1, dtype=tf.int32)
  for scale_index in tf.range(num_scales):
    if tf.reduce_any(tf.equal(global_scales_ind, scale_index)):
      initial_scale_index = scale_index
      break

  output_global = _ResizeAndExtract(initial_scale_index)

  # Loop over subsequent scales.
  for scale_index in tf.range(initial_scale_index + 1, num_scales):
    # Allow an undefined number of global feature scales to be extracted.
    tf.autograph.experimental.set_loop_options(
        shape_invariants=[(output_global, tf.TensorShape([None, None]))])

    if tf.reduce_any(tf.equal(global_scales_ind, scale_index)):
      global_descriptor = _ResizeAndExtract(scale_index)
      output_global = tf.concat([output_global, global_descriptor], 0)

  normalization_axis = 1
  if multi_scale_pool_type == 'average':
    output_global = tf.reduce_mean(
        output_global,
        axis=0,
        keepdims=False,
        name='multi_scale_average_pooling')
    normalization_axis = 0
  elif multi_scale_pool_type == 'sum':
    output_global = tf.reduce_sum(
        output_global, axis=0, keepdims=False, name='multi_scale_sum_pooling')
    normalization_axis = 0

  if normalize_global_descriptor:
    output_global = tf.nn.l2_normalize(
        output_global, axis=normalization_axis, name='l2_normalization')

  return output_global
