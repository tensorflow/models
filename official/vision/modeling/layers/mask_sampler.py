# Copyright 2024 The TensorFlow Authors. All Rights Reserved.
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

"""Contains definitions of mask sampler."""

# Import libraries
import tensorflow as tf, tf_keras

from official.vision.ops import spatial_transform_ops


def _sample_and_crop_foreground_masks(candidate_rois: tf.Tensor,
                                      candidate_gt_boxes: tf.Tensor,
                                      candidate_gt_classes: tf.Tensor,
                                      candidate_gt_indices: tf.Tensor,
                                      gt_masks: tf.Tensor,
                                      num_sampled_masks: int = 128,
                                      mask_target_size: int = 28):
  """Samples and creates cropped foreground masks for training.

  Args:
    candidate_rois: A `tf.Tensor` of shape of [batch_size, N, 4], where N is the
      number of candidate RoIs to be considered for mask sampling. It includes
      both positive and negative RoIs. The `num_mask_samples_per_image` positive
      RoIs will be sampled to create mask training targets.
    candidate_gt_boxes: A `tf.Tensor` of shape of [batch_size, N, 4], storing
      the corresponding groundtruth boxes to the `candidate_rois`.
    candidate_gt_classes: A `tf.Tensor` of shape of [batch_size, N], storing the
      corresponding groundtruth classes to the `candidate_rois`. 0 in the tensor
      corresponds to the background class, i.e. negative RoIs.
    candidate_gt_indices: A `tf.Tensor` of shape [batch_size, N], storing the
      corresponding groundtruth instance indices to the `candidate_gt_boxes`,
      i.e. gt_boxes[candidate_gt_indices[:, i]] = candidate_gt_boxes[:, i] and
      gt_boxes which is of shape [batch_size, MAX_INSTANCES, 4], M >= N, is
      the superset of candidate_gt_boxes.
    gt_masks: A `tf.Tensor` of [batch_size, MAX_INSTANCES, mask_height,
      mask_width] containing all the groundtruth masks which sample masks are
      drawn from.
    num_sampled_masks: An `int` that specifies the number of masks to sample.
    mask_target_size: An `int` that specifies the final cropped mask size after
      sampling. The output masks are resized w.r.t the sampled RoIs.

  Returns:
    foreground_rois: A `tf.Tensor` of shape of [batch_size, K, 4] storing the
      RoI that corresponds to the sampled foreground masks, where
      K = num_mask_samples_per_image.
    foreground_classes: A `tf.Tensor` of shape of [batch_size, K] storing the
      classes corresponding to the sampled foreground masks.
    cropoped_foreground_masks: A `tf.Tensor` of shape of
      [batch_size, K, mask_target_size, mask_target_size] storing the cropped
      foreground masks used for training.
  """
  _, fg_instance_indices = tf.nn.top_k(
      tf.cast(tf.greater(candidate_gt_classes, 0), dtype=tf.int32),
      k=num_sampled_masks)

  fg_instance_indices_shape = tf.shape(fg_instance_indices)
  batch_indices = (
      tf.expand_dims(tf.range(fg_instance_indices_shape[0]), axis=-1) *
      tf.ones([1, fg_instance_indices_shape[-1]], dtype=tf.int32))

  gather_nd_instance_indices = tf.stack(
      [batch_indices, fg_instance_indices], axis=-1)
  foreground_rois = tf.gather_nd(
      candidate_rois, gather_nd_instance_indices)
  foreground_boxes = tf.gather_nd(
      candidate_gt_boxes, gather_nd_instance_indices)
  foreground_classes = tf.gather_nd(
      candidate_gt_classes, gather_nd_instance_indices)
  foreground_gt_indices = tf.gather_nd(
      candidate_gt_indices, gather_nd_instance_indices)
  foreground_gt_indices = tf.where(
      tf.equal(foreground_gt_indices, -1),
      tf.zeros_like(foreground_gt_indices),
      foreground_gt_indices)

  foreground_gt_indices_shape = tf.shape(foreground_gt_indices)
  batch_indices = (
      tf.expand_dims(tf.range(foreground_gt_indices_shape[0]), axis=-1) *
      tf.ones([1, foreground_gt_indices_shape[-1]], dtype=tf.int32))
  gather_nd_gt_indices = tf.stack(
      [batch_indices, foreground_gt_indices], axis=-1)
  foreground_masks = tf.gather_nd(gt_masks, gather_nd_gt_indices)

  cropped_foreground_masks = spatial_transform_ops.crop_mask_in_target_box(
      foreground_masks, foreground_boxes, foreground_rois, mask_target_size,
      sample_offset=0.5)

  return foreground_rois, foreground_classes, cropped_foreground_masks


@tf_keras.utils.register_keras_serializable(package='Vision')
class MaskSampler(tf_keras.layers.Layer):
  """Samples and creates mask training targets."""

  def __init__(self, mask_target_size: int, num_sampled_masks: int, **kwargs):
    self._config_dict = {
        'mask_target_size': mask_target_size,
        'num_sampled_masks': num_sampled_masks,
    }
    super(MaskSampler, self).__init__(**kwargs)

  def call(self, candidate_rois: tf.Tensor, candidate_gt_boxes: tf.Tensor,
           candidate_gt_classes: tf.Tensor, candidate_gt_indices: tf.Tensor,
           gt_masks: tf.Tensor):
    """Samples and creates mask targets for training.

    Args:
      candidate_rois: A `tf.Tensor` of shape of [batch_size, N, 4], where N is
        the number of candidate RoIs to be considered for mask sampling. It
        includes both positive and negative RoIs. The
        `num_mask_samples_per_image` positive RoIs will be sampled to create
        mask training targets.
      candidate_gt_boxes: A `tf.Tensor` of shape of [batch_size, N, 4], storing
        the corresponding groundtruth boxes to the `candidate_rois`.
      candidate_gt_classes: A `tf.Tensor` of shape of [batch_size, N], storing
        the corresponding groundtruth classes to the `candidate_rois`. 0 in the
        tensor corresponds to the background class, i.e. negative RoIs.
      candidate_gt_indices: A `tf.Tensor` of shape [batch_size, N], storing the
        corresponding groundtruth instance indices to the `candidate_gt_boxes`,
        i.e. gt_boxes[candidate_gt_indices[:, i]] = candidate_gt_boxes[:, i],
          where gt_boxes which is of shape [batch_size, MAX_INSTANCES, 4], M >=
          N, is the superset of candidate_gt_boxes.
      gt_masks: A `tf.Tensor` of [batch_size, MAX_INSTANCES, mask_height,
        mask_width] containing all the groundtruth masks which sample masks are
        drawn from. after sampling. The output masks are resized w.r.t the
        sampled RoIs.

    Returns:
      foreground_rois: A `tf.Tensor` of shape of [batch_size, K, 4] storing the
        RoI that corresponds to the sampled foreground masks, where
        K = num_mask_samples_per_image.
      foreground_classes: A `tf.Tensor` of shape of [batch_size, K] storing the
        classes corresponding to the sampled foreground masks.
      cropoped_foreground_masks: A `tf.Tensor` of shape of
        [batch_size, K, mask_target_size, mask_target_size] storing the
        cropped foreground masks used for training.
    """
    foreground_rois, foreground_classes, cropped_foreground_masks = (
        _sample_and_crop_foreground_masks(
            candidate_rois,
            candidate_gt_boxes,
            candidate_gt_classes,
            candidate_gt_indices,
            gt_masks,
            self._config_dict['num_sampled_masks'],
            self._config_dict['mask_target_size']))
    return foreground_rois, foreground_classes, cropped_foreground_masks

  def get_config(self):
    return self._config_dict

  @classmethod
  def from_config(cls, config):
    return cls(**config)
