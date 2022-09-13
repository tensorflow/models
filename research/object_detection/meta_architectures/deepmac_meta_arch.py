"""Deep Mask heads above CenterNet (DeepMAC)[1] architecture.

[1]: https://arxiv.org/abs/2104.00613
"""

import collections

from absl import logging
import numpy as np
import tensorflow as tf

from object_detection.builders import losses_builder
from object_detection.core import box_list
from object_detection.core import box_list_ops
from object_detection.core import losses
from object_detection.core import preprocessor
from object_detection.core import standard_fields as fields
from object_detection.meta_architectures import center_net_meta_arch
from object_detection.models.keras_models import hourglass_network
from object_detection.models.keras_models import resnet_v1
from object_detection.protos import center_net_pb2
from object_detection.protos import losses_pb2
from object_detection.utils import shape_utils
from object_detection.utils import spatial_transform_ops
from object_detection.utils import tf_version

if tf_version.is_tf2():
  import tensorflow_io as tfio  # pylint:disable=g-import-not-at-top


INSTANCE_EMBEDDING = 'INSTANCE_EMBEDDING'
PIXEL_EMBEDDING = 'PIXEL_EMBEDDING'
MASK_LOGITS_GT_BOXES = 'MASK_LOGITS_GT_BOXES'
DEEP_MASK_ESTIMATION = 'deep_mask_estimation'
DEEP_MASK_BOX_CONSISTENCY = 'deep_mask_box_consistency'
DEEP_MASK_FEATURE_CONSISTENCY = 'deep_mask_feature_consistency'
DEEP_MASK_POINTLY_SUPERVISED = 'deep_mask_pointly_supervised'
SELF_SUPERVISED_DEAUGMENTED_MASK_LOGITS = (
    'SELF_SUPERVISED_DEAUGMENTED_MASK_LOGITS')
DEEP_MASK_AUGMENTED_SELF_SUPERVISION = 'deep_mask_augmented_self_supervision'
CONSISTENCY_FEATURE_MAP = 'CONSISTENCY_FEATURE_MAP'
LOSS_KEY_PREFIX = center_net_meta_arch.LOSS_KEY_PREFIX
NEIGHBORS_2D = [[-1, -1], [-1, 0], [-1, 1],
                [0, -1], [0, 1],
                [1, -1], [1, 0], [1, 1]]

WEAK_LOSSES = [DEEP_MASK_BOX_CONSISTENCY, DEEP_MASK_FEATURE_CONSISTENCY,
               DEEP_MASK_AUGMENTED_SELF_SUPERVISION,
               DEEP_MASK_POINTLY_SUPERVISED]

MASK_LOSSES = WEAK_LOSSES + [DEEP_MASK_ESTIMATION]


DeepMACParams = collections.namedtuple('DeepMACParams', [
        'classification_loss', 'dim', 'task_loss_weight', 'pixel_embedding_dim',
        'allowed_masked_classes_ids', 'mask_size', 'mask_num_subsamples',
        'use_xy', 'network_type', 'use_instance_embedding', 'num_init_channels',
        'predict_full_resolution_masks', 'postprocess_crop_size',
        'max_roi_jitter_ratio', 'roi_jitter_mode',
        'box_consistency_loss_weight', 'feature_consistency_threshold',
        'feature_consistency_dilation', 'feature_consistency_loss_weight',
        'box_consistency_loss_normalize', 'box_consistency_tightness',
        'feature_consistency_warmup_steps', 'feature_consistency_warmup_start',
        'use_only_last_stage', 'augmented_self_supervision_max_translation',
        'augmented_self_supervision_loss_weight',
        'augmented_self_supervision_flip_probability',
        'augmented_self_supervision_warmup_start',
        'augmented_self_supervision_warmup_steps',
        'augmented_self_supervision_loss',
        'augmented_self_supervision_scale_min',
        'augmented_self_supervision_scale_max',
        'pointly_supervised_keypoint_loss_weight',
        'ignore_per_class_box_overlap',
        'feature_consistency_type',
        'feature_consistency_comparison'
    ])


def _get_loss_weight(loss_name, config):
  """Utility function to get loss weights by name."""
  if loss_name == DEEP_MASK_ESTIMATION:
    return config.task_loss_weight
  elif loss_name == DEEP_MASK_FEATURE_CONSISTENCY:
    return config.feature_consistency_loss_weight
  elif loss_name == DEEP_MASK_BOX_CONSISTENCY:
    return config.box_consistency_loss_weight
  elif loss_name == DEEP_MASK_AUGMENTED_SELF_SUPERVISION:
    return config.augmented_self_supervision_loss_weight
  elif loss_name == DEEP_MASK_POINTLY_SUPERVISED:
    return config.pointly_supervised_keypoint_loss_weight
  else:
    raise ValueError('Unknown loss - {}'.format(loss_name))


def subsample_instances(classes, weights, boxes, masks, num_subsamples):
  """Randomly subsamples instances to the desired number.

  Args:
    classes: [num_instances, num_classes] float tensor of one-hot encoded
      classes.
    weights: [num_instances] float tensor of weights of each instance.
    boxes: [num_instances, 4] tensor of box coordinates.
    masks: [num_instances, height, width] tensor of per-instance masks.
    num_subsamples: int, the desired number of samples.

  Returns:
    classes: [num_subsamples, num_classes] float tensor of classes.
    weights: [num_subsamples] float tensor of weights.
    boxes: [num_subsamples, 4] float tensor of box coordinates.
    masks: [num_subsamples, height, width] float tensor of per-instance masks.

  """

  if num_subsamples <= -1:
    return classes, weights, boxes, masks

  num_instances = tf.reduce_sum(tf.cast(weights > 0.5, tf.int32))

  if num_instances <= num_subsamples:
    return (classes[:num_subsamples], weights[:num_subsamples],
            boxes[:num_subsamples], masks[:num_subsamples])

  else:
    random_index = tf.random.uniform([num_subsamples], 0, num_instances,
                                     dtype=tf.int32)

    return (tf.gather(classes, random_index), tf.gather(weights, random_index),
            tf.gather(boxes, random_index), tf.gather(masks, random_index))


def _get_deepmac_network_by_type(name, num_init_channels, mask_size=None):
  """Get DeepMAC network model given a string type."""

  if name.startswith('hourglass'):
    if name == 'hourglass10':
      return hourglass_network.hourglass_10(num_init_channels,
                                            initial_downsample=False)
    elif name == 'hourglass20':
      return hourglass_network.hourglass_20(num_init_channels,
                                            initial_downsample=False)
    elif name == 'hourglass32':
      return hourglass_network.hourglass_32(num_init_channels,
                                            initial_downsample=False)
    elif name == 'hourglass52':
      return hourglass_network.hourglass_52(num_init_channels,
                                            initial_downsample=False)
    elif name == 'hourglass100':
      return hourglass_network.hourglass_100(num_init_channels,
                                             initial_downsample=False)
    elif name == 'hourglass20_uniform_size':
      return hourglass_network.hourglass_20_uniform_size(num_init_channels)

    elif name == 'hourglass20_no_shortcut':
      return hourglass_network.hourglass_20_no_shortcut(num_init_channels)

  elif name == 'fully_connected':
    if not mask_size:
      raise ValueError('Mask size must be set.')
    return FullyConnectedMaskHead(num_init_channels, mask_size)

  elif _is_mask_head_param_free(name):
    return tf.keras.layers.Lambda(lambda x: x)

  elif name.startswith('resnet'):
    return ResNetMaskNetwork(name, num_init_channels)

  raise ValueError('Unknown network type {}'.format(name))


def boxes_batch_normalized_to_absolute_coordinates(boxes, height, width):
  ymin, xmin, ymax, xmax = tf.unstack(boxes, axis=2)
  height, width = tf.cast(height, tf.float32), tf.cast(width, tf.float32)
  ymin *= height
  ymax *= height
  xmin *= width
  xmax *= width

  return tf.stack([ymin, xmin, ymax, xmax], axis=2)


def boxes_batch_absolute_to_normalized_coordinates(boxes, height, width):
  ymin, xmin, ymax, xmax = tf.unstack(boxes, axis=2)
  height, width = tf.cast(height, tf.float32), tf.cast(width, tf.float32)
  ymin /= height
  ymax /= height
  xmin /= width
  xmax /= width

  return tf.stack([ymin, xmin, ymax, xmax], axis=2)


def _resize_instance_masks_non_empty(masks, shape):
  """Resize a non-empty tensor of masks to the given shape."""
  height, width = shape
  flattened_masks, batch_size, num_instances = flatten_first2_dims(masks)
  flattened_masks = flattened_masks[:, :, :, tf.newaxis]
  flattened_masks = tf.image.resize(
      flattened_masks, (height, width),
      method=tf.image.ResizeMethod.BILINEAR)
  return unpack_first2_dims(
      flattened_masks[:, :, :, 0], batch_size, num_instances)


def resize_instance_masks(masks, shape):
  batch_size, num_instances = tf.shape(masks)[0], tf.shape(masks)[1]
  return tf.cond(
      tf.shape(masks)[1] == 0,
      lambda: tf.zeros((batch_size, num_instances, shape[0], shape[1])),
      lambda: _resize_instance_masks_non_empty(masks, shape))


def filter_masked_classes(masked_class_ids, classes, weights, masks):
  """Filter out masks whose class IDs are not present in masked_class_ids.

  Args:
    masked_class_ids: A list of class IDs allowed to have masks. These class IDs
      are 1-indexed.
    classes: A [batch_size, num_instances, num_classes] float tensor containing
      the one-hot encoded classes.
    weights: A [batch_size, num_instances] float tensor containing the weights
      of each sample.
    masks: A [batch_size, num_instances, height, width] tensor containing the
      mask per instance.

  Returns:
    classes_filtered: A [batch_size, num_instances, num_classes] float tensor
       containing the one-hot encoded classes with classes not in
       masked_class_ids zeroed out.
    weights_filtered: A [batch_size, num_instances] float tensor containing the
      weights of each sample with instances whose classes aren't in
      masked_class_ids zeroed out.
    masks_filtered: A [batch_size, num_instances, height, width] tensor
      containing the mask per instance with masks not belonging to
      masked_class_ids zeroed out.
  """

  if len(masked_class_ids) == 0:  # pylint:disable=g-explicit-length-test
    return classes, weights, masks

  if tf.shape(classes)[1] == 0:
    return classes, weights, masks

  masked_class_ids = tf.constant(np.array(masked_class_ids, dtype=np.int32))
  label_id_offset = 1
  masked_class_ids -= label_id_offset
  class_ids = tf.argmax(classes, axis=2, output_type=tf.int32)
  matched_classes = tf.equal(
      class_ids[:, :, tf.newaxis], masked_class_ids[tf.newaxis, tf.newaxis, :]
  )

  matched_classes = tf.reduce_any(matched_classes, axis=2)
  matched_classes = tf.cast(matched_classes, tf.float32)

  return (
      classes * matched_classes[:, :, tf.newaxis],
      weights * matched_classes,
      masks * matched_classes[:, :, tf.newaxis, tf.newaxis]
  )


def per_instance_no_class_overlap(classes, boxes, height, width):
  """Returns 1s inside boxes but overlapping boxes of same class are zeroed out.

  Args:
    classes: A [batch_size, num_instances, num_classes] float tensor containing
      the one-hot encoded classes.
    boxes: A [batch_size, num_instances, 4] shaped float tensor of normalized
      boxes.
    height: int, height of the desired mask.
    width: int, width of the desired mask.

  Returns:
    mask: A [batch_size, num_instances, height, width] float tensor of 0s and
      1s.
  """
  box_mask = fill_boxes(boxes, height, width)
  per_class_box_mask = (
      box_mask[:, :, tf.newaxis, :, :] *
      classes[:, :, :, tf.newaxis, tf.newaxis])

  per_class_instance_count = tf.reduce_sum(per_class_box_mask, axis=1)
  per_class_valid_map = per_class_instance_count < 2
  class_indices = tf.argmax(classes, axis=2)

  per_instance_valid_map = tf.gather(
      per_class_valid_map, class_indices, batch_dims=1)

  return tf.cast(per_instance_valid_map, tf.float32)


def flatten_first2_dims(tensor):
  """Flatten first 2 dimensions of a tensor.

  Args:
    tensor: A tensor with shape [M, N, ....]

  Returns:
    flattened_tensor: A tensor of shape [M * N, ...]
    M: int, the length of the first dimension of the input.
    N: int, the length of the second dimension of the input.
  """
  shape = tf.shape(tensor)
  d1, d2, rest = shape[0], shape[1], shape[2:]

  tensor = tf.reshape(
      tensor, tf.concat([[d1 * d2], rest], axis=0))
  return tensor, d1, d2


def unpack_first2_dims(tensor, dim1, dim2):
  """Unpack the flattened first dimension of the tensor into 2 dimensions.

  Args:
    tensor: A tensor of shape [dim1 * dim2, ...]
    dim1: int, the size of the first dimension.
    dim2: int, the size of the second dimension.

  Returns:
    unflattened_tensor: A tensor of shape [dim1, dim2, ...].
  """
  shape = tf.shape(tensor)
  result_shape = tf.concat([[dim1, dim2], shape[1:]], axis=0)
  return tf.reshape(tensor, result_shape)


def crop_and_resize_instance_masks(masks, boxes, mask_size):
  """Crop and resize each mask according to the given boxes.

  Args:
    masks: A [B, N, H, W] float tensor.
    boxes: A [B, N, 4] float tensor of normalized boxes.
    mask_size: int, the size of the output masks.

  Returns:
    masks: A [B, N, mask_size, mask_size] float tensor of cropped and resized
      instance masks.
  """

  masks, batch_size, num_instances = flatten_first2_dims(masks)
  boxes, _, _ = flatten_first2_dims(boxes)
  cropped_masks = spatial_transform_ops.matmul_crop_and_resize(
      masks[:, :, :, tf.newaxis], boxes[:, tf.newaxis, :],
      [mask_size, mask_size])
  cropped_masks = tf.squeeze(cropped_masks, axis=[1, 4])
  return unpack_first2_dims(cropped_masks, batch_size, num_instances)


def fill_boxes(boxes, height, width):
  """Fills the area included in the boxes with 1s.

  Args:
    boxes: A [batch_size, num_instances, 4] shaped float tensor of boxes given
      in the normalized coordinate space.
    height: int, height of the output image.
    width: int, width of the output image.

  Returns:
    filled_boxes: A [batch_size, num_instances, height, width] shaped float
      tensor with 1s in the area that falls inside each box.
  """

  boxes_abs = boxes_batch_normalized_to_absolute_coordinates(
      boxes, height, width)
  ymin, xmin, ymax, xmax = tf.unstack(
      boxes_abs[:, :, tf.newaxis, tf.newaxis, :], 4, axis=4)

  ygrid, xgrid = tf.meshgrid(tf.range(height), tf.range(width), indexing='ij')
  ygrid, xgrid = tf.cast(ygrid, tf.float32), tf.cast(xgrid, tf.float32)
  ygrid, xgrid = (ygrid[tf.newaxis, tf.newaxis, :, :],
                  xgrid[tf.newaxis, tf.newaxis, :, :])

  filled_boxes = tf.logical_and(
      tf.logical_and(ygrid >= ymin, ygrid <= ymax),
      tf.logical_and(xgrid >= xmin, xgrid <= xmax))

  return tf.cast(filled_boxes, tf.float32)


def embedding_projection(x, y):
  """Compute dot product between two given embeddings.

  Args:
    x: [num_instances, height, width, dimension] float tensor input.
    y: [num_instances, height, width, dimension] or
      [num_instances, 1, 1, dimension] float tensor input. When the height
      and width dimensions are 1, TF will broadcast it.

  Returns:
    dist: [num_instances, height, width, 1] A float tensor returning
      the per-pixel embedding projection.
  """

  dot = tf.reduce_sum(x * y, axis=3, keepdims=True)
  return  dot


def _get_2d_neighbors_kernel():
  """Returns a conv. kernel that when applies generates 2D neighbors.

  Returns:
    kernel: A float tensor of shape [3, 3, 1, 8]
  """

  kernel = np.zeros((3, 3, 1, 8))

  for i, (y, x) in enumerate(NEIGHBORS_2D):
    kernel[1 + y, 1 + x, 0, i] = 1.0

  return tf.constant(kernel, dtype=tf.float32)


def generate_2d_neighbors(input_tensor, dilation=2):
  """Generate a feature map of 2D neighbors.

  Note: This op makes 8 (# of neighbors) as the leading dimension so that
  following ops on TPU won't have to pad the last dimension to 128.

  Args:
    input_tensor: A float tensor of shape [batch_size, height, width, channels].
    dilation: int, the dilation factor for considering neighbors.

  Returns:
    output: A float tensor of all 8 2-D neighbors. of shape
      [8, batch_size, height, width, channels].
  """

  # TODO(vighneshb) Minimize tranposing here to save memory.

  # input_tensor: [B, C, H, W]
  input_tensor = tf.transpose(input_tensor, (0, 3, 1, 2))
  # input_tensor: [B, C, H, W, 1]
  input_tensor = input_tensor[:, :, :, :, tf.newaxis]

  # input_tensor: [B * C, H, W, 1]
  input_tensor, batch_size, channels = flatten_first2_dims(input_tensor)

  kernel = _get_2d_neighbors_kernel()

  # output: [B * C, H, W, 8]
  output = tf.nn.atrous_conv2d(input_tensor, kernel, rate=dilation,
                               padding='SAME')
  # output: [B, C, H, W, 8]
  output = unpack_first2_dims(output, batch_size, channels)

  # return: [8, B, H, W, C]
  return tf.transpose(output, [4, 0, 2, 3, 1])


def normalize_feature_map(feature_map):
  return tf.math.l2_normalize(feature_map, axis=3, epsilon=1e-4)


def gaussian_pixel_similarity(a, b, theta):
  norm_difference = tf.linalg.norm(a - b, axis=-1)
  similarity = tf.exp(-norm_difference / theta)
  return similarity


def dotprod_pixel_similarity(a, b):
  return tf.reduce_sum(a * b, axis=-1)


def dilated_cross_pixel_similarity(feature_map, dilation=2, theta=2.0,
                                   method='gaussian'):
  """Dilated cross pixel similarity.

  method supports 2 values
  - 'gaussian' from https://arxiv.org/abs/2012.02310
  - 'dotprod' computes the dot product between feature vector for similarity.
     This assumes that the features are normalized.

  Args:
    feature_map: A float tensor of shape [batch_size, height, width, channels]
    dilation: int, the dilation factor.
    theta: The denominator while taking difference inside the gaussian.
    method: str, either 'gaussian' or 'dotprod'.

  Returns:
    dilated_similarity: A tensor of shape [8, batch_size, height, width]
  """
  neighbors = generate_2d_neighbors(feature_map, dilation)
  feature_map = feature_map[tf.newaxis]

  if method == 'gaussian':
    return gaussian_pixel_similarity(feature_map, neighbors, theta=theta)
  elif method == 'dotprod':
    return dotprod_pixel_similarity(feature_map, neighbors)
  else:
    raise ValueError('Unknown method for pixel sim %s' % method)


def dilated_cross_same_mask_label(instance_masks, dilation=2):
  """Dilated cross pixel similarity as defined in [1].

  [1]: https://arxiv.org/abs/2012.02310

  Args:
    instance_masks: A float tensor of shape [batch_size, num_instances,
      height, width]
    dilation: int, the dilation factor.

  Returns:
    dilated_same_label: A tensor of shape [8, batch_size, num_instances,
      height, width]
  """

  # instance_masks: [batch_size, height, width, num_instances]
  instance_masks = tf.transpose(instance_masks, (0, 2, 3, 1))

  # neighbors: [8, batch_size, height, width, num_instances]
  neighbors = generate_2d_neighbors(instance_masks, dilation)
  # instance_masks = [1, batch_size, height, width, num_instances]
  instance_masks = instance_masks[tf.newaxis]
  same_mask_prob = ((instance_masks * neighbors) +
                    ((1 - instance_masks) * (1 - neighbors)))

  return tf.transpose(same_mask_prob, (0, 1, 4, 2, 3))


def _per_pixel_single_conv(input_tensor, params, channels):
  """Convolve the given input with the given params.

  Args:
    input_tensor: A [num_instances, height, width, channels] shaped
      float tensor.
    params: A [num_instances, num_params] shaped float tensor.
    channels: int, number of channels in the convolution.

  Returns:
    output: A float tensor of shape [num_instances, height, width, channels]
  """

  input_channels = input_tensor.get_shape().as_list()[3]
  weights = params[:, :(input_channels * channels)]
  biases = params[:, (input_channels * channels):]
  num_instances = tf.shape(params)[0]

  weights = tf.reshape(weights, (num_instances, input_channels, channels))
  output = (input_tensor[:, :, tf.newaxis, :] @
            weights[:, tf.newaxis, tf.newaxis, :, :])

  output = output[:, :, 0, :, :]
  output = output + biases[:, tf.newaxis, tf.newaxis, :]
  return output


def per_pixel_conditional_conv(input_tensor, parameters, channels, depth):
  """Use parameters perform per-pixel convolutions with the given depth [1].

  [1]: https://arxiv.org/abs/2003.05664

  Args:
    input_tensor: float tensor of shape [num_instances, height,
      width, input_channels]
    parameters: A [num_instances, num_params] float tensor. If num_params
      is incomparible with the given channels and depth, a ValueError will
      be raised.
    channels: int, the number of channels in the convolution.
    depth: int, the number of layers of convolutions to perform.

  Returns:
    output: A [num_instances, height, width] tensor with the conditional
      conv applied according to each instance's parameters.
  """

  input_channels = input_tensor.get_shape().as_list()[3]
  num_params = parameters.get_shape().as_list()[1]

  input_convs = 1 if depth > 1 else 0
  intermediate_convs = depth - 2 if depth >= 2 else 0
  expected_weights = ((input_channels * channels * input_convs) +
                      (channels * channels * intermediate_convs) +
                      channels)  # final conv
  expected_biases = (channels * (depth - 1)) + 1

  if depth == 1:
    if input_channels != channels:
      raise ValueError(
          'When depth=1, input_channels({}) should be equal to'.format(
              input_channels) + ' channels({})'.format(channels))

  if num_params != (expected_weights + expected_biases):
    raise ValueError('Expected {} parameters at depth {}, but got {}'.format(
        expected_weights + expected_biases, depth, num_params))

  start = 0
  output = input_tensor
  for i in range(depth):

    is_last_layer = i == (depth - 1)
    if is_last_layer:
      channels = 1

    num_params_single_conv = channels * input_channels + channels
    params = parameters[:, start:start + num_params_single_conv]

    start += num_params_single_conv
    output = _per_pixel_single_conv(output, params, channels)

    if not is_last_layer:
      output = tf.nn.relu(output)

    input_channels = channels

  return output


def flip_boxes_left_right(boxes):
  ymin, xmin, ymax, xmax = tf.unstack(boxes, axis=2)

  return tf.stack(
      [ymin, 1.0 - xmax, ymax, 1.0 - xmin], axis=2
  )


def transform_images_and_boxes(images, boxes, tx, ty, sx, sy, flip):
  """Translate and scale a batch of images and boxes by the given amount.

  The function first translates and then scales the image and assumes the
  origin to be at the center of the image.

  Args:
    images: A [batch_size, height, width, 3] float tensor of images.
    boxes: optional, A [batch_size, num_instances, 4] shaped float tensor of
      normalized bounding boxes. If None, the second return value is always
      None.
    tx: A [batch_size] shaped float tensor of x translations.
    ty: A [batch_size] shaped float tensor of y translations.
    sx: A [batch_size] shaped float tensor of x scale factor.
    sy: A [batch_size] shaped float tensor of y scale factor.
    flip: A [batch_size] shaped bool tensor indicating whether or not we
      flip the image.

  Returns:
    transformed_images: Transfomed images of same shape as `images`.
    transformed_boxes: If `boxes` was not None, transformed boxes of same
      shape as boxes.

  """
  _, height, width, _ = shape_utils.combined_static_and_dynamic_shape(
      images)

  flip_selector = tf.cast(flip, tf.float32)
  flip_selector_4d = flip_selector[:, tf.newaxis, tf.newaxis, tf.newaxis]
  flip_selector_3d = flip_selector[:, tf.newaxis, tf.newaxis]
  flipped_images = tf.image.flip_left_right(images)
  images = flipped_images * flip_selector_4d + (1.0 - flip_selector_4d) * images

  cy = cx = tf.zeros_like(tx) + 0.5
  ymin = -ty*sy + cy - sy * 0.5
  xmin = -tx*sx + cx - sx * 0.5
  ymax = -ty*sy + cy + sy * 0.5
  xmax = -tx*sx + cx + sx * 0.5
  crop_box = tf.stack([ymin, xmin, ymax, xmax], axis=1)

  crop_box_expanded = crop_box[:, tf.newaxis, :]

  images_transformed = spatial_transform_ops.matmul_crop_and_resize(
      images, crop_box_expanded, (height, width)
  )
  images_transformed = images_transformed[:, 0, :, :, :]

  if boxes is not None:
    flipped_boxes = flip_boxes_left_right(boxes)
    boxes = flipped_boxes * flip_selector_3d + (1.0 - flip_selector_3d) * boxes
    win_height = ymax - ymin
    win_width = xmax - xmin
    win_height = win_height[:, tf.newaxis]
    win_width = win_width[:, tf.newaxis]
    boxes_transformed = (
        boxes - tf.stack([ymin, xmin, ymin, xmin], axis=1)[:, tf.newaxis, :])

    boxes_ymin, boxes_xmin, boxes_ymax, boxes_xmax = tf.unstack(
        boxes_transformed, axis=2)
    boxes_ymin *= 1.0 / win_height
    boxes_xmin *= 1.0 / win_width
    boxes_ymax *= 1.0 / win_height
    boxes_xmax *= 1.0 / win_width

    boxes = tf.stack([boxes_ymin, boxes_xmin, boxes_ymax, boxes_xmax], axis=2)

  return images_transformed, boxes


def transform_instance_masks(instance_masks, tx, ty, sx, sy, flip):
  """Transforms a batch of instances by the given amount.

  Args:
    instance_masks: A [batch_size, num_instances, height, width, 3] float
      tensor of instance masks.
    tx: A [batch_size] shaped float tensor of x translations.
    ty: A [batch_size] shaped float tensor of y translations.
    sx: A [batch_size] shaped float tensor of x scale factor.
    sy: A [batch_size] shaped float tensor of y scale factor.
    flip: A [batch_size] shaped bool tensor indicating whether or not we
      flip the image.

  Returns:
    transformed_images: Transfomed images of same shape as `images`.
    transformed_boxes: If `boxes` was not None, transformed boxes of same
      shape as boxes.

  """
  instance_masks, batch_size, num_instances = flatten_first2_dims(
      instance_masks)

  repeat = tf.zeros_like(tx, dtype=tf.int32) + num_instances
  tx = tf.repeat(tx, repeat)
  ty = tf.repeat(ty, repeat)
  sx = tf.repeat(sx, repeat)
  sy = tf.repeat(sy, repeat)
  flip = tf.repeat(flip, repeat)

  instance_masks = instance_masks[:, :, :, tf.newaxis]
  instance_masks, _ = transform_images_and_boxes(
      instance_masks, boxes=None, tx=tx, ty=ty, sx=sx, sy=sy, flip=flip)

  return unpack_first2_dims(
      instance_masks[:, :, :, 0], batch_size, num_instances)


class ResNetMaskNetwork(tf.keras.layers.Layer):
  """A small wrapper around ResNet blocks to predict masks."""

  def __init__(self, resnet_type, num_init_channels):
    """Creates the ResNet mask network.

    Args:
      resnet_type: A string of the for resnetN where N where N is in
        [4, 8, 12, 16, 20]
      num_init_channels: Number of filters in the ResNet block.
    """

    super(ResNetMaskNetwork, self).__init__()
    nc = num_init_channels

    if resnet_type == 'resnet4':
      channel_dims = [nc * 2]
      blocks = [2]
    elif resnet_type == 'resnet8':
      channel_dims = [nc * 2]
      blocks = [4]
    elif resnet_type == 'resnet12':
      channel_dims = [nc * 2]
      blocks = [6]
    elif resnet_type == 'resnet16':
      channel_dims = [nc * 2]
      blocks = [8]
    # Defined such that the channels are roughly similar to the hourglass20.
    elif resnet_type == 'resnet20':
      channel_dims = [nc * 2, nc * 3]
      blocks = [8, 2]
    else:
      raise ValueError('Unknown resnet type "{}"'.format(resnet_type))

    self.input_layer = tf.keras.layers.Conv2D(nc, 1, 1)

    # Last channel has to be defined so that batch norm can initialize properly.
    model_input = tf.keras.layers.Input([None, None, nc])
    output = model_input

    for i, (num_blocks, channels) in enumerate(zip(blocks, channel_dims)):
      output = resnet_v1.stack_basic(output, filters=channels,
                                     blocks=num_blocks, stride1=1,
                                     name='resnet_mask_block_%d' % i)
    self.model = tf.keras.Model(inputs=model_input, outputs=output)

  def __call__(self, inputs):
    return self.model(self.input_layer(inputs))


class FullyConnectedMaskHead(tf.keras.layers.Layer):
  """A 2 layer fully connected mask head."""

  def __init__(self, num_init_channels, mask_size):
    super(FullyConnectedMaskHead, self).__init__()
    self.fc1 = tf.keras.layers.Dense(units=1024, activation='relu')
    self.fc2 = tf.keras.layers.Dense(units=mask_size*mask_size)
    self.mask_size = mask_size
    self.num_input_channels = num_init_channels
    self.input_layer = tf.keras.layers.Conv2D(num_init_channels, 1, 1)
    model_input = tf.keras.layers.Input(
        [mask_size * mask_size * num_init_channels,])
    output = self.fc2(self.fc1(model_input))
    self.model = tf.keras.Model(inputs=model_input, outputs=output)

  def __call__(self, inputs):
    inputs = self.input_layer(inputs)
    inputs_shape = tf.shape(inputs)
    num_instances = inputs_shape[0]
    height = inputs_shape[1]
    width = inputs_shape[2]
    dims = inputs_shape[3]
    flattened_inputs = tf.reshape(inputs,
                                  [num_instances, height * width * dims])
    flattened_masks = self.model(flattened_inputs)
    return tf.reshape(flattened_masks,
                      [num_instances, self.mask_size, self.mask_size, 1])


class DenseResidualBlock(tf.keras.layers.Layer):
  """Residual block for 1D inputs.

  This class implemented the pre-activation version of the ResNet block.
  """

  def __init__(self, hidden_size, use_shortcut_linear):
    """Residual Block for 1D inputs.

    Args:
      hidden_size: size of the hidden layer.
      use_shortcut_linear: bool, whether or not to use a linear layer for
        shortcut.
    """

    super(DenseResidualBlock, self).__init__()

    self.bn_0 = tf.keras.layers.experimental.SyncBatchNormalization(axis=-1)
    self.bn_1 = tf.keras.layers.experimental.SyncBatchNormalization(axis=-1)

    self.fc_0 = tf.keras.layers.Dense(
        hidden_size, activation=None)
    self.fc_1 = tf.keras.layers.Dense(
        hidden_size, activation=None, kernel_initializer='zeros')

    self.activation = tf.keras.layers.Activation('relu')

    if use_shortcut_linear:
      self.shortcut = tf.keras.layers.Dense(
          hidden_size, activation=None, use_bias=False)
    else:
      self.shortcut = tf.keras.layers.Lambda(lambda x: x)

  def __call__(self, inputs):
    """Layer's forward pass.

    Args:
      inputs: input tensor.

    Returns:
      Tensor after residual block w/ CondBatchNorm.
    """
    out = self.fc_0(self.activation(self.bn_0(inputs)))
    residual_inp = self.fc_1(self.activation(self.bn_1(out)))

    skip = self.shortcut(inputs)

    return residual_inp + skip


class DenseResNet(tf.keras.layers.Layer):
  """Resnet with dense layers."""

  def __init__(self, num_layers, hidden_size, output_size):
    """Resnet with dense layers.

    Args:
      num_layers: int, the number of layers.
      hidden_size: size of the hidden layer.
      output_size: size of the output.
    """

    super(DenseResNet, self).__init__()

    self.input_proj = DenseResidualBlock(hidden_size, use_shortcut_linear=True)
    if num_layers < 4:
      raise ValueError(
          'Cannot construct a DenseResNet with less than 4 layers')

    num_blocks = (num_layers - 2) // 2

    if ((num_blocks * 2) + 2) != num_layers:
      raise ValueError(('DenseResNet depth has to be of the form (2n + 2). '
                        f'Found {num_layers}'))

    self._num_blocks = num_blocks
    blocks = [DenseResidualBlock(hidden_size, use_shortcut_linear=False)
              for _ in range(num_blocks)]
    self.resnet = tf.keras.Sequential(blocks)
    self.out_conv = tf.keras.layers.Dense(output_size)

  def __call__(self, inputs):
    net = self.input_proj(inputs)
    return self.out_conv(self.resnet(net))


def _is_mask_head_param_free(name):

  # Mask heads which don't have parameters of their own and instead rely
  # on the instance embedding.

  if name == 'embedding_projection' or name.startswith('cond_inst'):
    return True
  return False


class MaskHeadNetwork(tf.keras.layers.Layer):
  """Mask head class for DeepMAC."""

  def __init__(self, network_type, num_init_channels=64,
               use_instance_embedding=True, mask_size=None):
    """Initializes the network.

    Args:
      network_type: A string denoting the kind of network we want to use
        internally.
      num_init_channels: int, the number of channels in the first block. The
        number of channels in the following blocks depend on the network type
        used.
      use_instance_embedding: bool, if set, we concatenate the instance
        embedding to the input while predicting the mask.
      mask_size: int, size of the output mask. Required only with
        `fully_connected` mask type.
    """

    super(MaskHeadNetwork, self).__init__()

    self._net = _get_deepmac_network_by_type(
        network_type, num_init_channels, mask_size)
    self._use_instance_embedding = use_instance_embedding

    self._network_type = network_type
    self._num_init_channels = num_init_channels

    if (self._use_instance_embedding and
        (_is_mask_head_param_free(network_type))):
      raise ValueError(('Cannot feed instance embedding to mask head when '
                        'mask-head has no parameters.'))

    if _is_mask_head_param_free(network_type):
      self.project_out = tf.keras.layers.Lambda(lambda x: x)
    else:
      self.project_out = tf.keras.layers.Conv2D(
          filters=1, kernel_size=1, activation=None)

  def __call__(self, instance_embedding, pixel_embedding, training):
    """Returns mask logits given object center and spatial embeddings.

    Args:
      instance_embedding: A [num_instances, embedding_size] float tensor
        representing the center emedding vector of each instance.
      pixel_embedding: A [num_instances, height, width, pixel_embedding_size]
        float tensor representing the per-pixel spatial embedding for each
        instance.
      training: boolean flag indicating training or testing mode.

    Returns:
      mask: A [num_instances, height, width] float tensor containing the mask
        logits for each instance.
    """

    height = tf.shape(pixel_embedding)[1]
    width = tf.shape(pixel_embedding)[2]

    if self._use_instance_embedding:
      instance_embedding = instance_embedding[:, tf.newaxis, tf.newaxis, :]
      instance_embedding = tf.tile(instance_embedding, [1, height, width, 1])
      inputs = tf.concat([pixel_embedding, instance_embedding], axis=3)
    else:
      inputs = pixel_embedding

    out = self._net(inputs)
    if isinstance(out, list):
      out = out[-1]

    if self._network_type == 'embedding_projection':
      instance_embedding = instance_embedding[:, tf.newaxis, tf.newaxis, :]
      out = embedding_projection(instance_embedding, out)

    elif self._network_type.startswith('cond_inst'):
      depth = int(self._network_type.lstrip('cond_inst'))
      out = per_pixel_conditional_conv(out, instance_embedding,
                                       self._num_init_channels, depth)

    if out.shape[-1] > 1:
      out = self.project_out(out)

    return tf.squeeze(out, axis=-1)


def _batch_gt_list(gt_list):
  return tf.stack(gt_list, axis=0)


def deepmac_proto_to_params(deepmac_config):
  """Convert proto to named tuple."""

  loss = losses_pb2.Loss()
  # Add dummy localization loss to avoid the loss_builder throwing error.
  loss.localization_loss.weighted_l2.CopyFrom(
      losses_pb2.WeightedL2LocalizationLoss())

  loss.classification_loss.CopyFrom(deepmac_config.classification_loss)
  classification_loss, _, _, _, _, _, _ = (losses_builder.build(loss))

  deepmac_field_class = (
      center_net_pb2.CenterNet.DESCRIPTOR.nested_types_by_name[
          'DeepMACMaskEstimation'])

  params = {}
  for field in deepmac_field_class.fields:
    value = getattr(deepmac_config, field.name)
    if field.enum_type:
      params[field.name] = field.enum_type.values_by_number[value].name.lower()
    else:
      params[field.name] = value

  params['roi_jitter_mode'] = params.pop('jitter_mode')
  params['classification_loss'] = classification_loss
  return DeepMACParams(**params)


def _warmup_weight(current_training_step, warmup_start, warmup_steps):
  """Utility function for warming up loss weights."""

  if warmup_steps == 0:
    return 1.0

  training_step = tf.cast(current_training_step, tf.float32)
  warmup_steps = tf.cast(warmup_steps, tf.float32)
  start_step = tf.cast(warmup_start, tf.float32)
  warmup_weight = (training_step - start_step) / warmup_steps
  warmup_weight = tf.clip_by_value(warmup_weight, 0.0, 1.0)
  return warmup_weight


class DeepMACMetaArch(center_net_meta_arch.CenterNetMetaArch):
  """The experimental CenterNet DeepMAC[1] model.

  [1]: https://arxiv.org/abs/2104.00613
  """

  def __init__(self,
               is_training,
               add_summaries,
               num_classes,
               feature_extractor,
               image_resizer_fn,
               object_center_params,
               object_detection_params,
               deepmac_params: DeepMACParams,
               compute_heatmap_sparse=False):
    """Constructs the super class with object center & detection params only."""

    self._deepmac_params = deepmac_params
    if (self._deepmac_params.predict_full_resolution_masks and
        self._deepmac_params.max_roi_jitter_ratio > 0.0):
      raise ValueError('Jittering is not supported for full res masks.')

    if self._deepmac_params.mask_num_subsamples > 0:
      raise ValueError('Subsampling masks is currently not supported.')

    if self._deepmac_params.network_type == 'embedding_projection':
      if self._deepmac_params.use_xy:
        raise ValueError(
            'Cannot use x/y coordinates when using embedding projection.')

      pixel_embedding_dim = self._deepmac_params.pixel_embedding_dim
      dim = self._deepmac_params.dim
      if dim != pixel_embedding_dim:
        raise ValueError(
            'When using embedding projection mask head, '
            f'pixel_embedding_dim({pixel_embedding_dim}) '
            f'must be same as dim({dim}).')

    generator_class = tf.random.Generator
    self._self_supervised_rng = generator_class.from_non_deterministic_state()
    super(DeepMACMetaArch, self).__init__(
        is_training=is_training, add_summaries=add_summaries,
        num_classes=num_classes, feature_extractor=feature_extractor,
        image_resizer_fn=image_resizer_fn,
        object_center_params=object_center_params,
        object_detection_params=object_detection_params,
        compute_heatmap_sparse=compute_heatmap_sparse)

  def _construct_prediction_heads(self, num_classes, num_feature_outputs,
                                  class_prediction_bias_init):
    super_instance = super(DeepMACMetaArch, self)
    prediction_heads = super_instance._construct_prediction_heads(  # pylint:disable=protected-access
        num_classes, num_feature_outputs, class_prediction_bias_init)

    if self._deepmac_params is not None:
      prediction_heads[INSTANCE_EMBEDDING] = [
          center_net_meta_arch.make_prediction_net(self._deepmac_params.dim)
          for _ in range(num_feature_outputs)
      ]

      prediction_heads[PIXEL_EMBEDDING] = [
          center_net_meta_arch.make_prediction_net(
              self._deepmac_params.pixel_embedding_dim)
          for _ in range(num_feature_outputs)
      ]

      self._mask_net = MaskHeadNetwork(
          network_type=self._deepmac_params.network_type,
          use_instance_embedding=self._deepmac_params.use_instance_embedding,
          num_init_channels=self._deepmac_params.num_init_channels)

    return prediction_heads

  def _get_mask_head_input(self, boxes, pixel_embedding):
    """Get the input to the mask network, given bounding boxes.

    Args:
      boxes: A [batch_size, num_instances, 4] float tensor containing bounding
        boxes in normalized coordinates.
      pixel_embedding: A [batch_size, height, width, embedding_size] float
        tensor containing spatial pixel embeddings.

    Returns:
      embedding: A [batch_size, num_instances, mask_height, mask_width,
        embedding_size + 2] float tensor containing the inputs to the mask
        network. For each bounding box, we concatenate the normalized box
        coordinates to the cropped pixel embeddings. If
        predict_full_resolution_masks is set, mask_height and mask_width are
        the same as height and width of pixel_embedding. If not, mask_height
        and mask_width are the same as mask_size.
    """

    batch_size, num_instances = tf.shape(boxes)[0], tf.shape(boxes)[1]
    mask_size = self._deepmac_params.mask_size

    if self._deepmac_params.predict_full_resolution_masks:
      num_instances = tf.shape(boxes)[1]
      pixel_embedding = pixel_embedding[:, tf.newaxis, :, :, :]
      pixel_embeddings_processed = tf.tile(pixel_embedding,
                                           [1, num_instances, 1, 1, 1])
      image_shape = tf.shape(pixel_embeddings_processed)
      image_height, image_width = image_shape[2], image_shape[3]
      y_grid, x_grid = tf.meshgrid(tf.linspace(0.0, 1.0, image_height),
                                   tf.linspace(0.0, 1.0, image_width),
                                   indexing='ij')

      ycenter = (boxes[:, :, 0] + boxes[:, :, 2]) / 2.0
      xcenter = (boxes[:, :, 1] + boxes[:, :, 3]) / 2.0
      y_grid = y_grid[tf.newaxis, tf.newaxis, :, :]
      x_grid = x_grid[tf.newaxis, tf.newaxis, :, :]

      y_grid -= ycenter[:, :, tf.newaxis, tf.newaxis]
      x_grid -= xcenter[:, :, tf.newaxis, tf.newaxis]
      coords = tf.stack([y_grid, x_grid], axis=4)

    else:

      # TODO(vighneshb) Explore multilevel_roi_align and align_corners=False.
      embeddings = spatial_transform_ops.matmul_crop_and_resize(
          pixel_embedding, boxes, [mask_size, mask_size])
      pixel_embeddings_processed = embeddings
      mask_shape = tf.shape(pixel_embeddings_processed)
      mask_height, mask_width = mask_shape[2], mask_shape[3]

      y_grid, x_grid = tf.meshgrid(tf.linspace(-1.0, 1.0, mask_height),
                                   tf.linspace(-1.0, 1.0, mask_width),
                                   indexing='ij')
      coords = tf.stack([y_grid, x_grid], axis=2)
      coords = coords[tf.newaxis, tf.newaxis, :, :, :]
      coords = tf.tile(coords, [batch_size, num_instances, 1, 1, 1])

    if self._deepmac_params.use_xy:
      return tf.concat([coords, pixel_embeddings_processed], axis=4)
    else:
      return pixel_embeddings_processed

  def _get_instance_embeddings(self, boxes, instance_embedding):
    """Return the instance embeddings from bounding box centers.

    Args:
      boxes: A [batch_size, num_instances, 4] float tensor holding bounding
        boxes. The coordinates are in normalized input space.
      instance_embedding: A [batch_size, height, width, embedding_size] float
        tensor containing the instance embeddings.

    Returns:
      instance_embeddings: A [batch_size, num_instances, embedding_size]
        shaped float tensor containing the center embedding for each instance.
    """

    output_height = tf.cast(tf.shape(instance_embedding)[1], tf.float32)
    output_width = tf.cast(tf.shape(instance_embedding)[2], tf.float32)
    ymin = boxes[:, :, 0]
    xmin = boxes[:, :, 1]
    ymax = boxes[:, :, 2]
    xmax = boxes[:, :, 3]

    y_center_output = (ymin + ymax) * output_height / 2.0
    x_center_output = (xmin + xmax) * output_width / 2.0

    center_coords_output = tf.stack([y_center_output, x_center_output], axis=2)
    center_coords_output_int = tf.cast(center_coords_output, tf.int32)

    center_latents = tf.gather_nd(instance_embedding, center_coords_output_int,
                                  batch_dims=1)

    return center_latents

  def predict(self, preprocessed_inputs, true_image_shapes):
    prediction_dict = super(DeepMACMetaArch, self).predict(
        preprocessed_inputs, true_image_shapes)

    if self.groundtruth_has_field(fields.BoxListFields.boxes):
      mask_logits = self._predict_mask_logits_from_gt_boxes(prediction_dict)
      prediction_dict[MASK_LOGITS_GT_BOXES] = mask_logits

      if self._deepmac_params.augmented_self_supervision_loss_weight > 0.0:
        prediction_dict[SELF_SUPERVISED_DEAUGMENTED_MASK_LOGITS] = (
            self._predict_deaugmented_mask_logits_on_augmented_inputs(
                preprocessed_inputs, true_image_shapes))
    return prediction_dict

  def _predict_deaugmented_mask_logits_on_augmented_inputs(
      self, preprocessed_inputs, true_image_shapes):
    """Predicts masks on augmented images and reverses that augmentation.

    The masks are de-augmented so that they are aligned with the original image.

    Args:
      preprocessed_inputs: A batch of images of shape
        [batch_size, height, width, 3].
      true_image_shapes: True shape of the image in case there is any padding.

    Returns:
      mask_logits:
        A float tensor of shape [batch_size, num_instances,
          output_height, output_width, ]
    """

    batch_size = tf.shape(preprocessed_inputs)[0]
    gt_boxes = _batch_gt_list(
        self.groundtruth_lists(fields.BoxListFields.boxes))
    max_t = self._deepmac_params.augmented_self_supervision_max_translation
    tx = self._self_supervised_rng.uniform(
        [batch_size], minval=-max_t, maxval=max_t)
    ty = self._self_supervised_rng.uniform(
        [batch_size], minval=-max_t, maxval=max_t)

    scale_min = self._deepmac_params.augmented_self_supervision_scale_min
    scale_max = self._deepmac_params.augmented_self_supervision_scale_max
    sx = self._self_supervised_rng.uniform([batch_size], minval=scale_min,
                                           maxval=scale_max)
    sy = self._self_supervised_rng.uniform([batch_size], minval=scale_min,
                                           maxval=scale_max)
    flip = (self._self_supervised_rng.uniform(
        [batch_size], minval=0.0, maxval=1.0) <
            self._deepmac_params.augmented_self_supervision_flip_probability)

    augmented_inputs, augmented_boxes = transform_images_and_boxes(
        preprocessed_inputs, gt_boxes, tx=tx, ty=ty, sx=sx, sy=sy, flip=flip
    )

    augmented_prediction_dict = super(DeepMACMetaArch, self).predict(
        augmented_inputs, true_image_shapes)

    augmented_masks_lists = self._predict_mask_logits_from_boxes(
        augmented_prediction_dict, augmented_boxes)

    deaugmented_masks_list = []

    for mask_logits in augmented_masks_lists:
      deaugmented_masks = transform_instance_masks(
          mask_logits, tx=-tx, ty=-ty, sx=1.0/sx, sy=1.0/sy, flip=flip)
      deaugmented_masks = tf.stop_gradient(deaugmented_masks)
      deaugmented_masks_list.append(deaugmented_masks)

    return deaugmented_masks_list

  def _predict_mask_logits_from_embeddings(
      self, pixel_embedding, instance_embedding, boxes):
    mask_input = self._get_mask_head_input(boxes, pixel_embedding)
    mask_input, batch_size, num_instances = flatten_first2_dims(mask_input)

    instance_embeddings = self._get_instance_embeddings(
        boxes, instance_embedding)
    instance_embeddings, _, _ = flatten_first2_dims(instance_embeddings)

    mask_logits = self._mask_net(
        instance_embeddings, mask_input,
        training=tf.keras.backend.learning_phase())
    mask_logits = unpack_first2_dims(
        mask_logits, batch_size, num_instances)
    return mask_logits

  def _predict_mask_logits_from_boxes(self, prediction_dict, boxes):
    """Predict mask logits using the predict dict and the given set of boxes.

    Args:
      prediction_dict: a dict containing the keys INSTANCE_EMBEDDING and
        PIXEL_EMBEDDING, both expected to be list of tensors.
      boxes: A [batch_size, num_instances, 4] float tensor of boxes in the
        normalized coordinate system.
    Returns:
      mask_logits_list: A list of mask logits with the same spatial extents
        as prediction_dict[PIXEL_EMBEDDING].

    Returns:

    """
    mask_logits_list = []

    instance_embedding_list = prediction_dict[INSTANCE_EMBEDDING]
    pixel_embedding_list = prediction_dict[PIXEL_EMBEDDING]

    if self._deepmac_params.use_only_last_stage:
      instance_embedding_list = [instance_embedding_list[-1]]
      pixel_embedding_list = [pixel_embedding_list[-1]]

    for (instance_embedding, pixel_embedding) in zip(instance_embedding_list,
                                                     pixel_embedding_list):

      mask_logits_list.append(
          self._predict_mask_logits_from_embeddings(
              pixel_embedding, instance_embedding, boxes))

    return mask_logits_list

  def _predict_mask_logits_from_gt_boxes(self, prediction_dict):
    return self._predict_mask_logits_from_boxes(
        prediction_dict,
        _batch_gt_list(self.groundtruth_lists(fields.BoxListFields.boxes)))

  def _get_groundtruth_mask_output(self, boxes, masks):
    """Get the expected mask output for each box.

    Args:
      boxes: A [batch_size, num_instances, 4] float tensor containing bounding
        boxes in normalized coordinates.
      masks: A [batch_size, num_instances, height, width] float tensor
        containing binary ground truth masks.

    Returns:
      masks: If predict_full_resolution_masks is set, masks are not resized
      and the size of this tensor is [batch_size, num_instances,
      input_height, input_width]. Otherwise, returns a tensor of size
      [batch_size, num_instances, mask_size, mask_size].
    """

    mask_size = self._deepmac_params.mask_size
    if self._deepmac_params.predict_full_resolution_masks:
      return masks
    else:
      cropped_masks = crop_and_resize_instance_masks(
          masks, boxes, mask_size)
      cropped_masks = tf.stop_gradient(cropped_masks)

      # TODO(vighneshb) should we discretize masks?
      return cropped_masks

  def _resize_logits_like_gt(self, logits, gt):
    height, width = tf.shape(gt)[2], tf.shape(gt)[3]
    return resize_instance_masks(logits, (height, width))

  def _aggregate_classification_loss(self, loss, gt, pred, method):
    """Aggregates loss at a per-instance level.

    When this function is used with mask-heads, num_classes is usually 1.
    Args:
      loss: A [num_instances, num_pixels, num_classes] or
        [num_instances, num_classes] tensor. If the tensor is of rank 2, i.e.,
        of the form [num_instances, num_classes], we will assume that the
        number of pixels have already been nornalized.
      gt: A [num_instances, num_pixels, num_classes] float tensor of
        groundtruths.
      pred: A [num_instances, num_pixels, num_classes] float tensor of
        preditions.
      method: A string in ['auto', 'groundtruth'].
        'auto': When `loss` is rank 2, aggregates by sum. Otherwise, aggregates
          by mean.
        'groundtruth_count': Aggreagates the loss by computing sum and dividing
          by the number of positive (1) groundtruth pixels.
        'balanced': Normalizes each pixel by the number of positive or negative
          pixels depending on the groundtruth.

    Returns:
      per_instance_loss: A [num_instances] float tensor.
    """

    rank = len(loss.get_shape().as_list())
    if rank == 2:
      axes = [1]
    else:
      axes = [1, 2]

    if method == 'normalize_auto':
      normalization = 1.0
      if rank == 2:
        return tf.reduce_sum(loss, axis=axes)
      else:
        return tf.reduce_mean(loss, axis=axes)

    elif method == 'normalize_groundtruth_count':
      normalization = tf.reduce_sum(gt, axis=axes)
      return tf.reduce_sum(loss, axis=axes) / normalization

    elif method == 'normalize_balanced':
      if rank != 3:
        raise ValueError('Cannot apply normalized_balanced aggregation '
                         f'to loss of rank {rank}')
      normalization = (
          (gt * tf.reduce_sum(gt, keepdims=True, axis=axes)) +
          (1 - gt) * tf.reduce_sum(1 - gt, keepdims=True, axis=axes))
      return tf.reduce_sum(loss / normalization, axis=axes)

    else:
      raise ValueError('Unknown loss aggregation - {}'.format(method))

  def _compute_mask_prediction_loss(
      self, boxes, mask_logits, mask_gt, classes):
    """Compute the per-instance mask loss.

    Args:
      boxes: A [batch_size, num_instances, 4] float tensor of GT boxes in
        normalized coordinates.
      mask_logits: A [batch_size, num_instances, height, width] float tensor of
        predicted masks
      mask_gt: The groundtruth mask of same shape as mask_logits.
      classes: A [batch_size, num_instances, num_classes] shaped tensor of
        one-hot encoded classes.

    Returns:
      loss: A [batch_size, num_instances] shaped tensor with the loss for each
        instance.
    """

    if mask_gt is None:
      logging.info('No mask GT provided, mask loss is 0.')
      return tf.zeros_like(boxes[:, :, 0])

    batch_size, num_instances = tf.shape(boxes)[0], tf.shape(boxes)[1]
    mask_logits = self._resize_logits_like_gt(mask_logits, mask_gt)
    height, width = tf.shape(mask_logits)[2], tf.shape(mask_logits)[3]

    if self._deepmac_params.ignore_per_class_box_overlap:
      mask_logits *= per_instance_no_class_overlap(
          classes, boxes, height, width)

      height, wdith = tf.shape(mask_gt)[2], tf.shape(mask_gt)[3]
      mask_logits *= per_instance_no_class_overlap(
          classes, boxes, height, wdith)

    mask_logits = tf.reshape(mask_logits, [batch_size * num_instances, -1, 1])
    mask_gt = tf.reshape(mask_gt, [batch_size * num_instances, -1, 1])

    loss = self._deepmac_params.classification_loss(
        prediction_tensor=mask_logits,
        target_tensor=mask_gt,
        weights=tf.ones_like(mask_logits))

    loss = self._aggregate_classification_loss(
        loss, mask_gt, mask_logits, 'normalize_auto')
    return tf.reshape(loss, [batch_size, num_instances])

  def _compute_box_consistency_loss(
      self, boxes_gt, boxes_for_crop, mask_logits):
    """Compute the per-instance box consistency loss.

    Args:
      boxes_gt: A [batch_size, num_instances, 4] float tensor of GT boxes.
      boxes_for_crop: A [batch_size, num_instances, 4] float tensor of
        augmented boxes, to be used when using crop-and-resize based mask head.
      mask_logits: A [batch_size, num_instances, height, width]
        float tensor of predicted masks.

    Returns:
      loss: A [batch_size, num_instances] shaped tensor with the loss for
        each instance in the batch.
    """

    shape = tf.shape(mask_logits)
    batch_size, num_instances, height, width = (
        shape[0], shape[1], shape[2], shape[3])
    filled_boxes = fill_boxes(boxes_gt, height, width)[:, :, :, :, tf.newaxis]
    mask_logits = mask_logits[:, :, :, :, tf.newaxis]

    if self._deepmac_params.predict_full_resolution_masks:
      gt_crop = filled_boxes[:, :, :, :, 0]
      pred_crop = mask_logits[:, :, :, :, 0]
    else:
      gt_crop = crop_and_resize_instance_masks(
          filled_boxes, boxes_for_crop, self._deepmac_params.mask_size)
      pred_crop = crop_and_resize_instance_masks(
          mask_logits, boxes_for_crop, self._deepmac_params.mask_size)

    loss = 0.0
    for axis in [2, 3]:

      if self._deepmac_params.box_consistency_tightness:
        pred_max_raw = tf.reduce_max(pred_crop, axis=axis)
        pred_max_within_box = tf.reduce_max(pred_crop * gt_crop, axis=axis)
        box_1d = tf.reduce_max(gt_crop, axis=axis)
        pred_max = ((box_1d * pred_max_within_box) +
                    ((1 - box_1d) * pred_max_raw))

      else:
        pred_max = tf.reduce_max(pred_crop, axis=axis)

      pred_max = pred_max[:, :, :, tf.newaxis]
      gt_max = tf.reduce_max(gt_crop, axis=axis)[:, :, :, tf.newaxis]

      flat_pred, batch_size, num_instances = flatten_first2_dims(pred_max)
      flat_gt, _, _ = flatten_first2_dims(gt_max)

      # We use flat tensors while calling loss functions because we
      # want the loss per-instance to later multiply with the per-instance
      # weight. Flattening the first 2 dims allows us to represent each instance
      # in each batch as though they were samples in a larger batch.
      raw_loss = self._deepmac_params.classification_loss(
          prediction_tensor=flat_pred,
          target_tensor=flat_gt,
          weights=tf.ones_like(flat_pred))

      agg_loss = self._aggregate_classification_loss(
          raw_loss, flat_gt, flat_pred,
          self._deepmac_params.box_consistency_loss_normalize)
      loss += unpack_first2_dims(agg_loss, batch_size, num_instances)

    return loss

  def _compute_feature_consistency_loss(
      self, boxes, consistency_feature_map, mask_logits):
    """Compute the per-instance feature consistency loss.

    Args:
      boxes: A [batch_size, num_instances, 4] float tensor of GT boxes.
      consistency_feature_map: A [batch_size, height, width, 3]
        float tensor containing the feature map to use for consistency.
      mask_logits: A [batch_size, num_instances, height, width] float tensor of
        predicted masks.

    Returns:
      loss: A [batch_size, num_instances] shaped tensor with the loss for each
        instance fpr each sample in the batch.
    """

    if not self._deepmac_params.predict_full_resolution_masks:
      logging.info('Feature consistency is not implemented with RoIAlign '
                   ', i.e, fixed sized masks. Returning 0 loss.')
      return tf.zeros(tf.shape(boxes)[:2])

    dilation = self._deepmac_params.feature_consistency_dilation

    height, width = (tf.shape(consistency_feature_map)[1],
                     tf.shape(consistency_feature_map)[2])

    comparison = self._deepmac_params.feature_consistency_comparison
    if comparison == 'comparison_default_gaussian':
      similarity = dilated_cross_pixel_similarity(
          consistency_feature_map, dilation=dilation, theta=2.0,
          method='gaussian')
    elif comparison == 'comparison_normalized_dotprod':
      consistency_feature_map = normalize_feature_map(consistency_feature_map)
      similarity = dilated_cross_pixel_similarity(
          consistency_feature_map, dilation=dilation, theta=2.0,
          method='dotprod')

    else:
      raise ValueError('Unknown comparison type - %s' % comparison)

    mask_probs = tf.nn.sigmoid(mask_logits)
    same_mask_label_probability = dilated_cross_same_mask_label(
        mask_probs, dilation=dilation)
    same_mask_label_probability = tf.clip_by_value(
        same_mask_label_probability, 1e-3, 1.0)

    similarity_mask = (
        similarity > self._deepmac_params.feature_consistency_threshold)
    similarity_mask = tf.cast(
        similarity_mask[:, :, tf.newaxis, :, :], tf.float32)
    per_pixel_loss = -(similarity_mask *
                       tf.math.log(same_mask_label_probability))
    # TODO(vighneshb) explore if shrinking the box by 1px helps.
    box_mask = fill_boxes(boxes, height, width)
    box_mask_expanded = box_mask[tf.newaxis]

    per_pixel_loss = per_pixel_loss * box_mask_expanded
    loss = tf.reduce_sum(per_pixel_loss, axis=[0, 3, 4])
    num_box_pixels = tf.maximum(1.0, tf.reduce_sum(box_mask, axis=[2, 3]))
    loss = loss / num_box_pixels

    if tf.keras.backend.learning_phase():
      loss *= _warmup_weight(
          current_training_step=self._training_step,
          warmup_start=self._deepmac_params.feature_consistency_warmup_start,
          warmup_steps=self._deepmac_params.feature_consistency_warmup_steps)

    return loss

  def _self_supervision_loss(
      self, predicted_logits, self_supervised_logits, boxes, loss_name):
    original_shape = tf.shape(predicted_logits)
    batch_size, num_instances = original_shape[0], original_shape[1]
    box_mask = fill_boxes(boxes, original_shape[2], original_shape[3])

    loss_tensor_shape = [batch_size * num_instances, -1, 1]
    weights = tf.reshape(box_mask, loss_tensor_shape)

    predicted_logits = tf.reshape(predicted_logits, loss_tensor_shape)
    self_supervised_logits = tf.reshape(self_supervised_logits,
                                        loss_tensor_shape)
    self_supervised_probs = tf.nn.sigmoid(self_supervised_logits)
    predicted_probs = tf.nn.sigmoid(predicted_logits)
    num_box_pixels = tf.reduce_sum(weights, axis=[1, 2])
    num_box_pixels = tf.maximum(num_box_pixels, 1.0)

    if loss_name == 'loss_dice':
      self_supervised_binary_probs = tf.cast(
          self_supervised_logits > 0.0, tf.float32)

      loss_class = losses.WeightedDiceClassificationLoss(
          squared_normalization=False)
      loss = loss_class(prediction_tensor=predicted_logits,
                        target_tensor=self_supervised_binary_probs,
                        weights=weights)
      agg_loss = self._aggregate_classification_loss(
          loss, gt=self_supervised_probs, pred=predicted_logits,
          method='normalize_auto')
    elif loss_name == 'loss_mse':
      diff = self_supervised_probs - predicted_probs
      diff_sq = (diff * diff)

      diff_sq_sum = tf.reduce_sum(diff_sq * weights, axis=[1, 2])

      agg_loss = diff_sq_sum / num_box_pixels

    elif loss_name == 'loss_kl_div':
      loss_class = tf.keras.losses.KLDivergence(
          reduction=tf.keras.losses.Reduction.NONE)
      predicted_2class_probability = tf.stack(
          [predicted_probs, 1 - predicted_probs], axis=2
      )
      target_2class_probability = tf.stack(
          [self_supervised_probs, 1 - self_supervised_probs], axis=2
      )

      loss = loss_class(
          y_pred=predicted_2class_probability,
          y_true=target_2class_probability)
      agg_loss = tf.reduce_sum(loss * weights, axis=[1, 2]) / num_box_pixels
    else:
      raise RuntimeError('Unknown self-supervision loss %s' % loss_name)

    return tf.reshape(agg_loss, [batch_size, num_instances])

  def _compute_self_supervised_augmented_loss(
      self, original_logits, deaugmented_logits, boxes):

    if deaugmented_logits is None:
      logging.info('No self supervised masks provided. '
                   'Returning 0 self-supervised loss,')
      return tf.zeros(tf.shape(original_logits)[:2])

    loss = self._self_supervision_loss(
        predicted_logits=original_logits,
        self_supervised_logits=deaugmented_logits,
        boxes=boxes,
        loss_name=self._deepmac_params.augmented_self_supervision_loss)

    if tf.keras.backend.learning_phase():
      loss *= _warmup_weight(
          current_training_step=self._training_step,
          warmup_start=
          self._deepmac_params.augmented_self_supervision_warmup_start,
          warmup_steps=
          self._deepmac_params.augmented_self_supervision_warmup_steps)

    return loss

  def _compute_pointly_supervised_loss_from_keypoints(
      self, mask_logits, keypoints_gt, keypoints_depth_gt):
    """Computes per-point mask loss from keypoints.

    Args:
      mask_logits: A [batch_size, num_instances, height, width] float tensor
        denoting predicted masks.
      keypoints_gt: A [batch_size, num_instances, num_keypoints, 2] float tensor
        of normalize keypoint coordinates.
      keypoints_depth_gt: A [batch_size, num_instances, num_keyponts] float
        tensor of keypoint depths. We assume that +1 is foreground and -1
        is background.
    Returns:
      loss: Pointly supervised loss with shape [batch_size, num_instances].
    """

    if keypoints_gt is None:
      logging.info(('Returning 0 pointly supervised loss because '
                    'keypoints are not given.'))
      return tf.zeros(tf.shape(mask_logits)[:2])

    if keypoints_depth_gt is None:
      logging.info(('Returning 0 pointly supervised loss because '
                    'keypoint depths are not given.'))
      return tf.zeros(tf.shape(mask_logits)[:2])

    if not self._deepmac_params.predict_full_resolution_masks:
      raise NotImplementedError(
          'Pointly supervised loss not implemented with RoIAlign.')

    num_keypoints = tf.shape(keypoints_gt)[2]
    keypoints_nan = tf.math.is_nan(keypoints_gt)
    keypoints_gt = tf.where(
        keypoints_nan, tf.zeros_like(keypoints_gt), keypoints_gt)
    weights = tf.cast(
        tf.logical_not(tf.reduce_any(keypoints_nan, axis=3)), tf.float32)

    height, width = tf.shape(mask_logits)[2], tf.shape(mask_logits)[3]
    ky, kx = tf.unstack(keypoints_gt, axis=3)
    height_f, width_f = tf.cast(height, tf.float32), tf.cast(width, tf.float32)

    ky = tf.clip_by_value(tf.cast(ky * height_f, tf.int32), 0, height - 1)
    kx = tf.clip_by_value(tf.cast(kx * width_f, tf.int32), 0, width - 1)
    keypoints_gt_int = tf.stack([ky, kx], axis=3)

    mask_logits_flat, batch_size, num_instances = flatten_first2_dims(
        mask_logits)
    keypoints_gt_int_flat, _, _ = flatten_first2_dims(keypoints_gt_int)
    keypoint_depths_flat, _, _ = flatten_first2_dims(keypoints_depth_gt)
    weights_flat = tf.logical_not(
        tf.reduce_any(keypoints_nan, axis=2))
    weights_flat, _, _ = flatten_first2_dims(weights)

    # TODO(vighneshb): Replace with bilinear interpolation
    point_mask_logits = tf.gather_nd(
        mask_logits_flat, keypoints_gt_int_flat, batch_dims=1)

    point_mask_logits = tf.reshape(
        point_mask_logits, [batch_size * num_instances, num_keypoints, 1])

    labels = tf.cast(keypoint_depths_flat > 0.0, tf.float32)
    labels = tf.reshape(
        labels, [batch_size * num_instances, num_keypoints, 1])
    weights_flat = tf.reshape(
        weights_flat, [batch_size * num_instances, num_keypoints, 1])

    loss = self._deepmac_params.classification_loss(
        prediction_tensor=point_mask_logits, target_tensor=labels,
        weights=weights_flat
    )

    loss = self._aggregate_classification_loss(
        loss, gt=labels, pred=point_mask_logits, method='normalize_auto')

    return tf.reshape(loss, [batch_size, num_instances])

  def _compute_deepmac_losses(
      self, boxes, masks_logits, masks_gt, classes, consistency_feature_map,
      self_supervised_masks_logits=None, keypoints_gt=None,
      keypoints_depth_gt=None):
    """Returns the mask loss per instance.

    Args:
      boxes: A [batch_size, num_instances, 4] float tensor holding bounding
        boxes. The coordinates are in normalized input space.
      masks_logits: A [batch_size, num_instances, output_height, output_height].
        float tensor containing the instance mask predictions in their logit
        form.
      masks_gt: A [batch_size, num_instances, output_height, output_width] float
        tensor containing the groundtruth masks. If masks_gt is None,
        DEEP_MASK_ESTIMATION is filled with 0s.
      classes: A [batch_size, num_instances, num_classes] tensor of one-hot
        encoded classes.
      consistency_feature_map: [batch_size, output_height, output_width,
        channels] float tensor denoting the image to use for consistency.
      self_supervised_masks_logits: Optional self-supervised mask logits to
        compare against of same shape as mask_logits.
      keypoints_gt: A float tensor of shape
        [batch_size, num_instances, num_keypoints, 2], representing the points
        where we have mask supervision.
      keypoints_depth_gt: A float tensor of shape
        [batch_size, num_instances, num_keypoints] of keypoint depths which
        indicate the mask label at the keypoint locations. depth=+1 is
        foreground and depth=-1 is background.

    Returns:
      tensor_dict: A dictionary with 4 keys, each mapping to a tensor of shape
        [batch_size, num_instances]. The 4 keys are:
          - DEEP_MASK_ESTIMATION
          - DEEP_MASK_BOX_CONSISTENCY
          - DEEP_MASK_FEATURE_CONSISTENCY
          - DEEP_MASK_AUGMENTED_SELF_SUPERVISION
          - DEEP_MASK_POINTLY_SUPERVISED
    """

    if tf.keras.backend.learning_phase():
      boxes = tf.stop_gradient(boxes)
      def jitter_func(boxes):
        return preprocessor.random_jitter_boxes(
            boxes, self._deepmac_params.max_roi_jitter_ratio,
            jitter_mode=self._deepmac_params.roi_jitter_mode)

      boxes_for_crop = tf.map_fn(jitter_func,
                                 boxes, parallel_iterations=128)
    else:
      boxes_for_crop = boxes

    if masks_gt is not None:
      masks_gt = self._get_groundtruth_mask_output(
          boxes_for_crop, masks_gt)
    mask_prediction_loss = self._compute_mask_prediction_loss(
        boxes_for_crop, masks_logits, masks_gt, classes)

    box_consistency_loss = self._compute_box_consistency_loss(
        boxes, boxes_for_crop, masks_logits)

    feature_consistency_loss = self._compute_feature_consistency_loss(
        boxes, consistency_feature_map, masks_logits)

    self_supervised_loss = self._compute_self_supervised_augmented_loss(
        masks_logits, self_supervised_masks_logits, boxes,
    )

    pointly_supervised_loss = (
        self._compute_pointly_supervised_loss_from_keypoints(
            masks_logits, keypoints_gt, keypoints_depth_gt))

    return {
        DEEP_MASK_ESTIMATION: mask_prediction_loss,
        DEEP_MASK_BOX_CONSISTENCY: box_consistency_loss,
        DEEP_MASK_FEATURE_CONSISTENCY: feature_consistency_loss,
        DEEP_MASK_AUGMENTED_SELF_SUPERVISION: self_supervised_loss,
        DEEP_MASK_POINTLY_SUPERVISED: pointly_supervised_loss,
    }

  def _get_lab_image(self, preprocessed_image):
    raw_image = self._feature_extractor.preprocess_reverse(
        preprocessed_image)
    raw_image = raw_image / 255.0

    if tf_version.is_tf1():
      raise NotImplementedError(('RGB-to-LAB conversion required for the color'
                                 ' consistency loss is not supported in TF1.'))
    return tfio.experimental.color.rgb_to_lab(raw_image)

  def _maybe_get_gt_batch(self, field):
    """Returns a batch of groundtruth tensors if available, else None."""
    if self.groundtruth_has_field(field):
      return _batch_gt_list(self.groundtruth_lists(field))
    else:
      return None

  def _get_consistency_feature_map(self, prediction_dict):

    prediction_shape = tf.shape(prediction_dict[MASK_LOGITS_GT_BOXES][0])
    height, width = prediction_shape[2], prediction_shape[3]

    consistency_type = self._deepmac_params.feature_consistency_type
    if consistency_type == 'consistency_default_lab':
      preprocessed_image = tf.image.resize(
          prediction_dict['preprocessed_inputs'], (height, width))
      consistency_feature_map = self._get_lab_image(preprocessed_image)
    elif consistency_type == 'consistency_feature_map':
      consistency_feature_map = prediction_dict['extracted_features'][-1]
      consistency_feature_map = tf.image.resize(
          consistency_feature_map, (height, width))
    else:
      raise ValueError('Unknown feature consistency type - {}.'.format(
          self._deepmac_params.feature_consistency_type))

    return tf.stop_gradient(consistency_feature_map)

  def _compute_masks_loss(self, prediction_dict):
    """Computes the mask loss.

    Args:
      prediction_dict: dict from predict() method containing
        INSTANCE_EMBEDDING and PIXEL_EMBEDDING prediction.
        Both of these are lists of tensors, each of size
        [batch_size, height, width, embedding_size].

    Returns:
      loss_dict: A dict mapping string (loss names) to scalar floats.
    """

    allowed_masked_classes_ids = (
        self._deepmac_params.allowed_masked_classes_ids)

    loss_dict = {}
    for loss_name in MASK_LOSSES:
      loss_dict[loss_name] = 0.0

    gt_boxes = self._maybe_get_gt_batch(fields.BoxListFields.boxes)
    gt_weights = self._maybe_get_gt_batch(fields.BoxListFields.weights)
    gt_classes = self._maybe_get_gt_batch(fields.BoxListFields.classes)
    gt_masks = self._maybe_get_gt_batch(fields.BoxListFields.masks)
    gt_keypoints = self._maybe_get_gt_batch(fields.BoxListFields.keypoints)
    gt_depths = self._maybe_get_gt_batch(fields.BoxListFields.keypoint_depths)

    mask_logits_list = prediction_dict[MASK_LOGITS_GT_BOXES]
    self_supervised_mask_logits_list = prediction_dict.get(
        SELF_SUPERVISED_DEAUGMENTED_MASK_LOGITS,
        [None] * len(mask_logits_list))

    assert len(mask_logits_list) == len(self_supervised_mask_logits_list)
    consistency_feature_map = self._get_consistency_feature_map(prediction_dict)

    # Iterate over multiple preidctions by backbone (for hourglass length=2)
    for (mask_logits, self_supervised_mask_logits) in zip(
        mask_logits_list, self_supervised_mask_logits_list):

      # TODO(vighneshb) Add sub-sampling back if required.
      _, valid_mask_weights, gt_masks = filter_masked_classes(
          allowed_masked_classes_ids, gt_classes,
          gt_weights, gt_masks)

      sample_loss_dict = self._compute_deepmac_losses(
          boxes=gt_boxes, masks_logits=mask_logits, masks_gt=gt_masks,
          classes=gt_classes, consistency_feature_map=consistency_feature_map,
          self_supervised_masks_logits=self_supervised_mask_logits,
          keypoints_gt=gt_keypoints, keypoints_depth_gt=gt_depths)

      sample_loss_dict[DEEP_MASK_ESTIMATION] *= valid_mask_weights

      for loss_name in WEAK_LOSSES:
        sample_loss_dict[loss_name] *= gt_weights

      num_instances = tf.maximum(tf.reduce_sum(gt_weights), 1.0)
      num_instances_allowed = tf.maximum(
          tf.reduce_sum(valid_mask_weights), 1.0)

      loss_dict[DEEP_MASK_ESTIMATION] += (
          tf.reduce_sum(sample_loss_dict[DEEP_MASK_ESTIMATION]) /
          num_instances_allowed)

      for loss_name in WEAK_LOSSES:
        loss_dict[loss_name] += (tf.reduce_sum(sample_loss_dict[loss_name]) /
                                 num_instances)

    num_predictions = len(mask_logits_list)

    return dict((key, loss / float(num_predictions))
                for key, loss in loss_dict.items())

  def loss(self, prediction_dict, true_image_shapes, scope=None):

    losses_dict = super(DeepMACMetaArch, self).loss(
        prediction_dict, true_image_shapes, scope)

    if self._deepmac_params is not None:
      mask_loss_dict = self._compute_masks_loss(
          prediction_dict=prediction_dict)

      for loss_name in MASK_LOSSES:
        loss_weight = _get_loss_weight(loss_name, self._deepmac_params)
        if loss_weight > 0.0:
          losses_dict[LOSS_KEY_PREFIX + '/' + loss_name] = (
              loss_weight * mask_loss_dict[loss_name])

    return losses_dict

  def postprocess(self, prediction_dict, true_image_shapes, **params):
    """Produces boxes given a prediction dict returned by predict().

    Args:
      prediction_dict: a dictionary holding predicted tensors from "predict"
        function.
      true_image_shapes: int32 tensor of shape [batch, 3] where each row is of
      the form [height, width, channels] indicating the shapes of true images
        in the resized images, as resized images can be padded with zeros.
      **params: Currently ignored.

    Returns:
      detections: a dictionary containing the following fields
        detection_masks: (Optional) A uint8 tensor of shape [batch,
          max_detections, mask_height, mask_width] with masks for each
          detection. Background is specified with 0, and foreground is specified
          with positive integers (1 for standard instance segmentation mask, and
          1-indexed parts for DensePose task).
        And all other fields returned by the super class method.
    """
    postprocess_dict = super(DeepMACMetaArch, self).postprocess(
        prediction_dict, true_image_shapes, **params)
    boxes_strided = postprocess_dict['detection_boxes_strided']

    if self._deepmac_params is not None:
      masks = self._postprocess_masks(
          boxes_strided, prediction_dict[INSTANCE_EMBEDDING][-1],
          prediction_dict[PIXEL_EMBEDDING][-1])
      postprocess_dict[fields.DetectionResultFields.detection_masks] = masks

    return postprocess_dict

  def _postprocess_masks(self, boxes_output_stride,
                         instance_embedding, pixel_embedding):
    """Postprocess masks with the deep mask network.

    Args:
      boxes_output_stride: A [batch_size, num_instances, 4] float tensor
        containing the batch of boxes in the absolute output space of the
        feature extractor.
      instance_embedding: A [batch_size, output_height, output_width,
        embedding_size] float tensor containing instance embeddings.
      pixel_embedding: A [batch_size, output_height, output_width,
        pixel_embedding_size] float tensor containing the per-pixel embedding.

    Returns:
      masks: A float tensor of size [batch_size, num_instances, mask_size,
        mask_size] containing binary per-box instance masks.
    """

    height, width = (tf.shape(instance_embedding)[1],
                     tf.shape(instance_embedding)[2])
    boxes = boxes_batch_absolute_to_normalized_coordinates(
        boxes_output_stride, height, width)

    mask_logits = self._predict_mask_logits_from_embeddings(
        pixel_embedding, instance_embedding, boxes)

    # TODO(vighneshb) Explore sweeping mask thresholds.

    if self._deepmac_params.predict_full_resolution_masks:

      height, width = tf.shape(mask_logits)[1], tf.shape(mask_logits)[2]
      height *= self._stride
      width *= self._stride
      mask_logits = resize_instance_masks(mask_logits, (height, width))

      mask_logits = crop_and_resize_instance_masks(
          mask_logits, boxes, self._deepmac_params.postprocess_crop_size)

    masks_prob = tf.nn.sigmoid(mask_logits)

    return masks_prob

  def _transform_boxes_to_feature_coordinates(self, provided_boxes,
                                              true_image_shapes,
                                              resized_image_shape,
                                              instance_embedding):
    """Transforms normalzied boxes to feature map coordinates.

    Args:
      provided_boxes: A [batch, num_instances, 4] float tensor containing
        normalized bounding boxes.
      true_image_shapes: int32 tensor of shape [batch, 3] where each row is of
        the form [height, width, channels] indicating the shapes of true images
        in the resized images, as resized images can be padded with zeros.
      resized_image_shape: A 4D int32 tensor containing shapes of the
        preprocessed inputs (N, H, W, C).
      instance_embedding: A [batch, output_height, output_width, embedding_size]
        float tensor containing instance embeddings.

    Returns:
      A float tensor of size [batch, num_instances, 4] containing boxes whose
        coordinates have been transformed to the absolute output space of the
        feature extractor.
    """
    # Input boxes must be normalized.
    shape_utils.assert_box_normalized(provided_boxes)

    # Transform the provided boxes to the absolute output space of the feature
    # extractor.
    height, width = (tf.shape(instance_embedding)[1],
                     tf.shape(instance_embedding)[2])

    resized_image_height = resized_image_shape[1]
    resized_image_width = resized_image_shape[2]

    def transform_boxes(elems):
      boxes_per_image, true_image_shape = elems
      blist = box_list.BoxList(boxes_per_image)
      # First transform boxes from image space to resized image space since
      # there may have paddings in the resized images.
      blist = box_list_ops.scale(blist,
                                 true_image_shape[0] / resized_image_height,
                                 true_image_shape[1] / resized_image_width)
      # Then transform boxes from resized image space (normalized) to the
      # feature map space (absolute).
      blist = box_list_ops.to_absolute_coordinates(
          blist, height, width, check_range=False)
      return blist.get()

    return tf.map_fn(
        transform_boxes, [provided_boxes, true_image_shapes], dtype=tf.float32)

  def predict_masks_from_boxes(self, prediction_dict, true_image_shapes,
                               provided_boxes, **params):
    """Produces masks for the provided boxes.

    Args:
      prediction_dict: a dictionary holding predicted tensors from "predict"
        function.
      true_image_shapes: int32 tensor of shape [batch, 3] where each row is of
        the form [height, width, channels] indicating the shapes of true images
        in the resized images, as resized images can be padded with zeros.
      provided_boxes: float tensor of shape [batch, num_boxes, 4] containing
        boxes coordinates (normalized) from which we will produce masks.
      **params: Currently ignored.

    Returns:
      detections: a dictionary containing the following fields
        detection_masks: (Optional) A uint8 tensor of shape [batch,
          max_detections, mask_height, mask_width] with masks for each
          detection. Background is specified with 0, and foreground is specified
          with positive integers (1 for standard instance segmentation mask, and
          1-indexed parts for DensePose task).
        And all other fields returned by the super class method.
    """
    postprocess_dict = super(DeepMACMetaArch,
                             self).postprocess(prediction_dict,
                                               true_image_shapes, **params)

    instance_embedding = prediction_dict[INSTANCE_EMBEDDING][-1]
    resized_image_shapes = shape_utils.combined_static_and_dynamic_shape(
        prediction_dict['preprocessed_inputs'])
    boxes_strided = self._transform_boxes_to_feature_coordinates(
        provided_boxes, true_image_shapes, resized_image_shapes,
        instance_embedding)

    if self._deepmac_params is not None:
      masks = self._postprocess_masks(
          boxes_strided, instance_embedding,
          prediction_dict[PIXEL_EMBEDDING][-1])
      postprocess_dict[fields.DetectionResultFields.detection_masks] = masks

    return postprocess_dict
