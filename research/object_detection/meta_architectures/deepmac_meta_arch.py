"""Deep Mask heads above CenterNet (DeepMAC) architecture.

TODO(vighneshb) Add link to paper when done.
"""

import collections

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
from object_detection.protos import losses_pb2
from object_detection.protos import preprocessor_pb2
from object_detection.utils import shape_utils
from object_detection.utils import spatial_transform_ops
from object_detection.utils import tf_version

if tf_version.is_tf2():
  import tensorflow_io as tfio  # pylint:disable=g-import-not-at-top


INSTANCE_EMBEDDING = 'INSTANCE_EMBEDDING'
PIXEL_EMBEDDING = 'PIXEL_EMBEDDING'
DEEP_MASK_ESTIMATION = 'deep_mask_estimation'
DEEP_MASK_BOX_CONSISTENCY = 'deep_mask_box_consistency'
DEEP_MASK_COLOR_CONSISTENCY = 'deep_mask_color_consistency'
LOSS_KEY_PREFIX = center_net_meta_arch.LOSS_KEY_PREFIX
NEIGHBORS_2D = [[-1, -1], [-1, 0], [-1, 1],
                [0, -1], [0, 1],
                [1, -1], [1, 0], [1, 1]]


class DeepMACParams(
    collections.namedtuple('DeepMACParams', [
        'classification_loss', 'dim', 'task_loss_weight', 'pixel_embedding_dim',
        'allowed_masked_classes_ids', 'mask_size', 'mask_num_subsamples',
        'use_xy', 'network_type', 'use_instance_embedding', 'num_init_channels',
        'predict_full_resolution_masks', 'postprocess_crop_size',
        'max_roi_jitter_ratio', 'roi_jitter_mode',
        'box_consistency_loss_weight', 'color_consistency_threshold',
        'color_consistency_dilation', 'color_consistency_loss_weight'
    ])):
  """Class holding the DeepMAC network configutration."""

  __slots__ = ()

  def __new__(cls, classification_loss, dim, task_loss_weight,
              pixel_embedding_dim, allowed_masked_classes_ids, mask_size,
              mask_num_subsamples, use_xy, network_type, use_instance_embedding,
              num_init_channels, predict_full_resolution_masks,
              postprocess_crop_size, max_roi_jitter_ratio,
              roi_jitter_mode, box_consistency_loss_weight,
              color_consistency_threshold, color_consistency_dilation,
              color_consistency_loss_weight):
    return super(DeepMACParams,
                 cls).__new__(cls, classification_loss, dim,
                              task_loss_weight, pixel_embedding_dim,
                              allowed_masked_classes_ids, mask_size,
                              mask_num_subsamples, use_xy, network_type,
                              use_instance_embedding, num_init_channels,
                              predict_full_resolution_masks,
                              postprocess_crop_size, max_roi_jitter_ratio,
                              roi_jitter_mode, box_consistency_loss_weight,
                              color_consistency_threshold,
                              color_consistency_dilation,
                              color_consistency_loss_weight)


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

  elif name == 'embedding_projection':
    return tf.keras.layers.Lambda(lambda x: x)

  elif name.startswith('resnet'):
    return ResNetMaskNetwork(name, num_init_channels)

  raise ValueError('Unknown network type {}'.format(name))


def crop_masks_within_boxes(masks, boxes, output_size):
  """Crops masks to lie tightly within the boxes.

  Args:
    masks: A [num_instances, height, width] float tensor of masks.
    boxes: A [num_instances, 4] sized tensor of normalized bounding boxes.
    output_size: The height and width of the output masks.

  Returns:
    masks: A [num_instances, output_size, output_size] tensor of masks which
      are cropped to be tightly within the gives boxes and resized.

  """
  masks = spatial_transform_ops.matmul_crop_and_resize(
      masks[:, :, :, tf.newaxis], boxes[:, tf.newaxis, :],
      [output_size, output_size])
  return masks[:, 0, :, :, 0]


def resize_instance_masks(masks, shape):
  height, width = shape
  masks_ex = masks[:, :, :, tf.newaxis]
  masks_ex = tf.image.resize(masks_ex, (height, width),
                             method=tf.image.ResizeMethod.BILINEAR)
  masks = masks_ex[:, :, :, 0]

  return masks


def filter_masked_classes(masked_class_ids, classes, weights, masks):
  """Filter out masks whose class IDs are not present in masked_class_ids.

  Args:
    masked_class_ids: A list of class IDs allowed to have masks. These class IDs
      are 1-indexed.
    classes: A [num_instances, num_classes] float tensor containing the one-hot
      encoded classes.
    weights: A [num_instances] float tensor containing the weights of each
      sample.
    masks: A [num_instances, height, width] tensor containing the mask per
      instance.

  Returns:
    classes_filtered: A [num_instances, num_classes] float tensor containing the
       one-hot encoded classes with classes not in masked_class_ids zeroed out.
    weights_filtered: A [num_instances] float tensor containing the weights of
      each sample with instances whose classes aren't in masked_class_ids
      zeroed out.
    masks_filtered: A [num_instances, height, width] tensor containing the mask
      per instance with masks not belonging to masked_class_ids zeroed out.
  """

  if len(masked_class_ids) == 0:  # pylint:disable=g-explicit-length-test
    return classes, weights, masks

  if tf.shape(classes)[0] == 0:
    return classes, weights, masks

  masked_class_ids = tf.constant(np.array(masked_class_ids, dtype=np.int32))
  label_id_offset = 1
  masked_class_ids -= label_id_offset
  class_ids = tf.argmax(classes, axis=1, output_type=tf.int32)
  matched_classes = tf.equal(
      class_ids[:, tf.newaxis], masked_class_ids[tf.newaxis, :]
  )

  matched_classes = tf.reduce_any(matched_classes, axis=1)
  matched_classes = tf.cast(matched_classes, tf.float32)

  return (
      classes * matched_classes[:, tf.newaxis],
      weights * matched_classes,
      masks * matched_classes[:, tf.newaxis, tf.newaxis]
  )


def crop_and_resize_feature_map(features, boxes, size):
  """Crop and resize regions from a single feature map given a set of boxes.

  Args:
    features: A [H, W, C] float tensor.
    boxes: A [N, 4] tensor of norrmalized boxes.
    size: int, the size of the output features.

  Returns:
    per_box_features: A [N, size, size, C] tensor of cropped and resized
      features.
  """
  return spatial_transform_ops.matmul_crop_and_resize(
      features[tf.newaxis], boxes[tf.newaxis], [size, size])[0]


def crop_and_resize_instance_masks(masks, boxes, mask_size):
  """Crop and resize each mask according to the given boxes.

  Args:
    masks: A [N, H, W] float tensor.
    boxes: A [N, 4] float tensor of normalized boxes.
    mask_size: int, the size of the output masks.

  Returns:
    masks: A [N, mask_size, mask_size] float tensor of cropped and resized
      instance masks.
  """
  cropped_masks = spatial_transform_ops.matmul_crop_and_resize(
      masks[:, :, :, tf.newaxis], boxes[:, tf.newaxis, :],
      [mask_size, mask_size])
  cropped_masks = tf.squeeze(cropped_masks, axis=[1, 4])

  return cropped_masks


def fill_boxes(boxes, height, width):
  """Fills the area included in the box."""
  blist = box_list.BoxList(boxes)
  blist = box_list_ops.to_absolute_coordinates(blist, height, width)
  boxes = blist.get()
  ymin, xmin, ymax, xmax = tf.unstack(
      boxes[:, tf.newaxis, tf.newaxis, :], 4, axis=3)

  ygrid, xgrid = tf.meshgrid(tf.range(height), tf.range(width), indexing='ij')
  ygrid, xgrid = tf.cast(ygrid, tf.float32), tf.cast(xgrid, tf.float32)
  ygrid, xgrid = ygrid[tf.newaxis, :, :], xgrid[tf.newaxis, :, :]

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


def _get_2d_neighbors_kenel():
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
    input_tensor: A float tensor of shape [height, width, channels].
    dilation: int, the dilation factor for considering neighbors.

  Returns:
    output: A float tensor of all 8 2-D neighbors. of shape
      [8, height, width, channels].
  """
  input_tensor = tf.transpose(input_tensor, (2, 0, 1))
  input_tensor = input_tensor[:, :, :, tf.newaxis]

  kernel = _get_2d_neighbors_kenel()
  output = tf.nn.atrous_conv2d(input_tensor, kernel, rate=dilation,
                               padding='SAME')
  return tf.transpose(output, [3, 1, 2, 0])


def gaussian_pixel_similarity(a, b, theta):
  norm_difference = tf.linalg.norm(a - b, axis=-1)
  similarity = tf.exp(-norm_difference / theta)
  return similarity


def dilated_cross_pixel_similarity(feature_map, dilation=2, theta=2.0):
  """Dilated cross pixel similarity as defined in [1].

  [1]: https://arxiv.org/abs/2012.02310

  Args:
    feature_map: A float tensor of shape [height, width, channels]
    dilation: int, the dilation factor.
    theta: The denominator while taking difference inside the gaussian.

  Returns:
    dilated_similarity: A tensor of shape [8, height, width]
  """
  neighbors = generate_2d_neighbors(feature_map, dilation)
  feature_map = feature_map[tf.newaxis]

  return gaussian_pixel_similarity(feature_map, neighbors, theta=theta)


def dilated_cross_same_mask_label(instance_masks, dilation=2):
  """Dilated cross pixel similarity as defined in [1].

  [1]: https://arxiv.org/abs/2012.02310

  Args:
    instance_masks: A float tensor of shape [num_instances, height, width]
    dilation: int, the dilation factor.

  Returns:
    dilated_same_label: A tensor of shape [8, num_instances, height, width]
  """

  instance_masks = tf.transpose(instance_masks, (1, 2, 0))

  neighbors = generate_2d_neighbors(instance_masks, dilation)
  instance_masks = instance_masks[tf.newaxis]
  same_mask_prob = ((instance_masks * neighbors) +
                    ((1 - instance_masks) * (1 - neighbors)))

  return tf.transpose(same_mask_prob, (0, 3, 1, 2))


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

    if (self._use_instance_embedding and
        (self._network_type == 'embedding_projection')):
      raise ValueError(('Cannot feed instance embedding to mask head when '
                        'computing embedding projection.'))

    if network_type == 'embedding_projection':
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

    if out.shape[-1] > 1:
      out = self.project_out(out)

    return tf.squeeze(out, axis=-1)


def deepmac_proto_to_params(deepmac_config):
  """Convert proto to named tuple."""

  loss = losses_pb2.Loss()
  # Add dummy localization loss to avoid the loss_builder throwing error.
  loss.localization_loss.weighted_l2.CopyFrom(
      losses_pb2.WeightedL2LocalizationLoss())
  loss.classification_loss.CopyFrom(deepmac_config.classification_loss)
  classification_loss, _, _, _, _, _, _ = (losses_builder.build(loss))

  jitter_mode = preprocessor_pb2.RandomJitterBoxes.JitterMode.Name(
      deepmac_config.jitter_mode).lower()

  return DeepMACParams(
      dim=deepmac_config.dim,
      classification_loss=classification_loss,
      task_loss_weight=deepmac_config.task_loss_weight,
      pixel_embedding_dim=deepmac_config.pixel_embedding_dim,
      allowed_masked_classes_ids=deepmac_config.allowed_masked_classes_ids,
      mask_size=deepmac_config.mask_size,
      mask_num_subsamples=deepmac_config.mask_num_subsamples,
      use_xy=deepmac_config.use_xy,
      network_type=deepmac_config.network_type,
      use_instance_embedding=deepmac_config.use_instance_embedding,
      num_init_channels=deepmac_config.num_init_channels,
      predict_full_resolution_masks=
      deepmac_config.predict_full_resolution_masks,
      postprocess_crop_size=deepmac_config.postprocess_crop_size,
      max_roi_jitter_ratio=deepmac_config.max_roi_jitter_ratio,
      roi_jitter_mode=jitter_mode,
      box_consistency_loss_weight=deepmac_config.box_consistency_loss_weight,
      color_consistency_threshold=deepmac_config.color_consistency_threshold,
      color_consistency_dilation=deepmac_config.color_consistency_dilation,
      color_consistency_loss_weight=deepmac_config.color_consistency_loss_weight
  )


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
               deepmac_params,
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

      loss = self._deepmac_params.classification_loss

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
      boxes: A [num_instances, 4] float tensor containing bounding boxes in
        normalized coordinates.
      pixel_embedding: A [height, width, embedding_size] float tensor
        containing spatial pixel embeddings.

    Returns:
      embedding: A [num_instances, mask_height, mask_width, embedding_size + 2]
        float tensor containing the inputs to the mask network. For each
        bounding box, we concatenate the normalized box coordinates to the
        cropped pixel embeddings. If predict_full_resolution_masks is set,
        mask_height and mask_width are the same as height and width of
        pixel_embedding. If not, mask_height and mask_width are the same as
        mask_size.
    """

    num_instances = tf.shape(boxes)[0]
    mask_size = self._deepmac_params.mask_size

    if self._deepmac_params.predict_full_resolution_masks:
      num_instances = tf.shape(boxes)[0]
      pixel_embedding = pixel_embedding[tf.newaxis, :, :, :]
      pixel_embeddings_processed = tf.tile(pixel_embedding,
                                           [num_instances, 1, 1, 1])
      image_shape = tf.shape(pixel_embeddings_processed)
      image_height, image_width = image_shape[1], image_shape[2]
      y_grid, x_grid = tf.meshgrid(tf.linspace(0.0, 1.0, image_height),
                                   tf.linspace(0.0, 1.0, image_width),
                                   indexing='ij')

      blist = box_list.BoxList(boxes)
      ycenter, xcenter, _, _ = blist.get_center_coordinates_and_sizes()
      y_grid = y_grid[tf.newaxis, :, :]
      x_grid = x_grid[tf.newaxis, :, :]

      y_grid -= ycenter[:, tf.newaxis, tf.newaxis]
      x_grid -= xcenter[:, tf.newaxis, tf.newaxis]
      coords = tf.stack([y_grid, x_grid], axis=3)

    else:
      # TODO(vighneshb) Explore multilevel_roi_align and align_corners=False.
      pixel_embeddings_processed = crop_and_resize_feature_map(
          pixel_embedding, boxes, mask_size)
      mask_shape = tf.shape(pixel_embeddings_processed)
      mask_height, mask_width = mask_shape[1], mask_shape[2]
      y_grid, x_grid = tf.meshgrid(tf.linspace(-1.0, 1.0, mask_height),
                                   tf.linspace(-1.0, 1.0, mask_width),
                                   indexing='ij')

      coords = tf.stack([y_grid, x_grid], axis=2)
      coords = coords[tf.newaxis, :, :, :]
      coords = tf.tile(coords, [num_instances, 1, 1, 1])

    if self._deepmac_params.use_xy:
      return tf.concat([coords, pixel_embeddings_processed], axis=3)
    else:
      return pixel_embeddings_processed

  def _get_instance_embeddings(self, boxes, instance_embedding):
    """Return the instance embeddings from bounding box centers.

    Args:
      boxes: A [num_instances, 4] float tensor holding bounding boxes. The
        coordinates are in normalized input space.
      instance_embedding: A [height, width, embedding_size] float tensor
        containing the instance embeddings.

    Returns:
      instance_embeddings: A [num_instances, embedding_size] shaped float tensor
        containing the center embedding for each instance.
    """
    blist = box_list.BoxList(boxes)
    output_height = tf.shape(instance_embedding)[0]
    output_width = tf.shape(instance_embedding)[1]

    blist_output = box_list_ops.to_absolute_coordinates(
        blist, output_height, output_width, check_range=False)
    (y_center_output, x_center_output,
     _, _) = blist_output.get_center_coordinates_and_sizes()
    center_coords_output = tf.stack([y_center_output, x_center_output], axis=1)
    center_coords_output_int = tf.cast(center_coords_output, tf.int32)
    center_latents = tf.gather_nd(instance_embedding, center_coords_output_int)

    return center_latents

  def _get_groundtruth_mask_output(self, boxes, masks):
    """Get the expected mask output for each box.

    Args:
      boxes: A [num_instances, 4] float tensor containing bounding boxes in
        normalized coordinates.
      masks: A [num_instances, height, width] float tensor containing binary
        ground truth masks.

    Returns:
      masks: If predict_full_resolution_masks is set, masks are not resized
      and the size of this tensor is [num_instances, input_height, input_width].
      Otherwise, returns a tensor of size [num_instances, mask_size, mask_size].
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

    height, width = tf.shape(gt)[1], tf.shape(gt)[2]

    return resize_instance_masks(logits, (height, width))

  def _compute_per_instance_mask_prediction_loss(
      self, boxes, mask_logits, mask_gt):
    """Compute the per-instance mask loss.

    Args:
      boxes: A [num_instances, 4] float tensor of GT boxes.
      mask_logits: A [num_instances, height, width] float tensor of predicted
        masks
      mask_gt: The groundtruth mask.

    Returns:
      loss: A [num_instances] shaped tensor with the loss for each instance.
    """
    num_instances = tf.shape(boxes)[0]
    mask_logits = self._resize_logits_like_gt(mask_logits, mask_gt)

    mask_logits = tf.reshape(mask_logits, [num_instances, -1, 1])
    mask_gt = tf.reshape(mask_gt, [num_instances, -1, 1])
    loss = self._deepmac_params.classification_loss(
        prediction_tensor=mask_logits,
        target_tensor=mask_gt,
        weights=tf.ones_like(mask_logits))

    # TODO(vighneshb) Make this configurable via config.
    # Skip normalization for dice loss because the denominator term already
    # does normalization.
    if isinstance(self._deepmac_params.classification_loss,
                  losses.WeightedDiceClassificationLoss):
      return tf.reduce_sum(loss, axis=1)
    else:
      return tf.reduce_mean(loss, axis=[1, 2])

  def _compute_per_instance_box_consistency_loss(
      self, boxes_gt, boxes_for_crop, mask_logits):
    """Compute the per-instance box consistency loss.

    Args:
      boxes_gt: A [num_instances, 4] float tensor of GT boxes.
      boxes_for_crop: A [num_instances, 4] float tensor of augmented boxes,
        to be used when using crop-and-resize based mask head.
      mask_logits: A [num_instances, height, width] float tensor of predicted
        masks.

    Returns:
      loss: A [num_instances] shaped tensor with the loss for each instance.
    """

    height, width = tf.shape(mask_logits)[1], tf.shape(mask_logits)[2]
    filled_boxes = fill_boxes(boxes_gt, height, width)[:, :, :, tf.newaxis]
    mask_logits = mask_logits[:, :, :, tf.newaxis]

    if self._deepmac_params.predict_full_resolution_masks:
      gt_crop = filled_boxes[:, :, :, 0]
      pred_crop = mask_logits[:, :, :, 0]
    else:
      gt_crop = crop_and_resize_instance_masks(
          filled_boxes, boxes_for_crop, self._deepmac_params.mask_size)
      pred_crop = crop_and_resize_instance_masks(
          mask_logits, boxes_for_crop, self._deepmac_params.mask_size)

    loss = 0.0
    for axis in [1, 2]:
      pred_max = tf.reduce_max(pred_crop, axis=axis)[:, :, tf.newaxis]
      gt_max = tf.reduce_max(gt_crop, axis=axis)[:, :, tf.newaxis]

      axis_loss = self._deepmac_params.classification_loss(
          prediction_tensor=pred_max,
          target_tensor=gt_max,
          weights=tf.ones_like(pred_max))
      loss += axis_loss

    # Skip normalization for dice loss because the denominator term already
    # does normalization.
    # TODO(vighneshb) Make this configurable via config.
    if isinstance(self._deepmac_params.classification_loss,
                  losses.WeightedDiceClassificationLoss):
      return tf.reduce_sum(loss, axis=1)
    else:
      return tf.reduce_mean(loss, axis=[1, 2])

  def _compute_per_instance_color_consistency_loss(
      self, boxes, preprocessed_image, mask_logits):
    """Compute the per-instance color consistency loss.

    Args:
      boxes: A [num_instances, 4] float tensor of GT boxes.
      preprocessed_image: A [height, width, 3] float tensor containing the
        preprocessed image.
      mask_logits: A [num_instances, height, width] float tensor of predicted
        masks.

    Returns:
      loss: A [num_instances] shaped tensor with the loss for each instance.
    """

    dilation = self._deepmac_params.color_consistency_dilation

    height, width = (tf.shape(preprocessed_image)[0],
                     tf.shape(preprocessed_image)[1])
    color_similarity = dilated_cross_pixel_similarity(
        preprocessed_image, dilation=dilation, theta=2.0)
    mask_probs = tf.nn.sigmoid(mask_logits)
    same_mask_label_probability = dilated_cross_same_mask_label(
        mask_probs, dilation=dilation)
    same_mask_label_probability = tf.clip_by_value(
        same_mask_label_probability, 1e-3, 1.0)

    color_similarity_mask = (
        color_similarity > self._deepmac_params.color_consistency_threshold)
    color_similarity_mask = tf.cast(
        color_similarity_mask[:, tf.newaxis, :, :], tf.float32)
    per_pixel_loss = -(color_similarity_mask *
                       tf.math.log(same_mask_label_probability))
    # TODO(vighneshb) explore if shrinking the box by 1px helps.
    box_mask = fill_boxes(boxes, height, width)
    box_mask_expanded = box_mask[tf.newaxis, :, :, :]

    per_pixel_loss = per_pixel_loss * box_mask_expanded
    loss = tf.reduce_sum(per_pixel_loss, axis=[0, 2, 3])
    num_box_pixels = tf.maximum(1.0, tf.reduce_sum(box_mask, axis=[1, 2]))
    loss = loss / num_box_pixels

    return loss

  def _compute_per_instance_deepmac_losses(
      self, boxes, masks, instance_embedding, pixel_embedding,
      image):
    """Returns the mask loss per instance.

    Args:
      boxes: A [num_instances, 4] float tensor holding bounding boxes. The
        coordinates are in normalized input space.
      masks: A [num_instances, input_height, input_width] float tensor
        containing the instance masks.
      instance_embedding: A [output_height, output_width, embedding_size]
        float tensor containing the instance embeddings.
      pixel_embedding: optional [output_height, output_width,
        pixel_embedding_size] float tensor containing the per-pixel embeddings.
      image: [output_height, output_width, channels] float tensor
        denoting the input image.

    Returns:
      mask_prediction_loss: A [num_instances] shaped float tensor containing the
        mask loss for each instance.
      box_consistency_loss: A [num_instances] shaped float tensor containing
        the box consistency loss for each instance.
      box_consistency_loss: A [num_instances] shaped float tensor containing
        the color consistency loss.
    """

    if tf.keras.backend.learning_phase():
      boxes_for_crop = preprocessor.random_jitter_boxes(
          boxes, self._deepmac_params.max_roi_jitter_ratio,
          jitter_mode=self._deepmac_params.roi_jitter_mode)
    else:
      boxes_for_crop = boxes

    mask_input = self._get_mask_head_input(
        boxes_for_crop, pixel_embedding)
    instance_embeddings = self._get_instance_embeddings(
        boxes_for_crop, instance_embedding)
    mask_logits = self._mask_net(
        instance_embeddings, mask_input,
        training=tf.keras.backend.learning_phase())
    mask_gt = self._get_groundtruth_mask_output(boxes_for_crop, masks)

    mask_prediction_loss = self._compute_per_instance_mask_prediction_loss(
        boxes_for_crop, mask_logits, mask_gt)

    box_consistency_loss = self._compute_per_instance_box_consistency_loss(
        boxes, boxes_for_crop, mask_logits)

    color_consistency_loss = self._compute_per_instance_color_consistency_loss(
        boxes, image, mask_logits)

    return mask_prediction_loss, box_consistency_loss, color_consistency_loss

  def _get_lab_image(self, preprocessed_image):
    raw_image = self._feature_extractor.preprocess_reverse(
        preprocessed_image)
    raw_image = raw_image / 255.0

    if tf_version.is_tf1():
      raise NotImplementedError(('RGB-to-LAB conversion required for the color'
                                 ' consistency loss is not supported in TF1.'))
    return tfio.experimental.color.rgb_to_lab(raw_image)

  def _compute_instance_masks_loss(self, prediction_dict):
    """Computes the mask loss.

    Args:
      prediction_dict: dict from predict() method containing
        INSTANCE_EMBEDDING and PIXEL_EMBEDDING prediction.
        Both of these are lists of tensors, each of size
        [batch_size, height, width, embedding_size].

    Returns:
      loss_dict: A dict mapping string (loss names) to scalar floats.
    """
    gt_boxes_list = self.groundtruth_lists(fields.BoxListFields.boxes)
    gt_weights_list = self.groundtruth_lists(fields.BoxListFields.weights)
    gt_masks_list = self.groundtruth_lists(fields.BoxListFields.masks)
    gt_classes_list = self.groundtruth_lists(fields.BoxListFields.classes)

    allowed_masked_classes_ids = (
        self._deepmac_params.allowed_masked_classes_ids)

    loss_dict = {
        DEEP_MASK_ESTIMATION: 0.0,
        DEEP_MASK_BOX_CONSISTENCY: 0.0,
        DEEP_MASK_COLOR_CONSISTENCY: 0.0
    }

    prediction_shape = tf.shape(prediction_dict[INSTANCE_EMBEDDING][0])
    height, width = prediction_shape[1], prediction_shape[2]

    preprocessed_image = tf.image.resize(
        prediction_dict['preprocessed_inputs'], (height, width))
    image = self._get_lab_image(preprocessed_image)

    # TODO(vighneshb) See if we can save memory by only using the final
    # prediction
    # Iterate over multiple preidctions by backbone (for hourglass length=2)
    for instance_pred, pixel_pred in zip(
        prediction_dict[INSTANCE_EMBEDDING],
        prediction_dict[PIXEL_EMBEDDING]):
      # Iterate over samples in batch
      # TODO(vighneshb) find out how autograph is handling this. Converting
      # to a single op may give speed improvements
      for i, (boxes, weights, classes, masks) in enumerate(
          zip(gt_boxes_list, gt_weights_list, gt_classes_list, gt_masks_list)):

        # TODO(vighneshb) Add sub-sampling back if required.
        classes, valid_mask_weights, masks = filter_masked_classes(
            allowed_masked_classes_ids, classes, weights, masks)

        (per_instance_mask_loss, per_instance_consistency_loss,
         per_instance_color_consistency_loss) = (
             self._compute_per_instance_deepmac_losses(
                 boxes, masks, instance_pred[i], pixel_pred[i],
                 image[i]))
        per_instance_mask_loss *= valid_mask_weights
        per_instance_consistency_loss *= weights

        num_instances = tf.maximum(tf.reduce_sum(weights), 1.0)
        num_instances_allowed = tf.maximum(
            tf.reduce_sum(valid_mask_weights), 1.0)

        loss_dict[DEEP_MASK_ESTIMATION] += (
            tf.reduce_sum(per_instance_mask_loss) / num_instances_allowed)

        loss_dict[DEEP_MASK_BOX_CONSISTENCY] += (
            tf.reduce_sum(per_instance_consistency_loss) / num_instances)

        loss_dict[DEEP_MASK_COLOR_CONSISTENCY] += (
            tf.reduce_sum(per_instance_color_consistency_loss) / num_instances)

    batch_size = len(gt_boxes_list)
    num_predictions = len(prediction_dict[INSTANCE_EMBEDDING])

    return dict((key, loss / float(batch_size * num_predictions))
                for key, loss in loss_dict.items())

  def loss(self, prediction_dict, true_image_shapes, scope=None):

    losses_dict = super(DeepMACMetaArch, self).loss(
        prediction_dict, true_image_shapes, scope)

    if self._deepmac_params is not None:
      mask_loss_dict = self._compute_instance_masks_loss(
          prediction_dict=prediction_dict)

      losses_dict[LOSS_KEY_PREFIX + '/' + DEEP_MASK_ESTIMATION] = (
          self._deepmac_params.task_loss_weight * mask_loss_dict[
              DEEP_MASK_ESTIMATION]
      )

      if self._deepmac_params.box_consistency_loss_weight > 0.0:
        losses_dict[LOSS_KEY_PREFIX + '/' + DEEP_MASK_BOX_CONSISTENCY] = (
            self._deepmac_params.box_consistency_loss_weight * mask_loss_dict[
                DEEP_MASK_BOX_CONSISTENCY]
        )

      if self._deepmac_params.color_consistency_loss_weight > 0.0:
        losses_dict[LOSS_KEY_PREFIX + '/' + DEEP_MASK_COLOR_CONSISTENCY] = (
            self._deepmac_params.box_consistency_loss_weight * mask_loss_dict[
                DEEP_MASK_COLOR_CONSISTENCY]
        )
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

    def process(elems):
      boxes, instance_embedding, pixel_embedding = elems
      return self._postprocess_sample(boxes, instance_embedding,
                                      pixel_embedding)

    max_instances = self._center_params.max_box_predictions
    return tf.map_fn(process, [boxes_output_stride, instance_embedding,
                               pixel_embedding],
                     dtype=tf.float32, parallel_iterations=max_instances)

  def _postprocess_sample(self, boxes_output_stride,
                          instance_embedding, pixel_embedding):
    """Post process masks for a single sample.

    Args:
      boxes_output_stride: A [num_instances, 4] float tensor containing
        bounding boxes in the absolute output space.
      instance_embedding: A [output_height, output_width, embedding_size]
        float tensor containing instance embeddings.
      pixel_embedding: A [batch_size, output_height, output_width,
        pixel_embedding_size] float tensor containing the per-pixel embedding.

    Returns:
      masks: A float tensor of size [num_instances, mask_height, mask_width]
        containing binary per-box instance masks. If
        predict_full_resolution_masks is set, the masks will be resized to
        postprocess_crop_size. Otherwise, mask_height=mask_width=mask_size
    """

    height, width = (tf.shape(instance_embedding)[0],
                     tf.shape(instance_embedding)[1])
    height, width = tf.cast(height, tf.float32), tf.cast(width, tf.float32)
    blist = box_list.BoxList(boxes_output_stride)
    blist = box_list_ops.to_normalized_coordinates(
        blist, height, width, check_range=False)
    boxes = blist.get()

    mask_input = self._get_mask_head_input(boxes, pixel_embedding)
    instance_embeddings = self._get_instance_embeddings(
        boxes, instance_embedding)

    mask_logits = self._mask_net(
        instance_embeddings, mask_input,
        training=tf.keras.backend.learning_phase())

    # TODO(vighneshb) Explore sweeping mask thresholds.

    if self._deepmac_params.predict_full_resolution_masks:

      height, width = tf.shape(mask_logits)[1], tf.shape(mask_logits)[2]
      height *= self._stride
      width *= self._stride
      mask_logits = resize_instance_masks(mask_logits, (height, width))
      mask_logits = crop_masks_within_boxes(
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
