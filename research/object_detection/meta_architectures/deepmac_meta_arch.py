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


INSTANCE_EMBEDDING = 'INSTANCE_EMBEDDING'
PIXEL_EMBEDDING = 'PIXEL_EMBEDDING'
DEEP_MASK_ESTIMATION = 'deep_mask_estimation'
LOSS_KEY_PREFIX = center_net_meta_arch.LOSS_KEY_PREFIX


class DeepMACParams(
    collections.namedtuple('DeepMACParams', [
        'classification_loss', 'dim', 'task_loss_weight', 'pixel_embedding_dim',
        'allowed_masked_classes_ids', 'mask_size', 'mask_num_subsamples',
        'use_xy', 'network_type', 'use_instance_embedding', 'num_init_channels',
        'predict_full_resolution_masks', 'postprocess_crop_size',
        'max_roi_jitter_ratio', 'roi_jitter_mode'
    ])):
  """Class holding the DeepMAC network configutration."""

  __slots__ = ()

  def __new__(cls, classification_loss, dim, task_loss_weight,
              pixel_embedding_dim, allowed_masked_classes_ids, mask_size,
              mask_num_subsamples, use_xy, network_type, use_instance_embedding,
              num_init_channels, predict_full_resolution_masks,
              postprocess_crop_size, max_roi_jitter_ratio,
              roi_jitter_mode):
    return super(DeepMACParams,
                 cls).__new__(cls, classification_loss, dim,
                              task_loss_weight, pixel_embedding_dim,
                              allowed_masked_classes_ids, mask_size,
                              mask_num_subsamples, use_xy, network_type,
                              use_instance_embedding, num_init_channels,
                              predict_full_resolution_masks,
                              postprocess_crop_size, max_roi_jitter_ratio,
                              roi_jitter_mode)


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

    instance_embedding = instance_embedding[:, tf.newaxis, tf.newaxis, :]
    instance_embedding = tf.tile(instance_embedding, [1, height, width, 1])

    if self._use_instance_embedding:
      inputs = tf.concat([pixel_embedding, instance_embedding], axis=3)
    else:
      inputs = pixel_embedding

    out = self._net(inputs)
    if isinstance(out, list):
      out = out[-1]

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
      roi_jitter_mode=jitter_mode
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
    else:
      # TODO(vighneshb) Explore multilevel_roi_align and align_corners=False.
      pixel_embeddings_cropped = spatial_transform_ops.matmul_crop_and_resize(
          pixel_embedding[tf.newaxis], boxes[tf.newaxis],
          [mask_size, mask_size])
      pixel_embeddings_processed = pixel_embeddings_cropped[0]

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
      cropped_masks = spatial_transform_ops.matmul_crop_and_resize(
          masks[:, :, :, tf.newaxis], boxes[:, tf.newaxis, :],
          [mask_size, mask_size])
      cropped_masks = tf.stop_gradient(cropped_masks)
      cropped_masks = tf.squeeze(cropped_masks, axis=[1, 4])

      # TODO(vighneshb) should we discretize masks?
      return cropped_masks

  def _resize_logits_like_gt(self, logits, gt):

    height, width = tf.shape(gt)[1], tf.shape(gt)[2]

    return resize_instance_masks(logits, (height, width))

  def _compute_per_instance_mask_loss(
      self, boxes, masks, instance_embedding, pixel_embedding):
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

    Returns:
      mask_loss: A [num_instances] shaped float tensor containing the
        mask loss for each instance.
    """

    num_instances = tf.shape(boxes)[0]

    if tf.keras.backend.learning_phase():
      boxes = preprocessor.random_jitter_boxes(
          boxes, self._deepmac_params.max_roi_jitter_ratio,
          jitter_mode=self._deepmac_params.roi_jitter_mode)
    mask_input = self._get_mask_head_input(
        boxes, pixel_embedding)
    instance_embeddings = self._get_instance_embeddings(
        boxes, instance_embedding)

    mask_logits = self._mask_net(
        instance_embeddings, mask_input,
        training=tf.keras.backend.learning_phase())
    mask_gt = self._get_groundtruth_mask_output(boxes, masks)
    mask_logits = self._resize_logits_like_gt(mask_logits, mask_gt)

    mask_logits = tf.reshape(mask_logits, [num_instances, -1, 1])
    mask_gt = tf.reshape(mask_gt, [num_instances, -1, 1])
    loss = self._deepmac_params.classification_loss(
        prediction_tensor=mask_logits,
        target_tensor=mask_gt,
        weights=tf.ones_like(mask_logits))

    # TODO(vighneshb) Make this configurable via config.
    if isinstance(self._deepmac_params.classification_loss,
                  losses.WeightedDiceClassificationLoss):
      return tf.reduce_sum(loss, axis=1)
    else:
      return tf.reduce_mean(loss, axis=[1, 2])

  def _compute_instance_masks_loss(self, prediction_dict):
    """Computes the mask loss.

    Args:
      prediction_dict: dict from predict() method containing
        INSTANCE_EMBEDDING and PIXEL_EMBEDDING prediction.
        Both of these are lists of tensors, each of size
        [batch_size, height, width, embedding_size].

    Returns:
      loss: float, the mask loss as a scalar.
    """
    gt_boxes_list = self.groundtruth_lists(fields.BoxListFields.boxes)
    gt_weights_list = self.groundtruth_lists(fields.BoxListFields.weights)
    gt_masks_list = self.groundtruth_lists(fields.BoxListFields.masks)
    gt_classes_list = self.groundtruth_lists(fields.BoxListFields.classes)

    allowed_masked_classes_ids = (
        self._deepmac_params.allowed_masked_classes_ids)

    total_loss = 0.0

    # Iterate over multiple preidctions by backbone (for hourglass length=2)
    for instance_pred, pixel_pred in zip(
        prediction_dict[INSTANCE_EMBEDDING],
        prediction_dict[PIXEL_EMBEDDING]):
      # Iterate over samples in batch
      # TODO(vighneshb) find out how autograph is handling this. Converting
      # to a single op may give speed improvements
      for i, (boxes, weights, classes, masks) in enumerate(
          zip(gt_boxes_list, gt_weights_list, gt_classes_list, gt_masks_list)):

        _, weights, masks = filter_masked_classes(allowed_masked_classes_ids,
                                                  classes, weights, masks)
        num_subsample = self._deepmac_params.mask_num_subsamples
        _, weights, boxes, masks = subsample_instances(
            classes, weights, boxes, masks, num_subsample)

        per_instance_loss = self._compute_per_instance_mask_loss(
            boxes, masks, instance_pred[i], pixel_pred[i])
        per_instance_loss *= weights

        num_instances = tf.maximum(tf.reduce_sum(weights), 1.0)

        total_loss += tf.reduce_sum(per_instance_loss) / num_instances

    batch_size = len(gt_boxes_list)
    num_predictions = len(prediction_dict[INSTANCE_EMBEDDING])

    return total_loss / float(batch_size * num_predictions)

  def loss(self, prediction_dict, true_image_shapes, scope=None):

    losses_dict = super(DeepMACMetaArch, self).loss(
        prediction_dict, true_image_shapes, scope)

    if self._deepmac_params is not None:
      mask_loss = self._compute_instance_masks_loss(
          prediction_dict=prediction_dict)
      key = LOSS_KEY_PREFIX + '/' + DEEP_MASK_ESTIMATION
      losses_dict[key] = (
          self._deepmac_params.task_loss_weight * mask_loss
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
