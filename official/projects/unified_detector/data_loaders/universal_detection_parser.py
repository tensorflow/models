# Copyright 2023 The TensorFlow Authors. All Rights Reserved.
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

"""Data parser for universal detector."""

import enum
import functools
from typing import Any, Tuple

import gin
import tensorflow as tf

from official.projects.unified_detector.data_loaders import autoaugment
from official.projects.unified_detector.data_loaders import tf_example_decoder
from official.projects.unified_detector.utils import utilities
from official.projects.unified_detector.utils.typing import NestedTensorDict
from official.projects.unified_detector.utils.typing import TensorDict


@gin.constants_from_enum
class DetectionClass(enum.IntEnum):
  """As in `PageLayoutEntity.EntityType`."""
  WORD = 0
  LINE = 2
  PARAGRAPH = 3
  BLOCK = 4


NOT_ANNOTATED_ID = 8


def _erase(mask: tf.Tensor,
           feature: tf.Tensor,
           min_val: float = 0.,
           max_val: float = 256.) -> tf.Tensor:
  """Erase the feature maps with a mask.

  Erase feature maps with a mask and replace the erased area with uniform random
  noise. The mask can have different size from the feature maps.

  Args:
    mask: an (h, w) binay mask for pixels to erase with. Value 1 represents
      pixels to erase.
    feature: the (H, W, C) feature maps to erase from.
    min_val: The minimum value of random noise.
    max_val: The maximum value of random noise.

  Returns:
      The (H, W, C) feature maps, with pixels in mask replaced with noises. It's
    equal to mask * noise + (1 - mask) * feature.
  """
  h, w, c = utilities.resolve_shape(feature)
  resized_mask = tf.image.resize(
      tf.tile(tf.expand_dims(tf.cast(mask, tf.float32), -1), (1, 1, c)), (h, w))
  erased = tf.where(
      condition=(resized_mask > 0.5),
      x=tf.cast(tf.random.uniform((h, w, c), min_val, max_val), feature.dtype),
      y=feature)
  return erased


@gin.configurable(denylist=['is_training'])
class UniDetectorParserFn(object):
  """Data parser for universal detector."""

  def __init__(
      self,
      is_training: bool,
      output_dimension: int = 1025,
      mask_dimension: int = -1,
      max_num_instance: int = 128,
      rot90_probability: float = 0.5,
      use_color_distortion: bool = True,
      randaug_mag: float = 5.,
      randaug_std: float = 0.5,
      randaug_layer: int = 2,
      randaug_prob: float = 0.5,
      use_cropping: bool = True,
      crop_min_scale: float = 0.5,
      crop_max_scale: float = 1.5,
      crop_min_aspect: float = 4 / 5,
      crop_max_aspect: float = 5 / 4,
      is_shape_defined: bool = True,
      use_tpu: bool = True,
      detection_unit: DetectionClass = DetectionClass.LINE,
  ):
    """Constructor.

    Args:
      is_training: bool indicating TRAIN or EVAL.
      output_dimension: The size of input images.
      mask_dimension: The size of the output mask. If negative or zero, it will
        be set the same as output_dimension.
      max_num_instance: The maximum number of instances to output. If it's
        negative, padding or truncating will not be performed.
      rot90_probability: The probability of rotating multiples of 90 degrees.
      use_color_distortion: Whether to apply color distortions to images (via
        autoaugment).
      randaug_mag: (autoaugment parameter) Color distortion magnitude. Note
        that, this value should be set conservatively, as some color distortions
        can easily make text illegible e.g. posterize.
      randaug_std: (autoaugment parameter) Randomness in color distortion
        magnitude.
      randaug_layer: (autoaugment parameter) Number of color distortion
        operations.
      randaug_prob: (autoaugment parameter) Probabilily of applying each
        distortion operation.
      use_cropping: Bool, whether to use random cropping and resizing in
        training.
      crop_min_scale: The minimum scale of a random crop.
      crop_max_scale: The maximum scale of a random crop. If >1, it means the
        images are downsampled.
      crop_min_aspect: The minimum aspect ratio of a random crop.
      crop_max_aspect: The maximum aspect ratio of a random crop.
      is_shape_defined: Whether to define the static shapes for all features and
        labels. This must be set to True in TPU training as it requires static
        shapes for all tensors.
      use_tpu: Whether the inputs are fed to a TPU device.
      detection_unit: Whether word or line (or else) is regarded as an entity.
        The instance masks will be at word or line level.
    """
    if is_training and max_num_instance < 0:
      raise ValueError('In TRAIN mode, padding/truncation is required.')

    self._is_training = is_training
    self._output_dimension = output_dimension
    self._mask_dimension = (
        mask_dimension if mask_dimension > 0 else output_dimension)
    self._max_num_instance = max_num_instance
    self._decoder = tf_example_decoder.TfExampleDecoder(
        num_additional_channels=3, additional_class_names=['parent'])
    self._use_color_distortion = use_color_distortion
    self._rot90_probability = rot90_probability
    self._randaug_mag = randaug_mag
    self._randaug_std = randaug_std
    self._randaug_layer = randaug_layer
    self._randaug_prob = randaug_prob
    self._use_cropping = use_cropping
    self._crop_min_scale = crop_min_scale
    self._crop_max_scale = crop_max_scale
    self._crop_min_aspect = crop_min_aspect
    self._crop_max_aspect = crop_max_aspect
    self._is_shape_defined = is_shape_defined
    self._use_tpu = use_tpu
    self._detection_unit = detection_unit

  def __call__(self, value: str) -> Tuple[TensorDict, NestedTensorDict]:
    """Parsing the data.

    Args:
      value: The serialized data sample.

    Returns:
      Two dicts for features and labels.
      features:
        'source_id': id of the sample; only in EVAL mode
        'images': the normalized images, (output_dimension, output_dimension, 3)
      labels:
        See `_prepare_labels` for its content.
    """
    data = self._decoder.decode(value)
    features = {}
    labels = {}
    self._preprocess(data, features, labels)
    self._rot90k(data, features, labels)
    self._crop_and_resize(data, features, labels)
    self._color_distortion_and_normalize(data, features, labels)
    self._prepare_labels(data, features, labels)
    self._define_shapes(features, labels)
    return features, labels

  def _preprocess(self, data: TensorDict, features: TensorDict,
                  unused_labels: TensorDict):
    """All kinds of preprocessing of the decoded data dict."""
    # (1) Decode the entity_id_mask: a H*W*1 mask, each pixel equals to
    #     (1 + position) of the entity in the GT entity list. The IDs
    #     (which can be larger than 255) are stored in the last two channels.
    data['additional_channels'] = tf.cast(data['additional_channels'], tf.int32)
    entity_id_mask = (
        data['additional_channels'][:, :, -2:-1] * 256 +
        data['additional_channels'][:, :, -1:])
    data['entity_id_mask'] = entity_id_mask

    # (2) Write image id. Used in evaluation.
    if not self._use_tpu:
      features['source_id'] = data['source_id']

    # (3) Block mask: area without annotation
    data['image'] = _erase(
        data['additional_channels'][:, :, 0],
        data['image'],
        min_val=0.,
        max_val=256.)

  def _rot90k(self, data: TensorDict, unused_features: TensorDict,
              unused_labels: TensorDict):
    """Rotate the image, gt_bboxes, masks by 90k degrees."""
    if not self._is_training:
      return

    rotate_90_choice = tf.random.uniform([])

    def _rotate():
      """Rotation.

      These will be rotated:
        image,
        rbox,
        entity_id_mask,
      TODO(longshangbang): rotate vertices.

      Returns:
        The rotated tensors of the above fields.
      """
      k = tf.random.uniform([], 1, 4, dtype=tf.int32)
      h, w, _ = utilities.resolve_shape(data['image'])
      # Image
      rotated_img = tf.image.rot90(data['image'], k=k, name='image_rot90k')
      # Box
      rotate_box_op = functools.partial(
          utilities.rotate_rboxes90,
          rboxes=data['groundtruth_boxes'],
          image_width=w,
          image_height=h)
      rotated_boxes = tf.switch_case(
          k - 1,  # Indices start with 1.
          branch_fns=[
              lambda: rotate_box_op(rotation_count=1),
              lambda: rotate_box_op(rotation_count=2),
              lambda: rotate_box_op(rotation_count=3)
          ])
      # Mask
      rotated_mask = tf.image.rot90(
          data['entity_id_mask'], k=k, name='mask_rot90k')
      return rotated_img, rotated_boxes, rotated_mask

    # pylint: disable=g-long-lambda
    (data['image'], data['groundtruth_boxes'],
     data['entity_id_mask']) = tf.cond(
         rotate_90_choice < self._rot90_probability, _rotate, lambda:
         (data['image'], data['groundtruth_boxes'], data['entity_id_mask']))
    # pylint: enable=g-long-lambda

  def _crop_and_resize(self, data: TensorDict, unused_features: TensorDict,
                       unused_labels: TensorDict):
    """Perform random cropping and resizing."""
    # TODO(longshangbang): resize & translate box as well
    # TODO(longshangbang): resize & translate vertices as well

    # Get cropping target.
    h, w = utilities.resolve_shape(data['image'])[:2]
    left, top, crop_w, crop_h, pad_w, pad_h = self._get_crop_box(
        tf.cast(h, tf.float32), tf.cast(w, tf.float32))

    # Crop the image. (Pad the images if the crop box is larger than image.)
    if self._is_training:
      # padding left, top, right, bottom
      pad_left = tf.random.uniform([], 0, pad_w + 1, dtype=tf.int32)
      pad_top = tf.random.uniform([], 0, pad_h + 1, dtype=tf.int32)
    else:
      pad_left = 0
      pad_top = 0
    cropped_img = tf.image.crop_to_bounding_box(data['image'], top, left,
                                                crop_h, crop_w)
    padded_img = tf.pad(
        cropped_img,
        [[pad_top, pad_h - pad_top], [pad_left, pad_w - pad_left], [0, 0]],
        constant_values=127)

    # Resize images
    data['resized_image'] = tf.image.resize(
        padded_img, (self._output_dimension, self._output_dimension))
    data['resized_image'] = tf.cast(data['resized_image'], tf.uint8)

    # Crop the masks
    cropped_masks = tf.image.crop_to_bounding_box(data['entity_id_mask'], top,
                                                  left, crop_h, crop_w)
    padded_masks = tf.pad(
        cropped_masks,
        [[pad_top, pad_h - pad_top], [pad_left, pad_w - pad_left], [0, 0]])

    # Resize masks
    data['resized_masks'] = tf.image.resize(
        padded_masks, (self._mask_dimension, self._mask_dimension),
        method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    data['resized_masks'] = tf.squeeze(data['resized_masks'], -1)

  def _get_crop_box(
      self, h: tf.Tensor,
      w: tf.Tensor) -> Tuple[Any, Any, tf.Tensor, tf.Tensor, Any, Any]:
    """Get the cropping box.

    Args:
      h: The height of the image to crop. Should be float type.
      w: The width of the image to crop. Should be float type.

    Returns:
      A tuple representing (left, top, crop_w, crop_h, pad_w, pad_h).
      Then in `self._crop_and_resize`, a crop will be extracted with bounding
      box from top-left corner (left, top) and with size (crop_w, crop_h). This
      crop will then be padded with (pad_w, pad_h) to square sizes.
      The outputs also are re-cast to int32 type.
    """
    if not self._is_training or not self._use_cropping:
      # cast back to integers.
      w = tf.cast(w, tf.int32)
      h = tf.cast(h, tf.int32)
      side = tf.maximum(w, h)
      return 0, 0, w, h, side - w, side - h

    # Get box size
    scale = tf.random.uniform([], self._crop_min_scale, self._crop_max_scale)
    max_edge = tf.maximum(w, h)
    long_edge = max_edge * scale

    sqrt_aspect_ratio = tf.math.sqrt(
        tf.random.uniform([], self._crop_min_aspect, self._crop_max_aspect))
    box_h = long_edge / sqrt_aspect_ratio
    box_w = long_edge * sqrt_aspect_ratio

    # Get box location
    left = tf.random.uniform([], 0., tf.maximum(0., w - box_w))
    top = tf.random.uniform([], 0., tf.maximum(0., h - box_h))
    # Get crop & pad
    crop_w = tf.minimum(box_w, w - left)
    crop_h = tf.minimum(box_h, h - top)
    pad_w = box_w - crop_w
    pad_h = box_h - crop_h
    return (tf.cast(left, tf.int32), tf.cast(top, tf.int32),
            tf.cast(crop_w, tf.int32), tf.cast(crop_h, tf.int32),
            tf.cast(pad_w, tf.int32), tf.cast(pad_h, tf.int32))

  def _color_distortion_and_normalize(self, data: TensorDict,
                                      features: TensorDict,
                                      unused_labels: TensorDict):
    """Distort colors."""
    if self._is_training and self._use_color_distortion:
      data['resized_image'] = autoaugment.distort_image_with_randaugment(
          data['resized_image'], self._randaug_layer, self._randaug_mag,
          self._randaug_std, True, self._randaug_prob, True)
    # Normalize
    features['images'] = utilities.normalize_image_to_range(
        data['resized_image'])

  def _prepare_labels(self, data: TensorDict, features: TensorDict,
                      labels: TensorDict):
    """This function prepares the labels.

    These following targets are added to labels['segmentation_output']:
      'gt_word_score': A (h, w) float32 mask for textness score. 1 for word,
        0 for bkg.

    These following targets are added to labels['instance_labels']:
      'num_instance': A float scalar tensor for the total number of
        instances. It is bounded by the maximum number of instances allowed.
        It includes the special background instance, so it equals to
        (1 + entity numbers).
      'masks': A (h, w) int32 mask for entity IDs. The value of each pixel is
        the id of the entity it belongs to. A value of `0` means the bkg mask.
      'classes': A (max_num,) int tensor indicating the classes of each
        instance:
          2 for background
          1 for text entity
          0 for non-object
      'masks_sizes': A (max_num,) float tensor for the size of all masks.
      'gt_weights': Whether it's difficult / does not have text annotation.

    These following targets are added to labels['paragraph_labels']:
      'paragraph_ids': A (max_num,) integer tensor for paragprah id. if `-1`,
        then no paragraph label for this text.
      'has_para_ids': A float scalar; 1.0 if the sample has paragraph labels.

    Args:
      data: The data dictionary.
      features: The feature dict.
      labels: The label dict.
    """
    # Segmentation labels:
    self._get_segmentation_labels(data, features, labels)
    # Instance labels:
    self._get_instance_labels(data, features, labels)

  def _get_segmentation_labels(self, data: TensorDict,
                               unused_features: TensorDict,
                               labels: NestedTensorDict):
    labels['segmentation_output'] = {
        'gt_word_score': tf.cast((data['resized_masks'] > 0), tf.float32)
    }

  def _get_instance_labels(self, data: TensorDict, features: TensorDict,
                           labels: NestedTensorDict):
    """Generate the labels for text entity detection."""

    labels['instance_labels'] = {}
    # (1) Depending on `detection_unit`:
    #     Convert the word-id map to line-id map or use the word-id map directly
    # Word entity ids start from 1 in the map, so pad a -1 at the beginning of
    # the parent list to counter this offset.
    padded_parent = tf.concat(
        [tf.constant([-1]),
         tf.cast(data['groundtruth_parent'], tf.int32)], 0)
    if self._detection_unit == DetectionClass.WORD:
      entity_id_mask = data['resized_masks']
    elif self._detection_unit == DetectionClass.LINE:
      # The pixel value is entity_id + 1, shape = [H, W]; 0 for background.
      # correctness:
      # 0s in data['resized_masks'] --> padded_parent[0] == -1
      # i-th entity in plp.entities --> i+1 in data['resized_masks']
      #                             --> padded_parent[i+1]
      #                             --> data['groundtruth_parent'][i]
      #                             --> the parent of i-th entity
      entity_id_mask = tf.gather(padded_parent, data['resized_masks']) + 1
    elif self._detection_unit == DetectionClass.PARAGRAPH:
      # directly segmenting paragraphs; two hops here.
      entity_id_mask = tf.gather(padded_parent, data['resized_masks']) + 1
      entity_id_mask = tf.gather(padded_parent, entity_id_mask) + 1
    else:
      raise ValueError(f'No such detection unit: {self._detection_unit}')
    data['entity_id_mask'] = entity_id_mask

    # (2) Get individual masks for entities.
    entity_selection_mask = tf.equal(data['groundtruth_classes'],
                                     self._detection_unit)
    num_all_entity = utilities.resolve_shape(data['groundtruth_classes'])[0]
    # entity_ids is a 1-D tensor for IDs of all entities of a certain type.
    entity_ids = tf.boolean_mask(
        tf.range(num_all_entity, dtype=tf.int32), entity_selection_mask)  # (N,)
    # +1 to match the entity ids in entity_id_mask
    entity_ids = tf.reshape(entity_ids, (-1, 1, 1)) + 1
    individual_masks = tf.expand_dims(entity_id_mask, 0)
    individual_masks = tf.equal(entity_ids, individual_masks)  # (N, H, W), bool
    # TODO(longshangbang): replace with real mask sizes computing.
    # Currently, we use full-resolution masks for individual_masks. In order to
    # compute mask sizes, we need to convert individual_masks to int/float type.
    # This will cause OOM because the mask is too large.
    masks_sizes = tf.cast(
        tf.reduce_any(individual_masks, axis=[1, 2]), tf.float32)
    # remove empty masks (usually caused by cropping)
    non_empty_masks_ids = tf.not_equal(masks_sizes, 0)
    valid_masks = tf.boolean_mask(individual_masks, non_empty_masks_ids)
    valid_entity_ids = tf.boolean_mask(entity_ids, non_empty_masks_ids)[:, 0, 0]

    # (3) Write num of instance
    num_instance = tf.reduce_sum(tf.cast(non_empty_masks_ids, tf.float32))
    num_instance_and_bkg = num_instance + 1
    if self._max_num_instance >= 0:
      num_instance_and_bkg = tf.minimum(num_instance_and_bkg,
                                        self._max_num_instance)
    labels['instance_labels']['num_instance'] = num_instance_and_bkg

    # (4) Write instance masks
    num_entity_int = tf.cast(num_instance, tf.int32)
    max_num_entities = self._max_num_instance - 1  # Spare 1 for bkg.
    pad_num = tf.maximum(max_num_entities - num_entity_int, 0)
    padded_valid_masks = tf.pad(valid_masks, [[0, pad_num], [0, 0], [0, 0]])

    # If there are more instances than allowed, randomly sample some.
    # `random_selection_mask` is a 0/1 array; the maximum number of 1 is
    # `self._max_num_instance`; if not bound, it's an array with all 1s.
    if self._max_num_instance >= 0:
      padded_size = num_entity_int + pad_num
      random_selection = tf.random.uniform((padded_size,), dtype=tf.float32)
      selected_indices = tf.math.top_k(random_selection, k=max_num_entities)[1]
      random_selection_mask = tf.scatter_nd(
          indices=tf.expand_dims(selected_indices, axis=-1),
          updates=tf.ones((max_num_entities,), dtype=tf.bool),
          shape=(padded_size,))
    else:
      random_selection_mask = tf.ones((num_entity_int,), dtype=tf.bool)
    random_discard_mask = tf.logical_not(random_selection_mask)

    kept_masks = tf.boolean_mask(padded_valid_masks, random_selection_mask)
    erased_masks = tf.boolean_mask(padded_valid_masks, random_discard_mask)
    erased_masks = tf.cast(tf.reduce_any(erased_masks, axis=0), tf.float32)
    # erase text instances that are obmitted.
    features['images'] = _erase(erased_masks, features['images'], -1., 1.)
    labels['segmentation_output']['gt_word_score'] *= 1. - erased_masks
    kept_masks_and_bkg = tf.concat(
        [
            tf.math.logical_not(
                tf.reduce_any(kept_masks, axis=0, keepdims=True)),  # bkg
            kept_masks,
        ],
        0)
    labels['instance_labels']['masks'] = tf.argmax(kept_masks_and_bkg, axis=0)

    # (5) Write mask size
    # TODO(longshangbang): replace with real masks sizes
    masks_sizes = tf.cast(
        tf.reduce_any(kept_masks_and_bkg, axis=[1, 2]), tf.float32)
    labels['instance_labels']['masks_sizes'] = masks_sizes
    # (6) Write classes.
    classes = tf.ones((num_instance,), dtype=tf.int32)
    classes = tf.concat([tf.constant(2, tf.int32, (1,)), classes], 0)  # bkg
    if self._max_num_instance >= 0:
      classes = utilities.truncate_or_pad(classes, self._max_num_instance, 0)
    labels['instance_labels']['classes'] = classes

    # (7) gt-weights
    selected_ids = tf.boolean_mask(valid_entity_ids,
                                   random_selection_mask[:num_entity_int])

    if self._detection_unit != DetectionClass.PARAGRAPH:
      gt_text = tf.gather(data['groundtruth_text'], selected_ids - 1)
      gt_weights = tf.cast(tf.strings.length(gt_text) > 0, tf.float32)
    else:
      text_types = tf.concat(
          [
              tf.constant([8]),
              tf.cast(data['groundtruth_content_type'], tf.int32),
              # TODO(longshangbang): temp solution for tfes with no para labels
              tf.constant(8, shape=(1000,)),
          ],
          0)
      para_types = tf.gather(text_types, selected_ids)

      gt_weights = tf.cast(
          tf.not_equal(para_types, NOT_ANNOTATED_ID), tf.float32)

    gt_weights = tf.concat([tf.constant(1., shape=(1,)), gt_weights], 0)  # bkg
    if self._max_num_instance >= 0:
      gt_weights = utilities.truncate_or_pad(
          gt_weights, self._max_num_instance, 0)
    labels['instance_labels']['gt_weights'] = gt_weights

    # (8) get paragraph label
    # In this step, an array `{p_i}` is generated. `p_i` is an integer that
    # indicates the group of paragraph which i-th text belongs to. `p_i` == -1
    # if this instance is non-text or it has no paragraph labels.
    # word -> line -> paragraph
    if self._detection_unit == DetectionClass.WORD:
      num_hop = 2
    elif self._detection_unit == DetectionClass.LINE:
      num_hop = 1
    elif self._detection_unit == DetectionClass.PARAGRAPH:
      num_hop = 0
    else:
      raise ValueError(f'No such detection unit: {self._detection_unit}. '
                       'Note that this error should have been raised in '
                       'previous lines, not here!')
    para_ids = tf.identity(selected_ids)  # == id in plp + 1
    for _ in range(num_hop):
      para_ids = tf.gather(padded_parent, para_ids) + 1

    text_types = tf.concat(
        [
            tf.constant([8]),
            tf.cast(data['groundtruth_content_type'], tf.int32),
            # TODO(longshangbang): tricks for tfes that have not para labels
            tf.constant(8, shape=(1000,)),
        ],
        0)
    para_types = tf.gather(text_types, para_ids)

    para_ids = para_ids - 1  # revert to id in plp.entities; -1 for no labels
    valid_para = tf.cast(tf.not_equal(para_types, NOT_ANNOTATED_ID), tf.int32)
    para_ids = valid_para * para_ids + (1 - valid_para) * (-1)
    para_ids = tf.concat([tf.constant([-1]), para_ids], 0)  # add bkg

    has_para_ids = tf.cast(tf.reduce_sum(valid_para) > 0, tf.float32)

    if self._max_num_instance >= 0:
      para_ids = utilities.truncate_or_pad(
          para_ids, self._max_num_instance, 0, -1)
    labels['paragraph_labels'] = {
        'paragraph_ids': para_ids,
        'has_para_ids': has_para_ids
    }

  def _define_shapes(self, features: TensorDict, labels: TensorDict):
    """Define the tensor shapes for TPU compiling."""
    if not self._is_shape_defined:
      return
    features['images'] = tf.ensure_shape(
        features['images'], (self._output_dimension, self._output_dimension, 3))
    labels['segmentation_output']['gt_word_score'] = tf.ensure_shape(
        labels['segmentation_output']['gt_word_score'],
        (self._mask_dimension, self._mask_dimension))
    labels['instance_labels']['num_instance'] = tf.ensure_shape(
        labels['instance_labels']['num_instance'], [])
    if self._max_num_instance >= 0:
      labels['instance_labels']['masks_sizes'] = tf.ensure_shape(
          labels['instance_labels']['masks_sizes'], (self._max_num_instance,))
      labels['instance_labels']['masks'] = tf.ensure_shape(
          labels['instance_labels']['masks'],
          (self._mask_dimension, self._mask_dimension))
      labels['instance_labels']['classes'] = tf.ensure_shape(
          labels['instance_labels']['classes'], (self._max_num_instance,))
      labels['instance_labels']['gt_weights'] = tf.ensure_shape(
          labels['instance_labels']['gt_weights'], (self._max_num_instance,))
      labels['paragraph_labels']['paragraph_ids'] = tf.ensure_shape(
          labels['paragraph_labels']['paragraph_ids'],
          (self._max_num_instance,))
      labels['paragraph_labels']['has_para_ids'] = tf.ensure_shape(
          labels['paragraph_labels']['has_para_ids'], [])
