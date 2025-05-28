# Copyright 2025 The TensorFlow Authors. All Rights Reserved.
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

"""Builder class for preparing tf.train.Example in vision tasks."""

# https://www.python.org/dev/peps/pep-0563/#enabling-the-future-behavior-in-python-3-7
from __future__ import annotations

import hashlib
from typing import Optional, Sequence, Union
import numpy as np

from official.core import tf_example_builder
from official.vision.data import image_utils
from official.vision.data import tf_example_feature_key

BytesValueType = Union[bytes, Sequence[bytes], str, Sequence[str]]

_to_array = lambda v: [v] if not isinstance(v, (list, np.ndarray)) else v
_to_bytes = lambda v: v.encode() if isinstance(v, str) else v
_to_bytes_array = lambda v: list(map(_to_bytes, _to_array(v)))


class TfExampleBuilder(tf_example_builder.TfExampleBuilder):
  """Builder class for preparing tf.train.Example in vision task.

  Read API doc at https://www.tensorflow.org/api_docs/python/tf/train/Example.
  """

  def add_image_matrix_feature(
      self,
      image_matrix: np.ndarray,
      image_format: str = 'PNG',
      image_source_id: Optional[bytes] = None,
      feature_prefix: Optional[str] = None,
      label: Optional[Union[int, Sequence[int]]] = None) -> 'TfExampleBuilder':
    """Encodes and adds image features to the example.

    See `tf_example_feature_key.EncodedImageFeatureKey` for list of feature keys
    that will be added to the example.

    Example usages:
      >>> example_builder = TfExampleBuilder()
      * For adding RGB image feature with PNG encoding:
      >>> example_builder.add_image_matrix_feature(image_matrix)
      * For adding RGB image feature with a pre-generated source ID.
      >>> example_builder.add_image_matrix_feature(
              image_matrix, image_source_id=image_source_id)
      * For adding single-channel depth image feature with JPEG encoding:
      >>> example_builder.add_image_matrix_feature(
              image_matrix, image_format=ImageFormat.JPEG,
              feature_prefix='depth')

    Args:
      image_matrix: Numpy image matrix with shape (height, width, channels)
      image_format: Image format string, defaults to 'PNG'.
      image_source_id: Unique string ID to identify the image. Hashed image will
        be used if the field is not provided.
      feature_prefix: Feature prefix for image features.
      label: the label or a list of labels for the image.

    Returns:
      The builder object for subsequent method calls.
    """
    encoded_image = image_utils.encode_image(image_matrix, image_format)
    height, width, num_channels = image_matrix.shape

    return self.add_encoded_image_feature(encoded_image, image_format, height,
                                          width, num_channels, image_source_id,
                                          feature_prefix, label)

  def add_encoded_image_feature(
      self,
      encoded_image: bytes,
      image_format: Optional[str] = None,
      height: Optional[int] = None,
      width: Optional[int] = None,
      num_channels: Optional[int] = None,
      image_source_id: Optional[bytes] = None,
      feature_prefix: Optional[str] = None,
      label: Optional[Union[int, Sequence[int]]] = None) -> 'TfExampleBuilder':
    """Adds encoded image features to the example.

    See `tf_example_feature_key.EncodedImageFeatureKey` for list of feature keys
    that will be added to the example.

    Image format, height, width, and channels are inferred from the encoded
    image bytes if any of them is not provided. Hashed image will be used if
    pre-generated source ID is not provided.

    Example usages:
      >>> example_builder = TfExampleBuilder()
      * For adding RGB image feature:
      >>> example_builder.add_encoded_image_feature(image_bytes)
      * For adding RGB image feature with pre-generated source ID:
      >>> example_builder.add_encoded_image_feature(
              image_bytes, image_source_id=image_source_id)
      * For adding single-channel depth image feature:
      >>> example_builder.add_encoded_image_feature(
              image_bytes, feature_prefix='depth')

    Args:
      encoded_image: Encoded image string.
      image_format: Image format string.
      height: Number of rows.
      width: Number of columns.
      num_channels: Number of channels.
      image_source_id: Unique string ID to identify the image.
      feature_prefix: Feature prefix for image features.
      label: the label or a list of labels for the image.

    Returns:
      The builder object for subsequent method calls.
    """
    if image_format == 'RAW':
      if not (height and width and num_channels):
        raise ValueError('For raw image feature, height, width and '
                         'num_channels fields are required.')
    if not all((height, width, num_channels, image_format)):
      (height, width, num_channels, image_format) = (
          image_utils.decode_image_metadata(encoded_image))
    else:
      image_format = image_utils.validate_image_format(image_format)

    feature_key = tf_example_feature_key.EncodedImageFeatureKey(feature_prefix)

    # If source ID is not provided, we use hashed encoded image as the source
    # ID. Note that we only keep 24 bits to be consistent with the Model Garden
    # requirement, which will transform the source ID into float32.
    if not image_source_id:
      hashed_image = int(hashlib.blake2s(encoded_image).hexdigest(), 16)
      image_source_id = _to_bytes(str(hashed_image % ((1 << 24) + 1)))

    if label is not None:
      self.add_ints_feature(feature_key.label, label)

    return (
        self.add_bytes_feature(feature_key.encoded, encoded_image)
        .add_bytes_feature(feature_key.format, image_format)
        .add_ints_feature(feature_key.height, [height])
        .add_ints_feature(feature_key.width, [width])
        .add_ints_feature(feature_key.num_channels, num_channels)
        .add_bytes_feature(feature_key.source_id, image_source_id))

  def add_boxes_feature(
      self,
      xmins: Sequence[float],
      xmaxs: Sequence[float],
      ymins: Sequence[float],
      ymaxs: Sequence[float],
      labels: Sequence[int],
      confidences: Optional[Sequence[float]] = None,
      normalized: bool = True,
      feature_prefix: Optional[str] = None) -> 'TfExampleBuilder':
    """Adds box and label features to the example.

    Four features will be generated for xmin, ymin, xmax, and ymax. One feature
    will be generated for label. Different feature keys will be used for
    normalized boxes and pixel-value boxes, depending on the value of
    `normalized`.

    Example usages:
      >>> example_builder = TfExampleBuilder()
      >>> example_builder.add_boxes_feature(xmins, xmaxs, ymins, ymaxs, labels)

    Args:
      xmins: A list of minimum X coordinates.
      xmaxs: A list of maximum X coordinates.
      ymins: A list of minimum Y coordinates.
      ymaxs: A list of maximum Y coordinates.
      labels: The labels of added boxes.
      confidences: The confidences of added boxes.
      normalized: Indicate if the coordinates of boxes are normalized.
      feature_prefix: Feature prefix for added box features.

    Returns:
      The builder object for subsequent method calls.
    """
    if normalized:
      feature_key = tf_example_feature_key.BoxFeatureKey(feature_prefix)
    else:
      feature_key = tf_example_feature_key.BoxPixelFeatureKey(feature_prefix)

    self.add_floats_feature(feature_key.xmin, xmins)
    self.add_floats_feature(feature_key.xmax, xmaxs)
    self.add_floats_feature(feature_key.ymin, ymins)
    self.add_floats_feature(feature_key.ymax, ymaxs)
    self.add_ints_feature(feature_key.label, labels)
    if confidences is not None:
      self.add_floats_feature(feature_key.confidence, confidences)
    return self

  def _compute_mask_areas(
      self, instance_mask_matrices: np.ndarray) -> Sequence[float]:
    return np.sum(
        instance_mask_matrices, axis=(1, 2, 3),
        dtype=float).flatten().tolist()

  def add_instance_mask_matrices_feature(
      self,
      instance_mask_matrices: np.ndarray,
      feature_prefix: Optional[str] = None) -> 'TfExampleBuilder':
    """Encodes and adds instance mask features to the example.

    See `tf_example_feature_key.EncodedInstanceMaskFeatureKey` for list of
    feature keys that will be added to the example. Please note that all masks
    will be encoded as PNG images.

    Example usages:
      >>> example_builder = TfExampleBuilder()
      >>> example_builder.add_instance_mask_matrices_feature(
              instance_mask_matrices)

    TODO(b/223653024): Provide a way to generate visualization mask from
    feature mask.

    Args:
      instance_mask_matrices: Numpy instance mask matrices with shape
        (num_instance, height, width, 1) or (num_instance, height, width).
      feature_prefix: Feature prefix for instance mask features.

    Returns:
      The builder object for subsequent method calls.
    """
    if len(instance_mask_matrices.shape) == 3:
      instance_mask_matrices = instance_mask_matrices[..., np.newaxis]

    mask_areas = self._compute_mask_areas(instance_mask_matrices)
    encoded_instance_masks = list(
        map(lambda x: image_utils.encode_image(x, 'PNG'),
            instance_mask_matrices))

    return self.add_encoded_instance_masks_feature(encoded_instance_masks,
                                                   mask_areas, feature_prefix)

  def add_encoded_instance_masks_feature(
      self,
      encoded_instance_masks: Sequence[bytes],
      mask_areas: Optional[Sequence[float]] = None,
      feature_prefix: Optional[str] = None) -> 'TfExampleBuilder':
    """Adds encoded instance mask features to the example.

    See `tf_example_feature_key.EncodedInstanceMaskFeatureKey` for list of
    feature keys that will be added to the example.

    Image area is inferred from the encoded instance mask bytes if not provided.

    Example usages:
      >>> example_builder = TfExampleBuilder()
      >>> example_builder.add_encoded_instance_masks_feature(
              instance_mask_bytes)

    TODO(b/223653024): Provide a way to generate visualization mask from
    feature mask.

    Args:
      encoded_instance_masks: A list of encoded instance mask string. Note that
        the encoding is not changed in this function and it always assumes the
        image is in "PNG" format.
      mask_areas: Areas for each instance masks.
      feature_prefix: Feature prefix for instance mask features.

    Returns:
      The builder object for subsequent method calls.
    """
    encoded_instance_masks = _to_bytes_array(encoded_instance_masks)

    if mask_areas is None:
      instance_mask_matrices = np.array(
          list(map(image_utils.decode_image, encoded_instance_masks)))
      mask_areas = self._compute_mask_areas(instance_mask_matrices)

    feature_key = tf_example_feature_key.EncodedInstanceMaskFeatureKey(
        feature_prefix)
    return (
        self.add_bytes_feature(feature_key.mask, encoded_instance_masks)
        .add_floats_feature(feature_key.area, mask_areas))

  def add_semantic_mask_matrix_feature(
      self,
      mask_matrix: np.ndarray,
      mask_format: str = 'PNG',
      visualization_mask_matrix: Optional[np.ndarray] = None,
      visualization_mask_format: str = 'PNG',
      feature_prefix: Optional[str] = None) -> 'TfExampleBuilder':
    """Encodes and adds semantic mask features to the example.

    See `tf_example_feature_key.EncodedSemanticMaskFeatureKey` for list of
    feature keys that will be added to the example.

    Example usages:
      >>> example_builder = TfExampleBuilder()
      * For adding semantic mask feature:
      >>> example_builder.add_semantic_mask_matrix_feature(
              semantic_mask_matrix)
      * For adding semantic mask feature and visualization mask feature:
      >>> example_builder.add_semantic_mask_matrix_feature(
              semantic_mask_matrix,
              visualization_mask_matrix=visualization_mask_matrix)
      * For adding predicted semantic mask feature with visualization mask:
      >>> example_builder.add_encoded_semantic_mask_feature(
              predicted_mask_matrix,
              visualization_mask_matrix=predicted_visualization_mask_matrix,
              feature_prefix='predicted')

    TODO(b/223653024): Provide a way to generate visualization mask from
    feature mask.

    Args:
      mask_matrix: Numpy semantic mask matrix with shape (height, width, 1) or
        (height, width).
      mask_format: Mask format string, defaults to 'PNG'.
      visualization_mask_matrix: Numpy visualization mask matrix for semantic
        mask with shape (height, width, 3).
      visualization_mask_format: Visualization mask format string, defaults to
        'PNG'.
      feature_prefix: Feature prefix for semantic mask features.

    Returns:
      The builder object for subsequent method calls.
    """
    if len(mask_matrix.shape) == 2:
      mask_matrix = mask_matrix[..., np.newaxis]
    encoded_mask = image_utils.encode_image(mask_matrix, mask_format)

    encoded_visualization_mask = None
    if visualization_mask_matrix is not None:
      encoded_visualization_mask = image_utils.encode_image(
          visualization_mask_matrix, visualization_mask_format)

    return self.add_encoded_semantic_mask_feature(encoded_mask, mask_format,
                                                  encoded_visualization_mask,
                                                  visualization_mask_format,
                                                  feature_prefix)

  def add_encoded_semantic_mask_feature(
      self, encoded_mask: bytes,
      mask_format: str = 'PNG',
      encoded_visualization_mask: Optional[bytes] = None,
      visualization_mask_format: str = 'PNG',
      feature_prefix: Optional[str] = None) -> 'TfExampleBuilder':
    """Adds encoded semantic mask features to the example.

    See `tf_example_feature_key.EncodedSemanticMaskFeatureKey` for list of
    feature keys that will be added to the example.

    Example usages:
      >>> example_builder = TfExampleBuilder()
      * For adding semantic mask feature:
      >>> example_builder.add_encoded_semantic_mask_feature(semantic_mask_bytes)
      * For adding semantic mask feature and visualization mask feature:
      >>> example_builder.add_encoded_semantic_mask_feature(
              semantic_mask_bytes,
              encoded_visualization_mask=visualization_mask_bytes)
      * For adding predicted semantic mask feature with visualization mask:
      >>> example_builder.add_encoded_semantic_mask_feature(
              predicted_mask_bytes,
              encoded_visualization_mask=predicted_visualization_mask_bytes,
              feature_prefix='predicted')

    TODO(b/223653024): Provide a way to generate visualization mask from
    feature mask.

    Args:
      encoded_mask: Encoded semantic mask string.
      mask_format: Semantic mask format string, defaults to 'PNG'.
      encoded_visualization_mask: Encoded visualization mask string.
      visualization_mask_format: Visualization mask format string, defaults to
        'PNG'.
      feature_prefix: Feature prefix for semantic mask features.

    Returns:
      The builder object for subsequent method calls.
    """
    feature_key = tf_example_feature_key.EncodedSemanticMaskFeatureKey(
        feature_prefix)
    example_builder = (
        self.add_bytes_feature(feature_key.mask, encoded_mask)
        .add_bytes_feature(feature_key.mask_format, mask_format))
    if encoded_visualization_mask is not None:
      example_builder = (
          example_builder.add_bytes_feature(
              feature_key.visualization_mask, encoded_visualization_mask)
          .add_bytes_feature(
              feature_key.visualization_mask_format, visualization_mask_format))
    return example_builder

  def add_panoptic_mask_matrix_feature(
      self,
      panoptic_category_mask_matrix: np.ndarray,
      panoptic_instance_mask_matrix: np.ndarray,
      panoptic_category_mask_format: str = 'PNG',
      panoptic_instance_mask_format: str = 'PNG',
      feature_prefix: Optional[str] = None) -> 'TfExampleBuilder':
    """Encodes and adds panoptic mask features to the example.

    See `tf_example_feature_key.EncodedPanopticMaskFeatureKey` for list of
    feature keys that will be added to the example.

    Example usages:
      >>> example_builder = TfExampleBuilder()
      >>> example_builder.add_panoptic_mask_matrix_feature(
              panoptic_category_mask_matrix, panoptic_instance_mask_matrix)

    TODO(b/223653024): Provide a way to generate visualization mask from
    feature mask.

    Args:
      panoptic_category_mask_matrix: Numpy panoptic category mask matrix with
        shape (height, width, 1) or (height, width).
      panoptic_instance_mask_matrix: Numpy panoptic instance mask matrix with
        shape (height, width, 1) or (height, width).
      panoptic_category_mask_format: Panoptic category mask format string,
        defaults to 'PNG'.
      panoptic_instance_mask_format: Panoptic instance mask format string,
        defaults to 'PNG'.
      feature_prefix: Feature prefix for panoptic mask features.

    Returns:
      The builder object for subsequent method calls.
    """
    if len(panoptic_category_mask_matrix.shape) == 2:
      panoptic_category_mask_matrix = (
          panoptic_category_mask_matrix[..., np.newaxis])
    if len(panoptic_instance_mask_matrix.shape) == 2:
      panoptic_instance_mask_matrix = (
          panoptic_instance_mask_matrix[..., np.newaxis])
    encoded_panoptic_category_mask = image_utils.encode_image(
        panoptic_category_mask_matrix, panoptic_category_mask_format)
    encoded_panoptic_instance_mask = image_utils.encode_image(
        panoptic_instance_mask_matrix, panoptic_instance_mask_format)

    return self.add_encoded_panoptic_mask_feature(
        encoded_panoptic_category_mask, encoded_panoptic_instance_mask,
        panoptic_category_mask_format, panoptic_instance_mask_format,
        feature_prefix)

  def add_encoded_panoptic_mask_feature(
      self,
      encoded_panoptic_category_mask: bytes,
      encoded_panoptic_instance_mask: bytes,
      panoptic_category_mask_format: str = 'PNG',
      panoptic_instance_mask_format: str = 'PNG',
      feature_prefix: Optional[str] = None) -> 'TfExampleBuilder':
    """Adds encoded panoptic mask features to the example.

    See `tf_example_feature_key.EncodedPanopticMaskFeatureKey` for list of
    feature keys that will be added to the example.

    Example usages:
      >>> example_builder = TfExampleBuilder()
      >>> example_builder.add_encoded_panoptic_mask_feature(
              encoded_panoptic_category_mask, encoded_panoptic_instance_mask)

    TODO(b/223653024): Provide a way to generate visualization mask from
    feature mask.

    Args:
      encoded_panoptic_category_mask: Encoded panoptic category mask string.
      encoded_panoptic_instance_mask: Encoded panoptic instance mask string.
      panoptic_category_mask_format: Panoptic category mask format string,
        defaults to 'PNG'.
      panoptic_instance_mask_format: Panoptic instance mask format string,
        defaults to 'PNG'.
      feature_prefix: Feature prefix for panoptic mask features.

    Returns:
      The builder object for subsequent method calls.
    """
    feature_key = tf_example_feature_key.EncodedPanopticMaskFeatureKey(
        feature_prefix)
    return (
        self.add_bytes_feature(
            feature_key.category_mask, encoded_panoptic_category_mask)
        .add_bytes_feature(
            feature_key.category_mask_format, panoptic_category_mask_format)
        .add_bytes_feature(
            feature_key.instance_mask, encoded_panoptic_instance_mask)
        .add_bytes_feature(
            feature_key.instance_mask_format, panoptic_instance_mask_format))

