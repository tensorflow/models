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
"""Tensorflow Example proto decoder for object detection.

A decoder to decode string tensors containing serialized tensorflow.Example
protos for object detection.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import enum
import numpy as np
from six.moves import zip
import tensorflow.compat.v1 as tf
from tf_slim import tfexample_decoder as slim_example_decoder
from object_detection.core import data_decoder
from object_detection.core import standard_fields as fields
from object_detection.protos import input_reader_pb2
from object_detection.utils import label_map_util
from object_detection.utils import shape_utils

# pylint: disable=g-import-not-at-top
try:
  from tensorflow.contrib import lookup as contrib_lookup

except ImportError:
  # TF 2.0 doesn't ship with contrib.
  pass
# pylint: enable=g-import-not-at-top

_LABEL_OFFSET = 1


class Visibility(enum.Enum):
  """Visibility definitions.

  This follows the MS Coco convention (http://cocodataset.org/#format-data).
  """
  # Keypoint is not labeled.
  UNLABELED = 0
  # Keypoint is labeled but falls outside the object segment (e.g. occluded).
  NOT_VISIBLE = 1
  # Keypoint is labeled and visible.
  VISIBLE = 2


class _ClassTensorHandler(slim_example_decoder.Tensor):
  """An ItemHandler to fetch class ids from class text."""

  def __init__(self,
               tensor_key,
               label_map_proto_file,
               shape_keys=None,
               shape=None,
               default_value=''):
    """Initializes the LookupTensor handler.

    Simply calls a vocabulary (most often, a label mapping) lookup.

    Args:
      tensor_key: the name of the `TFExample` feature to read the tensor from.
      label_map_proto_file: File path to a text format LabelMapProto message
        mapping class text to id.
      shape_keys: Optional name or list of names of the TF-Example feature in
        which the tensor shape is stored. If a list, then each corresponds to
        one dimension of the shape.
      shape: Optional output shape of the `Tensor`. If provided, the `Tensor` is
        reshaped accordingly.
      default_value: The value used when the `tensor_key` is not found in a
        particular `TFExample`.

    Raises:
      ValueError: if both `shape_keys` and `shape` are specified.
    """
    name_to_id = label_map_util.get_label_map_dict(
        label_map_proto_file, use_display_name=False)
    # We use a default_value of -1, but we expect all labels to be contained
    # in the label map.
    try:
      # Dynamically try to load the tf v2 lookup, falling back to contrib
      lookup = tf.compat.v2.lookup
      hash_table_class = tf.compat.v2.lookup.StaticHashTable
    except AttributeError:
      lookup = contrib_lookup
      hash_table_class = contrib_lookup.HashTable
    name_to_id_table = hash_table_class(
        initializer=lookup.KeyValueTensorInitializer(
            keys=tf.constant(list(name_to_id.keys())),
            values=tf.constant(list(name_to_id.values()), dtype=tf.int64)),
        default_value=-1)
    display_name_to_id = label_map_util.get_label_map_dict(
        label_map_proto_file, use_display_name=True)
    # We use a default_value of -1, but we expect all labels to be contained
    # in the label map.
    display_name_to_id_table = hash_table_class(
        initializer=lookup.KeyValueTensorInitializer(
            keys=tf.constant(list(display_name_to_id.keys())),
            values=tf.constant(
                list(display_name_to_id.values()), dtype=tf.int64)),
        default_value=-1)

    self._name_to_id_table = name_to_id_table
    self._display_name_to_id_table = display_name_to_id_table
    super(_ClassTensorHandler, self).__init__(tensor_key, shape_keys, shape,
                                              default_value)

  def tensors_to_item(self, keys_to_tensors):
    unmapped_tensor = super(_ClassTensorHandler,
                            self).tensors_to_item(keys_to_tensors)
    return tf.maximum(self._name_to_id_table.lookup(unmapped_tensor),
                      self._display_name_to_id_table.lookup(unmapped_tensor))


class TfExampleDecoder(data_decoder.DataDecoder):
  """Tensorflow Example proto decoder."""

  def __init__(self,
               load_instance_masks=False,
               instance_mask_type=input_reader_pb2.NUMERICAL_MASKS,
               label_map_proto_file=None,
               use_display_name=False,
               dct_method='',
               num_keypoints=0,
               num_additional_channels=0,
               load_multiclass_scores=False,
               load_context_features=False,
               expand_hierarchy_labels=False,
               load_dense_pose=False,
               load_track_id=False):
    """Constructor sets keys_to_features and items_to_handlers.

    Args:
      load_instance_masks: whether or not to load and handle instance masks.
      instance_mask_type: type of instance masks. Options are provided in
        input_reader.proto. This is only used if `load_instance_masks` is True.
      label_map_proto_file: a file path to a
        object_detection.protos.StringIntLabelMap proto. If provided, then the
        mapped IDs of 'image/object/class/text' will take precedence over the
        existing 'image/object/class/label' ID.  Also, if provided, it is
        assumed that 'image/object/class/text' will be in the data.
      use_display_name: whether or not to use the `display_name` for label
        mapping (instead of `name`).  Only used if label_map_proto_file is
        provided.
      dct_method: An optional string. Defaults to None. It only takes
        effect when image format is jpeg, used to specify a hint about the
        algorithm used for jpeg decompression. Currently valid values
        are ['INTEGER_FAST', 'INTEGER_ACCURATE']. The hint may be ignored, for
        example, the jpeg library does not have that specific option.
      num_keypoints: the number of keypoints per object.
      num_additional_channels: how many additional channels to use.
      load_multiclass_scores: Whether to load multiclass scores associated with
        boxes.
      load_context_features: Whether to load information from context_features,
        to provide additional context to a detection model for training and/or
        inference.
      expand_hierarchy_labels: Expands the object and image labels taking into
        account the provided hierarchy in the label_map_proto_file. For positive
        classes, the labels are extended to ancestor. For negative classes,
        the labels are expanded to descendants.
      load_dense_pose: Whether to load DensePose annotations.
      load_track_id: Whether to load tracking annotations.

    Raises:
      ValueError: If `instance_mask_type` option is not one of
        input_reader_pb2.DEFAULT, input_reader_pb2.NUMERICAL, or
        input_reader_pb2.PNG_MASKS.
      ValueError: If `expand_labels_hierarchy` is True, but the
        `label_map_proto_file` is not provided.
    """
    # TODO(rathodv): delete unused `use_display_name` argument once we change
    # other decoders to handle label maps similarly.
    del use_display_name
    self.keys_to_features = {
        'image/encoded':
            tf.FixedLenFeature((), tf.string, default_value=''),
        'image/format':
            tf.FixedLenFeature((), tf.string, default_value='jpeg'),
        'image/filename':
            tf.FixedLenFeature((), tf.string, default_value=''),
        'image/key/sha256':
            tf.FixedLenFeature((), tf.string, default_value=''),
        'image/source_id':
            tf.FixedLenFeature((), tf.string, default_value=''),
        'image/height':
            tf.FixedLenFeature((), tf.int64, default_value=1),
        'image/width':
            tf.FixedLenFeature((), tf.int64, default_value=1),
        # Image-level labels.
        'image/class/text':
            tf.VarLenFeature(tf.string),
        'image/class/label':
            tf.VarLenFeature(tf.int64),
        'image/neg_category_ids':
            tf.VarLenFeature(tf.int64),
        'image/not_exhaustive_category_ids':
            tf.VarLenFeature(tf.int64),
        'image/class/confidence':
            tf.VarLenFeature(tf.float32),
        # Object boxes and classes.
        'image/object/bbox/xmin':
            tf.VarLenFeature(tf.float32),
        'image/object/bbox/xmax':
            tf.VarLenFeature(tf.float32),
        'image/object/bbox/ymin':
            tf.VarLenFeature(tf.float32),
        'image/object/bbox/ymax':
            tf.VarLenFeature(tf.float32),
        'image/object/class/label':
            tf.VarLenFeature(tf.int64),
        'image/object/class/text':
            tf.VarLenFeature(tf.string),
        'image/object/area':
            tf.VarLenFeature(tf.float32),
        'image/object/is_crowd':
            tf.VarLenFeature(tf.int64),
        'image/object/difficult':
            tf.VarLenFeature(tf.int64),
        'image/object/group_of':
            tf.VarLenFeature(tf.int64),
        'image/object/weight':
            tf.VarLenFeature(tf.float32),

    }
    # We are checking `dct_method` instead of passing it directly in order to
    # ensure TF version 1.6 compatibility.
    if dct_method:
      image = slim_example_decoder.Image(
          image_key='image/encoded',
          format_key='image/format',
          channels=3,
          dct_method=dct_method)
      additional_channel_image = slim_example_decoder.Image(
          image_key='image/additional_channels/encoded',
          format_key='image/format',
          channels=1,
          repeated=True,
          dct_method=dct_method)
    else:
      image = slim_example_decoder.Image(
          image_key='image/encoded', format_key='image/format', channels=3)
      additional_channel_image = slim_example_decoder.Image(
          image_key='image/additional_channels/encoded',
          format_key='image/format',
          channels=1,
          repeated=True)
    self.items_to_handlers = {
        fields.InputDataFields.image:
            image,
        fields.InputDataFields.source_id: (
            slim_example_decoder.Tensor('image/source_id')),
        fields.InputDataFields.key: (
            slim_example_decoder.Tensor('image/key/sha256')),
        fields.InputDataFields.filename: (
            slim_example_decoder.Tensor('image/filename')),
        # Image-level labels.
        fields.InputDataFields.groundtruth_image_confidences: (
            slim_example_decoder.Tensor('image/class/confidence')),
        fields.InputDataFields.groundtruth_verified_neg_classes: (
            slim_example_decoder.Tensor('image/neg_category_ids')),
        fields.InputDataFields.groundtruth_not_exhaustive_classes: (
            slim_example_decoder.Tensor('image/not_exhaustive_category_ids')),
        # Object boxes and classes.
        fields.InputDataFields.groundtruth_boxes: (
            slim_example_decoder.BoundingBox(['ymin', 'xmin', 'ymax', 'xmax'],
                                             'image/object/bbox/')),
        fields.InputDataFields.groundtruth_area:
            slim_example_decoder.Tensor('image/object/area'),
        fields.InputDataFields.groundtruth_is_crowd: (
            slim_example_decoder.Tensor('image/object/is_crowd')),
        fields.InputDataFields.groundtruth_difficult: (
            slim_example_decoder.Tensor('image/object/difficult')),
        fields.InputDataFields.groundtruth_group_of: (
            slim_example_decoder.Tensor('image/object/group_of')),
        fields.InputDataFields.groundtruth_weights: (
            slim_example_decoder.Tensor('image/object/weight')),

    }
    if load_multiclass_scores:
      self.keys_to_features[
          'image/object/class/multiclass_scores'] = tf.VarLenFeature(tf.float32)
      self.items_to_handlers[fields.InputDataFields.multiclass_scores] = (
          slim_example_decoder.Tensor('image/object/class/multiclass_scores'))

    if load_context_features:
      self.keys_to_features[
          'image/context_features'] = tf.VarLenFeature(tf.float32)
      self.items_to_handlers[fields.InputDataFields.context_features] = (
          slim_example_decoder.ItemHandlerCallback(
              ['image/context_features', 'image/context_feature_length'],
              self._reshape_context_features))

      self.keys_to_features[
          'image/context_feature_length'] = tf.FixedLenFeature((), tf.int64)
      self.items_to_handlers[fields.InputDataFields.context_feature_length] = (
          slim_example_decoder.Tensor('image/context_feature_length'))

    if num_additional_channels > 0:
      self.keys_to_features[
          'image/additional_channels/encoded'] = tf.FixedLenFeature(
              (num_additional_channels,), tf.string)
      self.items_to_handlers[
          fields.InputDataFields.
          image_additional_channels] = additional_channel_image
    self._num_keypoints = num_keypoints
    if num_keypoints > 0:
      self.keys_to_features['image/object/keypoint/x'] = (
          tf.VarLenFeature(tf.float32))
      self.keys_to_features['image/object/keypoint/y'] = (
          tf.VarLenFeature(tf.float32))
      self.keys_to_features['image/object/keypoint/visibility'] = (
          tf.VarLenFeature(tf.int64))
      self.items_to_handlers[fields.InputDataFields.groundtruth_keypoints] = (
          slim_example_decoder.ItemHandlerCallback(
              ['image/object/keypoint/y', 'image/object/keypoint/x'],
              self._reshape_keypoints))
      kpt_vis_field = fields.InputDataFields.groundtruth_keypoint_visibilities
      self.items_to_handlers[kpt_vis_field] = (
          slim_example_decoder.ItemHandlerCallback(
              ['image/object/keypoint/x', 'image/object/keypoint/visibility'],
              self._reshape_keypoint_visibilities))
    if load_instance_masks:
      if instance_mask_type in (input_reader_pb2.DEFAULT,
                                input_reader_pb2.NUMERICAL_MASKS):
        self.keys_to_features['image/object/mask'] = (
            tf.VarLenFeature(tf.float32))
        self.items_to_handlers[
            fields.InputDataFields.groundtruth_instance_masks] = (
                slim_example_decoder.ItemHandlerCallback(
                    ['image/object/mask', 'image/height', 'image/width'],
                    self._reshape_instance_masks))
      elif instance_mask_type == input_reader_pb2.PNG_MASKS:
        self.keys_to_features['image/object/mask'] = tf.VarLenFeature(tf.string)
        self.items_to_handlers[
            fields.InputDataFields.groundtruth_instance_masks] = (
                slim_example_decoder.ItemHandlerCallback(
                    ['image/object/mask', 'image/height', 'image/width'],
                    self._decode_png_instance_masks))
      else:
        raise ValueError('Did not recognize the `instance_mask_type` option.')
    if load_dense_pose:
      self.keys_to_features['image/object/densepose/num'] = (
          tf.VarLenFeature(tf.int64))
      self.keys_to_features['image/object/densepose/part_index'] = (
          tf.VarLenFeature(tf.int64))
      self.keys_to_features['image/object/densepose/x'] = (
          tf.VarLenFeature(tf.float32))
      self.keys_to_features['image/object/densepose/y'] = (
          tf.VarLenFeature(tf.float32))
      self.keys_to_features['image/object/densepose/u'] = (
          tf.VarLenFeature(tf.float32))
      self.keys_to_features['image/object/densepose/v'] = (
          tf.VarLenFeature(tf.float32))
      self.items_to_handlers[
          fields.InputDataFields.groundtruth_dp_num_points] = (
              slim_example_decoder.Tensor('image/object/densepose/num'))
      self.items_to_handlers[fields.InputDataFields.groundtruth_dp_part_ids] = (
          slim_example_decoder.ItemHandlerCallback(
              ['image/object/densepose/part_index',
               'image/object/densepose/num'], self._dense_pose_part_indices))
      self.items_to_handlers[
          fields.InputDataFields.groundtruth_dp_surface_coords] = (
              slim_example_decoder.ItemHandlerCallback(
                  ['image/object/densepose/x', 'image/object/densepose/y',
                   'image/object/densepose/u', 'image/object/densepose/v',
                   'image/object/densepose/num'],
                  self._dense_pose_surface_coordinates))
    if load_track_id:
      self.keys_to_features['image/object/track/label'] = (
          tf.VarLenFeature(tf.int64))
      self.items_to_handlers[
          fields.InputDataFields.groundtruth_track_ids] = (
              slim_example_decoder.Tensor('image/object/track/label'))

    if label_map_proto_file:
      # If the label_map_proto is provided, try to use it in conjunction with
      # the class text, and fall back to a materialized ID.
      label_handler = slim_example_decoder.BackupHandler(
          _ClassTensorHandler(
              'image/object/class/text', label_map_proto_file,
              default_value=''),
          slim_example_decoder.Tensor('image/object/class/label'))
      image_label_handler = slim_example_decoder.BackupHandler(
          _ClassTensorHandler(
              fields.TfExampleFields.image_class_text,
              label_map_proto_file,
              default_value=''),
          slim_example_decoder.Tensor(fields.TfExampleFields.image_class_label))
    else:
      label_handler = slim_example_decoder.Tensor('image/object/class/label')
      image_label_handler = slim_example_decoder.Tensor(
          fields.TfExampleFields.image_class_label)
    self.items_to_handlers[
        fields.InputDataFields.groundtruth_classes] = label_handler
    self.items_to_handlers[
        fields.InputDataFields.groundtruth_image_classes] = image_label_handler

    self._expand_hierarchy_labels = expand_hierarchy_labels
    self._ancestors_lut = None
    self._descendants_lut = None
    if expand_hierarchy_labels:
      if label_map_proto_file:
        ancestors_lut, descendants_lut = (
            label_map_util.get_label_map_hierarchy_lut(label_map_proto_file,
                                                       True))
        self._ancestors_lut = tf.constant(ancestors_lut, dtype=tf.int64)
        self._descendants_lut = tf.constant(descendants_lut, dtype=tf.int64)
      else:
        raise ValueError('In order to expand labels, the label_map_proto_file '
                         'has to be provided.')

  def decode(self, tf_example_string_tensor):
    """Decodes serialized tensorflow example and returns a tensor dictionary.

    Args:
      tf_example_string_tensor: a string tensor holding a serialized tensorflow
        example proto.

    Returns:
      A dictionary of the following tensors.
      fields.InputDataFields.image - 3D uint8 tensor of shape [None, None, 3]
        containing image.
      fields.InputDataFields.original_image_spatial_shape - 1D int32 tensor of
        shape [2] containing shape of the image.
      fields.InputDataFields.source_id - string tensor containing original
        image id.
      fields.InputDataFields.key - string tensor with unique sha256 hash key.
      fields.InputDataFields.filename - string tensor with original dataset
        filename.
      fields.InputDataFields.groundtruth_boxes - 2D float32 tensor of shape
        [None, 4] containing box corners.
      fields.InputDataFields.groundtruth_classes - 1D int64 tensor of shape
        [None] containing classes for the boxes.
      fields.InputDataFields.groundtruth_weights - 1D float32 tensor of
        shape [None] indicating the weights of groundtruth boxes.
      fields.InputDataFields.groundtruth_area - 1D float32 tensor of shape
        [None] containing containing object mask area in pixel squared.
      fields.InputDataFields.groundtruth_is_crowd - 1D bool tensor of shape
        [None] indicating if the boxes enclose a crowd.

    Optional:
      fields.InputDataFields.groundtruth_image_confidences - 1D float tensor of
        shape [None] indicating if a class is present in the image (1.0) or
        a class is not present in the image (0.0).
      fields.InputDataFields.image_additional_channels - 3D uint8 tensor of
        shape [None, None, num_additional_channels]. 1st dim is height; 2nd dim
        is width; 3rd dim is the number of additional channels.
      fields.InputDataFields.groundtruth_difficult - 1D bool tensor of shape
        [None] indicating if the boxes represent `difficult` instances.
      fields.InputDataFields.groundtruth_group_of - 1D bool tensor of shape
        [None] indicating if the boxes represent `group_of` instances.
      fields.InputDataFields.groundtruth_keypoints - 3D float32 tensor of
        shape [None, num_keypoints, 2] containing keypoints, where the
        coordinates of the keypoints are ordered (y, x).
      fields.InputDataFields.groundtruth_keypoint_visibilities - 2D bool
        tensor of shape [None, num_keypoints] containing keypoint visibilites.
      fields.InputDataFields.groundtruth_instance_masks - 3D float32 tensor of
        shape [None, None, None] containing instance masks.
      fields.InputDataFields.groundtruth_image_classes - 1D int64 of shape
        [None] containing classes for the boxes.
      fields.InputDataFields.multiclass_scores - 1D float32 tensor of shape
        [None * num_classes] containing flattened multiclass scores for
        groundtruth boxes.
      fields.InputDataFields.context_features - 1D float32 tensor of shape
        [context_feature_length * num_context_features]
      fields.InputDataFields.context_feature_length - int32 tensor specifying
        the length of each feature in context_features
    """
    serialized_example = tf.reshape(tf_example_string_tensor, shape=[])
    decoder = slim_example_decoder.TFExampleDecoder(self.keys_to_features,
                                                    self.items_to_handlers)
    keys = decoder.list_items()
    tensors = decoder.decode(serialized_example, items=keys)
    tensor_dict = dict(zip(keys, tensors))
    is_crowd = fields.InputDataFields.groundtruth_is_crowd
    tensor_dict[is_crowd] = tf.cast(tensor_dict[is_crowd], dtype=tf.bool)
    tensor_dict[fields.InputDataFields.image].set_shape([None, None, 3])
    tensor_dict[fields.InputDataFields.original_image_spatial_shape] = tf.shape(
        tensor_dict[fields.InputDataFields.image])[:2]

    if fields.InputDataFields.image_additional_channels in tensor_dict:
      channels = tensor_dict[fields.InputDataFields.image_additional_channels]
      channels = tf.squeeze(channels, axis=3)
      channels = tf.transpose(channels, perm=[1, 2, 0])
      tensor_dict[fields.InputDataFields.image_additional_channels] = channels

    def default_groundtruth_weights():
      return tf.ones(
          [tf.shape(tensor_dict[fields.InputDataFields.groundtruth_boxes])[0]],
          dtype=tf.float32)

    tensor_dict[fields.InputDataFields.groundtruth_weights] = tf.cond(
        tf.greater(
            tf.shape(
                tensor_dict[fields.InputDataFields.groundtruth_weights])[0],
            0), lambda: tensor_dict[fields.InputDataFields.groundtruth_weights],
        default_groundtruth_weights)

    if fields.InputDataFields.groundtruth_keypoints in tensor_dict:
      # Set all keypoints that are not labeled to NaN.
      gt_kpt_fld = fields.InputDataFields.groundtruth_keypoints
      gt_kpt_vis_fld = fields.InputDataFields.groundtruth_keypoint_visibilities
      visibilities_tiled = tf.tile(
          tf.expand_dims(tensor_dict[gt_kpt_vis_fld], -1),
          [1, 1, 2])
      tensor_dict[gt_kpt_fld] = tf.where(
          visibilities_tiled,
          tensor_dict[gt_kpt_fld],
          np.nan * tf.ones_like(tensor_dict[gt_kpt_fld]))

    if self._expand_hierarchy_labels:
      input_fields = fields.InputDataFields
      image_classes, image_confidences = self._expand_image_label_hierarchy(
          tensor_dict[input_fields.groundtruth_image_classes],
          tensor_dict[input_fields.groundtruth_image_confidences])
      tensor_dict[input_fields.groundtruth_image_classes] = image_classes
      tensor_dict[input_fields.groundtruth_image_confidences] = (
          image_confidences)

      box_fields = [
          fields.InputDataFields.groundtruth_group_of,
          fields.InputDataFields.groundtruth_is_crowd,
          fields.InputDataFields.groundtruth_difficult,
          fields.InputDataFields.groundtruth_area,
          fields.InputDataFields.groundtruth_boxes,
          fields.InputDataFields.groundtruth_weights,
      ]

      def expand_field(field_name):
        return self._expansion_box_field_labels(
            tensor_dict[input_fields.groundtruth_classes],
            tensor_dict[field_name])

      # pylint: disable=cell-var-from-loop
      for field in box_fields:
        if field in tensor_dict:
          tensor_dict[field] = tf.cond(
              tf.size(tensor_dict[field]) > 0, lambda: expand_field(field),
              lambda: tensor_dict[field])
      # pylint: enable=cell-var-from-loop

      tensor_dict[input_fields.groundtruth_classes] = (
          self._expansion_box_field_labels(
              tensor_dict[input_fields.groundtruth_classes],
              tensor_dict[input_fields.groundtruth_classes], True))

    if fields.InputDataFields.groundtruth_group_of in tensor_dict:
      group_of = fields.InputDataFields.groundtruth_group_of
      tensor_dict[group_of] = tf.cast(tensor_dict[group_of], dtype=tf.bool)

    if fields.InputDataFields.groundtruth_dp_num_points in tensor_dict:
      tensor_dict[fields.InputDataFields.groundtruth_dp_num_points] = tf.cast(
          tensor_dict[fields.InputDataFields.groundtruth_dp_num_points],
          dtype=tf.int32)
      tensor_dict[fields.InputDataFields.groundtruth_dp_part_ids] = tf.cast(
          tensor_dict[fields.InputDataFields.groundtruth_dp_part_ids],
          dtype=tf.int32)

    if fields.InputDataFields.groundtruth_track_ids in tensor_dict:
      tensor_dict[fields.InputDataFields.groundtruth_track_ids] = tf.cast(
          tensor_dict[fields.InputDataFields.groundtruth_track_ids],
          dtype=tf.int32)

    return tensor_dict

  def _reshape_keypoints(self, keys_to_tensors):
    """Reshape keypoints.

    The keypoints are reshaped to [num_instances, num_keypoints, 2].

    Args:
      keys_to_tensors: a dictionary from keys to tensors. Expected keys are:
        'image/object/keypoint/x'
        'image/object/keypoint/y'

    Returns:
      A 3-D float tensor of shape [num_instances, num_keypoints, 2] with values
        in [0, 1].
    """
    y = keys_to_tensors['image/object/keypoint/y']
    if isinstance(y, tf.SparseTensor):
      y = tf.sparse_tensor_to_dense(y)
    y = tf.expand_dims(y, 1)
    x = keys_to_tensors['image/object/keypoint/x']
    if isinstance(x, tf.SparseTensor):
      x = tf.sparse_tensor_to_dense(x)
    x = tf.expand_dims(x, 1)
    keypoints = tf.concat([y, x], 1)
    keypoints = tf.reshape(keypoints, [-1, self._num_keypoints, 2])
    return keypoints

  def _reshape_keypoint_visibilities(self, keys_to_tensors):
    """Reshape keypoint visibilities.

    The keypoint visibilities are reshaped to [num_instances,
    num_keypoints].

    The raw keypoint visibilities are expected to conform to the
    MSCoco definition. See Visibility enum.

    The returned boolean is True for the labeled case (either
    Visibility.NOT_VISIBLE or Visibility.VISIBLE). These are the same categories
    that COCO uses to evaluate keypoint detection performance:
    http://cocodataset.org/#keypoints-eval

    If image/object/keypoint/visibility is not provided, visibilities will be
    set to True for finite keypoint coordinate values, and 0 if the coordinates
    are NaN.

    Args:
      keys_to_tensors: a dictionary from keys to tensors. Expected keys are:
        'image/object/keypoint/x'
        'image/object/keypoint/visibility'

    Returns:
      A 2-D bool tensor of shape [num_instances, num_keypoints] with values
        in {0, 1}. 1 if the keypoint is labeled, 0 otherwise.
    """
    x = keys_to_tensors['image/object/keypoint/x']
    vis = keys_to_tensors['image/object/keypoint/visibility']
    if isinstance(vis, tf.SparseTensor):
      vis = tf.sparse_tensor_to_dense(vis)
    if isinstance(x, tf.SparseTensor):
      x = tf.sparse_tensor_to_dense(x)

    default_vis = tf.where(
        tf.math.is_nan(x),
        Visibility.UNLABELED.value * tf.ones_like(x, dtype=tf.int64),
        Visibility.VISIBLE.value * tf.ones_like(x, dtype=tf.int64))
    # Use visibility if provided, otherwise use the default visibility.
    vis = tf.cond(tf.equal(tf.size(x), tf.size(vis)),
                  true_fn=lambda: vis,
                  false_fn=lambda: default_vis)
    vis = tf.math.logical_or(
        tf.math.equal(vis, Visibility.NOT_VISIBLE.value),
        tf.math.equal(vis, Visibility.VISIBLE.value))
    vis = tf.reshape(vis, [-1, self._num_keypoints])
    return vis

  def _reshape_instance_masks(self, keys_to_tensors):
    """Reshape instance segmentation masks.

    The instance segmentation masks are reshaped to [num_instances, height,
    width].

    Args:
      keys_to_tensors: a dictionary from keys to tensors.

    Returns:
      A 3-D float tensor of shape [num_instances, height, width] with values
        in {0, 1}.
    """
    height = keys_to_tensors['image/height']
    width = keys_to_tensors['image/width']
    to_shape = tf.cast(tf.stack([-1, height, width]), tf.int32)
    masks = keys_to_tensors['image/object/mask']
    if isinstance(masks, tf.SparseTensor):
      masks = tf.sparse_tensor_to_dense(masks)
    masks = tf.reshape(
        tf.cast(tf.greater(masks, 0.0), dtype=tf.float32), to_shape)
    return tf.cast(masks, tf.float32)

  def _reshape_context_features(self, keys_to_tensors):
    """Reshape context features.

    The instance context_features are reshaped to
      [num_context_features, context_feature_length]

    Args:
      keys_to_tensors: a dictionary from keys to tensors.

    Returns:
      A 2-D float tensor of shape [num_context_features, context_feature_length]
    """
    context_feature_length = keys_to_tensors['image/context_feature_length']
    to_shape = tf.cast(tf.stack([-1, context_feature_length]), tf.int32)
    context_features = keys_to_tensors['image/context_features']
    if isinstance(context_features, tf.SparseTensor):
      context_features = tf.sparse_tensor_to_dense(context_features)
    context_features = tf.reshape(context_features, to_shape)
    return context_features

  def _decode_png_instance_masks(self, keys_to_tensors):
    """Decode PNG instance segmentation masks and stack into dense tensor.

    The instance segmentation masks are reshaped to [num_instances, height,
    width].

    Args:
      keys_to_tensors: a dictionary from keys to tensors.

    Returns:
      A 3-D float tensor of shape [num_instances, height, width] with values
        in {0, 1}.
    """

    def decode_png_mask(image_buffer):
      image = tf.squeeze(
          tf.image.decode_image(image_buffer, channels=1), axis=2)
      image.set_shape([None, None])
      image = tf.cast(tf.greater(image, 0), dtype=tf.float32)
      return image

    png_masks = keys_to_tensors['image/object/mask']
    height = keys_to_tensors['image/height']
    width = keys_to_tensors['image/width']
    if isinstance(png_masks, tf.SparseTensor):
      png_masks = tf.sparse_tensor_to_dense(png_masks, default_value='')
    return tf.cond(
        tf.greater(tf.size(png_masks), 0),
        lambda: tf.map_fn(decode_png_mask, png_masks, dtype=tf.float32),
        lambda: tf.zeros(tf.cast(tf.stack([0, height, width]), dtype=tf.int32)))

  def _dense_pose_part_indices(self, keys_to_tensors):
    """Creates a tensor that contains part indices for each DensePose point.

    Args:
      keys_to_tensors: a dictionary from keys to tensors.

    Returns:
      A 2-D int32 tensor of shape [num_instances, num_points] where each element
      contains the DensePose part index (0-23). The value `num_points`
      corresponds to the maximum number of sampled points across all instances
      in the image. Note that instances with less sampled points will be padded
      with zeros in the last dimension.
    """
    num_points_per_instances = keys_to_tensors['image/object/densepose/num']
    part_index = keys_to_tensors['image/object/densepose/part_index']
    if isinstance(num_points_per_instances, tf.SparseTensor):
      num_points_per_instances = tf.sparse_tensor_to_dense(
          num_points_per_instances)
    if isinstance(part_index, tf.SparseTensor):
      part_index = tf.sparse_tensor_to_dense(part_index)
    part_index = tf.cast(part_index, dtype=tf.int32)
    max_points_per_instance = tf.cast(
        tf.math.reduce_max(num_points_per_instances), dtype=tf.int32)
    num_points_cumulative = tf.concat([
        [0], tf.math.cumsum(num_points_per_instances)], axis=0)

    def pad_parts_tensor(instance_ind):
      points_range_start = num_points_cumulative[instance_ind]
      points_range_end = num_points_cumulative[instance_ind + 1]
      part_inds = part_index[points_range_start:points_range_end]
      return shape_utils.pad_or_clip_nd(part_inds,
                                        output_shape=[max_points_per_instance])

    return tf.map_fn(pad_parts_tensor,
                     tf.range(tf.size(num_points_per_instances)),
                     dtype=tf.int32)

  def _dense_pose_surface_coordinates(self, keys_to_tensors):
    """Creates a tensor that contains surface coords for each DensePose point.

    Args:
      keys_to_tensors: a dictionary from keys to tensors.

    Returns:
      A 3-D float32 tensor of shape [num_instances, num_points, 4] where each
      point contains (y, x, v, u) data for each sampled DensePose point. The
      (y, x) coordinate has normalized image locations for the point, and (v, u)
      contains the surface coordinate (also normalized) for the part. The value
      `num_points` corresponds to the maximum number of sampled points across
      all instances in the image. Note that instances with less sampled points
      will be padded with zeros in dim=1.
    """
    num_points_per_instances = keys_to_tensors['image/object/densepose/num']
    dp_y = keys_to_tensors['image/object/densepose/y']
    dp_x = keys_to_tensors['image/object/densepose/x']
    dp_v = keys_to_tensors['image/object/densepose/v']
    dp_u = keys_to_tensors['image/object/densepose/u']
    if isinstance(num_points_per_instances, tf.SparseTensor):
      num_points_per_instances = tf.sparse_tensor_to_dense(
          num_points_per_instances)
    if isinstance(dp_y, tf.SparseTensor):
      dp_y = tf.sparse_tensor_to_dense(dp_y)
    if isinstance(dp_x, tf.SparseTensor):
      dp_x = tf.sparse_tensor_to_dense(dp_x)
    if isinstance(dp_v, tf.SparseTensor):
      dp_v = tf.sparse_tensor_to_dense(dp_v)
    if isinstance(dp_u, tf.SparseTensor):
      dp_u = tf.sparse_tensor_to_dense(dp_u)
    max_points_per_instance = tf.cast(
        tf.math.reduce_max(num_points_per_instances), dtype=tf.int32)
    num_points_cumulative = tf.concat([
        [0], tf.math.cumsum(num_points_per_instances)], axis=0)

    def pad_surface_coordinates_tensor(instance_ind):
      """Pads DensePose surface coordinates for each instance."""
      points_range_start = num_points_cumulative[instance_ind]
      points_range_end = num_points_cumulative[instance_ind + 1]
      y = dp_y[points_range_start:points_range_end]
      x = dp_x[points_range_start:points_range_end]
      v = dp_v[points_range_start:points_range_end]
      u = dp_u[points_range_start:points_range_end]
      # Create [num_points_i, 4] tensor, where num_points_i is the number of
      # sampled points for instance i.
      unpadded_tensor = tf.stack([y, x, v, u], axis=1)
      return shape_utils.pad_or_clip_nd(
          unpadded_tensor, output_shape=[max_points_per_instance, 4])

    return tf.map_fn(pad_surface_coordinates_tensor,
                     tf.range(tf.size(num_points_per_instances)),
                     dtype=tf.float32)

  def _expand_image_label_hierarchy(self, image_classes, image_confidences):
    """Expand image level labels according to the hierarchy.

    Args:
      image_classes: Int64 tensor with the image level class ids for a sample.
      image_confidences: Float tensor signaling whether a class id is present in
        the image (1.0) or not present (0.0).

    Returns:
      new_image_classes: Int64 tensor equal to expanding image_classes.
      new_image_confidences: Float tensor equal to expanding image_confidences.
    """

    def expand_labels(relation_tensor, confidence_value):
      """Expand to ancestors or descendants depending on arguments."""
      mask = tf.equal(image_confidences, confidence_value)
      target_image_classes = tf.boolean_mask(image_classes, mask)
      expanded_indices = tf.reduce_any((tf.gather(
          relation_tensor, target_image_classes - _LABEL_OFFSET, axis=0) > 0),
                                       axis=0)
      expanded_indices = tf.where(expanded_indices)[:, 0] + _LABEL_OFFSET
      new_groundtruth_image_classes = (
          tf.concat([
              tf.boolean_mask(image_classes, tf.logical_not(mask)),
              expanded_indices,
          ],
                    axis=0))
      new_groundtruth_image_confidences = (
          tf.concat([
              tf.boolean_mask(image_confidences, tf.logical_not(mask)),
              tf.ones([tf.shape(expanded_indices)[0]],
                      dtype=image_confidences.dtype) * confidence_value,
          ],
                    axis=0))
      return new_groundtruth_image_classes, new_groundtruth_image_confidences

    image_classes, image_confidences = expand_labels(self._ancestors_lut, 1.0)
    new_image_classes, new_image_confidences = expand_labels(
        self._descendants_lut, 0.0)
    return new_image_classes, new_image_confidences

  def _expansion_box_field_labels(self,
                                  object_classes,
                                  object_field,
                                  copy_class_id=False):
    """Expand the labels of a specific object field according to the hierarchy.

    Args:
      object_classes: Int64 tensor with the class id for each element in
        object_field.
      object_field: Tensor to be expanded.
      copy_class_id: Boolean to choose whether to use class id values in the
        output tensor instead of replicating the original values.

    Returns:
      A tensor with the result of expanding object_field.
    """
    expanded_indices = tf.gather(
        self._ancestors_lut, object_classes - _LABEL_OFFSET, axis=0)
    if copy_class_id:
      new_object_field = tf.where(expanded_indices > 0)[:, 1] + _LABEL_OFFSET
    else:
      new_object_field = tf.repeat(
          object_field, tf.reduce_sum(expanded_indices, axis=1), axis=0)
    return new_object_field
