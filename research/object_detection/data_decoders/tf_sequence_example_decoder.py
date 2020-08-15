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
"""Sequence example decoder for object detection."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from six.moves import zip
import tensorflow.compat.v1 as tf
from tf_slim import tfexample_decoder as slim_example_decoder

from object_detection.core import data_decoder
from object_detection.core import standard_fields as fields
from object_detection.utils import label_map_util

# pylint: disable=g-import-not-at-top
try:
  from tensorflow.contrib import lookup as contrib_lookup
except ImportError:
  # TF 2.0 doesn't ship with contrib.
  pass
# pylint: enable=g-import-not-at-top


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

    self._name_to_id_table = name_to_id_table
    super(_ClassTensorHandler, self).__init__(tensor_key, shape_keys, shape,
                                              default_value)

  def tensors_to_item(self, keys_to_tensors):
    unmapped_tensor = super(_ClassTensorHandler,
                            self).tensors_to_item(keys_to_tensors)
    return self._name_to_id_table.lookup(unmapped_tensor)


class TfSequenceExampleDecoder(data_decoder.DataDecoder):
  """Tensorflow Sequence Example proto decoder for Object Detection.

  Sequence examples contain sequences of images which share common
  features. The structure of TfSequenceExamples can be seen in
  dataset_tools/seq_example_util.py

  For the TFODAPI, the following fields are required:
    Shared features:
      'image/format'
      'image/height'
      'image/width'

    Features with an entry for each image, where bounding box features can
    be empty lists if the image does not contain any objects:
      'image/encoded'
      'image/source_id'
      'region/bbox/xmin'
      'region/bbox/xmax'
      'region/bbox/ymin'
      'region/bbox/ymax'
      'region/label/string'

  Optionally, the sequence example can include context_features for use in
  Context R-CNN (see https://arxiv.org/abs/1912.03538):
    'image/context_features'
    'image/context_feature_length'
  """

  def __init__(self,
               label_map_proto_file,
               load_context_features=False,
               use_display_name=False,
               fully_annotated=False):
    """Constructs `TfSequenceExampleDecoder` object.

    Args:
      label_map_proto_file: a file path to a
        object_detection.protos.StringIntLabelMap proto. The
        label map will be used to map IDs of 'region/label/string'.
        It is assumed that 'region/label/string' will be in the data.
      load_context_features: Whether to load information from context_features,
        to provide additional context to a detection model for training and/or
        inference
      use_display_name: whether or not to use the `display_name` for label
        mapping (instead of `name`).  Only used if label_map_proto_file is
        provided.
      fully_annotated: If True, will assume that every frame (whether it has
        boxes or not), has been fully annotated. If False, a
        'region/is_annotated' field must be provided in the dataset which
        indicates which frames have annotations. Default False.
    """
    # Specifies how the tf.SequenceExamples are decoded.
    self._context_keys_to_features = {
        'image/format': tf.FixedLenFeature((), tf.string, default_value='jpeg'),
        'image/height': tf.FixedLenFeature((), tf.int64),
        'image/width': tf.FixedLenFeature((), tf.int64),
    }
    self._sequence_keys_to_feature_lists = {
        'image/encoded': tf.FixedLenSequenceFeature([], dtype=tf.string),
        'image/source_id': tf.FixedLenSequenceFeature([], dtype=tf.string),
        'region/bbox/xmin': tf.VarLenFeature(dtype=tf.float32),
        'region/bbox/xmax': tf.VarLenFeature(dtype=tf.float32),
        'region/bbox/ymin': tf.VarLenFeature(dtype=tf.float32),
        'region/bbox/ymax': tf.VarLenFeature(dtype=tf.float32),
        'region/label/string': tf.VarLenFeature(dtype=tf.string),
        'region/label/confidence': tf.VarLenFeature(dtype=tf.float32),
    }

    self._items_to_handlers = {
        # Context.
        fields.InputDataFields.image_height:
            slim_example_decoder.Tensor('image/height'),
        fields.InputDataFields.image_width:
            slim_example_decoder.Tensor('image/width'),

        # Sequence.
        fields.InputDataFields.num_groundtruth_boxes:
            slim_example_decoder.NumBoxesSequence('region/bbox/xmin'),
        fields.InputDataFields.groundtruth_boxes:
            slim_example_decoder.BoundingBoxSequence(
                prefix='region/bbox/', default_value=0.0),
        fields.InputDataFields.groundtruth_weights:
            slim_example_decoder.Tensor('region/label/confidence'),
    }

    # If the dataset is sparsely annotated, parse sequence features which
    # indicate which frames have been labeled.
    if not fully_annotated:
      self._sequence_keys_to_feature_lists['region/is_annotated'] = (
          tf.FixedLenSequenceFeature([], dtype=tf.int64))
      self._items_to_handlers[fields.InputDataFields.is_annotated] = (
          slim_example_decoder.Tensor('region/is_annotated'))

    self._items_to_handlers[fields.InputDataFields.image] = (
        slim_example_decoder.Tensor('image/encoded'))
    self._items_to_handlers[fields.InputDataFields.source_id] = (
        slim_example_decoder.Tensor('image/source_id'))

    label_handler = _ClassTensorHandler(
        'region/label/string', label_map_proto_file, default_value='')

    self._items_to_handlers[
        fields.InputDataFields.groundtruth_classes] = label_handler

    if load_context_features:
      self._context_keys_to_features['image/context_features'] = (
          tf.VarLenFeature(dtype=tf.float32))
      self._items_to_handlers[fields.InputDataFields.context_features] = (
          slim_example_decoder.ItemHandlerCallback(
              ['image/context_features', 'image/context_feature_length'],
              self._reshape_context_features))

      self._context_keys_to_features['image/context_feature_length'] = (
          tf.FixedLenFeature((), tf.int64))
      self._items_to_handlers[fields.InputDataFields.context_feature_length] = (
          slim_example_decoder.Tensor('image/context_feature_length'))
    self._fully_annotated = fully_annotated

  def decode(self, tf_seq_example_string_tensor):
    """Decodes serialized `tf.SequenceExample`s and returns a tensor dictionary.

    Args:
      tf_seq_example_string_tensor: a string tensor holding a serialized
        `tf.SequenceExample`.

    Returns:
      A list of dictionaries with (at least) the following tensors:
      fields.InputDataFields.source_id: a [num_frames] string tensor with a
        unique ID for each frame.
      fields.InputDataFields.num_groundtruth_boxes: a [num_frames] int32 tensor
        specifying the number of boxes in each frame.
      fields.InputDataFields.groundtruth_boxes: a [num_frames, num_boxes, 4]
        float32 tensor with bounding boxes for each frame. Note that num_boxes
        is the maximum boxes seen in any individual frame. Any frames with fewer
        boxes are padded with 0.0.
      fields.InputDataFields.groundtruth_classes: a [num_frames, num_boxes]
        int32 tensor with class indices for each box in each frame.
      fields.InputDataFields.groundtruth_weights: a [num_frames, num_boxes]
        float32 tensor with weights of the groundtruth boxes.
      fields.InputDataFields.is_annotated: a [num_frames] bool tensor specifying
        whether the image was annotated or not. If False, the corresponding
        entries in the groundtruth tensor will be ignored.
      fields.InputDataFields.context_features - 1D float32 tensor of shape
        [context_feature_length * num_context_features]
      fields.InputDataFields.context_feature_length - int32 tensor specifying
        the length of each feature in context_features
      fields.InputDataFields.image: a [num_frames] string tensor with
        the encoded images.
    """
    serialized_example = tf.reshape(tf_seq_example_string_tensor, shape=[])
    decoder = slim_example_decoder.TFSequenceExampleDecoder(
        self._context_keys_to_features, self._sequence_keys_to_feature_lists,
        self._items_to_handlers)
    keys = decoder.list_items()
    tensors = decoder.decode(serialized_example, items=keys)
    tensor_dict = dict(list(zip(keys, tensors)))
    tensor_dict[fields.InputDataFields.groundtruth_boxes].set_shape(
        [None, None, 4])
    tensor_dict[fields.InputDataFields.num_groundtruth_boxes] = tf.cast(
        tensor_dict[fields.InputDataFields.num_groundtruth_boxes],
        dtype=tf.int32)
    tensor_dict[fields.InputDataFields.groundtruth_classes] = tf.cast(
        tensor_dict[fields.InputDataFields.groundtruth_classes], dtype=tf.int32)
    tensor_dict[fields.InputDataFields.original_image_spatial_shape] = tf.cast(
        tf.stack([
            tensor_dict[fields.InputDataFields.image_height],
            tensor_dict[fields.InputDataFields.image_width]
        ]),
        dtype=tf.int32)
    tensor_dict.pop(fields.InputDataFields.image_height)
    tensor_dict.pop(fields.InputDataFields.image_width)

    def default_groundtruth_weights():
      """Produces weights of 1.0 for each valid box, and 0.0 otherwise."""
      num_boxes_per_frame = tensor_dict[
          fields.InputDataFields.num_groundtruth_boxes]
      max_num_boxes = tf.reduce_max(num_boxes_per_frame)
      num_boxes_per_frame_tiled = tf.tile(
          tf.expand_dims(num_boxes_per_frame, axis=-1),
          multiples=tf.stack([1, max_num_boxes]))
      range_tiled = tf.tile(
          tf.expand_dims(tf.range(max_num_boxes), axis=0),
          multiples=tf.stack([tf.shape(num_boxes_per_frame)[0], 1]))
      return tf.cast(
          tf.greater(num_boxes_per_frame_tiled, range_tiled), tf.float32)

    tensor_dict[fields.InputDataFields.groundtruth_weights] = tf.cond(
        tf.greater(
            tf.size(tensor_dict[fields.InputDataFields.groundtruth_weights]),
            0), lambda: tensor_dict[fields.InputDataFields.groundtruth_weights],
        default_groundtruth_weights)

    if self._fully_annotated:
      tensor_dict[fields.InputDataFields.is_annotated] = tf.ones_like(
          tensor_dict[fields.InputDataFields.num_groundtruth_boxes],
          dtype=tf.bool)
    else:
      tensor_dict[fields.InputDataFields.is_annotated] = tf.cast(
          tensor_dict[fields.InputDataFields.is_annotated], dtype=tf.bool)

    return tensor_dict

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
