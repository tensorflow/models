# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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

"""Tensorflow Sequence Example proto decoder.

A decoder to decode string tensors containing serialized
tensorflow.SequenceExample protos.
TODO(yinxiao): When TensorFlow object detection API officially supports
tensorflow.SequenceExample, merge this decoder.
"""
import tensorflow as tf
from object_detection.core import data_decoder
from object_detection.core import standard_fields as fields

tfexample_decoder = tf.contrib.slim.tfexample_decoder


class BoundingBoxSequence(tfexample_decoder.ItemHandler):
  """An ItemHandler that concatenates SparseTensors to Bounding Boxes.
  """

  def __init__(self, keys=None, prefix=None, return_dense=True,
               default_value=-1.0):
    """Initialize the bounding box handler.

    Args:
      keys: A list of four key names representing the ymin, xmin, ymax, xmax
        in the Example or SequenceExample.
      prefix: An optional prefix for each of the bounding box keys in the
        Example or SequenceExample. If provided, `prefix` is prepended to each
        key in `keys`.
      return_dense: if True, returns a dense tensor; if False, returns as
        sparse tensor.
      default_value: The value used when the `tensor_key` is not found in a
        particular `TFExample`.

    Raises:
      ValueError: if keys is not `None` and also not a list of exactly 4 keys
    """
    if keys is None:
      keys = ['ymin', 'xmin', 'ymax', 'xmax']
    elif len(keys) != 4:
      raise ValueError('BoundingBoxSequence expects 4 keys but got {}'.format(
          len(keys)))
    self._prefix = prefix
    self._keys = keys
    self._full_keys = [prefix + k for k in keys]
    self._return_dense = return_dense
    self._default_value = default_value
    super(BoundingBoxSequence, self).__init__(self._full_keys)

  def tensors_to_item(self, keys_to_tensors):
    """Maps the given dictionary of tensors to a concatenated list of bboxes.

    Args:
      keys_to_tensors: a mapping of TF-Example keys to parsed tensors.

    Returns:
      [time, num_boxes, 4] tensor of bounding box coordinates, in order
          [y_min, x_min, y_max, x_max]. Whether the tensor is a SparseTensor
          or a dense Tensor is determined by the return_dense parameter. Empty
          positions in the sparse tensor are filled with -1.0 values.
    """
    sides = []
    for key in self._full_keys:
      value = keys_to_tensors[key]
      expanded_dims = tf.concat(
          [tf.to_int64(tf.shape(value)),
           tf.constant([1], dtype=tf.int64)], 0)
      side = tf.sparse_reshape(value, expanded_dims)
      sides.append(side)
    bounding_boxes = tf.sparse_concat(2, sides)
    if self._return_dense:
      bounding_boxes = tf.sparse_tensor_to_dense(
          bounding_boxes, default_value=self._default_value)
    return bounding_boxes


class TFSequenceExampleDecoder(data_decoder.DataDecoder):
  """Tensorflow Sequence Example proto decoder."""

  def __init__(self):
    """Constructor sets keys_to_features and items_to_handlers."""
    self.keys_to_context_features = {
        'image/format':
            tf.FixedLenFeature((), tf.string, default_value='jpeg'),
        'image/filename':
            tf.FixedLenFeature((), tf.string, default_value=''),
        'image/key/sha256':
            tf.FixedLenFeature((), tf.string, default_value=''),
        'image/source_id':
            tf.FixedLenFeature((), tf.string, default_value=''),
        'image/height':
            tf.FixedLenFeature((), tf.int64, 1),
        'image/width':
            tf.FixedLenFeature((), tf.int64, 1),
    }
    self.keys_to_features = {
        'image/encoded': tf.FixedLenSequenceFeature((), tf.string),
        'bbox/xmin': tf.VarLenFeature(dtype=tf.float32),
        'bbox/xmax': tf.VarLenFeature(dtype=tf.float32),
        'bbox/ymin': tf.VarLenFeature(dtype=tf.float32),
        'bbox/ymax': tf.VarLenFeature(dtype=tf.float32),
        'bbox/label/index': tf.VarLenFeature(dtype=tf.int64),
        'bbox/label/string': tf.VarLenFeature(tf.string),
        'area': tf.VarLenFeature(tf.float32),
        'is_crowd': tf.VarLenFeature(tf.int64),
        'difficult': tf.VarLenFeature(tf.int64),
        'group_of': tf.VarLenFeature(tf.int64),
    }
    self.items_to_handlers = {
        fields.InputDataFields.image:
            tfexample_decoder.Image(
                image_key='image/encoded',
                format_key='image/format',
                channels=3,
                repeated=True),
        fields.InputDataFields.source_id: (
            tfexample_decoder.Tensor('image/source_id')),
        fields.InputDataFields.key: (
            tfexample_decoder.Tensor('image/key/sha256')),
        fields.InputDataFields.filename: (
            tfexample_decoder.Tensor('image/filename')),
        # Object boxes and classes.
        fields.InputDataFields.groundtruth_boxes:
            BoundingBoxSequence(prefix='bbox/'),
        fields.InputDataFields.groundtruth_classes: (
            tfexample_decoder.Tensor('bbox/label/index')),
        fields.InputDataFields.groundtruth_area:
            tfexample_decoder.Tensor('area'),
        fields.InputDataFields.groundtruth_is_crowd: (
            tfexample_decoder.Tensor('is_crowd')),
        fields.InputDataFields.groundtruth_difficult: (
            tfexample_decoder.Tensor('difficult')),
        fields.InputDataFields.groundtruth_group_of: (
            tfexample_decoder.Tensor('group_of'))
    }

  def decode(self, tf_seq_example_string_tensor, items=None):
    """Decodes serialized tf.SequenceExample and returns a tensor dictionary.

    Args:
      tf_seq_example_string_tensor: A string tensor holding a serialized
        tensorflow example proto.
      items: The list of items to decode. These must be a subset of the item
        keys in self._items_to_handlers. If `items` is left as None, then all
        of the items in self._items_to_handlers are decoded.

    Returns:
      A dictionary of the following tensors.
      fields.InputDataFields.image - 3D uint8 tensor of shape [None, None, seq]
        containing image(s).
      fields.InputDataFields.source_id - string tensor containing original
        image id.
      fields.InputDataFields.key - string tensor with unique sha256 hash key.
      fields.InputDataFields.filename - string tensor with original dataset
        filename.
      fields.InputDataFields.groundtruth_boxes - 2D float32 tensor of shape
        [None, 4] containing box corners.
      fields.InputDataFields.groundtruth_classes - 1D int64 tensor of shape
        [None] containing classes for the boxes.
      fields.InputDataFields.groundtruth_area - 1D float32 tensor of shape
        [None] containing object mask area in pixel squared.
      fields.InputDataFields.groundtruth_is_crowd - 1D bool tensor of shape
        [None] indicating if the boxes enclose a crowd.
      fields.InputDataFields.groundtruth_difficult - 1D bool tensor of shape
        [None] indicating if the boxes represent `difficult` instances.
    """
    serialized_example = tf.reshape(tf_seq_example_string_tensor, shape=[])
    decoder = TFSequenceExampleDecoderHelper(self.keys_to_context_features,
                                             self.keys_to_features,
                                             self.items_to_handlers)
    if not items:
      items = decoder.list_items()
    tensors = decoder.decode(serialized_example, items=items)
    tensor_dict = dict(zip(items, tensors))

    return tensor_dict


class TFSequenceExampleDecoderHelper(data_decoder.DataDecoder):
  """A decoder helper class for TensorFlow SequenceExamples.

  To perform this decoding operation, a SequenceExampleDecoder is given a list
  of ItemHandlers. Each ItemHandler indicates the set of features.
  """

  def __init__(self, keys_to_context_features, keys_to_sequence_features,
               items_to_handlers):
    """Constructs the decoder.

    Args:
      keys_to_context_features: A dictionary from TF-SequenceExample context
        keys to either tf.VarLenFeature or tf.FixedLenFeature instances.
        See tensorflow's parsing_ops.py.
      keys_to_sequence_features: A dictionary from TF-SequenceExample sequence
        keys to either tf.VarLenFeature or tf.FixedLenSequenceFeature instances.
      items_to_handlers: A dictionary from items (strings) to ItemHandler
        instances. Note that the ItemHandler's are provided the keys that they
        use to return the final item Tensors.
    Raises:
      ValueError: If the same key is present for context features and sequence
        features.
    """
    unique_keys = set()
    unique_keys.update(keys_to_context_features)
    unique_keys.update(keys_to_sequence_features)
    if len(unique_keys) != (
        len(keys_to_context_features) + len(keys_to_sequence_features)):
      # This situation is ambiguous in the decoder's keys_to_tensors variable.
      raise ValueError('Context and sequence keys are not unique. \n'
                       ' Context keys: %s \n Sequence keys: %s' %
                       (list(keys_to_context_features.keys()),
                        list(keys_to_sequence_features.keys())))
    self._keys_to_context_features = keys_to_context_features
    self._keys_to_sequence_features = keys_to_sequence_features
    self._items_to_handlers = items_to_handlers

  def list_items(self):
    """Returns keys of items."""
    return self._items_to_handlers.keys()

  def decode(self, serialized_example, items=None):
    """Decodes the given serialized TF-SequenceExample.

    Args:
      serialized_example: A serialized TF-SequenceExample tensor.
      items: The list of items to decode. These must be a subset of the item
        keys in self._items_to_handlers. If `items` is left as None, then all
        of the items in self._items_to_handlers are decoded.
    Returns:
      The decoded items, a list of tensor.
    """
    context, feature_list = tf.parse_single_sequence_example(
        serialized_example, self._keys_to_context_features,
        self._keys_to_sequence_features)
    # Reshape non-sparse elements just once:
    for k in self._keys_to_context_features:
      v = self._keys_to_context_features[k]
      if isinstance(v, tf.FixedLenFeature):
        context[k] = tf.reshape(context[k], v.shape)
    if not items:
      items = self._items_to_handlers.keys()
    outputs = []
    for item in items:
      handler = self._items_to_handlers[item]
      keys_to_tensors = {
          key: context[key] if key in context else feature_list[key]
          for key in handler.keys
      }
      outputs.append(handler.tensors_to_item(keys_to_tensors))
    return outputs
