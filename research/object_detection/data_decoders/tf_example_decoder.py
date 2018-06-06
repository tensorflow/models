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
import tensorflow as tf

from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from object_detection.core import data_decoder
from object_detection.core import standard_fields as fields
from object_detection.protos import input_reader_pb2
from object_detection.utils import label_map_util

slim_example_decoder = tf.contrib.slim.tfexample_decoder


# TODO(lzc): keep LookupTensor and BackupHandler in sync with
# tf.contrib.slim.tfexample_decoder version.
class LookupTensor(slim_example_decoder.Tensor):
  """An ItemHandler that returns a parsed Tensor, the result of a lookup."""

  def __init__(self,
               tensor_key,
               table,
               shape_keys=None,
               shape=None,
               default_value=''):
    """Initializes the LookupTensor handler.

    Simply calls a vocabulary (most often, a label mapping) lookup.

    Args:
      tensor_key: the name of the `TFExample` feature to read the tensor from.
      table: A tf.lookup table.
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
    self._table = table
    super(LookupTensor, self).__init__(tensor_key, shape_keys, shape,
                                       default_value)

  def tensors_to_item(self, keys_to_tensors):
    unmapped_tensor = super(LookupTensor, self).tensors_to_item(keys_to_tensors)
    return self._table.lookup(unmapped_tensor)


class BackupHandler(slim_example_decoder.ItemHandler):
  """An ItemHandler that tries two ItemHandlers in order."""

  def __init__(self, handler, backup):
    """Initializes the BackupHandler handler.

    If the first Handler's tensors_to_item returns a Tensor with no elements,
    the second Handler is used.

    Args:
      handler: The primary ItemHandler.
      backup: The backup ItemHandler.

    Raises:
      ValueError: if either is not an ItemHandler.
    """
    if not isinstance(handler, slim_example_decoder.ItemHandler):
      raise ValueError('Primary handler is of type %s instead of ItemHandler' %
                       type(handler))
    if not isinstance(backup, slim_example_decoder.ItemHandler):
      raise ValueError(
          'Backup handler is of type %s instead of ItemHandler' % type(backup))
    self._handler = handler
    self._backup = backup
    super(BackupHandler, self).__init__(handler.keys + backup.keys)

  def tensors_to_item(self, keys_to_tensors):
    item = self._handler.tensors_to_item(keys_to_tensors)
    return control_flow_ops.cond(
        pred=math_ops.equal(math_ops.reduce_prod(array_ops.shape(item)), 0),
        true_fn=lambda: self._backup.tensors_to_item(keys_to_tensors),
        false_fn=lambda: item)


class TfExampleDecoder(data_decoder.DataDecoder):
  """Tensorflow Example proto decoder."""

  def __init__(self,
               load_instance_masks=False,
               instance_mask_type=input_reader_pb2.NUMERICAL_MASKS,
               label_map_proto_file=None,
               use_display_name=False,
               dct_method='',
               num_keypoints=0,
               num_additional_channels=0):
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

    Raises:
      ValueError: If `instance_mask_type` option is not one of
        input_reader_pb2.DEFAULT, input_reader_pb2.NUMERICAL, or
        input_reader_pb2.PNG_MASKS.
    """
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
      self.items_to_handlers[fields.InputDataFields.groundtruth_keypoints] = (
          slim_example_decoder.ItemHandlerCallback(
              ['image/object/keypoint/y', 'image/object/keypoint/x'],
              self._reshape_keypoints))
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
    if label_map_proto_file:
      label_map = label_map_util.get_label_map_dict(label_map_proto_file,
                                                    use_display_name)
      # We use a default_value of -1, but we expect all labels to be contained
      # in the label map.
      table = tf.contrib.lookup.HashTable(
          initializer=tf.contrib.lookup.KeyValueTensorInitializer(
              keys=tf.constant(list(label_map.keys())),
              values=tf.constant(list(label_map.values()), dtype=tf.int64)),
          default_value=-1)
      # If the label_map_proto is provided, try to use it in conjunction with
      # the class text, and fall back to a materialized ID.
      # TODO(lzc): note that here we are using BackupHandler defined in this
      # file(which is branching slim_example_decoder.BackupHandler). Need to
      # switch back to slim_example_decoder.BackupHandler once tf 1.5 becomes
      # more popular.
      label_handler = BackupHandler(
          LookupTensor('image/object/class/text', table, default_value=''),
          slim_example_decoder.Tensor('image/object/class/label'))
    else:
      label_handler = slim_example_decoder.Tensor('image/object/class/label')
    self.items_to_handlers[
        fields.InputDataFields.groundtruth_classes] = label_handler

  def decode(self, tf_example_string_tensor):
    """Decodes serialized tensorflow example and returns a tensor dictionary.

    Args:
      tf_example_string_tensor: a string tensor holding a serialized tensorflow
        example proto.

    Returns:
      A dictionary of the following tensors.
      fields.InputDataFields.image - 3D uint8 tensor of shape [None, None, 3]
        containing image.
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
      fields.InputDataFields.num_groundtruth_boxes - int32 scalar indicating
        the number of groundtruth_boxes.
      fields.InputDataFields.groundtruth_area - 1D float32 tensor of shape
        [None] containing containing object mask area in pixel squared.
      fields.InputDataFields.groundtruth_is_crowd - 1D bool tensor of shape
        [None] indicating if the boxes enclose a crowd.

    Optional:
      fields.InputDataFields.image_additional_channels - 3D uint8 tensor of
        shape [None, None, num_additional_channels]. 1st dim is height; 2nd dim
        is width; 3rd dim is the number of additional channels.
      fields.InputDataFields.groundtruth_difficult - 1D bool tensor of shape
        [None] indicating if the boxes represent `difficult` instances.
      fields.InputDataFields.groundtruth_group_of - 1D bool tensor of shape
        [None] indicating if the boxes represent `group_of` instances.
      fields.InputDataFields.groundtruth_keypoints - 3D float32 tensor of
        shape [None, None, 2] containing keypoints, where the coordinates of
        the keypoints are ordered (y, x).
      fields.InputDataFields.groundtruth_instance_masks - 3D float32 tensor of
        shape [None, None, None] containing instance masks.
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
    tensor_dict[fields.InputDataFields.num_groundtruth_boxes] = tf.shape(
        tensor_dict[fields.InputDataFields.groundtruth_boxes])[0]

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
    return tensor_dict

  def _reshape_keypoints(self, keys_to_tensors):
    """Reshape keypoints.

    The instance segmentation masks are reshaped to [num_instances,
    num_keypoints, 2].

    Args:
      keys_to_tensors: a dictionary from keys to tensors.

    Returns:
      A 3-D float tensor of shape [num_instances, num_keypoints, 2] with values
        in {0, 1}.
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
    masks = tf.reshape(tf.to_float(tf.greater(masks, 0.0)), to_shape)
    return tf.cast(masks, tf.float32)

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
      image = tf.to_float(tf.greater(image, 0))
      return image

    png_masks = keys_to_tensors['image/object/mask']
    height = keys_to_tensors['image/height']
    width = keys_to_tensors['image/width']
    if isinstance(png_masks, tf.SparseTensor):
      png_masks = tf.sparse_tensor_to_dense(png_masks, default_value='')
    return tf.cond(
        tf.greater(tf.size(png_masks), 0),
        lambda: tf.map_fn(decode_png_mask, png_masks, dtype=tf.float32),
        lambda: tf.zeros(tf.to_int32(tf.stack([0, height, width]))))
