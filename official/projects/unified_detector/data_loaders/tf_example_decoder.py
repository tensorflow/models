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

"""Tensorflow Example proto decoder for GOCR."""

from typing import List, Optional, Sequence, Tuple, Union

import tensorflow as tf
from official.projects.unified_detector.utils.typing import TensorDict
from official.vision.dataloaders import decoder


class TfExampleDecoder(decoder.Decoder):
  """Tensorflow Example proto decoder."""

  def __init__(self,
               use_instance_mask: bool = False,
               additional_class_names: Optional[Sequence[str]] = None,
               additional_regression_names: Optional[Sequence[str]] = None,
               num_additional_channels: int = 0):
    """Constructor.

    keys_to_features is a dictionary mapping the names of the tf.Example
    fields to tf features, possibly with defaults.

    Uses fixed length for scalars and variable length for vectors.

    Args:
      use_instance_mask: if False, prevents decoding of the instance mask, which
        can take a lot of resources.
      additional_class_names: If not none, a list of additional class names. For
        additional class name n, named image/object/${n} are expected to be an
        int vector of length one, and are mapped to tensor dict key
        groundtruth_${n}.
      additional_regression_names: If not none, a list of additional regression
        output names. For additional class name n, named image/object/${n} are
        expected to be a float vector, and are mapped to tensor dict key
        groundtruth_${n}.
      num_additional_channels: The number of additional channels of information
        present in the tf.Example proto.
    """
    self._num_additional_channels = num_additional_channels
    self._use_instance_mask = use_instance_mask

    self.keys_to_features = {}
    # Map names in the final tensor dict (output of `self.decode()`) to names in
    # tf examples, e.g. 'groundtruth_text' -> 'image/object/text'
    self.name_to_key = {}

    if use_instance_mask:
      self.keys_to_features.update({
          'image/object/mask': tf.io.VarLenFeature(tf.string),
      })

    # Now we have lists of standard types.
    # To add new features, just add entries here.
    # The tuple elements are (example name, tensor name, default value).
    # If the items_to_handlers part is already set up use None for
    # the tensor name.
    # There are other tensor names listed as None which we probably
    # want to discuss and specify.
    scalar_strings = [
        ('image/encoded', None, ''),
        ('image/format', None, 'jpg'),
        ('image/additional_channels/encoded', None, ''),
        ('image/additional_channels/format', None, 'png'),
        ('image/label_type', 'label_type', ''),
        ('image/key', 'key', ''),
        ('image/source_id', 'source_id', ''),
    ]
    vector_strings = [
        ('image/attributes', None, ''),
        ('image/object/text', 'groundtruth_text', ''),
        ('image/object/encoded_text', 'groundtruth_encoded_text', ''),
        ('image/object/vertices', 'groundtruth_vertices', ''),
        ('image/object/object_type', None, ''),
        ('image/object/language', 'language', ''),
        ('image/object/reorderer_type', None, ''),
        ('image/label_map_path', 'label_map_path', '')
    ]
    scalar_ints = [
        ('image/height', None, 1),
        ('image/width', None, 1),
        ('image/channels', None, 3),
    ]
    vector_ints = [
        ('image/object/classes', 'groundtruth_classes', 0),
        ('image/object/frame_id', 'frame_id', 0),
        ('image/object/track_id', 'track_id', 0),
        ('image/object/content_type', 'groundtruth_content_type', 0),
    ]
    if additional_class_names:
      vector_ints += [('image/object/%s' % name, 'groundtruth_%s' % name, 0)
                      for name in additional_class_names]
    # This one is not yet needed:
    # scalar_floats = [
    # ]
    vector_floats = [
        ('image/object/weight', 'groundtruth_weight', 0),
        ('image/object/rbox_tl_x', None, 0),
        ('image/object/rbox_tl_y', None, 0),
        ('image/object/rbox_width', None, 0),
        ('image/object/rbox_height', None, 0),
        ('image/object/rbox_angle', None, 0),
        ('image/object/bbox/xmin', None, 0),
        ('image/object/bbox/xmax', None, 0),
        ('image/object/bbox/ymin', None, 0),
        ('image/object/bbox/ymax', None, 0),
    ]
    if additional_regression_names:
      vector_floats += [('image/object/%s' % name, 'groundtruth_%s' % name, 0)
                        for name in additional_regression_names]

    self._init_scalar_features(scalar_strings, tf.string)
    self._init_vector_features(vector_strings, tf.string)
    self._init_scalar_features(scalar_ints, tf.int64)
    self._init_vector_features(vector_ints, tf.int64)
    self._init_vector_features(vector_floats, tf.float32)

  def _init_scalar_features(
      self,
      feature_list: List[Tuple[str, Optional[str], Union[str, int, float]]],
      ftype: tf.dtypes.DType) -> None:
    for entry in feature_list:
      self.keys_to_features[entry[0]] = tf.io.FixedLenFeature(
          (), ftype, default_value=entry[2])
      if entry[1] is not None:
        self.name_to_key[entry[1]] = entry[0]

  def _init_vector_features(
      self,
      feature_list: List[Tuple[str, Optional[str], Union[str, int, float]]],
      ftype: tf.dtypes.DType) -> None:
    for entry in feature_list:
      self.keys_to_features[entry[0]] = tf.io.VarLenFeature(ftype)
      if entry[1] is not None:
        self.name_to_key[entry[1]] = entry[0]

  def _decode_png_instance_masks(self, keys_to_tensors: TensorDict)-> tf.Tensor:
    """Decode PNG instance segmentation masks and stack into dense tensor.

    The instance segmentation masks are reshaped to [num_instances, height,
    width].

    Args:
      keys_to_tensors: A dictionary from keys to tensors.

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

  def _decode_image(self,
                    parsed_tensors: TensorDict,
                    channel: int = 3) -> TensorDict:
    """Decodes the image and set its shape (H, W are dynamic; C is fixed)."""
    image = tf.io.decode_image(parsed_tensors['image/encoded'],
                               channels=channel)
    image.set_shape([None, None, channel])
    return {'image': image}

  def _decode_additional_channels(self,
                                  parsed_tensors: TensorDict,
                                  channel: int = 3) -> TensorDict:
    """Decodes the additional channels and set its static shape."""
    channels = tf.io.decode_image(
        parsed_tensors['image/additional_channels/encoded'], channels=channel)
    channels.set_shape([None, None, channel])
    return {'additional_channels': channels}

  def _decode_boxes(self, parsed_tensors: TensorDict) -> TensorDict:
    """Concat box coordinates in the format of [ymin, xmin, ymax, xmax]."""
    xmin = parsed_tensors['image/object/bbox/xmin']
    xmax = parsed_tensors['image/object/bbox/xmax']
    ymin = parsed_tensors['image/object/bbox/ymin']
    ymax = parsed_tensors['image/object/bbox/ymax']
    return {
        'groundtruth_aligned_boxes': tf.stack([ymin, xmin, ymax, xmax], axis=-1)
    }

  def _decode_rboxes(self, parsed_tensors: TensorDict) -> TensorDict:
    """Concat rbox coordinates: [left, top, box_width, box_height, angle]."""
    top_left_x = parsed_tensors['image/object/rbox_tl_x']
    top_left_y = parsed_tensors['image/object/rbox_tl_y']
    width = parsed_tensors['image/object/rbox_width']
    height = parsed_tensors['image/object/rbox_height']
    angle = parsed_tensors['image/object/rbox_angle']
    return {
        'groundtruth_boxes':
            tf.stack([top_left_x, top_left_y, width, height, angle], axis=-1)
    }

  def _decode_masks(self, parsed_tensors: TensorDict) -> TensorDict:
    """Decode a set of PNG masks to the tf.float32 tensors."""

    def _decode_png_mask(png_bytes):
      mask = tf.squeeze(
          tf.io.decode_png(png_bytes, channels=1, dtype=tf.uint8), axis=-1)
      mask = tf.cast(mask, dtype=tf.float32)
      mask.set_shape([None, None])
      return mask

    height = parsed_tensors['image/height']
    width = parsed_tensors['image/width']
    masks = parsed_tensors['image/object/mask']
    masks = tf.cond(
        pred=tf.greater(tf.size(input=masks), 0),
        true_fn=lambda: tf.map_fn(_decode_png_mask, masks, dtype=tf.float32),
        false_fn=lambda: tf.zeros([0, height, width], dtype=tf.float32))
    return {'groundtruth_instance_masks': masks}

  def decode(self, tf_example_string_tensor: tf.string):
    """Decodes serialized tensorflow example and returns a tensor dictionary.

    Args:
      tf_example_string_tensor: A string tensor holding a serialized tensorflow
        example proto.

    Returns:
      A dictionary contains a subset of the following, depends on the inputs:
        image: A uint8 tensor of shape [height, width, 3] containing the image.
        source_id: A string tensor contains image fingerprint.
        key: A string tensor contains the unique sha256 hash key.
        label_type: Either `full` or `partial`. `full` means all the text are
          fully labeled, `partial` otherwise. Currently, this is used by E2E
          model. If an input image is fully labeled, we update the weights of
          both the detection and the recognizer. Otherwise, only recognizer part
          of the model is trained.
        groundtruth_text: A string tensor list, the original transcriptions.
        groundtruth_encoded_text: A string tensor list, the class ids for the
          atoms in the text, after applying the reordering algorithm, in string
          form. For example "90,71,85,69,86,85,93,90,71,91,1,71,85,93,90,71".
          This depends on the class label map provided to the conversion
          program. These are 0 based, with -1 for OOV symbols.
        groundtruth_classes: A int32 tensor of shape [num_boxes] contains the
          class id. Note this is 1 based, 0 is reserved for background class.
        groundtruth_content_type: A int32 tensor of shape [num_boxes] contains
          the content type. Values correspond to PageLayoutEntity::ContentType.
        groundtruth_weight: A int32 tensor of shape [num_boxes], either 0 or 1.
          If a region has weight 0, it will be ignored when computing the
          losses.
        groundtruth_boxes: A float tensor of shape [num_boxes, 5] contains the
          groundtruth rotated rectangles. Each row is in [left, top, box_width,
          box_height, angle] order, absolute coordinates are used.
        groundtruth_aligned_boxes: A float tensor of shape [num_boxes, 4]
          contains the groundtruth axis-aligned rectangles. Each row is in
          [ymin, xmin, ymax, xmax] order. Currently, this is used to store
          groundtruth symbol boxes.
        groundtruth_vertices: A string tensor list contains encoded normalized
          box or polygon coordinates. E.g. `x1,y1,x2,y2,x3,y3,x4,y4`.
        groundtruth_instance_masks: A float tensor of shape [num_boxes, height,
          width] contains binarized image sized instance segmentation masks.
          `1.0` for positive region, `0.0` otherwise. None if not in tfe.
        frame_id: A int32 tensor of shape [num_boxes], either `0` or `1`.
          `0` means object comes from first image, `1` means second.
        track_id: A int32 tensor of shape [num_boxes], where value indicates
          identity across frame indices.
        additional_channels: A uint8 tensor of shape [H, W, C] representing some
          features.
    """
    parsed_tensors = tf.io.parse_single_example(
        serialized=tf_example_string_tensor, features=self.keys_to_features)
    for k in parsed_tensors:
      if isinstance(parsed_tensors[k], tf.SparseTensor):
        if parsed_tensors[k].dtype == tf.string:
          parsed_tensors[k] = tf.sparse.to_dense(
              parsed_tensors[k], default_value='')
        else:
          parsed_tensors[k] = tf.sparse.to_dense(
              parsed_tensors[k], default_value=0)

    decoded_tensors = {}
    decoded_tensors.update(self._decode_image(parsed_tensors))
    decoded_tensors.update(self._decode_rboxes(parsed_tensors))
    decoded_tensors.update(self._decode_boxes(parsed_tensors))
    if self._use_instance_mask:
      decoded_tensors[
          'groundtruth_instance_masks'] = self._decode_png_instance_masks(
              parsed_tensors)
    if self._num_additional_channels:
      decoded_tensors.update(self._decode_additional_channels(
          parsed_tensors, self._num_additional_channels))

    # other attributes:
    for key in self.name_to_key:
      if key not in decoded_tensors:
        decoded_tensors[key] = parsed_tensors[self.name_to_key[key]]

    if 'groundtruth_instance_masks' not in decoded_tensors:
      decoded_tensors['groundtruth_instance_masks'] = None

    return decoded_tensors
