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

"""Bounding Box List definition.

BoxList represents a list of bounding boxes as tensorflow
tensors, where each bounding box is represented as a row of 4 numbers,
[y_min, x_min, y_max, x_max].  It is assumed that all bounding boxes
within a given list correspond to a single image.  See also
box_list_ops.py for common box related operations (such as area, iou, etc).

Optionally, users can add additional related fields (such as weights).
We assume the following things to be true about fields:
* they correspond to boxes in the box_list along the 0th dimension
* they have inferrable rank at graph construction time
* all dimensions except for possibly the 0th can be inferred
  (i.e., not None) at graph construction time.

Some other notes:
  * Following tensorflow conventions, we use height, width ordering,
  and correspondingly, y,x (or ymin, xmin, ymax, xmax) ordering
  * Tensors are always provided as (flat) [N, 4] tensors.
"""

import tensorflow.compat.v1 as tf

from object_detection.utils import shape_utils


class BoxList(object):
  """Box collection."""

  def __init__(self, boxes):
    """Constructs box collection.

    Args:
      boxes: a tensor of shape [N, 4] representing box corners

    Raises:
      ValueError: if invalid dimensions for bbox data or if bbox data is not in
          float32 format.
    """
    if len(boxes.get_shape()) != 2 or boxes.get_shape()[-1] != 4:
      raise ValueError('Invalid dimensions for box data: {}'.format(
          boxes.shape))
    if boxes.dtype != tf.float32:
      raise ValueError('Invalid tensor type: should be tf.float32')
    self.data = {'boxes': boxes}

  def num_boxes(self):
    """Returns number of boxes held in collection.

    Returns:
      a tensor representing the number of boxes held in the collection.
    """
    return tf.shape(self.data['boxes'])[0]

  def num_boxes_static(self):
    """Returns number of boxes held in collection.

    This number is inferred at graph construction time rather than run-time.

    Returns:
      Number of boxes held in collection (integer) or None if this is not
        inferrable at graph construction time.
    """
    return shape_utils.get_dim_as_int(self.data['boxes'].get_shape()[0])

  def get_all_fields(self):
    """Returns all fields."""
    return self.data.keys()

  def get_extra_fields(self):
    """Returns all non-box fields (i.e., everything not named 'boxes')."""
    return [k for k in self.data.keys() if k != 'boxes']

  def add_field(self, field, field_data):
    """Add field to box list.

    This method can be used to add related box data such as
    weights/labels, etc.

    Args:
      field: a string key to access the data via `get`
      field_data: a tensor containing the data to store in the BoxList
    """
    self.data[field] = field_data

  def has_field(self, field):
    return field in self.data

  def get(self):
    """Convenience function for accessing box coordinates.

    Returns:
      a tensor with shape [N, 4] representing box coordinates.
    """
    return self.get_field('boxes')

  def set(self, boxes):
    """Convenience function for setting box coordinates.

    Args:
      boxes: a tensor of shape [N, 4] representing box corners

    Raises:
      ValueError: if invalid dimensions for bbox data
    """
    if len(boxes.get_shape()) != 2 or boxes.get_shape()[-1] != 4:
      raise ValueError('Invalid dimensions for box data.')
    self.data['boxes'] = boxes

  def get_field(self, field):
    """Accesses a box collection and associated fields.

    This function returns specified field with object; if no field is specified,
    it returns the box coordinates.

    Args:
      field: this optional string parameter can be used to specify
        a related field to be accessed.

    Returns:
      a tensor representing the box collection or an associated field.

    Raises:
      ValueError: if invalid field
    """
    if not self.has_field(field):
      raise ValueError('field ' + str(field) + ' does not exist')
    return self.data[field]

  def set_field(self, field, value):
    """Sets the value of a field.

    Updates the field of a box_list with a given value.

    Args:
      field: (string) name of the field to set value.
      value: the value to assign to the field.

    Raises:
      ValueError: if the box_list does not have specified field.
    """
    if not self.has_field(field):
      raise ValueError('field %s does not exist' % field)
    self.data[field] = value

  def get_center_coordinates_and_sizes(self, scope=None):
    """Computes the center coordinates, height and width of the boxes.

    Args:
      scope: name scope of the function.

    Returns:
      a list of 4 1-D tensors [ycenter, xcenter, height, width].
    """
    with tf.name_scope(scope, 'get_center_coordinates_and_sizes'):
      box_corners = self.get()
      ymin, xmin, ymax, xmax = tf.unstack(tf.transpose(box_corners))
      width = xmax - xmin
      height = ymax - ymin
      ycenter = ymin + height / 2.
      xcenter = xmin + width / 2.
      return [ycenter, xcenter, height, width]

  def transpose_coordinates(self, scope=None):
    """Transpose the coordinate representation in a boxlist.

    Args:
      scope: name scope of the function.
    """
    with tf.name_scope(scope, 'transpose_coordinates'):
      y_min, x_min, y_max, x_max = tf.split(
          value=self.get(), num_or_size_splits=4, axis=1)
      self.set(tf.concat([x_min, y_min, x_max, y_max], 1))

  def as_tensor_dict(self, fields=None):
    """Retrieves specified fields as a dictionary of tensors.

    Args:
      fields: (optional) list of fields to return in the dictionary.
        If None (default), all fields are returned.

    Returns:
      tensor_dict: A dictionary of tensors specified by fields.

    Raises:
      ValueError: if specified field is not contained in boxlist.
    """
    tensor_dict = {}
    if fields is None:
      fields = self.get_all_fields()
    for field in fields:
      if not self.has_field(field):
        raise ValueError('boxlist must contain all specified fields')
      tensor_dict[field] = self.get_field(field)
    return tensor_dict
