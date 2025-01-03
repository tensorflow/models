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

"""Contains functions which are convenient for unit testing."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from six.moves import range
from six.moves import zip
import tensorflow.compat.v1 as tf

from object_detection.core import anchor_generator
from object_detection.core import box_coder
from object_detection.core import box_list
from object_detection.core import box_predictor
from object_detection.core import matcher
from object_detection.utils import shape_utils
from object_detection.utils import tf_version

# Default size (both width and height) used for testing mask predictions.
DEFAULT_MASK_SIZE = 5


class MockBoxCoder(box_coder.BoxCoder):
  """Simple `difference` BoxCoder."""

  @property
  def code_size(self):
    return 4

  def _encode(self, boxes, anchors):
    return boxes.get() - anchors.get()

  def _decode(self, rel_codes, anchors):
    return box_list.BoxList(rel_codes + anchors.get())


class MockMaskHead(object):
  """Simple maskhead that returns all zeros as mask predictions."""

  def __init__(self, num_classes):
    self._num_classes = num_classes

  def predict(self, features):
    batch_size = tf.shape(features)[0]
    return tf.zeros((batch_size, 1, self._num_classes, DEFAULT_MASK_SIZE,
                     DEFAULT_MASK_SIZE),
                    dtype=tf.float32)


class MockBoxPredictor(box_predictor.BoxPredictor):
  """Simple box predictor that ignores inputs and outputs all zeros."""

  def __init__(self, is_training, num_classes, add_background_class=True):
    super(MockBoxPredictor, self).__init__(is_training, num_classes)
    self._add_background_class = add_background_class

  def _predict(self, image_features, num_predictions_per_location):
    image_feature = image_features[0]
    combined_feature_shape = shape_utils.combined_static_and_dynamic_shape(
        image_feature)
    batch_size = combined_feature_shape[0]
    num_anchors = (combined_feature_shape[1] * combined_feature_shape[2])
    code_size = 4
    zero = tf.reduce_sum(0 * image_feature)
    num_class_slots = self.num_classes
    if self._add_background_class:
      num_class_slots = num_class_slots + 1
    box_encodings = zero + tf.zeros(
        (batch_size, num_anchors, 1, code_size), dtype=tf.float32)
    class_predictions_with_background = zero + tf.zeros(
        (batch_size, num_anchors, num_class_slots), dtype=tf.float32)
    predictions_dict = {
        box_predictor.BOX_ENCODINGS:
            box_encodings,
        box_predictor.CLASS_PREDICTIONS_WITH_BACKGROUND:
            class_predictions_with_background
    }
    return predictions_dict


class MockKerasBoxPredictor(box_predictor.KerasBoxPredictor):
  """Simple box predictor that ignores inputs and outputs all zeros."""

  def __init__(self, is_training, num_classes, add_background_class=True):
    super(MockKerasBoxPredictor, self).__init__(
        is_training, num_classes, False, False)
    self._add_background_class = add_background_class

    # Dummy variable so that box predictor registers some variables.
    self._dummy_var = tf.Variable(0.0, trainable=True,
                                  name='box_predictor_var')

  def _predict(self, image_features, **kwargs):
    image_feature = image_features[0]
    combined_feature_shape = shape_utils.combined_static_and_dynamic_shape(
        image_feature)
    batch_size = combined_feature_shape[0]
    num_anchors = (combined_feature_shape[1] * combined_feature_shape[2])
    code_size = 4
    zero = tf.reduce_sum(0 * image_feature)
    num_class_slots = self.num_classes
    if self._add_background_class:
      num_class_slots = num_class_slots + 1
    box_encodings = zero + tf.zeros(
        (batch_size, num_anchors, 1, code_size), dtype=tf.float32)
    class_predictions_with_background = zero + tf.zeros(
        (batch_size, num_anchors, num_class_slots), dtype=tf.float32)
    predictions_dict = {
        box_predictor.BOX_ENCODINGS:
            box_encodings,
        box_predictor.CLASS_PREDICTIONS_WITH_BACKGROUND:
            class_predictions_with_background
    }
    return predictions_dict


class MockAnchorGenerator(anchor_generator.AnchorGenerator):
  """Mock anchor generator."""

  def name_scope(self):
    return 'MockAnchorGenerator'

  def num_anchors_per_location(self):
    return [1]

  def _generate(self, feature_map_shape_list):
    num_anchors = sum([shape[0] * shape[1] for shape in feature_map_shape_list])
    return box_list.BoxList(tf.zeros((num_anchors, 4), dtype=tf.float32))


class MockMatcher(matcher.Matcher):
  """Simple matcher that matches first anchor to first groundtruth box."""

  def _match(self, similarity_matrix, valid_rows):
    return tf.constant([0, -1, -1, -1], dtype=tf.int32)


def create_diagonal_gradient_image(height, width, depth):
  """Creates pyramid image. Useful for testing.

  For example, pyramid_image(5, 6, 1) looks like:
  # [[[ 5.  4.  3.  2.  1.  0.]
  #   [ 6.  5.  4.  3.  2.  1.]
  #   [ 7.  6.  5.  4.  3.  2.]
  #   [ 8.  7.  6.  5.  4.  3.]
  #   [ 9.  8.  7.  6.  5.  4.]]]

  Args:
    height: height of image
    width: width of image
    depth: depth of image

  Returns:
    pyramid image
  """
  row = np.arange(height)
  col = np.arange(width)[::-1]
  image_layer = np.expand_dims(row, 1) + col
  image_layer = np.expand_dims(image_layer, 2)

  image = image_layer
  for i in range(1, depth):
    image = np.concatenate((image, image_layer * pow(10, i)), 2)

  return image.astype(np.float32)


def create_random_boxes(num_boxes, max_height, max_width):
  """Creates random bounding boxes of specific maximum height and width.

  Args:
    num_boxes: number of boxes.
    max_height: maximum height of boxes.
    max_width: maximum width of boxes.

  Returns:
    boxes: numpy array of shape [num_boxes, 4]. Each row is in form
        [y_min, x_min, y_max, x_max].
  """

  y_1 = np.random.uniform(size=(1, num_boxes)) * max_height
  y_2 = np.random.uniform(size=(1, num_boxes)) * max_height
  x_1 = np.random.uniform(size=(1, num_boxes)) * max_width
  x_2 = np.random.uniform(size=(1, num_boxes)) * max_width

  boxes = np.zeros(shape=(num_boxes, 4))
  boxes[:, 0] = np.minimum(y_1, y_2)
  boxes[:, 1] = np.minimum(x_1, x_2)
  boxes[:, 2] = np.maximum(y_1, y_2)
  boxes[:, 3] = np.maximum(x_1, x_2)

  return boxes.astype(np.float32)


def first_rows_close_as_set(a, b, k=None, rtol=1e-6, atol=1e-6):
  """Checks if first K entries of two lists are close, up to permutation.

  Inputs to this assert are lists of items which can be compared via
  numpy.allclose(...) and can be sorted.

  Args:
    a: list of items which can be compared via numpy.allclose(...) and are
      sortable.
    b: list of items which can be compared via numpy.allclose(...) and are
      sortable.
    k: a non-negative integer.  If not provided, k is set to be len(a).
    rtol: relative tolerance.
    atol: absolute tolerance.

  Returns:
    boolean, True if input lists a and b have the same length and
    the first k entries of the inputs satisfy numpy.allclose() after
    sorting entries.
  """
  if not isinstance(a, list) or not isinstance(b, list) or len(a) != len(b):
    return False
  if not k:
    k = len(a)
  k = min(k, len(a))
  a_sorted = sorted(a[:k])
  b_sorted = sorted(b[:k])
  return all([
      np.allclose(entry_a, entry_b, rtol, atol)
      for (entry_a, entry_b) in zip(a_sorted, b_sorted)
  ])


class GraphContextOrNone(object):
  """A new Graph context for TF1.X and None for TF2.X.

  This is useful to write model tests that work with both TF1.X and TF2.X.

  Example test using this pattern:

  class ModelTest(test_case.TestCase):
    def test_model(self):
      with test_utils.GraphContextOrNone() as g:
        model = Model()
      def compute_fn():
        out = model.predict()
        return out['detection_boxes']
      boxes = self.execute(compute_fn, [], graph=g)
      self.assertAllClose(boxes, expected_boxes)
  """

  def __init__(self):
    if tf_version.is_tf2():
      self.graph = None
    else:
      self.graph = tf.Graph().as_default()

  def __enter__(self):
    if tf_version.is_tf2():
      return None
    else:
      return self.graph.__enter__()

  def __exit__(self, ttype, value, traceback):
    if tf_version.is_tf2():
      return False
    else:
      return self.graph.__exit__(ttype, value, traceback)


def image_with_dynamic_shape(height, width, channels):
  """Returns a single image with dynamic shape."""
  h = tf.random.uniform([], minval=height, maxval=height+1, dtype=tf.int32)
  w = tf.random.uniform([], minval=width, maxval=width+1, dtype=tf.int32)
  image = tf.random.uniform([h, w, channels])
  return image


def keypoints_with_dynamic_shape(num_instances, num_keypoints, num_coordinates):
  """Returns keypoints with dynamic shape."""
  n = tf.random.uniform([], minval=num_instances, maxval=num_instances+1,
                        dtype=tf.int32)
  keypoints = tf.random.uniform([n, num_keypoints, num_coordinates])
  return keypoints
