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

"""Tests for box_coder_builder."""

import tensorflow as tf

from google.protobuf import text_format
from object_detection.box_coders import faster_rcnn_box_coder
from object_detection.box_coders import mean_stddev_box_coder
from object_detection.box_coders import square_box_coder
from object_detection.builders import box_coder_builder
from object_detection.protos import box_coder_pb2


class BoxCoderBuilderTest(tf.test.TestCase):

  def test_build_faster_rcnn_box_coder_with_defaults(self):
    box_coder_text_proto = """
      faster_rcnn_box_coder {
      }
    """
    box_coder_proto = box_coder_pb2.BoxCoder()
    text_format.Merge(box_coder_text_proto, box_coder_proto)
    box_coder_object = box_coder_builder.build(box_coder_proto)
    self.assertTrue(isinstance(box_coder_object,
                               faster_rcnn_box_coder.FasterRcnnBoxCoder))
    self.assertEqual(box_coder_object._scale_factors, [10.0, 10.0, 5.0, 5.0])

  def test_build_faster_rcnn_box_coder_with_non_default_parameters(self):
    box_coder_text_proto = """
      faster_rcnn_box_coder {
        y_scale: 6.0
        x_scale: 3.0
        height_scale: 7.0
        width_scale: 8.0
      }
    """
    box_coder_proto = box_coder_pb2.BoxCoder()
    text_format.Merge(box_coder_text_proto, box_coder_proto)
    box_coder_object = box_coder_builder.build(box_coder_proto)
    self.assertTrue(isinstance(box_coder_object,
                               faster_rcnn_box_coder.FasterRcnnBoxCoder))
    self.assertEqual(box_coder_object._scale_factors, [6.0, 3.0, 7.0, 8.0])

  def test_build_mean_stddev_box_coder(self):
    box_coder_text_proto = """
      mean_stddev_box_coder {
      }
    """
    box_coder_proto = box_coder_pb2.BoxCoder()
    text_format.Merge(box_coder_text_proto, box_coder_proto)
    box_coder_object = box_coder_builder.build(box_coder_proto)
    self.assertTrue(
        isinstance(box_coder_object,
                   mean_stddev_box_coder.MeanStddevBoxCoder))

  def test_build_square_box_coder_with_defaults(self):
    box_coder_text_proto = """
      square_box_coder {
      }
    """
    box_coder_proto = box_coder_pb2.BoxCoder()
    text_format.Merge(box_coder_text_proto, box_coder_proto)
    box_coder_object = box_coder_builder.build(box_coder_proto)
    self.assertTrue(
        isinstance(box_coder_object, square_box_coder.SquareBoxCoder))
    self.assertEqual(box_coder_object._scale_factors, [10.0, 10.0, 5.0])

  def test_build_square_box_coder_with_non_default_parameters(self):
    box_coder_text_proto = """
      square_box_coder {
        y_scale: 6.0
        x_scale: 3.0
        length_scale: 7.0
      }
    """
    box_coder_proto = box_coder_pb2.BoxCoder()
    text_format.Merge(box_coder_text_proto, box_coder_proto)
    box_coder_object = box_coder_builder.build(box_coder_proto)
    self.assertTrue(
        isinstance(box_coder_object, square_box_coder.SquareBoxCoder))
    self.assertEqual(box_coder_object._scale_factors, [6.0, 3.0, 7.0])

  def test_raise_error_on_empty_box_coder(self):
    box_coder_text_proto = """
    """
    box_coder_proto = box_coder_pb2.BoxCoder()
    text_format.Merge(box_coder_text_proto, box_coder_proto)
    with self.assertRaises(ValueError):
      box_coder_builder.build(box_coder_proto)


if __name__ == '__main__':
  tf.test.main()
