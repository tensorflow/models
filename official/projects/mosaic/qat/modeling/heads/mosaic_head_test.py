# Copyright 2024 The TensorFlow Authors. All Rights Reserved.
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

"""Tests for mosaic_head."""

# Import libraries
from absl.testing import parameterized
import tensorflow as tf, tf_keras

from official.projects.mosaic.qat.modeling.heads import mosaic_head


class MosaicBlocksTest(parameterized.TestCase, tf.test.TestCase):

  def test_mosaic_head(self):
    decoder_head = mosaic_head.MosaicDecoderHeadQuantized(
        num_classes=32,
        decoder_input_levels=['3', '2'],
        decoder_stage_merge_styles=['concat_merge', 'sum_merge'],
        decoder_filters=[64, 64],
        decoder_projected_filters=[32, 32])
    inputs = [
        tf.ones([1, 32, 32, 128]), {
            '2': tf.ones([1, 128, 128, 64]),
            '3': tf.ones([1, 64, 64, 192])
        }
    ]
    outputs = decoder_head(inputs)
    self.assertAllEqual(outputs.shape, [1, 128, 128, 32])

  def test_mosaic_head_3laterals(self):
    decoder_head = mosaic_head.MosaicDecoderHeadQuantized(
        num_classes=32,
        decoder_input_levels=['3', '2', '1'],
        decoder_stage_merge_styles=[
            'concat_merge', 'concat_merge', 'sum_merge'
        ],
        decoder_filters=[64, 64, 64],
        decoder_projected_filters=[32, 32, 32])
    inputs = [
        tf.ones([1, 32, 32, 128]), {
            '1': tf.ones([1, 256, 256, 64]),
            '2': tf.ones([1, 128, 128, 64]),
            '3': tf.ones([1, 64, 64, 192])
        }
    ]
    outputs = decoder_head(inputs)
    self.assertAllEqual(outputs.shape, [1, 256, 256, 32])


if __name__ == '__main__':
  tf.test.main()
