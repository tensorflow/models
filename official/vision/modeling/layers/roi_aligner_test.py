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

"""Tests for roi_aligner.py."""

# Import libraries
import tensorflow as tf

from official.vision.modeling.layers import roi_aligner


class MultilevelROIAlignerTest(tf.test.TestCase):

  def test_serialize_deserialize(self):
    kwargs = dict(
        crop_size=7,
        sample_offset=0.5,
    )
    aligner = roi_aligner.MultilevelROIAligner(**kwargs)

    expected_config = dict(kwargs)
    self.assertEqual(aligner.get_config(), expected_config)

    new_aligner = roi_aligner.MultilevelROIAligner.from_config(
        aligner.get_config())

    self.assertAllEqual(aligner.get_config(), new_aligner.get_config())


if __name__ == '__main__':
  tf.test.main()
