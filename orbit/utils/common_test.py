# Copyright 2023 The Orbit Authors. All Rights Reserved.
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

"""Tests for orbit.utils.common."""

from orbit.utils import common

import tensorflow as tf


class UtilsTest(tf.test.TestCase):

  def test_create_global_step(self):
    step = common.create_global_step()
    self.assertEqual(step.name, "global_step:0")
    self.assertEqual(step.dtype, tf.int64)
    self.assertEqual(step, 0)
    step.assign_add(1)
    self.assertEqual(step, 1)


if __name__ == "__main__":
  tf.test.main()
