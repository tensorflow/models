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
"""Tests for tfgan.examples.mnist.eval."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function



import tensorflow as tf
import eval  # pylint:disable=redefined-builtin


class EvalTest(tf.test.TestCase):

  def _test_build_graph_helper(self, eval_real_images):
    tf.flags.FLAGS.eval_real_images = eval_real_images
    eval.main(None, run_eval_loop=False)

  def test_build_graph_realdata(self):
    self._test_build_graph_helper(True)

  def test_build_graph_generateddata(self):
    self._test_build_graph_helper(False)

if __name__ == '__main__':
  tf.test.main()
