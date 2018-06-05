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
"""Tests for gan.cifar.eval."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from absl import flags
from absl.testing import parameterized
import tensorflow as tf
import eval  # pylint:disable=redefined-builtin

FLAGS = flags.FLAGS
mock = tf.test.mock


class EvalTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.named_parameters(
      ('RealData', True, False),
      ('GeneratedData', False, False),
      ('GeneratedDataConditional', False, True))
  def test_build_graph(self, eval_real_images, conditional_eval):
    FLAGS.eval_real_images = eval_real_images
    FLAGS.conditional_eval = conditional_eval
    # Mock `frechet_inception_distance` and `inception_score`, which are
    # expensive.
    with mock.patch.object(
        eval.util, 'get_frechet_inception_distance') as mock_fid:
      with mock.patch.object(eval.util, 'get_inception_scores') as mock_iscore:
        mock_fid.return_value = 1.0
        mock_iscore.return_value = 1.0
        eval.main(None, run_eval_loop=False)


if __name__ == '__main__':
  tf.test.main()
