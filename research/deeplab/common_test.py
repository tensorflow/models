# Copyright 2018 The TensorFlow Authors All Rights Reserved.
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

"""Tests for common.py."""
import copy

import tensorflow as tf

from deeplab import common


class CommonTest(tf.test.TestCase):

  def testOutputsToNumClasses(self):
    num_classes = 21
    model_options = common.ModelOptions(
        outputs_to_num_classes={common.OUTPUT_TYPE: num_classes})
    self.assertEqual(model_options.outputs_to_num_classes[common.OUTPUT_TYPE],
                     num_classes)

  def testDeepcopy(self):
    num_classes = 21
    model_options = common.ModelOptions(
        outputs_to_num_classes={common.OUTPUT_TYPE: num_classes})
    model_options_new = copy.deepcopy(model_options)
    self.assertEqual((model_options_new.
                      outputs_to_num_classes[common.OUTPUT_TYPE]),
                     num_classes)

    num_classes_new = 22
    model_options_new.outputs_to_num_classes[common.OUTPUT_TYPE] = (
        num_classes_new)
    self.assertEqual(model_options.outputs_to_num_classes[common.OUTPUT_TYPE],
                     num_classes)
    self.assertEqual((model_options_new.
                      outputs_to_num_classes[common.OUTPUT_TYPE]),
                     num_classes_new)

if __name__ == '__main__':
  tf.test.main()
