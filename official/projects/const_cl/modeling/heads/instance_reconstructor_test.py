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

"""Tests for instance_reconstructor."""

import tensorflow as tf, tf_keras
from official.projects.const_cl.modeling.heads import instance_reconstructor


class InstanceReconstructorTest(tf.test.TestCase):

  def test_instance_reconstructor_return_shapes(self):
    decoder = instance_reconstructor.InstanceReconstructor()

    inputs = {
        'features': tf.ones([12, 5, 7, 7, 128]),
        'instances_position': tf.random.uniform([12, 5, 16, 4]),
        'instances_mask': tf.ones([12, 5, 16], tf.bool)
    }

    outputs = decoder(inputs, training=True)
    self.assertContainsSubset(
        list(outputs.keys()),
        ['inst_a2b', 'inst_b2a', 'inst_a', 'inst_b', 'masks_a', 'masks_b'])

    self.assertAllEqual(outputs['inst_a2b'].shape, [6, 16, 1024])
    self.assertAllEqual(outputs['inst_a'].shape, [6, 16, 128])
    self.assertAllEqual(outputs['inst_b2a'].shape, [6, 16, 1024])
    self.assertAllEqual(outputs['inst_b'].shape, [6, 16, 128])
    self.assertAllEqual(outputs['masks_b'].shape, [6, 16])
    self.assertAllEqual(outputs['masks_a'].shape, [6, 16])


if __name__ == '__main__':
  tf.test.main()
