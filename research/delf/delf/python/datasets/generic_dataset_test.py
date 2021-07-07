# Lint as: python3
# Copyright 2021 The TensorFlow Authors All Rights Reserved.
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
"""Tests for generic dataset."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from absl import flags

import numpy as np
from PIL import Image
import tensorflow as tf

from delf.python.datasets import generic_dataset

FLAGS = flags.FLAGS


class GenericDatasetTest(tf.test.TestCase):
  """Test functions for generic dataset."""

  def testGenericDataset(self):
    """Tests loading dummy images from list."""
    # Number of images to be created.
    n = 2
    image_names = []

    # Create and save `n` dummy images.
    for i in range(n):
      dummy_image = np.random.rand(1024, 750, 3) * 255
      img_out = Image.fromarray(dummy_image.astype('uint8')).convert('RGB')
      filename = os.path.join(FLAGS.test_tmpdir,
                              'test_image_{}.jpg'.format(i))
      img_out.save(filename)
      image_names.append('test_image_{}.jpg'.format(i))

    data = generic_dataset.ImagesFromList(root=FLAGS.test_tmpdir,
                                          image_paths=image_names,
                                          imsize=1024)
    self.assertLen(data, n)


if __name__ == '__main__':
  tf.test.main()
