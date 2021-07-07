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
"Tests for the tuples dataset module."

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from absl import flags
import numpy as np
from PIL import Image
import tensorflow as tf
import pickle

from delf.python.datasets import tuples_dataset
from delf.python.training.model import global_model

FLAGS = flags.FLAGS


class TuplesDatasetTest(tf.test.TestCase):
  """Tests for tuples dataset module."""

  def testCreateEpochTuples(self):
    """Tests epoch tuple creation."""
    # Create a tuples dataset instance.
    name = 'test_dataset'
    num_queries = 1
    pool_size = 5
    num_negatives = 2
    # Create a ground truth .pkl file.
    gnd = {
      'train': {'ids': [str(i) + '.png' for i in range(2 * num_queries + pool_size)],
                'cluster': [0, 0, 1, 2, 3, 4, 5],
                'qidxs': [0], 'pidxs': [1]}}
    gnd_name = name + '.pkl'
    with tf.io.gfile.GFile(os.path.join(FLAGS.test_tmpdir, gnd_name),
                           'wb') as gnd_file:
      pickle.dump(gnd, gnd_file)

    # Create random images for the dataset.
    for i in range(2 * num_queries + pool_size):
      dummy_image = np.random.rand(1024, 750, 3) * 255
      img_out = Image.fromarray(dummy_image.astype('uint8')).convert('RGB')
      filename = os.path.join(FLAGS.test_tmpdir, '{}.png'.format(i))
      img_out.save(filename)

    dataset = tuples_dataset.TuplesDataset(
      name=name,
      data_root=FLAGS.test_tmpdir,
      mode='train',
      imsize=1024,
      num_negatives=num_negatives,
      num_queries=num_queries,
      pool_size=pool_size
    )

    # Assert that initially no negative images are set.
    self.assertIsNone(dataset._nidxs)

    # Initialize a network for negative re-mining.
    model_params = {'architecture': 'ResNet101', 'pooling': 'gem',
                    'whitening': False, 'pretrained': True}
    model = global_model.GlobalFeatureNet(**model_params)

    avg_neg_distance = dataset.create_epoch_tuples(model)

    # Check that an appropriate number of negative images has been chosen per
    # query.
    self.assertAllEqual(tf.shape(dataset._nidxs), [num_queries, num_negatives])

if __name__ == '__main__':
  tf.test.main()
