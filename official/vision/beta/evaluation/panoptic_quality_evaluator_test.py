# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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

"""Tests for panoptic_quality_evaluator."""

import numpy as np
import tensorflow as tf

from official.vision.beta.evaluation import panoptic_quality_evaluator


class PanopticQualityEvaluatorTest(tf.test.TestCase):

  def test_multiple_batches(self):
    category_mask = np.zeros([6, 6], np.uint16)
    groundtruth_instance_mask = np.array([
        [1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1],
        [1, 1, 2, 2, 2, 1],
        [1, 2, 2, 2, 2, 1],
        [1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1],
    ],
                                         dtype=np.uint16)

    good_det_instance_mask = np.array([
        [1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1],
        [1, 2, 2, 2, 2, 1],
        [1, 2, 2, 2, 1, 1],
        [1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1],
    ],
                                      dtype=np.uint16)

    groundtruths = {
        'category_mask':
            tf.convert_to_tensor([category_mask]),
        'instance_mask':
            tf.convert_to_tensor([groundtruth_instance_mask]),
        'image_info':
            tf.convert_to_tensor([[[6, 6], [6, 6], [1.0, 1.0], [0, 0]]],
                                 dtype=tf.float32)
    }
    predictions = {
        'category_mask': tf.convert_to_tensor([category_mask]),
        'instance_mask': tf.convert_to_tensor([good_det_instance_mask])
    }

    pq_evaluator = panoptic_quality_evaluator.PanopticQualityEvaluator(
        num_categories=1,
        ignored_label=2,
        max_instances_per_category=16,
        offset=16,
        rescale_predictions=True)
    for _ in range(2):
      pq_evaluator.update_state(groundtruths, predictions)

    bad_det_instance_mask = np.array([
        [1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1],
        [1, 1, 1, 2, 2, 1],
        [1, 1, 1, 2, 2, 1],
        [1, 1, 1, 2, 2, 1],
        [1, 1, 1, 1, 1, 1],
    ],
                                     dtype=np.uint16)
    predictions['instance_mask'] = tf.convert_to_tensor([bad_det_instance_mask])
    for _ in range(2):
      pq_evaluator.update_state(groundtruths, predictions)

    results = pq_evaluator.result()
    np.testing.assert_array_equal(results['pq_per_class'],
                                  [((28 / 30 + 6 / 8) + (27 / 32)) / 2 / 2])
    np.testing.assert_array_equal(results['rq_per_class'], [3 / 4])
    np.testing.assert_array_equal(results['sq_per_class'],
                                  [((28 / 30 + 6 / 8) + (27 / 32)) / 3])
    self.assertAlmostEqual(results['All_pq'], 0.63177083)
    self.assertAlmostEqual(results['All_rq'], 0.75)
    self.assertAlmostEqual(results['All_sq'], 0.84236111)
    self.assertEqual(results['All_num_categories'], 1)


if __name__ == '__main__':
  tf.test.main()
