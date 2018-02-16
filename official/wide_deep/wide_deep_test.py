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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import tensorflow as tf

import wide_deep

tf.logging.set_verbosity(tf.logging.ERROR)

TEST_INPUT = ('18,Self-emp-not-inc,987,Bachelors,12,Married-civ-spouse,abc,'
    'Husband,zyx,wvu,34,56,78,tsr,<=50K')

TEST_INPUT_VALUES = {
    'age': 18,
    'education_num': 12,
    'capital_gain': 34,
    'capital_loss': 56,
    'hours_per_week': 78,
    'education': 'Bachelors',
    'marital_status': 'Married-civ-spouse',
    'relationship': 'Husband',
    'workclass': 'Self-emp-not-inc',
    'occupation': 'abc',
}

TEST_CSV = os.path.join(os.path.dirname(__file__), 'wide_deep_test.csv')


class BaseTest(tf.test.TestCase):

  def setUp(self):
    # Create temporary CSV file
    self.temp_dir = self.get_temp_dir()
    self.input_csv = os.path.join(self.temp_dir, 'test.csv')
    with tf.gfile.Open(self.input_csv, 'w') as temp_csv:
      temp_csv.write(TEST_INPUT)

  def test_input_fn(self):
    dataset = wide_deep.input_fn(self.input_csv, 1, False, 1)
    features, labels = dataset.make_one_shot_iterator().get_next()

    with tf.Session() as sess:
      features, labels = sess.run((features, labels))

      # Compare the two features dictionaries.
      for key in TEST_INPUT_VALUES:
        self.assertTrue(key in features)
        self.assertEqual(len(features[key]), 1)
        feature_value = features[key][0]

        # Convert from bytes to string for Python 3.
        if isinstance(feature_value, bytes):
          feature_value = feature_value.decode()

        self.assertEqual(TEST_INPUT_VALUES[key], feature_value)

      self.assertFalse(labels)

  def build_and_test_estimator(self, model_type):
    """Ensure that model trains and minimizes loss."""
    model = wide_deep.build_estimator(self.temp_dir, model_type)

    # Train for 1 step to initialize model and evaluate initial loss
    model.train(
        input_fn=lambda: wide_deep.input_fn(
            TEST_CSV, num_epochs=1, shuffle=True, batch_size=1),
        steps=1)
    initial_results = model.evaluate(
        input_fn=lambda: wide_deep.input_fn(
            TEST_CSV, num_epochs=1, shuffle=False, batch_size=1))

    # Train for 100 epochs at batch size 3 and evaluate final loss
    model.train(
        input_fn=lambda: wide_deep.input_fn(
            TEST_CSV, num_epochs=100, shuffle=True, batch_size=3))
    final_results = model.evaluate(
        input_fn=lambda: wide_deep.input_fn(
            TEST_CSV, num_epochs=1, shuffle=False, batch_size=1))

    print('%s initial results:' % model_type, initial_results)
    print('%s final results:' % model_type, final_results)

    # Ensure loss has decreased, while accuracy and both AUCs have increased.
    self.assertLess(final_results['loss'], initial_results['loss'])
    self.assertGreater(final_results['auc'], initial_results['auc'])
    self.assertGreater(final_results['auc_precision_recall'],
                       initial_results['auc_precision_recall'])
    self.assertGreater(final_results['accuracy'], initial_results['accuracy'])

  def test_wide_deep_estimator_training(self):
    self.build_and_test_estimator('wide_deep')


if __name__ == '__main__':
  tf.test.main()
