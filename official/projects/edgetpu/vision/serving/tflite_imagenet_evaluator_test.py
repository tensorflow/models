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

"""Tests for tflite_imagenet_evaluator."""

from unittest import mock
import tensorflow as tf

from official.core import exp_factory
from official.projects.edgetpu.vision.serving import tflite_imagenet_evaluator
from official.projects.edgetpu.vision.tasks import image_classification


class TfliteImagenetEvaluatorTest(tf.test.TestCase):

  # Only tests the parallelization aspect. Mocks image evaluation and dataset.
  def test_evaluate_all(self):
    batch_size = 8
    num_threads = 4
    global_batch_size = num_threads * batch_size

    config = exp_factory.get_exp_config('mobilenet_edgetpu_v2_xs')
    config.task.validation_data.global_batch_size = global_batch_size
    config.task.validation_data.dtype = 'float32'

    task = image_classification.EdgeTPUTask(config.task)
    dataset = task.build_inputs(config.task.validation_data)

    num_batches = 5
    with mock.patch.object(
        tflite_imagenet_evaluator.AccuracyEvaluator,
        'evaluate_single_image',
        return_value=True,
        autospec=True):
      evaluator = tflite_imagenet_evaluator.AccuracyEvaluator(
          model_content='MockModelContent'.encode('utf-8'),
          dataset=dataset.take(num_batches),
          num_threads=num_threads)
      num_evals, num_corrects = evaluator.evaluate_all()

    expected_evals = num_batches * num_threads * batch_size

    self.assertEqual(num_evals, expected_evals)
    self.assertEqual(num_corrects, expected_evals)


if __name__ == '__main__':
  tf.test.main()
