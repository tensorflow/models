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

r"""Evaluates image classification accuracy using tflite_imagenet_evaluator.

Usage:
tflite_imagenet_evaluator_run --tflite_model_path=/PATH/TO/MODEL.tflite
"""

from typing import Sequence
from absl import app
from absl import flags
import tensorflow as tf, tf_keras

from official.core import exp_factory
from official.projects.edgetpu.vision.serving import tflite_imagenet_evaluator
from official.projects.edgetpu.vision.tasks import image_classification


flags.DEFINE_string('tflite_model_path', None,
                    'Path to the tflite file to be evaluated.')
flags.DEFINE_integer('num_threads', 16, 'Number of local threads.')
flags.DEFINE_integer('batch_size', 256, 'Batch size per thread.')
flags.DEFINE_string(
    'model_name', 'mobilenet_edgetpu_v2_xs',
    'Model name to identify a registered data pipeline setup and use as the '
    'validation dataset.')

FLAGS = flags.FLAGS


def main(argv: Sequence[str]):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  with tf.io.gfile.GFile(FLAGS.tflite_model_path, 'rb') as f:
    model_content = f.read()

  config = exp_factory.get_exp_config(FLAGS.model_name)
  global_batch_size = FLAGS.num_threads * FLAGS.batch_size
  config.task.validation_data.global_batch_size = global_batch_size
  config.task.validation_data.dtype = 'float32'

  task = image_classification.EdgeTPUTask(config.task)
  dataset = task.build_inputs(config.task.validation_data)

  evaluator = tflite_imagenet_evaluator.AccuracyEvaluator(
      model_content=model_content,
      dataset=dataset,
      num_threads=FLAGS.num_threads)

  evals, corrects = evaluator.evaluate_all()
  accuracy = 100.0 * corrects / evals if evals > 0 else 0
  print('Final accuracy: {}, Evaluated: {}, Correct: {} '.format(
      accuracy, evals, corrects))


if __name__ == '__main__':
  flags.mark_flag_as_required('tflite_model_path')
  app.run(main)
