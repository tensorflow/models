# Copyright 2022 The TensorFlow Authors. All Rights Reserved.
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

"""Evalutes TriviaQA predictions."""
import json

from absl import app
from absl import flags
from absl import logging
import tensorflow as tf

from official.projects.triviaqa import evaluation

flags.DEFINE_string('gold_path', None,
                    'Path to golden validation, i.e. wikipedia-dev.json.')

flags.DEFINE_string('predictions_path', None,
                    'Path to predictions in JSON format')

FLAGS = flags.FLAGS


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')
  with tf.io.gfile.GFile(FLAGS.gold_path) as f:
    ground_truth = {
        datum['QuestionId']: datum['Answer'] for datum in json.load(f)['Data']
    }
  with tf.io.gfile.GFile(FLAGS.predictions_path) as f:
    predictions = json.load(f)
  logging.info(evaluation.evaluate_triviaqa(ground_truth, predictions))


if __name__ == '__main__':
  flags.mark_flag_as_required('predictions_path')
  app.run(main)
