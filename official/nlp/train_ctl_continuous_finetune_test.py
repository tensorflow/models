# Lint as: python3
# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
import os

# Import libraries

from absl import flags
from absl.testing import flagsaver
import tensorflow as tf
from official.common import flags as tfm_flags
from official.core import task_factory
from official.core import train_lib
from official.core import train_utils
from official.nlp import train_ctl_continuous_finetune

FLAGS = flags.FLAGS

tfm_flags.define_flags()


class ContinuousFinetuneTest(tf.test.TestCase):

  def setUp(self):
    super().setUp()
    self._model_dir = os.path.join(self.get_temp_dir(), 'model_dir')

  @flagsaver.flagsaver
  def testTrainCtl(self):
    src_model_dir = self.get_temp_dir()
    flags_dict = dict(
        experiment='mock',
        mode='continuous_train_and_eval',
        model_dir=self._model_dir,
        params_override={
            'task': {
                'init_checkpoint': src_model_dir,
            },
            'trainer': {
                'continuous_eval_timeout': 1,
                'steps_per_loop': 1,
                'train_steps': 1,
                'validation_steps': 1,
                'best_checkpoint_export_subdir': 'best_ckpt',
                'best_checkpoint_eval_metric': 'acc',
                'optimizer_config': {
                    'optimizer': {
                        'type': 'sgd'
                    },
                    'learning_rate': {
                        'type': 'constant'
                    }
                }
            }
        })

    with flagsaver.flagsaver(**flags_dict):
      # Train and save some checkpoints.
      params = train_utils.parse_configuration(flags.FLAGS)
      distribution_strategy = tf.distribute.get_strategy()
      with distribution_strategy.scope():
        task = task_factory.get_task(params.task, logging_dir=src_model_dir)
      _ = train_lib.run_experiment(
          distribution_strategy=distribution_strategy,
          task=task,
          mode='train',
          params=params,
          model_dir=src_model_dir)

      params = train_utils.parse_configuration(FLAGS)
      eval_metrics = train_ctl_continuous_finetune.run_continuous_finetune(
          FLAGS.mode, params, FLAGS.model_dir, run_post_eval=True)
      self.assertIn('best_acc', eval_metrics)


if __name__ == '__main__':
  tf.test.main()
