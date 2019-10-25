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

"""Run training and evaluation for CVT text models."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from base import configure
from base import utils
from training import trainer
from training import training_progress


FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('mode', 'train', '"train" or "eval')
tf.app.flags.DEFINE_string('model_name', 'default_model',
                           'A name identifying the model being '
                           'trained/evaluated')


def main():
  utils.heading('SETUP')
  config = configure.Config(mode=FLAGS.mode, model_name=FLAGS.model_name)
  config.write()
  with tf.Graph().as_default() as graph:
    model_trainer = trainer.Trainer(config)
    summary_writer = tf.summary.FileWriter(config.summaries_dir)
    checkpoints_saver = tf.train.Saver(max_to_keep=1)
    best_model_saver = tf.train.Saver(max_to_keep=1)
    init_op = tf.global_variables_initializer()
    graph.finalize()
    with tf.Session() as sess:
      sess.run(init_op)
      progress = training_progress.TrainingProgress(
          config, sess, checkpoints_saver, best_model_saver,
          config.mode == 'train')
      utils.log()
      if config.mode == 'train':
        utils.heading('START TRAINING ({:})'.format(config.model_name))
        model_trainer.train(sess, progress, summary_writer)
      elif config.mode == 'eval':
        utils.heading('RUN EVALUATION ({:})'.format(config.model_name))
        progress.best_model_saver.restore(sess, tf.train.latest_checkpoint(
            config.checkpoints_dir))
        model_trainer.evaluate_all_tasks(sess, summary_writer, None)
      else:
        raise ValueError('Mode must be "train" or "eval"')


if __name__ == '__main__':
  main()
