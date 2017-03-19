# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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
"""Model trainer for single or multi-replica training."""
from tensorflow import app
from tensorflow.python.platform import flags

import vgsl_model

flags.DEFINE_string('master', '', 'Name of the TensorFlow master to use.')
flags.DEFINE_string('train_dir', '/tmp/mdir',
                    'Directory where to write event logs.')
flags.DEFINE_string('model_str',
                    '1,150,600,3[S2(4x150)0,2 Ct5,5,16 Mp2,2 Ct5,5,64 Mp3,3'
                    '([Lrys64 Lbx128][Lbys64 Lbx128][Lfys64 Lbx128])S3(3x0)2,3'
                    'Lfx128 Lrx128 S0(1x4)0,3 Do Lfx256]O1c134',
                    'Network description.')
flags.DEFINE_integer('max_steps', 10000, 'Number of steps to train for.')
flags.DEFINE_integer('task', 0, 'Task id of the replica running the training.')
flags.DEFINE_integer('ps_tasks', 0, 'Number of tasks in the ps job.'
                     'If 0 no ps job is used.')
flags.DEFINE_string('train_data', None, 'Training data filepattern')
flags.DEFINE_float('initial_learning_rate', 0.00002, 'Initial learning rate')
flags.DEFINE_float('final_learning_rate', 0.00002, 'Final learning rate')
flags.DEFINE_integer('learning_rate_halflife', 1600000,
                     'Halflife of learning rate')
flags.DEFINE_string('optimizer_type', 'Adam',
                    'Optimizer from:GradientDescent, AdaGrad, Momentum, Adam')
flags.DEFINE_integer('num_preprocess_threads', 4, 'Number of input threads')

FLAGS = flags.FLAGS


def main(argv):
  del argv
  vgsl_model.Train(FLAGS.train_dir, FLAGS.model_str, FLAGS.train_data,
                   FLAGS.max_steps, FLAGS.master, FLAGS.task, FLAGS.ps_tasks,
                   FLAGS.initial_learning_rate, FLAGS.final_learning_rate,
                   FLAGS.learning_rate_halflife, FLAGS.optimizer_type,
                   FLAGS.num_preprocess_threads)


if __name__ == '__main__':
  app.run()
