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
"""Model eval separate from training."""
from tensorflow import app
from tensorflow.python.platform import flags

import vgsl_model

flags.DEFINE_string('eval_dir', '/tmp/mdir/eval',
                    'Directory where to write event logs.')
flags.DEFINE_string('graph_def_file', None,
                    'Output eval graph definition file.')
flags.DEFINE_string('train_dir', '/tmp/mdir',
                    'Directory where to find training checkpoints.')
flags.DEFINE_string('model_str',
                    '1,150,600,3[S2(4x150)0,2 Ct5,5,16 Mp2,2 Ct5,5,64 Mp3,3'
                    '([Lrys64 Lbx128][Lbys64 Lbx128][Lfys64 Lbx128])S3(3x0)2,3'
                    'Lfx128 Lrx128 S0(1x4)0,3 Do Lfx256]O1c134',
                    'Network description.')
flags.DEFINE_integer('num_steps', 1000, 'Number of steps to run evaluation.')
flags.DEFINE_integer('eval_interval_secs', 60,
                     'Time interval between eval runs.')
flags.DEFINE_string('eval_data', None, 'Evaluation data filepattern')
flags.DEFINE_string('decoder', None, 'Charset decoder')

FLAGS = flags.FLAGS


def main(argv):
  del argv
  vgsl_model.Eval(FLAGS.train_dir, FLAGS.eval_dir, FLAGS.model_str,
                  FLAGS.eval_data, FLAGS.decoder, FLAGS.num_steps,
                  FLAGS.graph_def_file, FLAGS.eval_interval_secs)


if __name__ == '__main__':
  app.run()
