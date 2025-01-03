# Copyright 2017 The TensorFlow Authors All Rights Reserved.
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

"""Define flags are common for both train.py and eval.py scripts."""
import logging
import sys

from tensorflow.compat.v1 import flags

import datasets
import model

FLAGS = flags.FLAGS

logging.basicConfig(
    level=logging.DEBUG,
    stream=sys.stderr,
    format='%(levelname)s '
    '%(asctime)s.%(msecs)06d: '
    '%(filename)s: '
    '%(lineno)d '
    '%(message)s',
    datefmt='%Y-%m-%d %H:%M:%S')


_common_flags_defined = False

def define():
  """Define common flags."""
  # yapf: disable
  # common_flags.define() may be called multiple times in unit tests.
  global _common_flags_defined
  if _common_flags_defined:
    return
  _common_flags_defined = True

  flags.DEFINE_integer('batch_size', 32,
                       'Batch size.')

  flags.DEFINE_integer('crop_width', None,
                       'Width of the central crop for images.')

  flags.DEFINE_integer('crop_height', None,
                       'Height of the central crop for images.')

  flags.DEFINE_string('train_log_dir', '/tmp/attention_ocr/train',
                      'Directory where to write event logs.')

  flags.DEFINE_string('dataset_name', 'fsns',
                      'Name of the dataset. Supported: fsns')

  flags.DEFINE_string('split_name', 'train',
                      'Dataset split name to run evaluation for: test,train.')

  flags.DEFINE_string('dataset_dir', None,
                      'Dataset root folder.')

  flags.DEFINE_string('checkpoint', '',
                      'Path for checkpoint to restore weights from.')

  flags.DEFINE_string('master',
                      '',
                      'BNS name of the TensorFlow master to use.')

  # Model hyper parameters
  flags.DEFINE_float('learning_rate', 0.004,
                     'learning rate')

  flags.DEFINE_string('optimizer', 'momentum',
                      'the optimizer to use')

  flags.DEFINE_float('momentum', 0.9,
                     'momentum value for the momentum optimizer if used')

  flags.DEFINE_bool('use_augment_input', True,
                    'If True will use image augmentation')

  # Method hyper parameters
  # conv_tower_fn
  flags.DEFINE_string('final_endpoint', 'Mixed_5d',
                      'Endpoint to cut inception tower')

  # sequence_logit_fn
  flags.DEFINE_bool('use_attention', True,
                    'If True will use the attention mechanism')

  flags.DEFINE_bool('use_autoregression', True,
                    'If True will use autoregression (a feedback link)')

  flags.DEFINE_integer('num_lstm_units', 256,
                       'number of LSTM units for sequence LSTM')

  flags.DEFINE_float('weight_decay', 0.00004,
                     'weight decay for char prediction FC layers')

  flags.DEFINE_float('lstm_state_clip_value', 10.0,
                     'cell state is clipped by this value prior to the cell'
                     ' output activation')

  # 'sequence_loss_fn'
  flags.DEFINE_float('label_smoothing', 0.1,
                     'weight for label smoothing')

  flags.DEFINE_bool('ignore_nulls', True,
                    'ignore null characters for computing the loss')

  flags.DEFINE_bool('average_across_timesteps', False,
                    'divide the returned cost by the total label weight')
  # yapf: enable


def get_crop_size():
  if FLAGS.crop_width and FLAGS.crop_height:
    return (FLAGS.crop_width, FLAGS.crop_height)
  else:
    return None


def create_dataset(split_name):
  ds_module = getattr(datasets, FLAGS.dataset_name)
  return ds_module.get_split(split_name, dataset_dir=FLAGS.dataset_dir)


def create_mparams():
  return {
      'conv_tower_fn':
      model.ConvTowerParams(final_endpoint=FLAGS.final_endpoint),
      'sequence_logit_fn':
      model.SequenceLogitsParams(
          use_attention=FLAGS.use_attention,
          use_autoregression=FLAGS.use_autoregression,
          num_lstm_units=FLAGS.num_lstm_units,
          weight_decay=FLAGS.weight_decay,
          lstm_state_clip_value=FLAGS.lstm_state_clip_value),
      'sequence_loss_fn':
      model.SequenceLossParams(
          label_smoothing=FLAGS.label_smoothing,
          ignore_nulls=FLAGS.ignore_nulls,
          average_across_timesteps=FLAGS.average_across_timesteps)
  }


def create_model(*args, **kwargs):
  ocr_model = model.Model(mparams=create_mparams(), *args, **kwargs)
  return ocr_model
