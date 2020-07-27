# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================

import logging
from typing import Text, Any, Mapping

import tensorflow as tf

from absl import app
from absl import flags

from research.mobilenet import mobilenet_trainer
from research.mobilenet.tf1_loader import v1_loader
from research.mobilenet.tf1_loader import v2_loader
from research.mobilenet.tf1_loader import v3_loader


def _get_model_loader() -> Mapping[Text, Any]:
  return {
    'mobilenet_v1': v1_loader.load_mobilenet_v1,
    'mobilenet_v2': v2_loader.load_mobilenet_v2,
    'mobilenet_v3_small': v3_loader.load_mobilenet_v3,
    'mobilenet_v3_large': v3_loader.load_mobilenet_v3,
    'mobilenet_v3_edge_tpu': v3_loader.load_mobilenet_v3
  }


def get_flags():
  """Initialize the data extraction parameters.

  Define the arguments with the default values and parses the arguments
  passed to the main program.

  """
  flags.DEFINE_string(
    'model_name',
    help='MobileNet version name: mobilenet_v1, mobilenet_v2, '
         'mobilenet_v3_small, mobilenet_v3_large, mobilenet_v3_edge_tpu',
    default='mobilenet_v1'
  )
  flags.DEFINE_string(
    'dataset_name',
    help='Dataset name from TDFS to train on: imagenette, imagenet2012',
    default='imagenette'
  )
  flags.DEFINE_string(
    'data_dir',
    help='Directory for evaluation data.',
    default=None
  )
  flags.DEFINE_string(
    'checkpoint_path',
    help='Path of tf1 checkpoint.',
    default=None
  )


def evaluate(params: flags.FlagValues):
  m_config = mobilenet_trainer._get_model_config().get(params.model_name)()
  d_config = mobilenet_trainer._get_dataset_config().get(params.dataset_name)()
  model_load_function = _get_model_loader().get(params.model_name)

  # build evaluation dataset
  d_config.split = 'validation'
  d_config.batch_size = 128
  d_config.one_hot = False
  if params.data_dir:
    d_config.data_dir = params.data_dir

  # the checkpoint is trained using slim
  eval_dataset = mobilenet_trainer.get_dataset(d_config, slim_preprocess=True)

  # build the model
  if not model_load_function:
    raise ValueError('The model {} is not supported.'.format(params.model_name))

  keras_model = model_load_function(
    checkpoint_path=params.checkpoint_path,
    config=m_config)

  # compile model
  if d_config.one_hot:
    loss_obj = tf.keras.losses.CategoricalCrossentropy()
  else:
    loss_obj = tf.keras.losses.SparseCategoricalCrossentropy()

  keras_model.compile(
    optimizer='rmsprop',
    loss=loss_obj,
    metrics=[mobilenet_trainer._get_metrics(one_hot=d_config.one_hot)['acc']])

  # run evaluation
  eval_result = keras_model.evaluate(eval_dataset)

  return eval_result


def main(_):
  eval_result = evaluate(flags.FLAGS)
  if eval_result:
    logging.info('The evaluation result is:{}'.format(eval_result))


if __name__ == '__main__':
  logging.basicConfig(
    format='%(asctime)-15s:%(levelname)s:%(module)s:%(message)s',
    level=logging.INFO)
  get_flags()
  app.run(main)
