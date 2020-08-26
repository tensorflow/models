# Copyright 2020 The TensorFlow Authors All Rights Reserved.
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

"""Script to train langid model.

The script builds language detection from wikipedia dataset,
builds SGNN model to train an on-device model to
predict the language of the given text.
"""

import os
from absl import app
from absl import flags
import numpy as np
import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds

import sgnn # import sequence_projection module

FLAGS = flags.FLAGS
flags.DEFINE_string('output_dir', '/tmp/langid',
                    'Path for the output directory.')

flags.DEFINE_integer('projection_size', 600, 'Size of projection layer.')
flags.DEFINE_integer('ngram_size', 3, 'Max size of ngram to project features.')
flags.DEFINE_string('fc_layer', '256,128',
                    'Size of fully connected layer, separated by comma.')

flags.DEFINE_integer('batch_size', 160, 'Batch size for training.')
flags.DEFINE_integer('epochs', 10, 'Num of epochs for training.')
flags.DEFINE_float('learning_rate', 2e-4, 'learning rate for optimizer.')

LANGIDS = ['ar', 'en', 'es', 'fr', 'ru', 'zh']


def dataset_fn(batch_size, is_training, split, try_gcs, max_input_len):
  """Creates dataset to train and evaluate.

  Args:
    batch_size: Batch size for training or evaluation.
    is_training: True if the dataset is for training.
    split: Split of dataset, follow the pattern defined in
      https://www.tensorflow.org/datasets/splits
    try_gcs: True if loading the data from gcs.
    max_input_len: Max length of input string.

  Returns:
    Dataset object.
  """

  def _get_text(item):
    return tf.strings.substr(item['text'], 0, max_input_len)

  all_data = []
  for idx, langid in enumerate(LANGIDS):
    dataset = tfds.load(
        'wikipedia/20190301.%s' % langid, try_gcs=try_gcs, split=split)

    map_fn = lambda item: (_get_text(item), idx)  # pylint: disable=cell-var-from-loop
    dataset = dataset.map(map_fn)
    all_data.append(dataset)

  datasets = tf.data.experimental.sample_from_datasets(
      all_data, [1. / len(all_data)] * len(LANGIDS))
  repeat_count = None if is_training else 1
  return datasets.cache().shuffle(100000).batch(batch_size).repeat(repeat_count)


def save_and_convert(model, output_dir):
  """Save keras model and convert to tflite."""
  saved_model_path = os.path.join(output_dir, 'saved_model')
  tf.saved_model.save(model, saved_model_path)
  converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_path)
  converter.allow_custom_ops = True
  converter.target_spec.supported_ops = [
      tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS
  ]
  data = converter.convert()
  with open(os.path.join(output_dir, 'model.tflite'), 'wb') as f:
    f.write(data)


def train_and_evaluate():
  """Train and evaluate the model."""
  hash_seed = np.random.uniform(-1, 1, FLAGS.projection_size) * 0x7FFFFFFF
  fc_layer = [int(fc) for fc in FLAGS.fc_layer.split(',')]
  fc_layer.append(len(LANGIDS) + 1)
  hparams = sgnn.Hparams(learning_rate=FLAGS.learning_rate)

  model = sgnn.keras_model(hash_seed, FLAGS.ngram_size, fc_layer, hparams)
  model.fit(
      dataset_fn(FLAGS.batch_size, True, 'train[:10%]', True, 100),
      epochs=FLAGS.epochs,
      steps_per_epoch=1000,
      validation_steps=100,
      validation_data=dataset_fn(FLAGS.batch_size, False, 'train[10:11%]', True,
                                 100),
  )
  save_and_convert(model, FLAGS.output_dir)


def main(_):
  if not os.path.exists(FLAGS.output_dir):
    os.mkdir(FLAGS.output_dir)

  train_and_evaluate()


if __name__ == '__main__':
  app.run(main)
