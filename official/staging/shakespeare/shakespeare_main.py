# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
"""Runs a character LSTM model trained on Shakespeare."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import os

# pylint: disable=wrong-import-order
from absl import app
from absl import flags
import numpy as np
import tensorflow as tf
# pylint: enable=wrong-import-order

from official.utils.flags import core as flags_core
from official.utils.misc import distribution_utils
from official.utils.misc import keras_utils

EMBEDDING_DIM = 256
RNN_UNITS = 1024
SEQ_LENGTH = 100
# Calculated by running batch_size=1
BATCHES_PER_EPOCH = 11043


def define_flags():
  """Define the flags for the Shakespeare character LSTM."""
  flags_core.define_base(data_dir=False,
                         clean=False,
                         train_epochs=True,
                         epochs_between_evals=False,
                         stop_threshold=False,
                         num_gpu=True,
                         hooks=False,
                         export_dir=False,
                         run_eagerly=True,
                         distribution_strategy=True)

  flags_core.define_performance(num_parallel_calls=False,
                                inter_op=False,
                                intra_op=False,
                                synthetic_data=False,
                                max_train_steps=False,
                                dtype=True,
                                loss_scale=True,
                                enable_xla=True,
                                force_v2_in_keras_compile=True)

  flags_core.set_defaults(train_epochs=43,
                          batch_size=64)

  flags.DEFINE_boolean(name='enable_eager', default=True, help='Enable eager?')
  flags.DEFINE_boolean(
      name='train', default=True,
      help='If true trains the model.')
  flags.DEFINE_string(
      name='predict_context', default=None,
      help='If set, makes a prediction with the given context.')
  flags.DEFINE_integer(
      name='predict_length', default=1000,
      help='Length of the predicted text including the context.')
  flags.DEFINE_integer(
      name='log_steps', default=100,
      help='For every log_steps, we log the timing information such as '
      'examples per second.')
  flags.DEFINE_string(
      name='training_data', default=None,
      help='Path to file containing the training data.')
  flags.DEFINE_boolean(name='cudnn', default=True, help='Use CuDNN LSTM.')


def get_dataset(path_to_file, batch_size=None, seq_length=SEQ_LENGTH):
  """Creates a dataset from a given text file.

  Args:
    path_to_file: The path to the training data.
    batch_size: Batch size to use.
    seq_length: The length of the LSTM sequence.

  Returns:
    A tuple, consisting of the Dataset and the class to character mapping
    and character to class mapping.
  """
  with tf.io.gfile.GFile(path_to_file, 'rb') as train_data:
    text = train_data.read().decode(encoding='utf-8')

  # Create vocab
  vocab = sorted(set(text))
  char2idx = {u: i for i, u in enumerate(vocab)}
  idx2char = np.array(vocab)

  # Split text into sequence length + 1 chucks to create examples
  text_as_int = np.array([char2idx[c] for c in text])
  char_dataset = tf.data.Dataset.from_tensor_slices(text_as_int)
  sequences = char_dataset.batch(seq_length+1, drop_remainder=True)

  def split_input_target(chunk):
    input_text = chunk[:-1]
    target_text = chunk[1:]
    return input_text, tf.one_hot(target_text, len(vocab))
  dataset = sequences.map(split_input_target)
  dataset = dataset.shuffle(10000).repeat()
  dataset = dataset.batch(batch_size, drop_remainder=True)

  return dataset, idx2char, char2idx


def build_model(vocab_size,
                embedding_dim=EMBEDDING_DIM,
                rnn_units=RNN_UNITS,
                batch_size=None,
                stateful=False,
                use_cudnn=True):
  """Builds the Shakespeare model.

  Args:
    vocab_size: The number of character classes in the input.
    embedding_dim: The dimension of the embedding space for each class.
    rnn_units: The number of RNN units in the layer.
    batch_size: When predicting, the batch size of the predictions.
    stateful: If true, the LSTM is stateful.

  Returns:
    A Keras Model.
  """
  assert keras_utils.is_v2_0()
  LSTM = functools.partial(tf.keras.layers.LSTM, implementation=2)

  # By indirecting the activation through a lambda layer, the logic to dispatch
  # to CuDNN in V2 doesn't trigger and we force the LSTM to run in non-CuDNN
  # mode.
  lstm_activation = ('tanh' if use_cudnn else
                     lambda x: tf.math.tanh(x))

  batch_shape = [batch_size if stateful else None, None]
  return tf.keras.Sequential([
      tf.keras.layers.Embedding(vocab_size, embedding_dim,
                                batch_input_shape=batch_shape),
      LSTM(rnn_units,
           activation=lstm_activation,
           return_sequences=True,
           stateful=stateful,
           recurrent_initializer='glorot_uniform'),
      tf.keras.layers.Dense(vocab_size),
      tf.keras.layers.Softmax(dtype=tf.float32)])


def train_model(flags_obj, dataset, vocab_size, strategy, checkpoint_dir=None):
  """Trains a Shakespeare model.

  Args:
    flags_obj: An object containing parsed flag values.s
    dataset: the training data set.
    vocab_size: the number of unique character classes.
    strategy: distribution strategy to use.
    checkpoint_dir: if not None, the directory in which to make checkpoints.

  Returns:
    The training history and callbacks.
  """
  train_steps = BATCHES_PER_EPOCH // flags_obj.batch_size
  strategy_scope = distribution_utils.get_strategy_scope(strategy)

  with strategy_scope:
    model = build_model(vocab_size=vocab_size, batch_size=flags_obj.batch_size,
                        use_cudnn=flags_obj.cudnn)

   # When keras_use_ctl is False, Model.fit() automatically applies
   # loss scaling so we don't need to create a LossScaleOptimizer.
    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=tf.keras.losses.CategoricalCrossentropy(),
        metrics=[tf.keras.metrics.Recall(top_k=1, name='RecallAt1'),
                 tf.keras.metrics.Recall(top_k=5, name='RecallAt5')],
        run_eagerly=flags_obj.run_eagerly,
        experimental_run_tf_function=flags_obj.force_v2_in_keras_compile)

  callbacks = []
  if checkpoint_dir:
    checkpoint_prefix = os.path.join(checkpoint_dir, 'ckpt_{epoch}')
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_prefix,
        save_weights_only=True)
    callbacks.append(checkpoint_callback)
  time_callback = keras_utils.TimeHistory(flags_obj.batch_size,
                                          flags_obj.log_steps)
  callbacks.append(time_callback)
  history = model.fit(dataset,
                      epochs=flags_obj.train_epochs,
                      steps_per_epoch=train_steps,
                      callbacks=callbacks,
                      verbose=2)
  return history, callbacks


def make_prediction(checkpoint_dir, length, context, idx2char, char2idx):
  """Make predictions from a Shakespeare model.

  Args:
    checkpoint_dir: the directory from which to load checkpoints
    length: the total length of the generated text (including the context).
    context: the initial text with which the LSTM is primed.
    idx2char: the character class to character mapping.
    char2idx: the character to character class mapping.

  Returns:
    A generated string of text of the given length.
  """
  prediction_model = build_model(
      vocab_size=len(idx2char), batch_size=1, stateful=True)
  prediction_model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))
  prediction_model.build(tf.TensorShape([1, None]))

  input_eval = [char2idx[s] for s in context]
  input_eval = tf.expand_dims(input_eval, 0)

  text_generated = []

  prediction_model.reset_states()
  for _ in range(length - len(context)):
    predictions = prediction_model(input_eval)
    predictions = tf.squeeze(predictions, 0)

    # We applied a softmax to the output of the model so that
    # tf.keras.metrics.Recall would work. We need logits for
    # tf.random.categorical, so we convert the probabilities back to log odds
    predictions = tf.math.log(predictions / (1 - predictions))

    random_output = tf.random.categorical(predictions, num_samples=1)
    selected_id = random_output[-1, 0].numpy()
    input_eval = tf.expand_dims([selected_id], 0)
    text_generated.append(idx2char[selected_id])

  return context + ''.join(text_generated)


def run(flags_obj):
  """Run Shakespeare training and predict.

  Args:
    flags_obj: An object containing parsed flag values.

  Returns:
    Dictionary with status from the run.
  """
  if not flags_obj.training_data:
    raise ValueError(
        'Must set the path to a training data file. e.g download the following '
        'https://storage.googleapis.com/download.tensorflow.org/data/'
        'shakespeare.txt')

  if flags_obj.dtype == 'fp16':
    policy = tf.keras.mixed_precision.experimental.Policy(
        'mixed_float16',
        loss_scale=flags_core.get_loss_scale(flags_obj,
                                             default_for_fp16='dynamic'))
    tf.keras.mixed_precision.experimental.set_policy(policy)

  keras_utils.set_session_config(
      enable_eager=flags_obj.enable_eager,
      enable_xla=flags_obj.enable_xla)

  strategy = distribution_utils.get_distribution_strategy(
      distribution_strategy=flags_obj.distribution_strategy,
      num_gpus=flags_obj.num_gpus)

  dataset, idx2char, char2idx = get_dataset(flags_obj.training_data,
                                            batch_size=flags_obj.batch_size)
  stats = {}
  if flags_obj.train:
    history, callbacks = train_model(flags_obj, dataset,
                                     len(idx2char), strategy,
                                     checkpoint_dir=flags_obj.model_dir)

    stats['history'] = history.history
    stats['callbacks'] = callbacks

  if flags_obj.predict_context:
    if not flags_obj.model_dir:
      raise ValueError('Must set model_dir to get predictions.')
    print(make_prediction(flags_obj.model_dir,
                          flags_obj.predict_length,
                          flags_obj.predict_context,
                          idx2char,
                          char2idx))

  return stats


def main(_):
  flags_obj = flags.FLAGS
  run(flags_obj)


if __name__ == '__main__':
  define_flags()
  app.run(main)
