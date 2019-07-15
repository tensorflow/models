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

import os
import numpy as np

from absl import app as absl_app
from absl import flags

import tensorflow as tf


BATCH_SIZE = 64
EPOCHS = 10
EMBEDDING_DIM = 256
RNN_UNITS = 1024
SEQ_LENGTH = 100


def define_flags():
  """Define the flags for the Shakespeare character LSTM."""
  flags.DEFINE_string(
      name='model_dir', default=None,
      help='Directory for model check points.')
  flags.DEFINE_boolean(
      name='train', default=True,
      help='If true trains the model.')
  flags.DEFINE_string(
      name='predict_context', default=None,
      help='If set, makes a prediction with the given context.')
  flags.DEFINE_integer(
      name='predict_length', default=1000,
      help='Length of the predicted text including the context.')
  flags.DEFINE_string(
      name='training_data', default=None,
      help='Path to file containing the training data.')


def get_dataset(path_to_file, seq_length=SEQ_LENGTH):
  """Creates a dataset from a given text file.

  Args:
    path_to_file: The path to the training data.
    seq_length: The length of the LSTM sequence.

  Returns:
    A tuple, consisting of the Dataset and the class to character mapping
    and character to class mapping.
  """
  text = open(path_to_file, 'rb').read().decode(encoding='utf-8')

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
  dataset = dataset.shuffle(10000).batch(BATCH_SIZE, drop_remainder=True)

  return dataset, idx2char, char2idx


def build_model(vocab_size,
                embedding_dim=EMBEDDING_DIM,
                rnn_units=RNN_UNITS,
                batch_size=BATCH_SIZE,
                stateful=False):
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
  batch_shape = [batch_size if stateful else None, None]
  return tf.keras.Sequential([
      tf.keras.layers.Embedding(vocab_size, embedding_dim,
                                batch_input_shape=batch_shape),
      tf.keras.layers.LSTM(rnn_units,
                           return_sequences=True,
                           stateful=stateful,
                           recurrent_initializer='glorot_uniform'),
      tf.keras.layers.Dense(vocab_size, activation='softmax')])


def train_model(dataset, vocab_size, checkpoint_dir=None):
  """Trains a Shakespeare model.

  Args:
    dataset: the training data set.
    vocab_size: the number of unique character classes.
    checkpoint_dir: if not None, the directory in which to make checkpoints.

  Returns:
    The training history.
  """
  strategy = tf.distribute.MirroredStrategy()

  with strategy.scope():
    model = build_model(vocab_size=vocab_size)
    model.compile(optimizer=tf.keras.optimizers.Adam(),
                  loss=tf.keras.losses.CategoricalCrossentropy(),
                  metrics=[
                      tf.keras.metrics.Recall(top_k=1, name='RecallAt1'),
                      tf.keras.metrics.Recall(top_k=5, name='RecallAt5')])

  callbacks = []
  if checkpoint_dir:
    checkpoint_prefix = os.path.join(checkpoint_dir, 'ckpt_{epoch}')
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_prefix,
        save_weights_only=True)
    callbacks.append(checkpoint_callback)

  return model.fit(dataset, epochs=EPOCHS, callbacks=callbacks)


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


def main(_):
  flags_obj = flags.FLAGS

  if not flags_obj.training_data:
    raise ValueError(
        'Must set the path to a training data file. e.g download the following '
        'https://storage.googleapis.com/download.tensorflow.org/data/'
        'shakespeare.txt')
  dataset, idx2char, char2idx = get_dataset(flags_obj.training_data)

  if flags_obj.train:
    train_model(dataset, len(idx2char), flags_obj.model_dir)

  if flags_obj.predict_context:
    if not flags_obj.model_dir:
      raise ValueError('Must set model_dir to get predictions.')
    print(make_prediction(flags_obj.model_dir,
                          flags_obj.predict_length,
                          flags_obj.predict_context,
                          idx2char,
                          char2idx))


if __name__ == '__main__':
  define_flags()
  absl_app.run(main)
