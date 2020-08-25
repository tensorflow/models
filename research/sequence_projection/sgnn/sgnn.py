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

"""Builds SGNN model.

[1] Sujith Ravi and Zornitsa Kozareva. 2018. "Self-governing neural networks for
on-device short text
classification." In Proceedings of the 2018 Conference on Empirical Methods in
Natural Language
Processing, pages 887-893. Association for Computational Linguistics

The model will be constructed in this way:
* Projects text to float features, the size is defined by projection_size
* Fully connected layer predicts the class of predictions.
"""

import collections
import tensorflow.compat.v2 as tf
import tensorflow_text as tf_text

from tensorflow_lite_support.custom_ops.python import tflite_text_api

# Hparam collections that will be used to tune the model.
Hparams = collections.namedtuple(
    'Hparams',
    [
        # Learning rate for the optimizer.
        'learning_rate'
    ])


def preprocess(text):
  """Normalize the text, and return tokens."""
  assert len(text.get_shape().as_list()) == 2
  assert text.get_shape().as_list()[-1] == 1
  text = tf.reshape(text, [-1])
  text = tf_text.case_fold_utf8(text)
  tokenizer = tflite_text_api.WhitespaceTokenizer()
  return tokenizer.tokenize(text)


def get_ngrams(tokens, n):
  """Generates character ngrams from tokens.

  Args:
    tokens: A string ragged tensor for tokens, in shape of [batch_size,
      num_token].
    n: ngram size for char ngrams.

  Returns:
    A string ragged tensor for ngrams, in shape of [batch_size, num_token,
    ngrams].
  """
  chars_split = tf.strings.unicode_split('^' + tokens + '$', 'UTF-8')
  chars_joined = tflite_text_api.ngrams(
      chars_split,
      width=n,
      axis=-1,
      reduction_type=tf_text.Reduction.STRING_JOIN,
      string_separator='')
  flat_row_splits = tf.nn.embedding_lookup(chars_joined.values.row_splits,
                                           chars_joined.row_splits)
  return tf.RaggedTensor.from_row_splits(chars_joined.values.values,
                                         flat_row_splits)


def project(ngrams, hash_seed, buckets):
  """Projects a ngram RaggedTensor to float tensor.

  Args:
    ngrams: A string ragged tensor, in shape of [batch_size, num_token, ngrams].
    hash_seed: A python int list, in shape of [num_hash].
    buckets: An int for the max value of projected integers.

  Returns:
    A float tensor that projects ngrams to the space represented by hash_seed,
    in shape of [batch_size, num_hash].
  """
  num_hash = len(hash_seed)
  # Hash ngrams string tensor to hash signatures.
  signatures = tf.ragged.map_flat_values(tf.strings.to_hash_bucket_fast, ngrams,
                                         buckets)

  # Each ngram signature will be multiplied by a different hash seed,
  # mod by hash buckets, and linear mapping.
  # value = abs(signature * seed % bucket)
  # if value > bucket / 2: value -= buckets
  hash_tensor = tf.constant(hash_seed, dtype=tf.int64)
  value = tf.math.floormod(
      tf.abs(signatures.values * tf.reshape(hash_tensor, [-1, 1])), buckets)
  value = value - tf.cast(tf.greater(value, buckets >> 1), tf.int64) * buckets

  # Wrap values to ragged tensor, and calculates
  # output_i,j = mean(value_i,j,k) for k-th ngram in i-th text
  # computed with j-th hash seed
  row_lengths = tf.repeat(
      tf.reshape(signatures.row_lengths(), [1, -1]), num_hash, axis=0)
  row_lengths = tf.cast(tf.reshape(row_lengths, [-1]), tf.int32)
  result = tf.RaggedTensor.from_row_lengths(
      tf.RaggedTensor.from_row_lengths(tf.reshape(value, [-1]), row_lengths),
      tf.repeat(tf.shape(signatures.row_lengths()), num_hash))
  result = tf.reduce_mean(result, 2) / (buckets >> 1)
  return tf.transpose(tf.reshape(result.values, [num_hash, -1]))


def fused_project(ngrams, hash_seed, buckets):
  """A wrapper to fuse project method when converting to TFLite model.

  Args:
    ngrams: A string ragged tensor, in shape of [batch_size, num_token, ngrams].
    hash_seed: A python int list, in shape of [num_hash].
    buckets: An int for the max value of projected integers.

  Returns:
    A float tensor that projects ngrams to the space represented by hash_seed,
    in shape of [batch_size, num_hash].
  """
  hash_seed_attr = ' '.join(['i: %d' % seed for seed in hash_seed])
  experimental_implements = [
      'name: "tftext:custom:SgnnProjection"',
      'attr { key: "hash_seed" value { list {%s} } }' % hash_seed_attr,
      'attr { key: "buckets" value { i: %d } }' % buckets,
  ]
  experimental_implements = ' '.join(experimental_implements)

  @tf.function(experimental_implements=experimental_implements)
  def func(ngrams_values, *ngrams_row_splits):
    ngrams = tf.RaggedTensor.from_nested_row_splits(
        flat_values=ngrams_values, nested_row_splits=ngrams_row_splits)
    return project(ngrams, hash_seed, buckets)
  return func(ngrams.flat_values, *ngrams.nested_row_splits)


def sgnn(texts, hash_seed, ngram_size):
  """Projects the string text to float features.

  It first generasts N ngrams of the tokens from given text,
  then projects each ngram tensor with a partion of the seeds.

  Args:
    texts: a string tensor, in shape of [batch_size].
    hash_seed: a list of integers, in shape of [projection_size].
    ngram_size: max size of ngram to generate features.

  Returns:
    A float tensor that projects ngrams to the space represented by hash_seed,
    in shape of [batch_size, projection_size].
  """
  projection_size = len(hash_seed)
  partition_size = int(projection_size / ((ngram_size + 1) * ngram_size / 2))
  if partition_size == 0:
    raise ValueError(
        'projection size %d is not enough for %d ngram partitions' %
        (projection_size, ngram_size))
  indices = [int(i * (i + 1) / 2) * partition_size for i in range(ngram_size)]
  indices.append(projection_size)
  projection_layer = []
  tokens = preprocess(texts)

  for i in range(ngram_size):
    ngram = get_ngrams(tokens, i + 1)
    projection = fused_project(ngram, hash_seed[indices[i]:indices[i + 1]],
                               0x7FFFFFFF)
    projection_layer.append(projection)

  return tf.cast(tf.concat(projection_layer, -1), tf.float32)


class ProjectLayer(tf.keras.layers.Layer):
  """Projects the texts to a fixed sized features."""

  def __init__(self, seed, ngram_size, **kwargs):
    self.seed = seed
    self.ngram_size = ngram_size
    super(ProjectLayer, self).__init__(**kwargs)

  def get_config(self):
    return {
        'seed': self.seed,
        'ngram_size': self.ngram_size,
    }

  def call(self, x):
    return sgnn(x, self.seed, self.ngram_size)

  def compute_output_shape(self, input_shape):
    return (input_shape[0], len(self.seed))


def keras_model(hash_seed, ngram_size, fc_size_list, hparams):
  """Compiles a keras model from projected features to labels.

  Args:
    hash_seed: a list of int used to project the feature.
    ngram_size: maximum size of ngram to generate features from texts.
    fc_size_list: a list of int, sizes of each fully connected layer.
    hparams: hyper parameters for the model.

  Returns:
    A keras model that predicts the language id.

  """
  if not fc_size_list:
    raise ValueError(
        'Must specify one or more fully connected layers via fc_size_list')
  model = tf.keras.Sequential()
  model.add(ProjectLayer(hash_seed, ngram_size))
  for size in fc_size_list[:-1]:
    model.add(tf.keras.layers.Dense(size))
  model.add(tf.keras.layers.Dense(fc_size_list[-1], activation='softmax'))

  model.compile(
      optimizer=tf.keras.optimizers.Adam(lr=hparams.learning_rate),
      loss=tf.keras.losses.SparseCategoricalCrossentropy(),
      metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])
  return model
