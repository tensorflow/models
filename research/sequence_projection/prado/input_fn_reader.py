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
# Lint as: python3
"""Methods related to input datasets and readers."""

import functools
import sys
from typing import Any, Callable, Mapping, Optional, Tuple, Dict

from absl import logging

import tensorflow.compat.v1 as tf
import tensorflow_datasets as tfds


def imdb_reviews(features, _):
  return features["text"], features["label"]


def civil_comments(features, runner_config):
  labels = runner_config["model_config"]["labels"]
  label_tensor = tf.stack([features[label] for label in labels], axis=1)
  label_tensor = tf.floor(label_tensor + 0.5)
  return features["text"], label_tensor


def goemotions(features, runner_config):
  labels = runner_config["model_config"]["labels"]
  label_tensor = tf.stack([features[label] for label in labels], axis=1)
  return features["comment_text"], tf.cast(label_tensor, tf.float32)


def random_substr(str_tensor, max_words):
  """Select random substring if the input has more than max_words."""
  word_batch_r = tf.strings.split(str_tensor, result_type="RaggedTensor")
  row_splits = word_batch_r.row_splits
  words = word_batch_r.values
  start_idx = row_splits[:-1]
  end_idx = row_splits[1:]
  words_per_example = end_idx - start_idx
  ones = tf.ones_like(end_idx)
  max_val = tf.maximum(ones, words_per_example - max_words)
  max_words_batch = tf.reduce_max(words_per_example)
  rnd = tf.random.uniform(
      tf.shape(start_idx), minval=0, maxval=max_words_batch, dtype=tf.int64)
  off_start_idx = tf.math.floormod(rnd, max_val)
  new_words_per_example = tf.where(
      tf.equal(max_val, 1), words_per_example, ones * max_words)
  new_start_idx = start_idx + off_start_idx
  new_end_idx = new_start_idx + new_words_per_example
  indices = tf.expand_dims(tf.range(tf.size(words), dtype=tf.int64), axis=0)
  within_limit = tf.logical_and(
      tf.greater_equal(indices, tf.expand_dims(new_start_idx, axis=1)),
      tf.less(indices, tf.expand_dims(new_end_idx, axis=1)))
  keep_indices = tf.reduce_any(within_limit, axis=0)
  keep_indices = tf.cast(keep_indices, dtype=tf.int32)
  _, selected_words = tf.dynamic_partition(words, keep_indices, 2)
  row_splits = tf.math.cumsum(new_words_per_example)
  row_splits = tf.concat([[0], row_splits], axis=0)
  new_tensor = tf.RaggedTensor.from_row_splits(
      values=selected_words, row_splits=row_splits)
  return tf.strings.reduce_join(new_tensor, axis=1, separator=" ")


def _post_processor(features, runner_config, mode, create_projection,
                    batch_size):
  """Post process the data to a form expected by model_fn."""
  data_processor = getattr(sys.modules[__name__], runner_config["dataset"])
  text, label = data_processor(features, runner_config)
  if "max_seq_len" in runner_config["model_config"]:
    max_seq_len = runner_config["model_config"]["max_seq_len"]
    logging.info("Truncating text to have at most %d tokens", max_seq_len)
    text = random_substr(text, max_seq_len)
  text = tf.reshape(text, [batch_size])
  num_classes = len(runner_config["model_config"]["labels"])
  label = tf.reshape(label, [batch_size, num_classes])
  projection, seq_length = create_projection(runner_config["model_config"],
                                             mode, text)
  return {"projection": projection, "seq_length": seq_length, "label": label}


def create_input_fn(runner_config: Dict[str, Any], create_projection: Callable,
                    mode: tf.estimator.ModeKeys, drop_remainder: bool):
  """Returns an input function to use in the instantiation of tf.estimator.*."""

  def _input_fn(
      params: Mapping[str, Any]
  ) -> Tuple[Mapping[str, tf.Tensor], Optional[Mapping[str, tf.Tensor]]]:
    """Method to be used for reading the data."""
    assert mode != tf.estimator.ModeKeys.PREDICT
    split = "train" if mode == tf.estimator.ModeKeys.TRAIN else "test"
    ds = tfds.load(runner_config["dataset"], split=split)
    ds = ds.batch(params["batch_size"], drop_remainder=drop_remainder)
    ds = ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    ds = ds.shuffle(buffer_size=100)
    ds = ds.repeat(count=1 if mode == tf.estimator.ModeKeys.EVAL else None)
    ds = ds.map(
        functools.partial(
            _post_processor,
            runner_config=runner_config,
            mode=mode,
            create_projection=create_projection,
            batch_size=params["batch_size"]),
        num_parallel_calls=tf.data.experimental.AUTOTUNE,
        deterministic=False)
    return ds

  return _input_fn
