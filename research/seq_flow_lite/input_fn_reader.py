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

from absl import logging

import tensorflow as tf
import tensorflow_datasets as tfds

from layers import projection_layers # import seq_flow_lite module
from utils import misc_utils # import seq_flow_lite module


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


def create_input_fn(runner_config, mode, drop_remainder):
  """Returns an input function to use in the instantiation of tf.estimator.*."""

  def _post_processor(features, batch_size):
    """Post process the data to a form expected by model_fn."""
    data_processor = getattr(sys.modules[__name__], runner_config["dataset"])
    text, label = data_processor(features, runner_config)
    model_config = runner_config["model_config"]
    if "max_seq_len" in model_config:
      max_seq_len = model_config["max_seq_len"]
      logging.info("Truncating text to have at most %d tokens", max_seq_len)
      text = misc_utils.random_substr(text, max_seq_len)
    text = tf.reshape(text, [batch_size])
    num_classes = len(model_config["labels"])
    label = tf.reshape(label, [batch_size, num_classes])
    prxlayer = projection_layers.ProjectionLayer(model_config, mode)
    projection, seq_length = prxlayer(text)
    return {"projection": projection, "seq_length": seq_length, "label": label}

  def _input_fn(params):
    """Method to be used for reading the data."""
    assert mode != tf.estimator.ModeKeys.PREDICT
    split = "train" if mode == tf.estimator.ModeKeys.TRAIN else "test"
    ds = tfds.load(runner_config["dataset"], split=split)
    ds = ds.batch(params["batch_size"], drop_remainder=drop_remainder)
    ds = ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    ds = ds.shuffle(buffer_size=100)
    ds = ds.repeat(count=1 if mode == tf.estimator.ModeKeys.EVAL else None)
    ds = ds.map(
        functools.partial(_post_processor, batch_size=params["batch_size"]),
        num_parallel_calls=tf.data.experimental.AUTOTUNE,
        deterministic=False)
    return ds

  return _input_fn
