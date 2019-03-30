# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
"""MNIST model training using TPUs.

This program demonstrates training of the convolutional neural network model
defined in mnist.py on Google Cloud TPUs (https://cloud.google.com/tpu/).

If you are not interested in TPUs, you should ignore this file.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys

from official.utils.flags import core as flags_core
from absl import flags
import tensorflow as tf  # pylint: disable=g-bad-import-order

# For open source environment, add grandparent directory for import
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(sys.path[0]))))

from official.mnist_upgraded import dataset  # pylint: disable=wrong-import-position
from official.mnist_upgraded import mnist  # pylint: disable=wrong-import-position

# Cloud TPU Cluster Resolver flags
flags.DEFINE_string(
    name="tpu", default=None,
    help="The Cloud TPU to use for training. This should be either the name "
    "used when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470 "
    "url.")
flags.DEFINE_string(
    name="tpu_zone", default=None,
    help="[Optional] GCE zone where the Cloud TPU is located in. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")
flags.DEFINE_string(
    name="gcp_project", default=None,
    help="[Optional] Project name for the Cloud TPU-enabled project. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

# Model specific parameters
flags.DEFINE_string(name="data_dir",default= "",
                       help="Path to directory containing the MNIST dataset")
flags.DEFINE_string(name="model_dir", default=None, help="Estimator model_dir")
flags.DEFINE_integer("batch_size",default= 1024,
                        help="Mini-batch size for the training. Note that this "
                        "is the global batch size and not the per-shard batch.")
flags.DEFINE_integer(name="train_steps",default= 1000,help= "Total number of training steps.")
flags.DEFINE_integer(name="eval_steps",default= 0,
                        help="Total number of evaluation steps. If `0`, evaluation "
                        "after training is skipped.")
flags.DEFINE_float(name="learning_rate",default= 0.05, help="Learning rate.")

flags.DEFINE_bool(name="use_tpu",default= True,help= "Use TPUs rather than plain CPUs")
flags.DEFINE_bool(name="enable_predict",default= True, help="Do some predictions at the end")
flags.DEFINE_integer(name="iterations",default= 50,
                        help="Number of iterations per TPU training loop.")
flags.DEFINE_integer(name="num_shards",default= 8, help="Number of shards (TPU chips).")

FLAGS = flags.FLAGS


def metric_fn(labels, logits):
  accuracy = tf.compat.v1.metrics.accuracy(
      labels=labels, predictions=tf.argmax(input=logits, axis=1))
  return {"accuracy": accuracy}


def model_fn(features, labels, mode, params):
  """model_fn constructs the ML model used to predict handwritten digits."""

  del params
  image = features
  if isinstance(image, dict):
    image = features["image"]

  model = mnist.create_model("channels_last")

  if mode == tf.estimator.ModeKeys.PREDICT:
    logits = model(image, training=False)
    predictions = {
        'class_ids': tf.argmax(input=logits, axis=1),
        'probabilities': tf.nn.softmax(logits),
    }
    return tf.contrib.tpu.TPUEstimatorSpec(mode, predictions=predictions)

  logits = model(image, training=(mode == tf.estimator.ModeKeys.TRAIN))
  loss = tf.compat.v1.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

  if mode == tf.estimator.ModeKeys.TRAIN:
    learning_rate = tf.compat.v1.train.exponential_decay(
        FLAGS.learning_rate,
        tf.compat.v1.train.get_global_step(),
        decay_steps=100000,
        decay_rate=0.96)
    optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=learning_rate)
    if FLAGS.use_tpu:
      optimizer = tf.contrib.tpu.CrossShardOptimizer(optimizer)
    return tf.contrib.tpu.TPUEstimatorSpec(
        mode=mode,
        loss=loss,
        train_op=optimizer.minimize(loss, tf.compat.v1.train.get_global_step()))

  if mode == tf.estimator.ModeKeys.EVAL:
    return tf.contrib.tpu.TPUEstimatorSpec(
        mode=mode, loss=loss, eval_metrics=(metric_fn, [labels, logits]))


def train_input_fn(params):
  """train_input_fn defines the input pipeline used for training."""
  batch_size = params["batch_size"]
  data_dir = params["data_dir"]
  # Retrieves the batch size for the current shard. The # of shards is
  # computed according to the input pipeline deployment. See
  # `tf.contrib.tpu.RunConfig` for details.
  ds = dataset.train(data_dir).cache().repeat().shuffle(
      buffer_size=50000).batch(batch_size, drop_remainder=True)
  return ds


def eval_input_fn(params):
  batch_size = params["batch_size"]
  data_dir = params["data_dir"]
  ds = dataset.test(data_dir).batch(batch_size, drop_remainder=True)
  return ds


def predict_input_fn(params):
  batch_size = params["batch_size"]
  data_dir = params["data_dir"]
  # Take out top 10 samples from test data to make the predictions.
  ds = dataset.test(data_dir).take(10).batch(batch_size)
  return ds


def main(argv):
  del argv  # Unused.
  tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)

  tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
      FLAGS.tpu,
      zone=FLAGS.tpu_zone,
      project=FLAGS.gcp_project
  )

  run_config = tf.contrib.tpu.RunConfig(
      cluster=tpu_cluster_resolver,
      model_dir=FLAGS.model_dir,
      session_config=tf.compat.v1.ConfigProto(
          allow_soft_placement=True, log_device_placement=True),
      tpu_config=tf.contrib.tpu.TPUConfig(FLAGS.iterations, FLAGS.num_shards),
  )

  estimator = tf.contrib.tpu.TPUEstimator(
      model_fn=model_fn,
      use_tpu=FLAGS.use_tpu,
      train_batch_size=FLAGS.batch_size,
      eval_batch_size=FLAGS.batch_size,
      predict_batch_size=FLAGS.batch_size,
      params={"data_dir": FLAGS.data_dir},
      config=run_config)
  # TPUEstimator.train *requires* a max_steps argument.
  estimator.train(input_fn=train_input_fn, max_steps=FLAGS.train_steps)
  # TPUEstimator.evaluate *requires* a steps argument.
  # Note that the number of examples used during evaluation is
  # --eval_steps * --batch_size.
  # So if you change --batch_size then change --eval_steps too.
  if FLAGS.eval_steps:
    estimator.evaluate(input_fn=eval_input_fn, steps=FLAGS.eval_steps)

  # Run prediction on top few samples of test data.
  if FLAGS.enable_predict:
    predictions = estimator.predict(input_fn=predict_input_fn)

    for pred_dict in predictions:
      template = ('Prediction is "{}" ({:.1f}%).')

      class_id = pred_dict['class_ids']
      probability = pred_dict['probabilities'][class_id]

      print(template.format(class_id, 100 * probability))


if __name__ == "__main__":
  tf.compat.v1.app.run()
