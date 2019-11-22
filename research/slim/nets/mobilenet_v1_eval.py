# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
"""Validate mobilenet_v1 with options for quantization."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import tensorflow as tf
from tensorflow.contrib import quantize as contrib_quantize
from tensorflow.contrib import slim as contrib_slim

from datasets import dataset_factory
from nets import mobilenet_v1
from preprocessing import preprocessing_factory

slim = contrib_slim

flags = tf.app.flags

flags.DEFINE_string('master', '', 'Session master')
flags.DEFINE_integer('batch_size', 250, 'Batch size')
flags.DEFINE_integer('num_classes', 1001, 'Number of classes to distinguish')
flags.DEFINE_integer('num_examples', 50000, 'Number of examples to evaluate')
flags.DEFINE_integer('image_size', 224, 'Input image resolution')
flags.DEFINE_float('depth_multiplier', 1.0, 'Depth multiplier for mobilenet')
flags.DEFINE_bool('quantize', False, 'Quantize training')
flags.DEFINE_string('checkpoint_dir', '', 'The directory for checkpoints')
flags.DEFINE_string('eval_dir', '', 'Directory for writing eval event logs')
flags.DEFINE_string('dataset_dir', '', 'Location of dataset')

FLAGS = flags.FLAGS


def imagenet_input(is_training):
  """Data reader for imagenet.

  Reads in imagenet data and performs pre-processing on the images.

  Args:
     is_training: bool specifying if train or validation dataset is needed.
  Returns:
     A batch of images and labels.
  """
  if is_training:
    dataset = dataset_factory.get_dataset('imagenet', 'train',
                                          FLAGS.dataset_dir)
  else:
    dataset = dataset_factory.get_dataset('imagenet', 'validation',
                                          FLAGS.dataset_dir)

  provider = slim.dataset_data_provider.DatasetDataProvider(
      dataset,
      shuffle=is_training,
      common_queue_capacity=2 * FLAGS.batch_size,
      common_queue_min=FLAGS.batch_size)
  [image, label] = provider.get(['image', 'label'])

  image_preprocessing_fn = preprocessing_factory.get_preprocessing(
      'mobilenet_v1', is_training=is_training)

  image = image_preprocessing_fn(image, FLAGS.image_size, FLAGS.image_size)

  images, labels = tf.train.batch(
      tensors=[image, label],
      batch_size=FLAGS.batch_size,
      num_threads=4,
      capacity=5 * FLAGS.batch_size)
  return images, labels


def metrics(logits, labels):
  """Specify the metrics for eval.

  Args:
    logits: Logits output from the graph.
    labels: Ground truth labels for inputs.

  Returns:
     Eval Op for the graph.
  """
  labels = tf.squeeze(labels)
  names_to_values, names_to_updates = slim.metrics.aggregate_metric_map({
      'Accuracy': tf.metrics.accuracy(tf.argmax(logits, 1), labels),
      'Recall_5': tf.metrics.recall_at_k(labels, logits, 5),
  })
  for name, value in names_to_values.iteritems():
    slim.summaries.add_scalar_summary(
        value, name, prefix='eval', print_summary=True)
  return names_to_updates.values()


def build_model():
  """Build the mobilenet_v1 model for evaluation.

  Returns:
    g: graph with rewrites after insertion of quantization ops and batch norm
    folding.
    eval_ops: eval ops for inference.
    variables_to_restore: List of variables to restore from checkpoint.
  """
  g = tf.Graph()
  with g.as_default():
    inputs, labels = imagenet_input(is_training=False)

    scope = mobilenet_v1.mobilenet_v1_arg_scope(
        is_training=False, weight_decay=0.0)
    with slim.arg_scope(scope):
      logits, _ = mobilenet_v1.mobilenet_v1(
          inputs,
          is_training=False,
          depth_multiplier=FLAGS.depth_multiplier,
          num_classes=FLAGS.num_classes)

    if FLAGS.quantize:
      contrib_quantize.create_eval_graph()

    eval_ops = metrics(logits, labels)

  return g, eval_ops


def eval_model():
  """Evaluates mobilenet_v1."""
  g, eval_ops = build_model()
  with g.as_default():
    num_batches = math.ceil(FLAGS.num_examples / float(FLAGS.batch_size))
    slim.evaluation.evaluate_once(
        FLAGS.master,
        FLAGS.checkpoint_dir,
        logdir=FLAGS.eval_dir,
        num_evals=num_batches,
        eval_op=eval_ops)


def main(unused_arg):
  eval_model()


if __name__ == '__main__':
  tf.app.run(main)
