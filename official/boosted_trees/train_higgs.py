r"""A script that builds boosted trees over higgs data.

If you haven't, please run data_download.py beforehand to prepare the data.

Don't forget to clean up model_dir before each training (or give a new dir).
Otherwise training stops immediately since n_trees is already reached in the
existing model_dir.

Usage:
$ python train_higgs.py --n_trees=100 --max_depth=6 --learning_rate=0.1 \
    --model_dir=/tmp/higgs_model

Note that BoostedTreesClassifier is available since Tensorflow 1.8.0.
So you need to install recent enough version of Tensorflow to use this example.

The training data is by default the first million examples out of 11M examples,
and eval data is by default the last million examples.
They are controlled by --train_start, --train_count, --eval_start, --eval_count.
e.g. to train over the first 10 million examples instead of 1 million:
$ python train_higgs.py --n_trees=100 --max_depth=6 --learning_rate=0.1 \
    --model_dir=/tmp/higgs_model --train_count=10000000

Training history and metrics can be inspected using tensorboard.
$ tensorboard --logdir=/tmp/higgs_model
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys

import numpy as np
import tensorflow as tf

NPZ_FILE = 'HIGGS.csv.gz.npz'  # numpy compressed file containing 'data' array


def parse_args():
  """Parses arguments and returns a tuple (known_args, unparsed_args)."""
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--data_dir', type=str, default='/tmp/higgs_data',
      help='Path to directory containing the higgs dataset.')
  parser.add_argument(
      '--model_dir', type=str, default='/tmp/higgs_model',
      help='The directory where the model will be stored.')
  parser.add_argument(
      '--train_start', type=int, default=0,
      help='Start index of train examples within the data.')
  parser.add_argument(
      '--train_count', type=int, default=1000000,
      help='Number of train examples within the data.')
  parser.add_argument(
      '--eval_start', type=int, default=10000000,
      help='Start index of eval examples within the data.')
  parser.add_argument(
      '--eval_count', type=int, default=1000000,
      help='Number of eval examples within the data.')

  parser.add_argument(
      '--n_trees', type=int, default=100, help='Number of trees to build.')
  parser.add_argument(
      '--max_depth', type=int, default=6, help='Maximum depths of each tree.')
  parser.add_argument(
      '--learning_rate', type=float, default=0.1,
      help='Maximum depths of each tree.')
  return parser.parse_known_args()


def read_higgs_data():
  """Reads higgs data from csv and returns train and eval data."""
  npz_filename = os.path.join(FLAGS.data_dir, NPZ_FILE)
  try:
    # gfile allows numpy to read data from network data sources as well.
    with tf.gfile.Open(npz_filename) as npz_file:
      with np.load(npz_file) as npz:
        data = npz['data']
  except Exception as e:
    raise RuntimeError(
        'Error loading data; use data_download.py to prepare the data:\n{}: {}'
        .format(type(e).__name__, e))
  return (data[FLAGS.train_start:FLAGS.train_start+FLAGS.train_count],
          data[FLAGS.eval_start:FLAGS.eval_start+FLAGS.eval_count])


def make_inputs_from_np_arrays(features_np, label_np):
  """Makes and returns input_fn and feature_columns from numpy arrays.

  The generated input_fn will return the tuple of the dictionary of features
  and a label, and feature_columns will consist of the list of
  tf.feature_column.BucketizedColumn.

  Note, for in-memory training, tf.data.Dataset should contain the whole data
  as a single tensor. Don't use batch.

  Args:
    features_np: a numpy ndarray (shape=[batch_size, num_features]) for
        float32 features.
    label_np: a numpy ndarray (shape=[batch_size, 1]) for labels.

  Returns:
    input_fn: a function returning a Dataset of feature dict and label.
    feature_column: a list of tf.feature_column.BucketizedColumn.
  """
  num_features = features_np.shape[1]
  features_np_list = np.split(features_np, num_features, axis=1)

  # Create source feature_columns and bucketized_columns.
  def get_bucket_boundaries(feature):
    """Returns bucket boundaries for feature by percentiles."""
    return np.unique(np.percentile(feature, range(0, 100))).tolist()
  source_columns = [
      tf.feature_column.numeric_column(
          # 1-based feature names.
          'feature_%02d' % (i + 1), dtype=tf.float32,
          # Although higgs data have no missing values, in general, default
          # could be set as 0 or some reasonable value for missing values.
          default_value=0.0)
      for i in range(num_features)
  ]
  bucketized_columns = [
      tf.feature_column.bucketized_column(
          source_columns[i],
          boundaries=get_bucket_boundaries(features_np_list[i]))
      for i in range(num_features)
  ]

  # Make an input_fn that extracts source features.
  def input_fn():
    """Returns features as a dictionary of numpy arrays, and a label."""
    features = {
        # Give source column name instead for easier track.
        source_columns[i].name: tf.constant(features_np_list[i])
        for i in range(num_features)
    }
    return tf.data.Dataset.zip((tf.data.Dataset.from_tensors(features),
                                tf.data.Dataset.from_tensors(label_np),))

  return input_fn, bucketized_columns


def main(unused_argv):
  print('## data loading..')
  train_data, eval_data = read_higgs_data()
  print('## data loaded; train: {}{}, eval: {}{}'.format(
      train_data.dtype, train_data.shape, eval_data.dtype, eval_data.shape))
  # data consists of one label column and 28 feature columns following.
  train_input_fn, feature_columns = make_inputs_from_np_arrays(
      features_np=train_data[:, 1:], label_np=train_data[:, 0:1])
  eval_input_fn, _ = make_inputs_from_np_arrays(
      features_np=eval_data[:, 1:], label_np=eval_data[:, 0:1])
  print('## features prepared. training starts..')

  # TODO(youngheek): consider remove the normalization factor part from l2.
  l2 = 1.0 / FLAGS.train_count

  # Though BoostedTreesClassifier is under tf.estimator, faster in-memory
  # training is yet provided as a contrib library.
  classifier = tf.contrib.estimator.boosted_trees_classifier_train_in_memory(
      train_input_fn,
      feature_columns,
      model_dir=FLAGS.model_dir or None,
      n_trees=FLAGS.n_trees,
      max_depth=FLAGS.max_depth,
      learning_rate=FLAGS.learning_rate,
      l2_regularization=l2)

  # Evaluation.
  eval_result = classifier.evaluate(eval_input_fn, steps=1)

  # Exporting the savedmodel.
  feature_spec = tf.estimator.export.build_parsing_serving_input_receiver_fn(
      tf.feature_column.make_parse_example_spec(feature_columns))
  classifier.export_savedmodel(os.path.join(FLAGS.model_dir, 'export'),
                               feature_spec)


if __name__ == '__main__':
  # Training progress and eval results are shown as logging.INFO; so enables it.
  tf.logging.set_verbosity(tf.logging.INFO)
  FLAGS, unparsed = parse_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
