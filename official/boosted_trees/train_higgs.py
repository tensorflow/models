r"""A script that builds boosted trees over higgs data.

If you haven't, please run data_download.py beforehand to prepare the data.

For some more details on this example, please refer to README.md as well.

Note that the model_dir is cleaned up before starting the training.

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
Set --logdir as the --model_dir set by flag when training
(or the default /tmp/higgs_model).
$ tensorboard --logdir=/tmp/higgs_model
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys

from absl import app as absl_app
from absl import flags
import numpy as np  # pylint: disable=wrong-import-order
import tensorflow as tf  # pylint: disable=wrong-import-order

from official.utils.flags import core as flags_core
from official.utils.flags._conventions import help_wrap


NPZ_FILE = 'HIGGS.csv.gz.npz'  # numpy compressed file containing 'data' array


def define_train_higgs_flags():
  """Add tree related flags as well as training/eval configuration."""
  flags_core.define_base(stop_threshold=False, batch_size=False, num_gpu=False)
  flags.adopt_module_key_flags(flags_core)

  flags.DEFINE_integer(
      name='train_start', default=0,
      help=help_wrap('Start index of train examples within the data.'))
  flags.DEFINE_integer(
      name='train_count', default=1000000,
      help=help_wrap('Number of train examples within the data.'))
  flags.DEFINE_integer(
      name='eval_start', default=10000000,
      help=help_wrap('Start index of eval examples within the data.'))
  flags.DEFINE_integer(
      name='eval_count', default=1000000,
      help=help_wrap('Number of eval examples within the data.'))

  flags.DEFINE_integer(
      'n_trees', default=100, help=help_wrap('Number of trees to build.'))
  flags.DEFINE_integer(
      'max_depth', default=6, help=help_wrap('Maximum depths of each tree.'))
  flags.DEFINE_float(
      'learning_rate', default=0.1,
      help=help_wrap('Maximum depths of each tree.'))

  flags_core.set_defaults(data_dir='/tmp/higgs_data',
                          model_dir='/tmp/higgs_model')



def read_higgs_data(data_dir, train_start, train_count, eval_start, eval_count):
  """Reads higgs data from csv and returns train and eval data."""
  npz_filename = os.path.join(data_dir, NPZ_FILE)
  try:
    # gfile allows numpy to read data from network data sources as well.
    with tf.gfile.Open(npz_filename, 'rb') as npz_file:
      with np.load(npz_file) as npz:
        data = npz['data']
  except Exception as e:
    raise RuntimeError(
        'Error loading data; use data_download.py to prepare the data:\n{}: {}'
        .format(type(e).__name__, e))
  return (data[train_start:train_start+train_count],
          data[eval_start:eval_start+eval_count])


# This showcases how to make input_fn when the input data is available in the
# form of numpy arrays.
def make_inputs_from_np_arrays(features_np, label_np):
  """Makes and returns input_fn and feature_columns from numpy arrays.

  The generated input_fn will return tf.data.Dataset of feature dictionary and a
  label, and feature_columns will consist of the list of
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
  # 1-based feature names.
  feature_names = ['feature_%02d' % (i + 1) for i in range(num_features)]

  # Create source feature_columns and bucketized_columns.
  def get_bucket_boundaries(feature):
    """Returns bucket boundaries for feature by percentiles."""
    return np.unique(np.percentile(feature, range(0, 100))).tolist()
  source_columns = [
      tf.feature_column.numeric_column(
          feature_name, dtype=tf.float32,
          # Although higgs data have no missing values, in general, default
          # could be set as 0 or some reasonable value for missing values.
          default_value=0.0)
      for feature_name in feature_names
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
        feature_name: tf.constant(features_np_list[i])
        for i, feature_name in enumerate(feature_names)
    }
    return tf.data.Dataset.zip((tf.data.Dataset.from_tensors(features),
                                tf.data.Dataset.from_tensors(label_np),))

  return input_fn, bucketized_columns


def make_eval_inputs_from_np_arrays(features_np, label_np):
  """Makes eval input as streaming batches."""
  num_features = features_np.shape[1]
  features_np_list = np.split(features_np, num_features, axis=1)
  # 1-based feature names.
  feature_names = ['feature_%02d' % (i + 1) for i in range(num_features)]

  def input_fn():
    features = {
        feature_name: tf.constant(features_np_list[i])
        for i, feature_name in enumerate(feature_names)
    }
    return tf.data.Dataset.zip(
        (tf.data.Dataset.from_tensor_slices(features),
         tf.data.Dataset.from_tensor_slices(label_np),)).batch(1000)

  return input_fn


def train_boosted_trees(flags_obj):
  """Train boosted_trees estimator on HIGGS data.

  Args:
    flags_obj: An object containing parsed flag values.
  """

  # Clean up the model directory if present.
  if tf.gfile.Exists(flags_obj.model_dir):
    tf.gfile.DeleteRecursively(flags_obj.model_dir)
  print('## data loading..')
  train_data, eval_data = read_higgs_data(
      flags_obj.data_dir, flags_obj.train_start, flags_obj.train_count,
      flags_obj.eval_start, flags_obj.eval_count)
  print('## data loaded; train: {}{}, eval: {}{}'.format(
      train_data.dtype, train_data.shape, eval_data.dtype, eval_data.shape))
  # data consists of one label column and 28 feature columns following.
  train_input_fn, feature_columns = make_inputs_from_np_arrays(
      features_np=train_data[:, 1:], label_np=train_data[:, 0:1])
  eval_input_fn = make_eval_inputs_from_np_arrays(
      features_np=eval_data[:, 1:], label_np=eval_data[:, 0:1])
  print('## features prepared. training starts..')

  # Though BoostedTreesClassifier is under tf.estimator, faster in-memory
  # training is yet provided as a contrib library.
  classifier = tf.contrib.estimator.boosted_trees_classifier_train_in_memory(
      train_input_fn,
      feature_columns,
      model_dir=flags_obj.model_dir or None,
      n_trees=flags_obj.n_trees,
      max_depth=flags_obj.max_depth,
      learning_rate=flags_obj.learning_rate)

  # Evaluation.
  eval_result = classifier.evaluate(eval_input_fn)

  # Exporting the savedmodel.
  if flags_obj.export_dir is not None:
    feature_spec = tf.estimator.export.build_parsing_serving_input_receiver_fn(
        tf.feature_column.make_parse_example_spec(feature_columns))
    classifier.export_savedmodel(flags_obj.export_dir, feature_spec)


def main(_):
  train_boosted_trees(flags.FLAGS)


if __name__ == '__main__':
  # Training progress and eval results are shown as logging.INFO; so enables it.
  tf.logging.set_verbosity(tf.logging.INFO)
  define_train_higgs_flags()
  absl_app.run(main)
