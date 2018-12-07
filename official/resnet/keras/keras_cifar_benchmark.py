"""Executes Keras benchmarks and accuracy tests."""

from __future__ import print_function

import os
import sys

from absl import app as absl_app
from absl import flags
import tensorflow as tf  # pylint: disable=g-bad-import-order

from official.resnet import cifar10_main as cifar_main
import official.resnet.keras.keras_cifar_main as keras_cifar_main

DATA_DIR = '/data/cifar10_data/'


class KerasCifar10BenchmarkTests():

  def keras_resnet56_1_gpu(self):
    self._setup()
    flags.FLAGS.num_gpus = 1
    flags.FLAGS.data_dir = DATA_DIR
    flags.FLAGS.batch_size = 128
    flags.FLAGS.train_epochs = 1
    flags.FLAGS.model_dir = self._get_model_dir('keras_resnet56_1_gpu')
    flags.FLAGS.resnet_size = 56
    flags.FLAGS.dtype = 'fp32'
    stats = keras_cifar_main.run_cifar_with_keras(flags.FLAGS)
    report_info = {}
    results = []
    results.append(self._create_result(stats['accuracy_top_1'].item(),
                                       'top_1',
                                       'quality'))

    results.append(self._create_result(stats['training_accuracy_top_1'].item(),
                                       'top_1_train_accuracy',
                                       'quality'))

    report_info['results'] = results
    return report_info

  def keras_resnet56_4_gpu(self):
    flags.FLAGS.num_gpus = 4
    flags.FLAGS.data_dir = DATA_DIR
    flags.FLAGS.batch_size = 128
    flags.FLAGS.train_epochs = 182
    flags.FLAGS.model_dir = ''
    flags.FLAGS.resnet_size = 56
    flags.FLAGS.dtype = 'fp32'
    keras_cifar_main.run_cifar_with_keras(flags.FLAGS)

  def keras_resnet56_no_dist_strat_1_gpu(self):
    self._setup()
    flags.dist_strat_off = True
    flags.FLAGS.num_gpus = 1
    flags.FLAGS.data_dir = DATA_DIR
    flags.FLAGS.batch_size = 128
    flags.FLAGS.train_epochs = 1
    flags.FLAGS.model_dir = ''
    flags.FLAGS.resnet_size = 56
    flags.FLAGS.dtype = 'fp32'
    stats = keras_cifar_main.run_cifar_with_keras(flags.FLAGS)
    report_info = {}
    results = []
    results.append(self._create_result(stats['accuracy_top_1'].item(),
                                       'top_1',
                                       'quality'))

    results.append(self._create_result(stats['training_accuracy_top_1'].item(),
                                       'top_1_train_accuracy',
                                       'quality'))

    report_info['results'] = results
    return report_info

  def _create_result(self, result, result_name, result_unit):
    res_dict = {}
    res_dict['result'] = result
    res_dict['result_name'] = result_name
    res_dict['result_unit'] = result_unit
    return res_dict

  def _get_model_dir(self, folder_name):
    return os.path.join('/workspace', folder_name)

  def _setup(self):
    tf.logging.set_verbosity(tf.logging.DEBUG)
    keras_cifar_main.define_keras_cifar_flags()
    cifar_main.define_cifar_flags()
    flags.FLAGS(['foo'])

  def run_tests(self, test_list):
    keras_benchmark = KerasCifar10BenchmarkTests()
    if test_list:
      for t in test_list:
        getattr(self, t)()
    else:
      print('Running all tests')
      keras_benchmark.keras_resnet56_1_gpu()
      keras_benchmark.keras_resnet56_no_dist_strat_1_gpu()
      keras_benchmark.keras_resnet56_4_gpu()


def main(_):
  keras_benchmark = KerasCifar10BenchmarkTests()
  keras_benchmark.run_tests(['keras_resnet56_1_gpu'])


if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.DEBUG)
  cifar_main.define_cifar_flags()
  absl_app.run(main)
