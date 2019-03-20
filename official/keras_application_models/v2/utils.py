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
"""Utils for retrain Keras Application models on CIFAR-10."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# pylint: disable=g-bad-import-order
import os
import numpy as np
import absl.logging
from absl import app as absl_app
from absl import flags
import tensorflow as tf
# pylint: enable=g-bad-import-order


class NotReallyADistributionStrategy():

  class FakeScope():
    def __enter__(self):
      pass
    def __exit__(self, ext_type, ext_val, tb):
      pass


  def scope(self):
    return self.FakeScope()


def define_flags():
  flags.DEFINE_integer(
      name="num_gpus", default=1, help="The number of gpus used to training.")

  flags.DEFINE_integer(
      name="train_epochs", default=2, help="The number of epochs for training.")

  flags.DEFINE_integer(
      name="batch_size", default=32, help="The size of batch in 1 iteration.")

  flags.DEFINE_boolean(
      name="no_pretrained_weights", default=False, help=
          "To train the model without pretrained ImageNet weights. It could "
          "slow down training and converge but could test if the model is "
          "able to train from the scratch.")

  flags.DEFINE_boolean(
      name="no_eager", default=False, help=
          "To disable eager execution. Note that if eager execution is "
          "disabled, only one GPU is utilized even if multiple GPUs are "
          "provided and multi_gpu_model is used.")

  flags.DEFINE_boolean(
      name="dist_strat", default=False, help=
          "To enable distribution strategy for model training and evaluation. "
          "Number of GPUs used for distribution strategy can be set by the "
          "argument --num_gpus.")

  flags.DEFINE_integer(
      name="limit_train_num", default=-1, help=
          "To limit train dataset size, usually for code validation or "
          "perf-only testing. Set to -1 to use full dataset.")

  flags.DEFINE_boolean(
      name="enable_model_saving", default=False, help=
          "To enable best model saving feature of Keras training.")

  flags.DEFINE_integer(
      name="num_dataset_private_threads", default=0, help=
          "Num of private threads for dataset to accelerate data loading.")

  # pylint: disable=unused-variable
  def _check_eager_dist_strat(flag_dict):
    return flag_dict["disable_eager"] or not flag_dict["dist_strat"]


def init_eager_execution(disable_eager):
  """Init eager execution, which is compat in TF1.0/TF2.0."""
  if disable_eager:
    absl.logging.info("Eager execution is disabled...")
    tf.compat.v1.disable_eager_execution()
  else:
    absl.logging.info("Eager execution is enabled...")
    tf.compat.v1.enable_eager_execution()


def add_global_regularization(model, l1=0., l2=0.):
  for layer in model.layers:
    if hasattr(layer, "kernel_regularizer"):
      layer.kerner_regularizer = tf.keras.regularizers.l1_l2(l1, l2)


def prepare_model_saving(name):
  """Prepare directory for saving model and returns the callback."""
  save_dir = os.path.join(os.getcwd(), '%s_keras_ckpt' % name)
  model_name = 'model.{epoch:03d}.h5'
  if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
  filepath = os.path.join(save_dir, model_name)
  checkpoint = tf.keras.callbacks.ModelCheckpoint(
      filepath=filepath, monitor='val_acc', verbose=1, save_best_only=True)
  return checkpoint


def _local_gpu_devices(num_gpus):
  """Return string representation of GPUs."""
  return (tuple("/device:GPU:%d" % i for i in range(num_gpus)) or
       ("/device:CPU:0",))


def get_distribution_strategy(num_gpus,
                              all_reduce_alg=None,
                              no_distribution_strategy=False):
  """Return a DistributionStrategy for running the model.

  Args:
    num_gpus: Number of GPUs to run this model. If num_gpus == 0, use CPU.
    all_reduce_alg: Specify which algorithm to use when performing all-reduce.
      See tf.contrib.distribute.AllReduceCrossDeviceOps for available
      algorithms. If None, DistributionStrategy will choose based on device
      topology.
    no_distribution_strategy: when set to True, do not use any
      distribution strategy. Note that when it is True, and num_gpus is
      larger than 1, it will raise a ValueError.

  Returns:
    tf.contrib.distribute.DistibutionStrategy object.
  Raises:
    ValueError: if no_distribution_strategy is True and num_gpus is
    larger than 1
  """
  try:
    import tensorflow.distribute as tf_distribute
  except ImportError:
    # v1 compat
    import tensorflow.contrib.distribute as tf_distribute


  if no_distribution_strategy:
    return NotReallyADistributionStrategy()
  else:
    devices = _local_gpu_devices(num_gpus)
    absl.logging.info("Using GPU devices: %s", devices)
    if all_reduce_alg:
      return tf_distribute.MirroredStrategy(
          devices,
          cross_device_ops=tf_distribute.AllReduceCrossDeviceOps(
              all_reduce_alg, num_packs=2))
    else:
      return tf_distribute.MirroredStrategy(devices)

