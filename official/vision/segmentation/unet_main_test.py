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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tempfile

from absl import flags
from absl.testing import flagsaver
from absl.testing import parameterized
import numpy as np
import tensorflow as tf

from tensorflow.contrib import cluster_resolver as contrib_cluster_resolver
from tensorflow.contrib.tpu.python.tpu import device_assignment as device_lib
from tensorflow.python.distribute import tpu_strategy as tpu_strategy_lib
from tensorflow.python.tpu import tpu_strategy_util
from official.modeling.hyperparams import params_dict
from official.vision.segmentation import unet_config
from official.vision.segmentation import unet_main as unet_main_lib
from official.vision.segmentation import unet_metrics
from official.vision.segmentation import unet_model as unet_model_lib

FLAGS = flags.FLAGS


def create_fake_input_fn(params,
                         features_size,
                         labels_size,
                         use_bfloat16=False):
  """Returns fake input function for testing."""

  def fake_data_input_fn(unused_ctx=None):
    """An input function for generating fake data."""
    batch_size = params.train_batch_size
    features = np.random.rand(64, *features_size)
    labels = np.random.randint(2, size=[64] + labels_size)

    # Convert the inputs to a Dataset.
    dataset = tf.data.Dataset.from_tensor_slices((features, labels))

    def _assign_dtype(features, labels):
      if use_bfloat16:
        features = tf.cast(features, tf.bfloat16)
        labels = tf.cast(labels, tf.bfloat16)
      else:
        features = tf.cast(features, tf.float32)
        labels = tf.cast(labels, tf.float32)
      return features, labels

    # Shuffle, repeat, and batch the examples.
    dataset = dataset.map(_assign_dtype)
    dataset = dataset.shuffle(64).repeat()
    dataset = dataset.batch(batch_size, drop_remainder=True)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

    # Return the dataset.
    return dataset

  return fake_data_input_fn


class UnetMainTest(parameterized.TestCase, tf.test.TestCase):

  def setUp(self):
    super(UnetMainTest, self).setUp()
    self._model_dir = os.path.join(tempfile.mkdtemp(), 'model_dir')
    tf.io.gfile.makedirs(self._model_dir)

  def tearDown(self):
    tf.io.gfile.rmtree(self._model_dir)
    super(UnetMainTest, self).tearDown()

  @flagsaver.flagsaver
  def testUnet3DModel(self):
    FLAGS.tpu = ''
    FLAGS.mode = 'train'
    params = params_dict.ParamsDict(unet_config.UNET_CONFIG,
                                    unet_config.UNET_RESTRICTIONS)
    params.override(
        {
            'input_image_size': [64, 64, 64],
            'train_item_count': 4,
            'eval_item_count': 4,
            'train_batch_size': 2,
            'eval_batch_size': 2,
            'batch_size': 2,
            'num_base_filters': 16,
            'dtype': 'bfloat16',
            'depth': 1,
            'train_steps': 2,
            'eval_steps': 2,
            'mode': FLAGS.mode,
            'tpu': FLAGS.tpu,
            'num_gpus': 0,
            'checkpoint_interval': 1,
            'use_tpu': True,
            'input_partition_dims': None,
        },
        is_strict=False)
    params.validate()
    params.lock()

    image_size = params.input_image_size + [params.num_channels]
    label_size = params.input_image_size + [params.num_classes]
    input_fn = create_fake_input_fn(
        params, features_size=image_size, labels_size=label_size)

    resolver = contrib_cluster_resolver.TPUClusterResolver(tpu=params.tpu)
    topology = tpu_strategy_util.initialize_tpu_system(resolver)
    device_assignment = None

    if params.input_partition_dims is not None:
      assert np.prod(
          params.input_partition_dims) == 2, 'invalid unit test configuration'
      computation_shape = [1, 1, 1, 2]
      partition_dimension = params.input_partition_dims
      num_replicas = resolver.get_tpu_system_metadata().num_cores // np.prod(
          partition_dimension)
      device_assignment = device_lib.device_assignment(
          topology,
          computation_shape=computation_shape,
          num_replicas=num_replicas)

    strategy = tpu_strategy_lib.TPUStrategy(
        resolver, device_assignment=device_assignment)

    with strategy.scope():
      model = unet_model_lib.build_unet_model(params)
      optimizer = unet_model_lib.create_optimizer(params.init_learning_rate,
                                                  params)
      loss_fn = unet_metrics.get_loss_fn(params.mode, params)
      model.compile(loss=loss_fn, optimizer=optimizer, metrics=[loss_fn])

      eval_ds = input_fn()
      iterator = iter(eval_ds)

      image, _ = next(iterator)
      logits = model(image, training=False)
      self.assertEqual(logits.shape[1:], params.input_image_size + [3])

  @parameterized.parameters(
      {
          'use_mlir': True,
          'dtype': 'bfloat16',
          'input_partition_dims': None,
      }, {
          'use_mlir': False,
          'dtype': 'bfloat16',
          'input_partition_dims': None,
      }, {
          'use_mlir': True,
          'dtype': 'bfloat16',
          'input_partition_dims': None,
      }, {
          'use_mlir': False,
          'dtype': 'bfloat16',
          'input_partition_dims': None,
      }, {
          'use_mlir': True,
          'dtype': 'bfloat16',
          'input_partition_dims': [1, 2, 1, 1, 1],
      }, {
          'use_mlir': False,
          'dtype': 'bfloat16',
          'input_partition_dims': [1, 2, 1, 1, 1],
      }, {
          'use_mlir': True,
          'dtype': 'bfloat16',
          'input_partition_dims': [1, 2, 1, 1, 1],
      }, {
          'use_mlir': False,
          'dtype': 'bfloat16',
          'input_partition_dims': [1, 2, 1, 1, 1]
      })
  @flagsaver.flagsaver
  def testUnetTrain(self, use_mlir, dtype, input_partition_dims):
    FLAGS.tpu = ''
    FLAGS.mode = 'train'

    if use_mlir:
      tf.config.experimental.enable_mlir_bridge()

    params = params_dict.ParamsDict(unet_config.UNET_CONFIG,
                                    unet_config.UNET_RESTRICTIONS)
    params.override(
        {
            'model_dir': self._model_dir,
            'input_image_size': [8, 8, 8],
            'train_item_count': 2,
            'eval_item_count': 2,
            'train_batch_size': 2,
            'eval_batch_size': 2,
            'batch_size': 2,
            'num_base_filters': 1,
            'dtype': 'bfloat16',
            'depth': 1,
            'epochs': 1,
            'checkpoint_interval': 1,
            'train_steps': 1,
            'eval_steps': 1,
            'mode': FLAGS.mode,
            'tpu': FLAGS.tpu,
            'use_tpu': True,
            'num_gpus': 0,
            'distribution_strategy': 'tpu',
            'steps_per_loop': 1,
            'input_partition_dims': input_partition_dims,
        },
        is_strict=False)
    params.validate()
    params.lock()

    image_size = params.input_image_size + [params.num_channels]
    label_size = params.input_image_size + [params.num_classes]
    input_fn = create_fake_input_fn(
        params, features_size=image_size, labels_size=label_size)

    input_dtype = params.dtype
    if input_dtype == 'float16' or input_dtype == 'bfloat16':
      policy = tf.keras.mixed_precision.experimental.Policy(
          'mixed_bfloat16' if input_dtype == 'bfloat16' else 'mixed_float16')
      tf.keras.mixed_precision.experimental.set_policy(policy)

    strategy = unet_main_lib.create_distribution_strategy(params)
    with strategy.scope():
      unet_model = unet_model_lib.build_unet_model(params)
      unet_main_lib.train(params, strategy, unet_model, input_fn, input_fn)


if __name__ == '__main__':
  unet_main_lib.define_unet3d_flags()
  tf.test.main()
