# Copyright 2023 The TensorFlow Authors. All Rights Reserved.
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

"""Tests for mobilenet_edgetpu model."""

import os

import tensorflow as tf, tf_keras
from official.legacy.image_classification import preprocessing
from official.projects.edgetpu.vision.modeling import common_modules
from official.projects.edgetpu.vision.modeling import mobilenet_edgetpu_v1_model
from official.projects.edgetpu.vision.modeling import mobilenet_edgetpu_v1_model_blocks

# TODO(b/151324383): Enable once training is supported for mobilenet-edgetpu
EXAMPLE_IMAGE = ('third_party/tensorflow_models/official/vision/'
                 'image_classification/testdata/panda.jpg')

CKPTS = 'gs://**/efficientnets'


def _copy_recursively(src: str, dst: str) -> None:
  """Recursively copy directory."""
  for src_dir, _, src_files in tf.io.gfile.walk(src):
    dst_dir = os.path.join(dst, os.path.relpath(src_dir, src))
    if not tf.io.gfile.exists(dst_dir):
      tf.io.gfile.makedirs(dst_dir)
    for src_file in src_files:
      tf.io.gfile.copy(
          os.path.join(src_dir, src_file),
          os.path.join(dst_dir, src_file),
          overwrite=True)


class MobilenetEdgeTPUBlocksTest(tf.test.TestCase):

  def setUp(self):
    super(tf.test.TestCase, self).setUp()
    # Ensure no model duplicates
    tf_keras.backend.clear_session()

  def test_bottleneck_block(self):
    """Test for creating a model with bottleneck block arguments."""
    images = tf.zeros((4, 224, 224, 3), dtype=tf.float32)

    tf_keras.backend.set_image_data_format('channels_last')

    blocks = [
        mobilenet_edgetpu_v1_model_blocks.BlockConfig.from_args(
            input_filters=3,
            output_filters=6,
            kernel_size=3,
            num_repeat=3,
            expand_ratio=6,
            strides=(2, 2),
            fused_conv=False,
        )
    ]
    config = mobilenet_edgetpu_v1_model.ModelConfig.from_args(
        blocks=blocks,
        num_classes=10,
        use_se=False,
    )

    model = mobilenet_edgetpu_v1_model.MobilenetEdgeTPU(config)
    outputs = model(images, training=True)
    self.assertEqual((4, 10), outputs.shape)

    ref_var_names = set([
        'stem_conv2d/kernel:0',
        'stem_bn/gamma:0',
        'stem_bn/beta:0',
        'stack_0/block_0/expand_conv2d/kernel:0',
        'stack_0/block_0/expand_bn/gamma:0',
        'stack_0/block_0/expand_bn/beta:0',
        'stack_0/block_0/depthwise_conv2d/depthwise_kernel:0',
        'stack_0/block_0/depthwise_bn/gamma:0',
        'stack_0/block_0/depthwise_bn/beta:0',
        'stack_0/block_0/project_conv2d/kernel:0',
        'stack_0/block_0/project_bn/gamma:0',
        'stack_0/block_0/project_bn/beta:0',
        'stack_0/block_1/expand_conv2d/kernel:0',
        'stack_0/block_1/expand_bn/gamma:0',
        'stack_0/block_1/expand_bn/beta:0',
        'stack_0/block_1/depthwise_conv2d/depthwise_kernel:0',
        'stack_0/block_1/depthwise_bn/gamma:0',
        'stack_0/block_1/depthwise_bn/beta:0',
        'stack_0/block_1/project_conv2d/kernel:0',
        'stack_0/block_1/project_bn/gamma:0',
        'stack_0/block_1/project_bn/beta:0',
        'stack_0/block_2/expand_conv2d/kernel:0',
        'stack_0/block_2/expand_bn/gamma:0',
        'stack_0/block_2/expand_bn/beta:0',
        'stack_0/block_2/depthwise_conv2d/depthwise_kernel:0',
        'stack_0/block_2/depthwise_bn/gamma:0',
        'stack_0/block_2/depthwise_bn/beta:0',
        'stack_0/block_2/project_conv2d/kernel:0',
        'stack_0/block_2/project_bn/gamma:0',
        'stack_0/block_2/project_bn/beta:0',
        'top_conv2d/kernel:0',
        'top_bn/gamma:0',
        'top_bn/beta:0',
        'logits/kernel:0',
        'logits/bias:0'
    ])

    var_names = set([var.name for var in model.trainable_variables])
    self.assertEqual(var_names, ref_var_names)

  def test_fused_bottleneck_block(self):
    """Test for creating a model with fused bottleneck block arguments."""
    images = tf.zeros((4, 224, 224, 3), dtype=tf.float32)

    tf_keras.backend.set_image_data_format('channels_last')

    blocks = [
        mobilenet_edgetpu_v1_model_blocks.BlockConfig.from_args(
            input_filters=3,
            output_filters=6,
            kernel_size=3,
            num_repeat=3,
            expand_ratio=6,
            strides=(2, 2),
            fused_conv=True,
        )
    ]
    config = mobilenet_edgetpu_v1_model.ModelConfig.from_args(
        blocks=blocks,
        num_classes=10,
        use_se=False,
    )

    model = mobilenet_edgetpu_v1_model.MobilenetEdgeTPU(config)

    outputs = model(images, training=True)
    self.assertEqual((4, 10), outputs.shape)

    var_names = {var.name for var in model.trainable_variables}

    ref_var_names = [
        'stack_0/block_0/fused_conv2d/kernel:0',
        'stack_0/block_1/fused_conv2d/kernel:0',
        'stack_0/block_2/fused_conv2d/kernel:0',
    ]

    for ref_var_name in ref_var_names:
      self.assertIn(ref_var_name, var_names)

  def test_variables(self):
    """Test for variables in blocks to be included in `model.variables`."""
    images = tf.zeros((4, 224, 224, 3), dtype=tf.float32)

    tf_keras.backend.set_image_data_format('channels_last')

    blocks = [
        mobilenet_edgetpu_v1_model_blocks.BlockConfig.from_args(
            input_filters=3,
            output_filters=6,
            kernel_size=3,
            num_repeat=3,
            expand_ratio=6,
            id_skip=False,
            strides=(2, 2),
            se_ratio=0.8,
            fused_conv=False,
        )
    ]
    config = mobilenet_edgetpu_v1_model.ModelConfig.from_args(
        blocks=blocks,
        num_classes=10,
        use_se=True,
    )

    model = mobilenet_edgetpu_v1_model.MobilenetEdgeTPU(config)

    _ = model(images, training=True)
    var_names = {var.name for var in model.variables}

    self.assertIn('stack_0/block_0/depthwise_conv2d/depthwise_kernel:0',
                  var_names)


class MobilenetEdgeTPUBuildTest(tf.test.TestCase):

  def setUp(self):
    super(tf.test.TestCase, self).setUp()
    # Ensure no model duplicates
    tf_keras.backend.clear_session()

  def test_create_mobilenet_edgetpu(self):
    model = mobilenet_edgetpu_v1_model.MobilenetEdgeTPU()
    self.assertEqual(common_modules.count_params(model), 4092713)


class MobilenetEdgeTPUPredictTest(tf.test.TestCase):

  def setUp(self):
    super(tf.test.TestCase, self).setUp()
    # Ensure no model duplicates
    tf_keras.backend.clear_session()

  def _copy_saved_model_to_local(self, model_ckpt):
    # Copy saved model to local first for speed
    tmp_path = '/tmp/saved_model'
    _copy_recursively(model_ckpt, tmp_path)
    return tmp_path

  def _test_prediction(self, model_name, image_size):
    model = mobilenet_edgetpu_v1_model.MobilenetEdgeTPU.from_name(model_name)

    # Predict image filled with zeros
    images = tf.zeros((4, image_size, image_size, 3), dtype=tf.float32)
    pred = model(images, training=False)
    self.assertEqual(pred.shape, (4, 1000))

    # Predict image with loaded weights
    images = preprocessing.load_eval_image(EXAMPLE_IMAGE, image_size)
    images = tf.expand_dims(images, axis=0)
    model_ckpt = os.path.join(CKPTS, model_name)
    model_ckpt = self._copy_saved_model_to_local(model_ckpt)
    model = mobilenet_edgetpu_v1_model.MobilenetEdgeTPU.from_name(
        model_name, model_weights_path=model_ckpt)

    pred = model(images, training=False)
    pred = pred[0].numpy()
    pred_idx, pred_prob = pred.argmax(), pred.max()

    # 388 is 'giant panda' (see labels_map_file)
    self.assertEqual(pred_idx, 388)
    self.assertGreater(pred_prob, 0.75)

  def test_mobilenet_edgetpu_image_shape(self):
    self.skipTest(
        'TODO(b/151324383): Enable once training is supported for mobilenet-edgetpu'
    )
    params = dict(input_channels=5, num_classes=20, rescale_input=False)
    model = mobilenet_edgetpu_v1_model.MobilenetEdgeTPU.from_name(
        'mobilenet_edgetpu', overrides=params)

    images = tf.zeros((6, 100, 38, 5), dtype=tf.float32)
    pred = model(images, training=False)

    self.assertEqual(pred.shape, (6, 20))

  def test_mobilenet_edgetpu_predict(self):
    self.skipTest(
        'TODO(b/151324383): Enable once training is supported for mobilenet-edgetpu'
    )
    self._test_prediction('mobilenet_edgetpu', 224)


if __name__ == '__main__':
  tf.test.main()
