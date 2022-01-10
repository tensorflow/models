# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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

"""Tests for nn_blocks."""

from pyglove.tensorflow import keras
from pyglove.tensorflow import selections
from pyglove.tensorflow.keras import layers
from pyglove.tensorflow.keras.layers import modeling_utils
import tensorflow as tf

from official.projects.tunas.modeling.layers import nn_blocks


class Conv2DTest(tf.test.TestCase):
  """Tests for `nn_blocks.conv2d`."""

  def setUp(self):
    super().setUp()
    bsz, h, w, c = 8, 32, 32, 32
    self.input_tensor = tf.random.uniform(shape=[bsz, h, w, c])

  def testBareConv2D(self):
    """Test for bare conv2d without normalization and activation."""
    self.assertAllClose(
        nn_blocks.conv2d(
            kernel_size=(3, 3),
            filters=8,
            strides=(1, 1),
            name='Conv',
            kernel_initializer=keras.initializers.ones(),
            kernel_regularizer=keras.regularizers.l2(0.1))(self.input_tensor),
        layers.Conv2D(
            kernel_size=(3, 3),
            filters=8,
            strides=(1, 1),
            padding='same',
            kernel_initializer=keras.initializers.ones(),
            kernel_regularizer=keras.regularizers.l2(0.1),
            name='Conv')(self.input_tensor))

  def testConv2DWithNormAndActivation(self):
    """Test conv2d with normalization and activation."""
    # Conv2d-BN-Relu using layers objects.
    self.assertAllClose(
        nn_blocks.conv2d(
            kernel_size=(3, 3),
            filters=8,
            strides=(2, 2),
            normalization=layers.BatchNormalization(),
            activation=layers.ReLU(),
            kernel_initializer=keras.initializers.ones(),
            kernel_regularizer=keras.regularizers.l2(0.1),
            name='Conv')(self.input_tensor),
        layers.Sequential([
            layers.Conv2D(
                kernel_size=(3, 3),
                filters=8,
                strides=(2, 2),
                padding='same',
                kernel_initializer=keras.initializers.ones(),
                kernel_regularizer=keras.regularizers.l2(0.1)),
            layers.BatchNormalization(),
            layers.ReLU()
        ], name='Conv')(self.input_tensor))

  def testConv2DWithTunableKernelSize(self):
    """Test conv2d with normalization and activation."""
    # Conv2d-BN-Relu using layers objects.
    kernel_size = selections.select(
        [(3, 3), (5, 5)], tf.constant(0, dtype=tf.int32))
    self.assertAllClose(
        nn_blocks.conv2d(
            kernel_size=kernel_size,
            filters=8,
            strides=(2, 2),
            normalization=layers.BatchNormalization(),
            activation=layers.ReLU(),
            kernel_initializer=keras.initializers.ones(),
            kernel_regularizer=keras.regularizers.l2(0.1),
            name='Conv')(self.input_tensor),
        layers.Switch(
            candidates=[
                layers.Sequential([
                    layers.Conv2D(
                        kernel_size=(3, 3),
                        filters=8,
                        strides=(2, 2),
                        padding='same',
                        kernel_initializer=keras.initializers.ones(),
                        kernel_regularizer=keras.regularizers.l2(0.1)),
                    layers.BatchNormalization(),
                    layers.ReLU()
                ], name='branch_0'),
                layers.Sequential([
                    layers.Conv2D(
                        kernel_size=(5, 5),
                        filters=8,
                        strides=(2, 2),
                        padding='same',
                        kernel_initializer=keras.initializers.ones(),
                        kernel_regularizer=keras.regularizers.l2(0.1)),
                    layers.BatchNormalization(),
                    layers.ReLU()
                ], name='branch_1')],
            selected_index=kernel_size.index,
            name='Conv')(self.input_tensor))

  def testConv2DWithTunableGroups(self):
    """Test conv2d with normalization and activation."""
    # Conv2d-BN-Relu using layers objects.
    groups = selections.select(
        [1, 2], tf.constant(0, dtype=tf.int32))
    self.assertAllClose(
        nn_blocks.conv2d(
            kernel_size=3,
            filters=8,
            strides=(2, 2),
            groups=groups,
            normalization=layers.BatchNormalization(),
            activation=layers.ReLU(),
            kernel_initializer=keras.initializers.ones(),
            kernel_regularizer=keras.regularizers.l2(0.1),
            name='Conv')(self.input_tensor),
        layers.Switch(
            candidates=[
                layers.Sequential([
                    layers.Conv2D(
                        kernel_size=3,
                        filters=8,
                        strides=(2, 2),
                        padding='same',
                        kernel_initializer=keras.initializers.ones(),
                        kernel_regularizer=keras.regularizers.l2(0.1),
                        groups=1),
                    layers.BatchNormalization(),
                    layers.ReLU()
                ], name='group_branch_0'),
                layers.Sequential([
                    layers.Conv2D(
                        kernel_size=3,
                        filters=8,
                        strides=(2, 2),
                        padding='same',
                        kernel_initializer=keras.initializers.ones(),
                        kernel_regularizer=keras.regularizers.l2(0.1),
                        groups=2),
                    layers.BatchNormalization(),
                    layers.ReLU()
                ], name='group_branch_1')],
            selected_index=groups.index,
            name='Conv')(self.input_tensor))


class DepthwiseConv2DTest(tf.test.TestCase):
  """Tests for `nn_blocks.depthwise_conv2d`."""

  def setUp(self):
    super().setUp()
    bsz, h, w, c = 8, 32, 32, 32
    self.input_tensor = tf.random.uniform(shape=[bsz, h, w, c])

  def testBareDepthwiseConv2D(self):
    """Test for depthwise_conv2d without normalization and activation."""
    self.assertAllClose(
        nn_blocks.depthwise_conv2d(
            kernel_size=(3, 3),
            strides=(1, 1),
            depthwise_initializer=keras.initializers.ones(),
            depthwise_regularizer=keras.regularizers.l2(0.1),
            name='DepthwiseConv')(self.input_tensor),
        layers.DepthwiseConv2D(
            kernel_size=(3, 3),
            strides=(1, 1),
            padding='same',
            depthwise_initializer=keras.initializers.ones(),
            depthwise_regularizer=keras.regularizers.l2(0.1),
            name='DepthwiseConv')(self.input_tensor))

  def testDepthwiseConvWithNormAndActivation(self):
    """Test for depthwise_conv2d with normalization and activation."""
    # DepthwiseConv2d-BN-Relu using layers.Object.
    self.assertAllClose(
        nn_blocks.depthwise_conv2d(
            kernel_size=(3, 3),
            strides=(1, 1),
            depthwise_initializer=keras.initializers.ones(),
            depthwise_regularizer=keras.regularizers.l2(0.1),
            normalization=layers.BatchNormalization(),
            activation=layers.ReLU(),
            name='DepthwiseConv'
        )(self.input_tensor),
        layers.Sequential([
            layers.DepthwiseConv2D(
                kernel_size=(3, 3),
                strides=(1, 1),
                padding='same',
                depthwise_initializer=keras.initializers.ones(),
                depthwise_regularizer=keras.regularizers.l2(0.1)),
            layers.BatchNormalization(),
            layers.ReLU()
        ], name='DepthwiseConv')(self.input_tensor))

  def testDepthwiseConv2DWithTunableKernelSize(self):
    """Test conv2d with normalization and activation."""
    # Conv2d-BN-Relu using layers objects.
    kernel_size = selections.select(
        [(3, 3), (5, 5)], tf.constant(0, dtype=tf.int32))
    self.assertAllClose(
        nn_blocks.depthwise_conv2d(
            kernel_size=kernel_size,
            strides=(1, 1),
            depthwise_initializer=keras.initializers.ones(),
            depthwise_regularizer=keras.regularizers.l2(0.1),
            normalization=layers.BatchNormalization(),
            activation=layers.ReLU(),
            name='DepthwiseConv'
        )(self.input_tensor),
        layers.Switch(
            candidates=[
                layers.Sequential([
                    layers.DepthwiseConv2D(
                        kernel_size=(3, 3),
                        strides=(1, 1),
                        padding='same',
                        depthwise_initializer=keras.initializers.ones(),
                        depthwise_regularizer=keras.regularizers.l2(0.1)),
                    layers.BatchNormalization(),
                    layers.ReLU()
                ], name='branch_0'),
                layers.Sequential([
                    layers.DepthwiseConv2D(
                        kernel_size=(5, 5),
                        strides=(1, 1),
                        padding='same',
                        depthwise_initializer=keras.initializers.ones(),
                        depthwise_regularizer=keras.regularizers.l2(0.1)),
                    layers.BatchNormalization(),
                    layers.ReLU()
                ], name='branch_1')],
            selected_index=kernel_size.index,
            name='DepthwiseConv')(self.input_tensor))


class InvertedBottleneckTest(tf.test.TestCase):
  """Tests for `nn_blocks.inverted_bottleneck`."""

  def setUp(self):
    super().setUp()
    bsz, h, w, c = 8, 32, 32, 32
    self.input_tensor = tf.random.uniform(shape=[bsz, h, w, c])

  def testRegularInvertedBottleneck(self):
    """Test regular inverted bottleneck without tunable hyper-parameters."""
    # Regular inverted bottleneck.
    layer = nn_blocks.inverted_bottleneck(
        kernel_size=(3, 3),
        filters=4,
        expansion_factor=2,
        normalization=layers.BatchNormalization(),
        kernel_initializer=keras.initializers.ones(),
        kernel_regularizer=keras.regularizers.l2(0.1),
        depthwise_initializer=keras.initializers.ones(),
        depthwise_regularizer=keras.regularizers.l2(0.1),
        post_expansion=layers.identity(),
        post_depthwise=layers.identity(),
        post_projection=layers.identity())  # pylint: disable=unnecessary-lambda
    self.assertAllClose(
        layer(self.input_tensor),
        layers.Sequential([
            nn_blocks.conv2d(
                kernel_size=(1, 1),
                filters=nn_blocks._expand_filters(expansion_factor=2),  # pylint: disable=no-value-for-parameter
                normalization=layers.BatchNormalization(),
                activation=layers.ReLU(),
                use_bias=False,
                kernel_initializer=keras.initializers.ones(),
                kernel_regularizer=keras.regularizers.l2(0.1)),
            layers.identity(),
            nn_blocks.depthwise_conv2d(
                kernel_size=(3, 3),
                normalization=layers.BatchNormalization(),
                activation=layers.ReLU(),
                depthwise_initializer=keras.initializers.ones(),
                depthwise_regularizer=keras.regularizers.l2(0.1),
                use_bias=False,),
            layers.identity(),
            nn_blocks.conv2d(
                kernel_size=(1, 1),
                filters=4,
                normalization=layers.BatchNormalization(),
                activation=None,
                use_bias=False,
                kernel_initializer=keras.initializers.ones(),
                kernel_regularizer=keras.regularizers.l2(0.1)),
            layers.identity(),
        ])(self.input_tensor))

  def testSeparateTowers(self):
    """Test `nn_blocks.inverted_bottleneck`."""
    op_sel = tf.constant(0, dtype=tf.int32)
    filters_sel = tf.constant(0, dtype=tf.int32)
    self.assertAllClose(
        nn_blocks.inverted_bottleneck(
            kernel_size=selections.select([(3, 3), (5, 5)], op_sel),
            filters=selections.select([2, 4], filters_sel),
            expansion_factor=3,
            normalization=layers.BatchNormalization(),
            kernel_initializer=keras.initializers.ones(),
            kernel_regularizer=keras.regularizers.l2(0.1),
            depthwise_initializer=keras.initializers.ones(),
            depthwise_regularizer=keras.regularizers.l2(0.1),
            )(self.input_tensor),
        layers.Switch([
            layers.Sequential([
                nn_blocks.conv2d(
                    kernel_size=(1, 1),
                    filters=nn_blocks._expand_filters(
                        expansion_factor=3),  # pylint: disable=no-value-for-parameter
                    normalization=layers.BatchNormalization(),
                    activation=layers.ReLU(),
                    use_bias=False,
                    kernel_initializer=keras.initializers.ones(),
                    kernel_regularizer=keras.regularizers.l2(0.1)),
                nn_blocks.depthwise_conv2d(
                    kernel_size=(3, 3),
                    normalization=layers.BatchNormalization(),
                    activation=layers.ReLU(),
                    use_bias=False,
                    depthwise_initializer=keras.initializers.ones(),
                    depthwise_regularizer=keras.regularizers.l2(0.1)),
                nn_blocks.conv2d(
                    kernel_size=(1, 1),
                    filters=selections.select([2, 4], filters_sel),
                    normalization=layers.BatchNormalization(),
                    activation=None,
                    use_bias=False,
                    kernel_initializer=keras.initializers.ones(),
                    kernel_regularizer=keras.regularizers.l2(0.1)),
            ], name='branch0'),
            layers.Sequential([
                nn_blocks.conv2d(
                    kernel_size=(1, 1),
                    filters=nn_blocks._expand_filters(
                        expansion_factor=3),  # pylint: disable=no-value-for-parameter
                    normalization=layers.BatchNormalization(),
                    activation=layers.ReLU(),
                    use_bias=False,
                    kernel_initializer=keras.initializers.ones(),
                    kernel_regularizer=keras.regularizers.l2(0.1)),
                nn_blocks.depthwise_conv2d(
                    kernel_size=(5, 5),
                    normalization=layers.BatchNormalization(),
                    activation=layers.ReLU(),
                    use_bias=False,
                    depthwise_initializer=keras.initializers.ones(),
                    depthwise_regularizer=keras.regularizers.l2(0.1)),
                nn_blocks.conv2d(
                    kernel_size=(1, 1),
                    filters=selections.select([2, 4], filters_sel),
                    normalization=layers.BatchNormalization(),
                    activation=None,
                    use_bias=False,
                    kernel_initializer=keras.initializers.ones(),
                    kernel_regularizer=keras.regularizers.l2(0.1)),
            ], name='branch1')
        ], selected_index=op_sel)(self.input_tensor))


class SqueezeAndExciteTest(tf.test.TestCase):
  """Tests for `nn_blocks.SqueezeExcitation`."""

  def testFixedRatio(self):
    """Test fixed ratio."""

    xlayer = nn_blocks.SqueezeExcitation(0.25)
    inputs = tf.ones(shape=(1, 2, 2, 3))
    xlayer(inputs)
    kernels = xlayer.trainable_variables
    self.assertEqual(kernels[0].shape.as_list(), [1, 1, 3, 8])
    self.assertEqual(kernels[1].shape.as_list(), [8])
    self.assertEqual(kernels[2].shape.as_list(), [1, 1, 8, 3])
    self.assertEqual(kernels[3].shape.as_list(), [3])


class FusedInvertedBottleneckTest(tf.test.TestCase):
  """Tests for `nn_blocks.fused_inverted_bottleneck`."""

  def setUp(self):
    super().setUp()
    bsz, h, w, c = 8, 32, 32, 32
    self.input_tensor = tf.random.uniform(shape=[bsz, h, w, c])

  def testRegularFusedInvertedBottleneck(self):
    """Test regular inverted bottleneck without tunable hyper-parameters."""
    # Regular inverted bottleneck.
    layer = nn_blocks.fused_inverted_bottleneck(
        kernel_size=(3, 3),
        filters=4,
        expansion_factor=2,
        normalization=layers.BatchNormalization(),
        kernel_initializer=keras.initializers.ones(),
        kernel_regularizer=keras.regularizers.l2(0.1),
        post_fusion=layers.identity(),
        post_projection=layers.identity())  # pylint: disable=unnecessary-lambda
    self.assertAllClose(
        layer(self.input_tensor),
        layers.Sequential([
            nn_blocks.conv2d(
                kernel_size=(3, 3),
                filters=nn_blocks._expand_filters(expansion_factor=2),  # pylint: disable=no-value-for-parameter
                normalization=layers.BatchNormalization(),
                activation=layers.ReLU(),
                use_bias=False,
                kernel_initializer=keras.initializers.ones(),
                kernel_regularizer=keras.regularizers.l2(0.1)),
            layers.identity(),
            nn_blocks.conv2d(
                kernel_size=(1, 1),
                filters=4,
                normalization=layers.BatchNormalization(),
                activation=None,
                use_bias=False,
                kernel_initializer=keras.initializers.ones(),
                kernel_regularizer=keras.regularizers.l2(0.1)),
            layers.identity(),
        ])(self.input_tensor))

  def testSeparateTowers(self):
    """Test `nn_blocks.inverted_bottleneck`."""
    op_sel = tf.constant(0, dtype=tf.int32)
    filters_sel = tf.constant(0, dtype=tf.int32)
    self.assertAllClose(
        nn_blocks.fused_inverted_bottleneck(
            kernel_size=selections.select([(3, 3), (5, 5)], op_sel),
            filters=selections.select([2, 4], filters_sel),
            expansion_factor=3,
            normalization=layers.BatchNormalization(),
            kernel_initializer=keras.initializers.ones(),
            kernel_regularizer=keras.regularizers.l2(0.1),
            )(self.input_tensor),
        layers.Switch([
            layers.Sequential([
                nn_blocks.conv2d(
                    kernel_size=(3, 3),
                    filters=nn_blocks._expand_filters(
                        expansion_factor=3),  # pylint: disable=no-value-for-parameter
                    normalization=layers.BatchNormalization(),
                    activation=layers.ReLU(),
                    use_bias=False,
                    kernel_initializer=keras.initializers.ones(),
                    kernel_regularizer=keras.regularizers.l2(0.1)),
                nn_blocks.conv2d(
                    kernel_size=(1, 1),
                    filters=selections.select([2, 4], filters_sel),
                    normalization=layers.BatchNormalization(),
                    activation=None,
                    use_bias=False,
                    kernel_initializer=keras.initializers.ones(),
                    kernel_regularizer=keras.regularizers.l2(0.1)),
            ], name='branch0'),
            layers.Sequential([
                nn_blocks.conv2d(
                    kernel_size=(5, 5),
                    filters=nn_blocks._expand_filters(
                        expansion_factor=3),  # pylint: disable=no-value-for-parameter
                    normalization=layers.BatchNormalization(),
                    activation=layers.ReLU(),
                    use_bias=False,
                    kernel_initializer=keras.initializers.ones(),
                    kernel_regularizer=keras.regularizers.l2(0.1)),
                nn_blocks.conv2d(
                    kernel_size=(1, 1),
                    filters=selections.select([2, 4], filters_sel),
                    normalization=layers.BatchNormalization(),
                    activation=None,
                    use_bias=False,
                    kernel_initializer=keras.initializers.ones(),
                    kernel_regularizer=keras.regularizers.l2(0.1)),
            ], name='branch1')
        ], selected_index=op_sel)(self.input_tensor))


class TuckerBottleneckTest(tf.test.TestCase):
  """Tests for `nn_blocks.inverted_bottleneck`."""

  def setUp(self):
    super().setUp()
    bsz, h, w, c = 8, 32, 32, 32
    self.input_tensor = tf.random.uniform(shape=[bsz, h, w, c])

  def testRegularTuckerBottleneck(self):
    """Test regular inverted bottleneck without tunable hyper-parameters."""
    # Regular inverted bottleneck.
    layer = nn_blocks.tucker_bottleneck(
        kernel_size=(3, 3),
        filters=4,
        input_scale_ratio=2.0,
        output_scale_ratio=4.0,
        activation=layers.ReLU(),
        normalization=layers.BatchNormalization(),
        kernel_initializer=keras.initializers.ones(),
        kernel_regularizer=keras.regularizers.l2(0.1))  # pylint: disable=unnecessary-lambda
    self.assertAllClose(
        layer(self.input_tensor),
        layers.Sequential([
            nn_blocks.conv2d(
                kernel_size=(1, 1),
                filters=nn_blocks._scale_filters(ratio=2.0, base=8),  # pylint: disable=no-value-for-parameter
                normalization=layers.BatchNormalization(),
                activation=layers.ReLU(),
                use_bias=False,
                kernel_initializer=keras.initializers.ones(),
                kernel_regularizer=keras.regularizers.l2(0.1)),
            nn_blocks.conv2d(
                kernel_size=(3, 3),
                filters=modeling_utils.scale_filters(4, 4.0, 8),  # pylint: disable=no-value-for-parameter
                normalization=layers.BatchNormalization(),
                activation=layers.ReLU(),
                use_bias=False,
                kernel_initializer=keras.initializers.ones(),
                kernel_regularizer=keras.regularizers.l2(0.1)),
            nn_blocks.conv2d(
                kernel_size=(1, 1),
                filters=4,
                normalization=layers.BatchNormalization(),
                activation=None,
                use_bias=False,
                kernel_initializer=keras.initializers.ones(),
                kernel_regularizer=keras.regularizers.l2(0.1)),
        ])(self.input_tensor))

if __name__ == '__main__':
  tf.test.main()
