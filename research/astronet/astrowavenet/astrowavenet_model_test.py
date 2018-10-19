# Copyright 2018 The TensorFlow Authors.
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

"""Tests for astrowavenet."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from astronet.util import configdict
from astrowavenet import astrowavenet_model


class AstrowavenetTest(tf.test.TestCase):

  def assertShapeEquals(self, shape, tensor_or_array):
    """Asserts that a Tensor or Numpy array has the expected shape.

    Args:
      shape: Numpy array or anything that can be converted to one.
      tensor_or_array: tf.Tensor, tf.Variable, or Numpy array.
    """
    if isinstance(tensor_or_array, (np.ndarray, np.generic)):
      self.assertAllEqual(shape, tensor_or_array.shape)
    elif isinstance(tensor_or_array, (tf.Tensor, tf.Variable)):
      self.assertAllEqual(shape, tensor_or_array.shape.as_list())
    else:
      raise TypeError("tensor_or_array must be a Tensor or Numpy ndarray")

  def test_build_model(self):
    time_series_length = 9
    input_num_features = 8
    context_num_features = 7

    input_placeholder = tf.placeholder(
        dtype=tf.float32,
        shape=[None, time_series_length, input_num_features],
        name="input")
    context_placeholder = tf.placeholder(
        dtype=tf.float32,
        shape=[None, time_series_length, context_num_features],
        name="context")
    features = {
        "autoregressive_input": input_placeholder,
        "conditioning_stack": context_placeholder
    }
    mode = tf.estimator.ModeKeys.TRAIN
    hparams = configdict.ConfigDict({
        "dilation_kernel_width": 2,
        "skip_output_dim": 6,
        "preprocess_output_size": 3,
        "preprocess_kernel_width": 5,
        "num_residual_blocks": 2,
        "dilation_rates": [1, 2, 4],
        "output_distribution": {
            "type": "normal",
            "min_scale": 0.001,
        }
    })

    model = astrowavenet_model.AstroWaveNet(features, hparams, mode)
    model.build()

    variables = {v.op.name: v for v in tf.trainable_variables()}

    # Verify variable shapes in two residual blocks.

    var = variables["preprocess/causal_conv/kernel"]
    self.assertShapeEquals((5, 8, 3), var)
    var = variables["preprocess/causal_conv/bias"]
    self.assertShapeEquals((3,), var)

    var = variables["block_0/dilation_1/filter/causal_conv/kernel"]
    self.assertShapeEquals((2, 3, 3), var)
    var = variables["block_0/dilation_1/filter/causal_conv/bias"]
    self.assertShapeEquals((3,), var)
    var = variables["block_0/dilation_1/filter/conv1x1/kernel"]
    self.assertShapeEquals((1, 7, 3), var)
    var = variables["block_0/dilation_1/filter/conv1x1/bias"]
    self.assertShapeEquals((3,), var)
    var = variables["block_0/dilation_1/gate/causal_conv/kernel"]
    self.assertShapeEquals((2, 3, 3), var)
    var = variables["block_0/dilation_1/gate/causal_conv/bias"]
    self.assertShapeEquals((3,), var)
    var = variables["block_0/dilation_1/gate/conv1x1/kernel"]
    self.assertShapeEquals((1, 7, 3), var)
    var = variables["block_0/dilation_1/gate/conv1x1/bias"]
    self.assertShapeEquals((3,), var)
    var = variables["block_0/dilation_1/residual/conv1x1/kernel"]
    self.assertShapeEquals((1, 3, 3), var)
    var = variables["block_0/dilation_1/residual/conv1x1/bias"]
    self.assertShapeEquals((3,), var)
    var = variables["block_0/dilation_1/skip/conv1x1/kernel"]
    self.assertShapeEquals((1, 3, 6), var)
    var = variables["block_0/dilation_1/skip/conv1x1/bias"]
    self.assertShapeEquals((6,), var)

    var = variables["block_1/dilation_4/filter/causal_conv/kernel"]
    self.assertShapeEquals((2, 3, 3), var)
    var = variables["block_1/dilation_4/filter/causal_conv/bias"]
    self.assertShapeEquals((3,), var)
    var = variables["block_1/dilation_4/filter/conv1x1/kernel"]
    self.assertShapeEquals((1, 7, 3), var)
    var = variables["block_1/dilation_4/filter/conv1x1/bias"]
    self.assertShapeEquals((3,), var)
    var = variables["block_1/dilation_4/gate/causal_conv/kernel"]
    self.assertShapeEquals((2, 3, 3), var)
    var = variables["block_1/dilation_4/gate/causal_conv/bias"]
    self.assertShapeEquals((3,), var)
    var = variables["block_1/dilation_4/gate/conv1x1/kernel"]
    self.assertShapeEquals((1, 7, 3), var)
    var = variables["block_1/dilation_4/gate/conv1x1/bias"]
    self.assertShapeEquals((3,), var)
    var = variables["block_1/dilation_4/residual/conv1x1/kernel"]
    self.assertShapeEquals((1, 3, 3), var)
    var = variables["block_1/dilation_4/residual/conv1x1/bias"]
    self.assertShapeEquals((3,), var)
    var = variables["block_1/dilation_4/skip/conv1x1/kernel"]
    self.assertShapeEquals((1, 3, 6), var)
    var = variables["block_1/dilation_4/skip/conv1x1/bias"]
    self.assertShapeEquals((6,), var)

    var = variables["postprocess/conv1x1/kernel"]
    self.assertShapeEquals((1, 6, 6), var)
    var = variables["postprocess/conv1x1/bias"]
    self.assertShapeEquals((6,), var)
    var = variables["dist_params/conv1x1/kernel"]
    self.assertShapeEquals((1, 6, 16), var)
    var = variables["dist_params/conv1x1/bias"]
    self.assertShapeEquals((16,), var)

    # Verify total number of trainable parameters.

    num_preprocess_params = (
        hparams.preprocess_kernel_width * input_num_features *
        hparams.preprocess_output_size + hparams.preprocess_output_size)

    num_gated_params = (
        hparams.dilation_kernel_width * hparams.preprocess_output_size *
        hparams.preprocess_output_size + hparams.preprocess_output_size +
        1 * context_num_features * hparams.preprocess_output_size +
        hparams.preprocess_output_size) * 2
    num_residual_params = (
        1 * hparams.preprocess_output_size * hparams.preprocess_output_size +
        hparams.preprocess_output_size)
    num_skip_params = (
        1 * hparams.preprocess_output_size * hparams.skip_output_dim +
        hparams.skip_output_dim)
    num_block_params = (
        num_gated_params + num_residual_params + num_skip_params) * len(
            hparams.dilation_rates) * hparams.num_residual_blocks

    num_postprocess_params = (
        1 * hparams.skip_output_dim * hparams.skip_output_dim +
        hparams.skip_output_dim)

    num_dist_params = (1 * hparams.skip_output_dim * 2 * input_num_features +
                       2 * input_num_features)

    total_params = (
        num_preprocess_params + num_block_params + num_postprocess_params +
        num_dist_params)

    total_retrieved_params = 0
    for v in tf.trainable_variables():
      total_retrieved_params += np.prod(v.shape)

    self.assertEqual(total_params, total_retrieved_params)

    # Verify model runs and outputs losses of correct shape.

    scaffold = tf.train.Scaffold()
    scaffold.finalize()
    with self.cached_session() as sess:
      sess.run([scaffold.init_op, scaffold.local_init_op])
      step = sess.run(model.global_step)
      self.assertEqual(0, step)

      batch_size = 11
      feed_dict = {
          input_placeholder:
              np.random.random((batch_size, time_series_length,
                                input_num_features)),
          context_placeholder:
              np.random.random((batch_size, time_series_length,
                                context_num_features))
      }
      batch_losses, per_example_loss, total_loss = sess.run(
          [model.batch_losses, model.per_example_loss, model.total_loss],
          feed_dict=feed_dict)
      self.assertShapeEquals(
          (batch_size, time_series_length, input_num_features), batch_losses)
      self.assertShapeEquals((batch_size,), per_example_loss)
      self.assertShapeEquals((), total_loss)

  def test_build_model_categorical(self):
    time_series_length = 9
    input_num_features = 8
    context_num_features = 7

    input_placeholder = tf.placeholder(
        dtype=tf.float32,
        shape=[None, time_series_length, input_num_features],
        name="input")
    context_placeholder = tf.placeholder(
        dtype=tf.float32,
        shape=[None, time_series_length, context_num_features],
        name="context")
    features = {
        "autoregressive_input": input_placeholder,
        "conditioning_stack": context_placeholder
    }
    mode = tf.estimator.ModeKeys.TRAIN
    hparams = configdict.ConfigDict({
        "dilation_kernel_width": 2,
        "skip_output_dim": 6,
        "preprocess_output_size": 3,
        "preprocess_kernel_width": 5,
        "num_residual_blocks": 2,
        "dilation_rates": [1, 2, 4],
        "output_distribution": {
            "type": "categorical",
            "num_classes": 256,
            "min_quantization_value": -1,
            "max_quantization_value": 1
        }
    })

    model = astrowavenet_model.AstroWaveNet(features, hparams, mode)
    model.build()

    variables = {v.op.name: v for v in tf.trainable_variables()}

    var = variables["dist_params/conv1x1/kernel"]
    self.assertShapeEquals(
        (1, hparams.skip_output_dim,
         hparams.output_distribution.num_classes * input_num_features), var)
    var = variables["dist_params/conv1x1/bias"]
    self.assertShapeEquals(
        (hparams.output_distribution.num_classes * input_num_features,), var)

    # Verify model runs and outputs losses of correct shape.

    scaffold = tf.train.Scaffold()
    scaffold.finalize()
    with self.cached_session() as sess:
      sess.run([scaffold.init_op, scaffold.local_init_op])
      step = sess.run(model.global_step)
      self.assertEqual(0, step)

      batch_size = 11
      feed_dict = {
          input_placeholder:
              np.random.random((batch_size, time_series_length,
                                input_num_features)),
          context_placeholder:
              np.random.random((batch_size, time_series_length,
                                context_num_features))
      }
      batch_losses, per_example_loss, total_loss = sess.run(
          [model.batch_losses, model.per_example_loss, model.total_loss],
          feed_dict=feed_dict)
      self.assertShapeEquals(
          (batch_size, time_series_length, input_num_features), batch_losses)
      self.assertShapeEquals((batch_size,), per_example_loss)
      self.assertShapeEquals((), total_loss)

  def test_output_normal(self):
    time_series_length = 6
    input_num_features = 2
    context_num_features = 7

    input_placeholder = tf.placeholder(
        dtype=tf.float32,
        shape=[None, time_series_length, input_num_features],
        name="input")
    context_placeholder = tf.placeholder(
        dtype=tf.float32,
        shape=[None, time_series_length, context_num_features],
        name="context")
    features = {
        "autoregressive_input": input_placeholder,
        "conditioning_stack": context_placeholder
    }
    mode = tf.estimator.ModeKeys.TRAIN
    hparams = configdict.ConfigDict({
        "dilation_kernel_width": 2,
        "skip_output_dim": 6,
        "preprocess_output_size": 3,
        "preprocess_kernel_width": 5,
        "num_residual_blocks": 2,
        "dilation_rates": [1, 2, 4],
        "output_distribution": {
            "type": "normal",
            "min_scale": 0,
        }
    })

    model = astrowavenet_model.AstroWaveNet(features, hparams, mode)
    model.build()

    # Model predicts loc and scale.
    self.assertItemsEqual(["loc", "scale"], model.dist_params.keys())
    self.assertShapeEquals((None, time_series_length, input_num_features),
                           model.dist_params["loc"])
    self.assertShapeEquals((None, time_series_length, input_num_features),
                           model.dist_params["scale"])

    scaffold = tf.train.Scaffold()
    scaffold.finalize()
    with self.cached_session() as sess:
      sess.run([scaffold.init_op, scaffold.local_init_op])
      step = sess.run(model.global_step)
      self.assertEqual(0, step)

      feed_dict = {
          input_placeholder: [
              [[1, 9], [1, 9], [1, 9], [1, 9], [1, 9], [1, 9]],
              [[2, 8], [2, 8], [2, 8], [2, 8], [2, 8], [2, 8]],
          ],
          # Context is not needed since we explicitly feed the dist params.
          model.dist_params["loc"]: [
              [[1, 8], [1, 8], [1, 8], [1, 8], [1, 8], [1, 8]],
              [[2, 9], [2, 9], [2, 9], [2, 9], [2, 9], [2, 9]],
          ],
          model.dist_params["scale"]: [
              [[0.1, 0.1], [0.2, 0.2], [0.5, 0.5], [1, 1], [2, 2], [5, 5]],
              [[0.1, 0.1], [0.2, 0.2], [0.5, 0.5], [1, 1], [2, 2], [5, 5]],
          ],
      }
      batch_losses, per_example_loss, num_examples, total_loss = sess.run(
          [
              model.batch_losses, model.per_example_loss,
              model.num_nonzero_weight_examples, model.total_loss
          ],
          feed_dict=feed_dict)
      np.testing.assert_array_almost_equal(
          [[[-1.38364656, 48.61635344], [-0.69049938, 11.80950062],
            [0.22579135, 2.22579135], [0.91893853, 1.41893853],
            [1.61208571, 1.73708571], [2.52837645, 2.54837645]],
           [[-1.38364656, 48.61635344], [-0.69049938, 11.80950062],
            [0.22579135, 2.22579135], [0.91893853, 1.41893853],
            [1.61208571, 1.73708571], [2.52837645, 2.54837645]]], batch_losses)
      np.testing.assert_array_almost_equal([5.96392435, 5.96392435],
                                           per_example_loss)
      np.testing.assert_almost_equal(2, num_examples)
      np.testing.assert_almost_equal(5.96392435, total_loss)

  def test_output_categorical(self):
    time_series_length = 3
    input_num_features = 1
    context_num_features = 7
    num_classes = 4  # For quantized categorical output predictions.

    input_placeholder = tf.placeholder(
        dtype=tf.float32,
        shape=[None, time_series_length, input_num_features],
        name="input")
    context_placeholder = tf.placeholder(
        dtype=tf.float32,
        shape=[None, time_series_length, context_num_features],
        name="context")
    features = {
        "autoregressive_input": input_placeholder,
        "conditioning_stack": context_placeholder
    }
    mode = tf.estimator.ModeKeys.TRAIN
    hparams = configdict.ConfigDict({
        "dilation_kernel_width": 2,
        "skip_output_dim": 6,
        "preprocess_output_size": 3,
        "preprocess_kernel_width": 5,
        "num_residual_blocks": 2,
        "dilation_rates": [1, 2, 4],
        "output_distribution": {
            "type": "categorical",
            "min_scale": 0,
            "num_classes": num_classes,
            "min_quantization_value": 0,
            "max_quantization_value": 1
        }
    })

    model = astrowavenet_model.AstroWaveNet(features, hparams, mode)
    model.build()

    self.assertItemsEqual(["logits"], model.dist_params.keys())
    self.assertShapeEquals(
        (None, time_series_length, input_num_features, num_classes),
        model.dist_params["logits"])

    scaffold = tf.train.Scaffold()
    scaffold.finalize()
    with self.cached_session() as sess:
      sess.run([scaffold.init_op, scaffold.local_init_op])
      step = sess.run(model.global_step)
      self.assertEqual(0, step)

      feed_dict = {
          input_placeholder: [
              [[0], [0], [0]],  # min_quantization_value
              [[0.2], [0.2], [0.2]],  # Within bucket.
              [[0.25], [0.25], [0.25]],  # On bucket boundary.
              [[0.5], [0.5], [0.5]],  # On bucket boundary.
              [[0.8], [0.8], [0.8]],  # Within bucket.
              [[1], [1], [1]],  # max_quantization_value
              [[-0.1], [1.5], [200]],  # Outside range: will be clipped.
          ],
          # Context is not needed since we explicitly feed the dist params.
          model.dist_params["logits"]: [
              [[[1, 0, 0, 0]], [[0, 1, 0, 0]], [[0, 0, 0, 1]]],
              [[[1, 0, 0, 0]], [[0, 1, 0, 0]], [[0, 0, 0, 1]]],
              [[[0, 1, 0, 0]], [[1, 0, 0, 0]], [[0, 0, 1, 0]]],
              [[[0, 0, 1, 0]], [[0, 1, 0, 0]], [[0, 0, 0, 1]]],
              [[[0, 0, 0, 1]], [[1, 0, 0, 0]], [[1, 0, 0, 0]]],
              [[[0, 0, 0, 1]], [[0, 1, 0, 0]], [[0, 0, 1, 0]]],
              [[[1, 0, 0, 0]], [[0, 0, 1, 0]], [[0, 1, 0, 0]]],
          ],
      }
      (target, batch_losses, per_example_loss, num_examples,
       total_loss) = sess.run([
           model.autoregressive_target, model.batch_losses,
           model.per_example_loss, model.num_nonzero_weight_examples,
           model.total_loss
       ],
                              feed_dict=feed_dict)
      np.testing.assert_array_almost_equal([
          [[0], [0], [0]],
          [[0], [0], [0]],
          [[1], [1], [1]],
          [[2], [2], [2]],
          [[3], [3], [3]],
          [[3], [3], [3]],
          [[0], [3], [3]],
      ], target)
      np.testing.assert_array_almost_equal([
          [[0.74366838], [1.74366838], [1.74366838]],
          [[0.74366838], [1.74366838], [1.74366838]],
          [[0.74366838], [1.74366838], [1.74366838]],
          [[0.74366838], [1.74366838], [1.74366838]],
          [[0.74366838], [1.74366838], [1.74366838]],
          [[0.74366838], [1.74366838], [1.74366838]],
          [[0.74366838], [1.74366838], [1.74366838]],
      ], batch_losses)
      np.testing.assert_array_almost_equal([
          1.41033504, 1.41033504, 1.41033504, 1.41033504, 1.41033504,
          1.41033504, 1.41033504
      ], per_example_loss)
      np.testing.assert_almost_equal(7, num_examples)
      np.testing.assert_almost_equal(1.41033504, total_loss)

  def test_output_weighted(self):
    time_series_length = 6
    input_num_features = 2
    context_num_features = 7

    input_placeholder = tf.placeholder(
        dtype=tf.float32,
        shape=[None, time_series_length, input_num_features],
        name="input")
    weights_placeholder = tf.placeholder(
        dtype=tf.float32,
        shape=[None, time_series_length, input_num_features],
        name="input")
    context_placeholder = tf.placeholder(
        dtype=tf.float32,
        shape=[None, time_series_length, context_num_features],
        name="context")
    features = {
        "autoregressive_input": input_placeholder,
        "weights": weights_placeholder,
        "conditioning_stack": context_placeholder
    }
    mode = tf.estimator.ModeKeys.TRAIN
    hparams = configdict.ConfigDict({
        "dilation_kernel_width": 2,
        "skip_output_dim": 6,
        "preprocess_output_size": 3,
        "preprocess_kernel_width": 5,
        "num_residual_blocks": 2,
        "dilation_rates": [1, 2, 4],
        "output_distribution": {
            "type": "normal",
            "min_scale": 0,
        }
    })

    model = astrowavenet_model.AstroWaveNet(features, hparams, mode)
    model.build()

    scaffold = tf.train.Scaffold()
    scaffold.finalize()
    with self.cached_session() as sess:
      sess.run([scaffold.init_op, scaffold.local_init_op])
      step = sess.run(model.global_step)
      self.assertEqual(0, step)

      feed_dict = {
          input_placeholder: [
              [[1, 9], [1, 9], [1, 9], [1, 9], [1, 9], [1, 9]],
              [[2, 8], [2, 8], [2, 8], [2, 8], [2, 8], [2, 8]],
              [[3, 7], [3, 7], [3, 7], [3, 7], [3, 7], [3, 7]],
          ],
          weights_placeholder: [
              [[1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1]],
              [[1, 0], [1, 1], [1, 1], [0, 1], [0, 1], [0, 0]],
              [[0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]],
          ],
          # Context is not needed since we explicitly feed the dist params.
          model.dist_params["loc"]: [
              [[1, 8], [1, 8], [1, 8], [1, 8], [1, 8], [1, 8]],
              [[2, 9], [2, 9], [2, 9], [2, 9], [2, 9], [2, 9]],
              [[3, 6], [3, 6], [3, 6], [3, 6], [3, 6], [3, 6]],
          ],
          model.dist_params["scale"]: [
              [[0.1, 0.1], [0.2, 0.2], [0.5, 0.5], [1, 1], [2, 2], [5, 5]],
              [[0.1, 0.1], [0.2, 0.2], [0.5, 0.5], [1, 1], [2, 2], [5, 5]],
              [[0.1, 0.1], [0.2, 0.2], [0.5, 0.5], [1, 1], [2, 2], [5, 5]],
          ],
      }
      batch_losses, per_example_loss, num_examples, total_loss = sess.run(
          [
              model.batch_losses, model.per_example_loss,
              model.num_nonzero_weight_examples, model.total_loss
          ],
          feed_dict=feed_dict)
      np.testing.assert_array_almost_equal(
          [[[-1.38364656, 48.61635344], [-0.69049938, 11.80950062],
            [0.22579135, 2.22579135], [0.91893853, 1.41893853],
            [1.61208571, 1.73708571], [2.52837645, 2.54837645]],
           [[-1.38364656, 0], [-0.69049938, 11.80950062],
            [0.22579135, 2.22579135], [0, 1.41893853], [0, 1.73708571], [0, 0]],
           [[0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]]], batch_losses)
      np.testing.assert_array_almost_equal([5.96392435, 2.19185166, 0],
                                           per_example_loss)
      np.testing.assert_almost_equal(2, num_examples)
      np.testing.assert_almost_equal(4.07788801, total_loss)

  def test_causality(self):
    time_series_length = 7
    input_num_features = 1
    context_num_features = 1

    input_placeholder = tf.placeholder(
        dtype=tf.float32,
        shape=[None, time_series_length, input_num_features],
        name="input")
    context_placeholder = tf.placeholder(
        dtype=tf.float32,
        shape=[None, time_series_length, context_num_features],
        name="context")
    features = {
        "autoregressive_input": input_placeholder,
        "conditioning_stack": context_placeholder
    }
    mode = tf.estimator.ModeKeys.TRAIN
    hparams = configdict.ConfigDict({
        "dilation_kernel_width": 1,
        "skip_output_dim": 1,
        "preprocess_output_size": 1,
        "preprocess_kernel_width": 1,
        "num_residual_blocks": 1,
        "dilation_rates": [1],
        "output_distribution": {
            "type": "normal",
            "min_scale": 0.001,
        }
    })

    model = astrowavenet_model.AstroWaveNet(features, hparams, mode)
    model.build()

    scaffold = tf.train.Scaffold()
    scaffold.finalize()
    with self.cached_session() as sess:
      sess.run([scaffold.init_op, scaffold.local_init_op])
      step = sess.run(model.global_step)
      self.assertEqual(0, step)

      feed_dict = {
          input_placeholder: [
              [[0], [0], [0], [0], [0], [0], [0]],
              [[1], [0], [0], [0], [0], [0], [0]],
              [[0], [0], [0], [1], [0], [0], [0]],
              [[0], [0], [0], [0], [0], [0], [1]],
              [[0], [0], [0], [0], [0], [0], [0]],
              [[0], [0], [0], [0], [0], [0], [0]],
              [[0], [0], [0], [0], [0], [0], [0]],
          ],
          context_placeholder: [
              [[0], [0], [0], [0], [0], [0], [0]],
              [[0], [0], [0], [0], [0], [0], [0]],
              [[0], [0], [0], [0], [0], [0], [0]],
              [[0], [0], [0], [0], [0], [0], [0]],
              [[1], [0], [0], [0], [0], [0], [0]],
              [[0], [0], [0], [1], [0], [0], [0]],
              [[0], [0], [0], [0], [0], [0], [1]],
          ],
      }
      network_output = sess.run(model.network_output, feed_dict=feed_dict)
      np.testing.assert_array_equal(
          [
              [[0], [0], [0], [0], [0], [0], [0]],
              # Input elements are used to predict the next timestamp.
              [[0], [1], [0], [0], [0], [0], [0]],
              [[0], [0], [0], [0], [1], [0], [0]],
              [[0], [0], [0], [0], [0], [0], [0]],
              # Context elements are used to predict the current timestamp.
              [[1], [0], [0], [0], [0], [0], [0]],
              [[0], [0], [0], [1], [0], [0], [0]],
              [[0], [0], [0], [0], [0], [0], [1]],
          ],
          np.greater(np.abs(network_output), 0))


if __name__ == "__main__":
  tf.test.main()
