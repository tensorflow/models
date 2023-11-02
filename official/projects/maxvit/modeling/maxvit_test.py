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

"""Tests for MaxViT."""
import collections
from typing import Optional, Sequence

from absl.testing import parameterized
import numpy as np
import tensorflow as tf, tf_keras

from official.projects.maxvit.configs import backbones
from official.projects.maxvit.modeling import maxvit
from official.vision.configs import common


class MaxViTBlockTest(tf.test.TestCase):
  """Test the layers of MaxViT."""

  def testMaxViTBlockCreation(self) -> None:
    """Ensures that layers can be constructed and forward-props can run."""
    inputs_shape = [2, 64, 64, 3]
    inp = tf.random.uniform(
        shape=inputs_shape, minval=-1.0, maxval=1.0, dtype=tf.float32
    )

    model = maxvit.MaxViTBlock(
        hidden_size=8, head_size=4, window_size=4, grid_size=4
    )
    out = model(inp, training=False)

    self.assertAllEqual([2, 64, 64, 8], out.get_shape().as_list())
    self.assertDTypeEqual(tf.reduce_mean(out).numpy(), np.float32)


class MaxViTTest(tf.test.TestCase, parameterized.TestCase):
  """Test the layers of MaxViT."""

  @parameterized.named_parameters(
      collections.OrderedDict(
          testcase_name='MaxViTTest',
          input_shape=[2, 64, 64, 3],
          input_dtype=tf.float32,
          training=False,
          stem_hsize=[12, 12],
          num_blocks=[2, 2, 2, 2],
          window_size=2,
          grid_size=2,
          block_type=['maxvit', 'maxvit', 'maxvit'],
          hidden_size=[16, 32, 64],
          expected_shape=[2, 4, 4, 64],
          name='maxvit_test',
      ),
      collections.OrderedDict(
          testcase_name='MaxViTTiny',
          input_shape=[2, 64, 64, 3],
          input_dtype=tf.float32,
          training=False,
          block_type=['maxvit', 'maxvit', 'maxvit', 'maxvit'],
          stem_hsize=[64, 64],
          num_blocks=[2, 3, 5, 2],
          window_size=2,
          grid_size=2,
          hidden_size=[96, 192, 384, 768],
          expected_shape=[2, 2, 2, 768],
          name='maxvit_tiny',
      ),
      collections.OrderedDict(
          testcase_name='MaxViTTinyWithPrelogits',
          input_shape=[2, 64, 64, 3],
          input_dtype=tf.float32,
          training=False,
          representation_size=16,
          add_gap_layer_norm=True,
          block_type=['maxvit', 'maxvit', 'maxvit', 'maxvit'],
          stem_hsize=[64, 64],
          num_blocks=[2, 3, 5, 2],
          window_size=2,
          grid_size=2,
          hidden_size=[96, 192, 384, 768],
          expected_shape=[2, 2, 2, 768],
          name='maxvit_tiny',
      ),
  )
  def testForward(
      self,
      input_shape: Sequence[int],
      input_dtype: Optional[tf.DType] = tf.float32,
      **kwargs
  ) -> None:
    """Ensures that layers can be constructed and forward-props can run."""

    inp = tf.random.uniform(
        input_shape,
        minval=-1.0,
        maxval=1.0,
        dtype=input_dtype,
    )

    model = maxvit.MaxViT(**kwargs)
    out = model(inp, training=kwargs.get('training', None))

    add_gap_layer_norm = kwargs.get('add_gap_layer_norm', False)
    if add_gap_layer_norm:
      self.assertAllEqual([input_shape[0], kwargs['representation_size']],
                          out['pre_logits'].get_shape().as_list())

    # Remove `pre_logits` if exists.
    out.pop('pre_logits', None)
    out = out[max(out.keys())]
    self.assertAllEqual(kwargs['expected_shape'], out.get_shape().as_list())
    self.assertDTypeEqual(tf.reduce_mean(out).numpy(), np.float32)

  def testBuildMaxViTWithConfig(self):
    backbone_config = backbones.Backbone(
        type='maxvit',
        maxvit=backbones.MaxViT(
            stem_hsize=[32, 32],
            num_blocks=[2, 3, 5, 2],
            window_size=2,
            grid_size=2,
            hidden_size=[32, 32, 32, 32],
        ),
    )
    backbone = maxvit.build_maxvit(
        input_specs=tf_keras.layers.InputSpec(shape=[None] + [64, 64, 3]),
        backbone_config=backbone_config,
        norm_activation_config=common.NormActivation(),
    )

    self.assertSetEqual(
        set(['2', '3', '4', '5']), set(backbone.output_specs.keys())
    )


if __name__ == '__main__':
  tf.test.main()
