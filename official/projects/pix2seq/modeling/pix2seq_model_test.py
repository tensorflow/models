# Copyright 2024 The TensorFlow Authors. All Rights Reserved.
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

"""Tests for Pix2Seq model."""

from absl.testing import parameterized
import numpy as np
import tensorflow as tf, tf_keras
from official.projects.pix2seq.modeling import pix2seq_model
from official.vision.modeling.backbones import resnet


class Pix2SeqTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.named_parameters(
      ('One backbone', 1),
      ('Two backbones', 2),
  )
  def test_forward(self, num_backbones: int):
    hidden_size = 256
    num_heads = 8
    max_seq_len = 50
    vocab_size = 164
    image_size = 224
    batch_size = 2
    backbones = [
        resnet.ResNet(50, bn_trainable=False) for _ in range(num_backbones)
    ]
    backbone_endpoint_names = ['5' for _ in range(num_backbones)]
    model = pix2seq_model.Pix2Seq(
        backbones,
        backbone_endpoint_names,
        max_seq_len,
        vocab_size,
        hidden_size,
        num_heads=num_heads,
        encoded_feature_dropout_rates=[0.1] * num_backbones,
    )
    _, outs = model(
        tf.ones((batch_size, num_backbones, image_size, image_size, 3)),
        tf.ones((batch_size, max_seq_len), tf.int64),
        True,
    )

    self.assertLen(outs, 2)  # intermediate decoded outputs.

  @parameterized.named_parameters(
      ('One backbone', 1),
      ('Two backbones', 2),
  )
  def test_forward_infer_teacher_forcing(self, num_backbones: int):
    hidden_size = 256
    num_heads = 8
    max_seq_len = 50
    vocab_size = 164
    image_size = 224
    batch_size = 2
    backbones = [
        resnet.ResNet(50, bn_trainable=False) for _ in range(num_backbones)
    ]
    backbone_endpoint_names = ['5' for _ in range(num_backbones)]
    model = pix2seq_model.Pix2Seq(
        backbones,
        backbone_endpoint_names,
        max_seq_len,
        vocab_size,
        hidden_size,
        num_heads=num_heads,
        encoded_feature_dropout_rates=[0.1] * num_backbones,
    )
    _, outs = model(
        tf.ones((batch_size, num_backbones, image_size, image_size, 3)),
        tf.ones((batch_size, max_seq_len), tf.int64),
        training=False,
        use_teacher_forcing_for_eval=True,
    )

    self.assertLen(outs, 2)  # intermediate decoded outputs.

  @parameterized.named_parameters(
      ('One backbone', 1),
      ('Two backbones', 2),
  )
  def test_forward_infer(self, num_backbones: int):
    hidden_size = 256
    num_heads = 8
    max_seq_len = 50
    vocab_size = 600
    image_size = 640
    batch_size = 2
    backbones = [
        resnet.ResNet(50, bn_trainable=False) for _ in range(num_backbones)
    ]
    backbone_endpoint_names = ['5' for _ in range(num_backbones)]
    model = pix2seq_model.Pix2Seq(
        backbones,
        backbone_endpoint_names,
        max_seq_len,
        vocab_size,
        hidden_size,
        num_heads=num_heads,
        encoded_feature_dropout_rates=[0.1] * num_backbones,
    )
    tokens, _ = model(
        tf.ones((batch_size, num_backbones, image_size, image_size, 3)),
        tf.ones((batch_size, 1), tf.int64) * 10,
        False,
    )

    self.assertLen(tokens, 2)  # intermediate decoded outputs.

  def test_forward_infer_with_early_stopping(self):
    hidden_size = 256
    num_heads = 8
    max_seq_len = 50
    vocab_size = 600
    image_size = 640
    batch_size = 2
    backbone = resnet.ResNet(50, bn_trainable=False)
    backbone_endpoint_names = ['5']
    model = pix2seq_model.Pix2Seq(
        [backbone],
        backbone_endpoint_names,
        max_seq_len,
        vocab_size,
        hidden_size,
        num_heads=num_heads,
        early_stopping_token=0,
    )
    tokens, _ = model(
        tf.ones((batch_size, 1, image_size, image_size, 3)),
        tf.ones((batch_size, 1), tf.int64) * 10,
        False,
    )

    self.assertLen(tokens, 2)  # intermediate decoded outputs.

  def test_forward_infer_with_long_prompt(self):
    hidden_size = 256
    num_heads = 8
    max_seq_len = 50
    vocab_size = 600
    image_size = 640
    batch_size = 2
    backbone = resnet.ResNet(50, bn_trainable=False)
    backbone_endpoint_names = ['5']
    model = pix2seq_model.Pix2Seq(
        [backbone],
        backbone_endpoint_names,
        max_seq_len,
        vocab_size,
        hidden_size,
        num_heads=num_heads,
    )
    tokens, _ = model(
        tf.ones((batch_size, 1, image_size, image_size, 3)),
        tf.ones((batch_size, 2), tf.int64) * 10,
        False,
    )

    self.assertLen(tokens, 2)  # intermediate decoded outputs.
    self.assertShapeEqual(tokens, np.ndarray([batch_size, max_seq_len - 2 + 1]))

  def test_cond_fn_without_early_stopping(self):
    tokens = tf.constant(
        # pyformat: disable
        [
            [0, 0, 0],
            [0, 0, 0],
            [0, 1, 0],
            [1, 0, 0],
            [0, 0, 1],  # Should not stop early.
            [0, 0, 0],
            [0, 0, 0],  # Should stop inference here.
        ],
        # pyformat: enable
        dtype=tf.int64
    )
    cond = pix2seq_model._create_cond_fn(
        seq_len=tokens.shape[0],
        early_stopping_token=None,
        prompt_len=1,
    )
    expected_results = [True, True, True, True, True, True, False]

    self.assertLen(expected_results, tokens.shape[0])
    for step, expected_result in enumerate(expected_results):
      self.assertEqual(
          expected_result,
          cond(step, None, tokens, None),
          msg=f'step={step}',
      )

  def test_cond_fn_with_early_stopping(self):
    tokens = tf.constant(
        # pyformat: disable
        [
            [0, 0, 0],
            [0, 0, 0],
            [0, 1, 0],
            [1, 0, 0],
            [0, 0, 1],  # Should stop inference here.
            [0, 0, 0],
            [0, 0, 0],
        ],
        # pyformat: enable
        dtype=tf.int64
    )
    cond = pix2seq_model._create_cond_fn(
        seq_len=tokens.shape[0],
        early_stopping_token=1,
        prompt_len=1,
    )
    expected_results = [True, True, True, True, True, False, False]

    self.assertLen(expected_results, tokens.shape[0])
    for step, expected_result in enumerate(expected_results):
      self.assertEqual(
          expected_result,
          cond(step, None, tokens, None),
          msg=f'step={step}',
      )

  def test_cond_fn_with_early_stopping_keep_inference_to_end(self):
    tokens = tf.constant(
        # pyformat: disable
        [
            [1, 1, 1],  # Early stopping token within prompt should be ignored.
            [0, 0, 0],
            [0, 1, 0],
            [1, 0, 0],  # Should keep inferencing until the end.
        ],
        # pyformat: enable
        dtype=tf.int64
    )
    cond = pix2seq_model._create_cond_fn(
        seq_len=tokens.shape[0],
        early_stopping_token=1,
        prompt_len=1,
    )
    expected_results = [True, True, True, False]

    self.assertLen(expected_results, tokens.shape[0])
    for step, expected_result in enumerate(expected_results):
      self.assertEqual(
          expected_result,
          cond(step, None, tokens, None),
          msg=f'step={step}',
      )


if __name__ == '__main__':
  tf.test.main()
