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

"""Tests for Pix2Seq model."""
import tensorflow as tf
from official.projects.pix2seq.modeling import pix2seq_model
from official.vision.modeling.backbones import resnet


class Pix2SeqTest(tf.test.TestCase):

  def test_forward(self):
    hidden_size = 256
    max_seq_len = 50
    vocab_size = 164
    image_size = 224
    batch_size = 2
    backbone = resnet.ResNet(50, bn_trainable=False)
    backbone_endpoint_name = '5'
    model = pix2seq_model.Pix2Seq(
        backbone, backbone_endpoint_name, max_seq_len, vocab_size, hidden_size
    )
    _, outs = model(
        tf.ones((batch_size, image_size, image_size, 3)),
        tf.ones((batch_size, max_seq_len), tf.int64),
        True,
    )

    self.assertLen(outs, 2)  # intermediate decoded outputs.

  def test_forward_infer(self):
    hidden_size = 256
    max_seq_len = 50
    vocab_size = 600
    image_size = 640
    batch_size = 2
    backbone = resnet.ResNet(50, bn_trainable=False)
    backbone_endpoint_name = '5'
    model = pix2seq_model.Pix2Seq(
        backbone, backbone_endpoint_name, max_seq_len, vocab_size, hidden_size
    )
    tokens, _ = model(
        tf.ones((batch_size, image_size, image_size, 3)),
        tf.ones((batch_size, 1), tf.int64) * 10,
        False,
    )

    self.assertLen(tokens, 2)  # intermediate decoded outputs.


if __name__ == '__main__':
  tf.test.main()
