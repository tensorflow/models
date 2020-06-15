# Lint as: python3
# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for BERT configurations and models instantiation."""

import tensorflow as tf

from official.nlp.configs import bert
from official.nlp.configs import encoders


class BertModelsTest(tf.test.TestCase):

  def test_network_invocation(self):
    config = bert.BertPretrainerConfig(
        encoder=encoders.TransformerEncoderConfig(vocab_size=10, num_layers=1))
    _ = bert.instantiate_from_cfg(config)

    # Invokes with classification heads.
    config = bert.BertPretrainerConfig(
        encoder=encoders.TransformerEncoderConfig(vocab_size=10, num_layers=1),
        cls_heads=[
            bert.ClsHeadConfig(
                inner_dim=10, num_classes=2, name="next_sentence")
        ])
    _ = bert.instantiate_from_cfg(config)

    with self.assertRaises(ValueError):
      config = bert.BertPretrainerConfig(
          encoder=encoders.TransformerEncoderConfig(
              vocab_size=10, num_layers=1),
          cls_heads=[
              bert.ClsHeadConfig(
                  inner_dim=10, num_classes=2, name="next_sentence"),
              bert.ClsHeadConfig(
                  inner_dim=10, num_classes=2, name="next_sentence")
          ])
      _ = bert.instantiate_from_cfg(config)

  def test_checkpoint_items(self):
    config = bert.BertPretrainerConfig(
        encoder=encoders.TransformerEncoderConfig(vocab_size=10, num_layers=1),
        cls_heads=[
            bert.ClsHeadConfig(
                inner_dim=10, num_classes=2, name="next_sentence")
        ])
    encoder = bert.instantiate_from_cfg(config)
    self.assertSameElements(encoder.checkpoint_items.keys(),
                            ["encoder", "next_sentence.pooler_dense"])


if __name__ == "__main__":
  tf.test.main()
