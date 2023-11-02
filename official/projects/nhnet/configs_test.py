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

"""Tests for configs."""

import tensorflow as tf, tf_keras
from official.projects.nhnet import configs

BERT2BERT_CONFIG = {
    "vocab_size": 30522,
    "hidden_size": 768,
    "num_hidden_layers": 12,
    "num_attention_heads": 12,
    "intermediate_size": 3072,
    "hidden_act": "gelu",
    "hidden_dropout_prob": 0.1,
    "attention_probs_dropout_prob": 0.1,
    "max_position_embeddings": 512,
    "type_vocab_size": 2,
    "initializer_range": 0.02,

    # model params
    "decoder_intermediate_size": 3072,
    "num_decoder_attn_heads": 12,
    "num_decoder_layers": 12,

    # training params
    "label_smoothing": 0.1,
    "learning_rate": 0.05,
    "learning_rate_warmup_steps": 20000,
    "optimizer": "Adam",
    "adam_beta1": 0.9,
    "adam_beta2": 0.997,
    "adam_epsilon": 1e-09,

    # predict params
    "beam_size": 5,
    "alpha": 0.6,
    "initializer_gain": 1.0,
    "use_cache": True,

    # input params
    "input_sharding": False,
    "input_data_not_padded": False,
    "pad_token_id": 0,
    "end_token_id": 102,
    "start_token_id": 101,
}

NHNET_CONFIG = {
    "vocab_size": 30522,
    "hidden_size": 768,
    "num_hidden_layers": 12,
    "num_attention_heads": 12,
    "intermediate_size": 3072,
    "hidden_act": "gelu",
    "hidden_dropout_prob": 0.1,
    "attention_probs_dropout_prob": 0.1,
    "max_position_embeddings": 512,
    "type_vocab_size": 2,
    "initializer_range": 0.02,

    # model params
    "decoder_intermediate_size": 3072,
    "num_decoder_attn_heads": 12,
    "num_decoder_layers": 12,
    "multi_channel_cross_attention": True,

    # training params
    "label_smoothing": 0.1,
    "learning_rate": 0.05,
    "learning_rate_warmup_steps": 20000,
    "optimizer": "Adam",
    "adam_beta1": 0.9,
    "adam_beta2": 0.997,
    "adam_epsilon": 1e-09,

    # predict params
    "beam_size": 5,
    "alpha": 0.6,
    "initializer_gain": 1.0,
    "use_cache": True,

    # input params
    "passage_list": ["b", "c", "d", "e", "f"],
    "input_sharding": False,
    "input_data_not_padded": False,
    "pad_token_id": 0,
    "end_token_id": 102,
    "start_token_id": 101,
    "init_from_bert2bert": True,
}


class ConfigsTest(tf.test.TestCase):

  def test_configs(self):
    cfg = configs.BERT2BERTConfig()
    cfg.validate()
    self.assertEqual(cfg.as_dict(), BERT2BERT_CONFIG)

  def test_nhnet_config(self):
    cfg = configs.NHNetConfig()
    cfg.validate()
    self.assertEqual(cfg.as_dict(), NHNET_CONFIG)


if __name__ == "__main__":
  tf.test.main()
