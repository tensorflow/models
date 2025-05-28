# Copyright 2025 The TensorFlow Authors. All Rights Reserved.
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

# pylint: disable=g-doc-return-or-yield,line-too-long
"""WMT translation configurations."""

from official.core import config_definitions as cfg
from official.core import exp_factory
from official.modeling import optimization
from official.nlp.data import wmt_dataloader
from official.nlp.tasks import translation


@exp_factory.register_config_factory('wmt_transformer/large')
def wmt_transformer_large() -> cfg.ExperimentConfig:
  """WMT Transformer Large.

  Please refer to
  tensorflow_models/official/nlp/data/train_sentencepiece.py
  to generate sentencepiece_model
  and pass
  --params_override=task.sentencepiece_model_path='YOUR_PATH'
  to the train script.
  """
  learning_rate = 2.0
  hidden_size = 1024
  learning_rate *= (hidden_size**-0.5)
  warmup_steps = 16000
  train_steps = 300000
  token_batch_size = 24576
  encdecoder = translation.EncDecoder(
      num_attention_heads=16, intermediate_size=hidden_size * 4)
  config = cfg.ExperimentConfig(
      runtime=cfg.RuntimeConfig(enable_xla=True),
      task=translation.TranslationConfig(
          model=translation.ModelConfig(
              encoder=encdecoder,
              decoder=encdecoder,
              embedding_width=hidden_size,
              padded_decode=True,
              decode_max_length=100),
          train_data=wmt_dataloader.WMTDataConfig(
              tfds_name='wmt14_translate/de-en',
              tfds_split='train',
              src_lang='en',
              tgt_lang='de',
              is_training=True,
              global_batch_size=token_batch_size,
              static_batch=True,
              max_seq_length=64
          ),
          validation_data=wmt_dataloader.WMTDataConfig(
              tfds_name='wmt14_translate/de-en',
              tfds_split='test',
              src_lang='en',
              tgt_lang='de',
              is_training=False,
              global_batch_size=32,
              static_batch=True,
              max_seq_length=100,
          ),
          sentencepiece_model_path=None,
      ),
      trainer=cfg.TrainerConfig(
          train_steps=train_steps,
          validation_steps=-1,
          steps_per_loop=1000,
          summary_interval=1000,
          checkpoint_interval=5000,
          validation_interval=5000,
          max_to_keep=1,
          optimizer_config=optimization.OptimizationConfig({
              'optimizer': {
                  'type': 'adam',
                  'adam': {
                      'beta_2': 0.997,
                      'epsilon': 1e-9,
                  },
              },
              'learning_rate': {
                  'type': 'power',
                  'power': {
                      'initial_learning_rate': learning_rate,
                      'power': -0.5,
                  }
              },
              'warmup': {
                  'type': 'linear',
                  'linear': {
                      'warmup_steps': warmup_steps,
                      'warmup_learning_rate': 0.0
                  }
              }
          })),
      restrictions=[
          'task.train_data.is_training != None',
          'task.sentencepiece_model_path != None',
      ])
  return config
