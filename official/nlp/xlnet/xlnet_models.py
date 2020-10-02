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
"""XLNet models that are compatible with TF 2.x."""
import tensorflow as tf

from official.nlp.modeling import models
from official.nlp.modeling import networks
from official.nlp.xlnet import xlnet_config


def _get_initializer(
    initialization_method: str,
    initialization_range: float,
    initialization_std: float) -> tf.keras.initializers.Initializer:
  """Gets variable initializer."""
  if initialization_method == 'uniform':
    initializer = tf.keras.initializers.RandomUniform(
        minval=-initialization_range, maxval=initialization_range)
  elif initialization_method == 'normal':
    initializer = tf.keras.initializers.RandomNormal(stddev=initialization_std)
  else:
    raise ValueError('Initializer {} not supported'.format(
        initialization_method))
  return initializer


def get_xlnet_base(model_config: xlnet_config.XLNetConfig,
                   run_config: xlnet_config.RunConfig,
                   attention_type: str,
                   two_stream: bool,
                   use_cls_mask: bool) -> tf.keras.Model:
  """Gets an 'XLNetBase' object.

  Args:
    model_config: the config that defines the core XLNet model.
    run_config: separate runtime configuration with extra parameters.
    attention_type: the attention type for the base XLNet model, "uni" or "bi".
    two_stream: whether or not to use two strema attention.
    use_cls_mask: whether or not cls mask is included in the input sequences.

  Returns:
    An XLNetBase object.
  """
  initializer = _get_initializer(initialization_method=run_config.init_method,
                                 initialization_range=run_config.init_range,
                                 initialization_std=run_config.init_std)
  kwargs = dict(
      vocab_size=model_config.n_token,
      num_layers=model_config.n_layer,
      hidden_size=model_config.d_model,
      num_attention_heads=model_config.n_head,
      head_size=model_config.d_head,
      inner_size=model_config.d_inner,
      dropout_rate=run_config.dropout,
      attention_dropout_rate=run_config.dropout_att,
      attention_type=attention_type,
      bi_data=run_config.bi_data,
      initializer=initializer,
      two_stream=two_stream,
      tie_attention_biases=not model_config.untie_r,
      memory_length=run_config.mem_len,
      clamp_length=run_config.clamp_len,
      reuse_length=run_config.reuse_len,
      inner_activation=model_config.ff_activation,
      use_cls_mask=use_cls_mask)
  return networks.XLNetBase(**kwargs)


def classifier_model(
    model_config: xlnet_config.XLNetConfig,
    run_config: xlnet_config.RunConfig,
    num_labels: int,
    final_layer_initializer: tf.keras.initializers.Initializer = None
    ) -> tf.keras.Model:
  """Returns a TF2 Keras XLNet classifier model.

  Construct a Keras model for predicting `num_labels` outputs from an input with
  maximum sequence length `max_seq_length`.

  Args:
    model_config: the config that defines the core XLNet model.
    run_config: separate runtime configuration with extra parameters.
    num_labels: integer, the number of classes.
    final_layer_initializer: Initializer for final dense layer. If `None`, then
      it defaults to the one specified in `run_config`.

  Returns:
    Combined prediction model inputs -> (one-hot labels)
    XLNet sub-model inputs -> (xlnet_outputs)
    where inputs are:
      (words, segments, mask, permutation mask,
       target mapping, masked tokens)
  """
  if final_layer_initializer is not None:
    initializer = final_layer_initializer
  else:
    initializer = tf.keras.initializers.RandomNormal(
        mean=0., stddev=.02)
  xlnet_base = get_xlnet_base(
      model_config=model_config,
      run_config=run_config,
      attention_type='bi',
      two_stream=False,
      use_cls_mask=False)
  return models.XLNetClassifier(
      network=xlnet_base,
      num_classes=num_labels,
      dropout_rate=run_config.dropout,
      summary_type='last',
      initializer=initializer), xlnet_base
