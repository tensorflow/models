# Copyright 2020 The TensorFlow Authors All Rights Reserved.
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
"""Tensorflow graph creator for PRADO model."""

import collections
import functools

from typing import Mapping, Dict, Any
from absl import logging
import tensorflow.compat.v1 as tf

from prado import common_layer # import sequence_projection module
from tf_ops import sequence_string_projection_op as ssp # import sequence_projection module

_NGRAM_INFO = [
    {
        "name": "unigram",
        "padding": 0,
        "kernel_size": [1, 1],
        "mask": None
    },
    {
        "name": "bigram",
        "padding": 1,
        "kernel_size": [2, 1],
        "mask": None
    },
    {
        "name": "trigram",
        "padding": 2,
        "kernel_size": [3, 1],
        "mask": None
    },
    {
        "name": "bigramskip1",
        "padding": 2,
        "kernel_size": [3, 1],
        "mask": [[[[1]]], [[[0]]], [[[1]]]]
    },
    {
        "name": "bigramskip2",
        "padding": 3,
        "kernel_size": [4, 1],
        "mask": [[[[1]]], [[[0]]], [[[0]]], [[[1]]]]
    },
    {
        "name": "fourgram",
        "padding": 3,
        "kernel_size": [4, 1],
        "mask": None
    },
    {
        "name": "fivegram",
        "padding": 4,
        "kernel_size": [5, 1],
        "mask": None
    },
]


def _get_params(model_config, varname, default_value=None):
  value = model_config[varname] if varname in model_config else default_value
  logging.info("%s = %s", varname, value)
  return value


def create_projection(model_config, mode, inputs):
  """Create projection."""
  feature_size = _get_params(model_config, "feature_size")
  text_distortion_probability = _get_params(model_config,
                                            "text_distortion_probability", 0.0)
  max_seq_len = _get_params(model_config, "max_seq_len", 0)
  add_eos_tag = _get_params(model_config, "add_eos_tag")
  is_training = mode == tf.estimator.ModeKeys.TRAIN
  distortion_probability = text_distortion_probability if is_training else 0.0
  raw_string = tf.identity(inputs, "Input")
  features, _, seq_length = ssp.sequence_string_projection(
      input=raw_string,
      feature_size=feature_size,
      max_splits=max_seq_len - 1,
      distortion_probability=distortion_probability,
      split_on_space=True,
      add_eos_tag=add_eos_tag,
      vocabulary="")

  if mode != tf.estimator.ModeKeys.PREDICT and max_seq_len > 0:
    pad_value = [[0, 0], [0, max_seq_len - tf.shape(features)[1]], [0, 0]]
    features = tf.pad(features, pad_value)
    batch_size = inputs.get_shape().as_list()[0]
    features = tf.reshape(features,
                          [batch_size, max_seq_len, feature_size])
  return features, seq_length


def _fully_connected(pod_layers, tensor, num_features, mode, bsz, keep_prob):
  """Fully connected layer."""
  tensor_out = pod_layers.fully_connected(tensor, num_features)
  if mode == tf.estimator.ModeKeys.TRAIN:
    tensor_out = tf.nn.dropout(tensor_out, rate=(1 - keep_prob))
  return tf.reshape(tensor_out, [bsz, -1, 1, num_features])


def _get_convolutional_layer(pod_layers, head_type, channels, valid_step_mask,
                             tensor, invalid_value):
  """Get convolutional layer."""
  info = _NGRAM_INFO[head_type]
  pad = info["padding"]
  weight_mask = info["mask"]
  kernel_size = info["kernel_size"]
  paddings = [[0, 0], [0, pad], [0, 0], [0, 0]]
  # Padding before convolution and using 'valid' instead of 'same' padding
  # structure ensures that the convolution output is identical between
  # train/eval and inference models. It also ensures that they lineup
  # correctly with the valid_step_mask.
  tensor = tf.pad(tensor, paddings) if pad != 0 else tensor
  # Not using activation allows a bigram feature to de-emphasize a feature
  # that triggers positive for unigram for example. The output weights
  # should be allowed to be positve or negative for this to happen.
  tensor = pod_layers.convolution2d(
      tensor,
      kernel_size,
      channels,
      padding="VALID",
      weight_mask=weight_mask,
      activation=None)
  if valid_step_mask is not None:
    tensor = tensor * valid_step_mask + (1 - valid_step_mask) * invalid_value
  return tensor


def _get_predictions(pod_layers, head_type, keys, values, channels,
                     valid_step_mask):
  """Get predictions using one ngram head."""
  conv_layer = functools.partial(_get_convolutional_layer, pod_layers,
                                 head_type, channels, valid_step_mask)
  return conv_layer(keys, -100), conv_layer(values, 0)


def reduce_tensors(pod_layers, bsz, attention_logits, values):
  """Reduce information using attention."""
  channels = attention_logits.get_shape().as_list()[-1]
  attention_logits = tf.reshape(attention_logits, [bsz, -1, channels])
  values = tf.reshape(values, [bsz, -1, channels])

  with tf.variable_scope("attention_expected_value"):
    attention_logits = tf.identity(attention_logits, "attention_logits_in")
    values = tf.identity(values, "values_in")
    attention_logits = tf.transpose(attention_logits, [0, 2, 1])
    values = tf.transpose(values, [0, 2, 1])
    attention = tf.nn.softmax(attention_logits, axis=2)
    evalue = tf.reduce_sum(attention * values, axis=[2])
    evalue = tf.identity(evalue, "expected_value_out")
  return pod_layers.quantization(evalue)


def ngram_attention_args_v2(projection, seq_length, mode, num_classes,
                            model_args):
  """Implements an ngram attention network.

  Args:
    projection: Projection features from text.
    seq_length: Sequence length.
    mode: Model creation mode (train, eval or predict).
    num_classes: Number of classes to be predicted.
    model_args: A namedtuple containing all model arguments.

  Returns:
    A tensor corresponding to the logits of the graph.
  """

  pod_layers = common_layer.CommonLayers(
      mode, quantization_enabled=model_args.quantize)

  features = pod_layers.qrange_tanh(projection)
  bsz = features.get_shape().as_list()[0] or tf.shape(features)[0]

  # Regularizer just for the word embedding.
  pod_layers.set_regularizer_scale(model_args.embedding_regularizer_scale)
  values = _fully_connected(pod_layers, features, model_args.embedding_size,
                            mode, bsz, model_args.keep_prob)
  keys = _fully_connected(pod_layers, features, model_args.embedding_size, mode,
                          bsz, model_args.keep_prob)

  # Regularizer for the rest of the network.
  pod_layers.set_regularizer_scale(model_args.network_regularizer_scale)

  valid_step_mask = None
  if mode != tf.estimator.ModeKeys.PREDICT:
    valid_step_mask = pod_layers.zero_beyond_sequence_length(
        seq_length, features)
    valid_step_mask = tf.expand_dims(valid_step_mask, 3)
    # Mask out the sentence beyond valid sequence length for training graph.
    # This ensures that these values are all zeroed out. Without masking, the
    # fully connected layer before will make them take an arbitrary constant
    # value during training/eval in the minibatches. But these values won't
    # be present during inference as the inference is not batched.
    keys = valid_step_mask * keys
    values = valid_step_mask * values
    pod_layers.set_variable_length_moment_fn(seq_length, tf.shape(features)[1])

  multi_head_predictions = []
  for head_type, head in zip(model_args.head_types, model_args.heads):
    if not head:
      continue
    att_logits, att_values = _get_predictions(pod_layers, head_type, keys,
                                              values, head, valid_step_mask)
    multi_head_predictions.append(
        reduce_tensors(pod_layers, bsz, att_logits, att_values))
  multi_head_predictions = tf.concat(multi_head_predictions, axis=1)
  multi_head_predictions = pod_layers.quantization(multi_head_predictions)
  # Sequence dimension has been summed out, so we don't need special moment
  # function.
  pod_layers.set_moment_fn(None)

  output = multi_head_predictions

  # Add FC layers before the logits.
  for fc_layer_size in model_args.pre_logits_fc_layers:
    output = pod_layers.fully_connected(
        output, fc_layer_size, activation=tf.nn.relu)

  return pod_layers.fully_connected(output, num_classes, activation=None)


def create_encoder(model_config: Dict[str, Any], projection: tf.Tensor,
                   seq_length: tf.Tensor,
                   mode: tf.estimator.ModeKeys) -> Mapping[str, tf.Tensor]:
  """Implements a simple attention network for brand safety."""

  args = {}

  def _get_params(varname, default_value=None):
    value = model_config[varname] if varname in model_config else default_value
    logging.info("%s = %s", varname, value)
    args[varname] = value

  _get_params("labels")
  _get_params("quantize", True)
  _get_params("max_seq_len", 0)
  _get_params("max_seq_len_inference", 0)
  _get_params("split_on_space", True)
  _get_params("exclude_nonalphaspace_unicodes", False)
  _get_params("embedding_regularizer_scale", 35e-3)
  _get_params("embedding_size", 64)
  _get_params("heads", [0, 64, 64, 0, 0])
  _get_params("feature_size", 512)
  _get_params("network_regularizer_scale", 1e-4)
  _get_params("keep_prob", 0.5)
  _get_params("word_novelty_bits", 0)
  _get_params("doc_size_levels", 0)
  _get_params("pre_logits_fc_layers", [])
  args["head_types"] = list(range(len(args["heads"])))
  args["text_distortion_probability"] = 0.0
  if mode == tf.estimator.ModeKeys.TRAIN:
    _get_params("text_distortion_probability", 0.25)
  model_args = collections.namedtuple("ModelArgs", sorted(args))(**args)
  num_classes = len(model_args.labels)
  logits = ngram_attention_args_v2(
      projection=projection,
      seq_length=seq_length,
      mode=mode,
      num_classes=num_classes,
      model_args=model_args)
  outputs = {
      "logits":
          tf.identity(logits, "Logits"),
      "label_map":
          tf.constant(list(model_args.labels), tf.string, name="LabelMap")
  }
  return outputs
