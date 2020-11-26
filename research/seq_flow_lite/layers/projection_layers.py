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
"""Tensorflow projection creator for PRADO model."""

from absl import logging
import tensorflow as tf

from layers import base_layers # import seq_flow_lite module
from tf_ops import sequence_string_projection_op as ssp # import seq_flow_lite module
from tf_ops import sequence_string_projection_op_v2 as sspv2 # import seq_flow_lite module


class ProjectionLayer(base_layers.BaseLayer):
  """Base class for encoders."""

  def __init__(self, model_config, mode):
    """Create projection."""

    def _get_params(varname, default_value=None):
      value = model_config[varname] if varname in model_config else default_value
      default = "" if varname in model_config else " (default)"
      logging.info("%s = %s%s", varname, value, default)
      setattr(self, varname, value)

    self.mode = mode
    _get_params("feature_size")
    _get_params("max_seq_len", 0)
    _get_params("add_eos_tag", False)
    _get_params("add_bos_tag", False)
    _get_params("hashtype", "murmur")
    _get_params("split_on_space", True)
    _get_params("token_separators", "")
    _get_params("vocabulary", "")
    _get_params("quantize")
    _get_params("word_novelty_bits", 0)
    _get_params("doc_size_levels", 0)
    self.distortion_probability = 0.0
    if mode == base_layers.TRAIN:
      _get_params("distortion_probability", 0.0)
    parameters = base_layers.Parameters(mode, self.quantize)
    super(ProjectionLayer, self).__init__(parameters=parameters)

  def call(self, inputs):
    projection, _, seq_length = ssp.sequence_string_projection(
        input=inputs,
        feature_size=self.feature_size,
        max_splits=self.max_seq_len - 1,
        hashtype=self.hashtype,
        distortion_probability=self.distortion_probability,
        split_on_space=self.split_on_space,
        token_separators=self.token_separators,
        word_novelty_bits=self.word_novelty_bits,
        doc_size_levels=self.doc_size_levels,
        add_eos_tag=self.add_eos_tag,
        add_bos_tag=self.add_bos_tag,
        vocabulary=self.vocabulary)

    modes = [base_layers.PREDICT, base_layers.TFLITE]
    if self.mode not in modes and self.max_seq_len > 0:
      short_by = self.max_seq_len - tf.shape(projection)[1]
      projection = tf.pad(projection, [[0, 0], [0, short_by], [0, 0]])
      batch_size = self.get_batch_dimension(inputs)
      projection = tf.reshape(projection,
                              [batch_size, self.max_seq_len, self.feature_size])
    if self.mode in modes:
      projection = self.qrange_tanh(projection)
    return projection, seq_length


class ProjectionLayerPreSegmented(base_layers.BaseLayer):
  """Base class for encoders."""

  def __init__(self, model_config, mode):
    """Create projection."""

    def _get_params(varname, default_value=None):
      value = model_config[varname] if varname in model_config else default_value
      default = "" if varname in model_config else " (default)"
      logging.info("%s = %s%s", varname, value, default)
      setattr(self, varname, value)

    self.mode = mode
    _get_params("feature_size")
    _get_params("add_eos_tag", False)
    _get_params("add_bos_tag", False)
    _get_params("vocabulary", "")
    _get_params("quantize")
    self.distortion_probability = 0.0
    if mode == base_layers.TRAIN:
      _get_params("distortion_probability", 0.0)
    parameters = base_layers.Parameters(mode, self.quantize)
    super(ProjectionLayerPreSegmented, self).__init__(parameters=parameters)

  def call(self, inputs, sequence_length):
    projection = sspv2.sequence_string_projection_v2(
        input=inputs,
        sequence_length=sequence_length,
        feature_size=self.feature_size,
        distortion_probability=self.distortion_probability,
        add_eos_tag=self.add_eos_tag,
        add_bos_tag=self.add_bos_tag,
        vocabulary=self.vocabulary)

    modes = [base_layers.PREDICT, base_layers.TFLITE]
    if self.mode in modes:
      projection = self.qrange_tanh(projection)
    return projection
