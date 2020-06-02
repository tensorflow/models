# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
"""Functions for quantized training and evaluation."""

import tensorflow.compat.v1 as tf
import tf_slim as slim
# pylint: disable=g-import-not-at-top
try:
  from tensorflow.contrib import quantize as contrib_quantize
except ImportError:
  # TF 2.0 doesn't ship with contrib.
  pass
# pylint: enable=g-import-not-at-top


def build(graph_rewriter_config, is_training):
  """Returns a function that modifies default graph based on options.

  Args:
    graph_rewriter_config: graph_rewriter_pb2.GraphRewriter proto.
    is_training: whether in training of eval mode.
  """
  def graph_rewrite_fn():
    """Function to quantize weights and activation of the default graph."""
    if (graph_rewriter_config.quantization.weight_bits != 8 or
        graph_rewriter_config.quantization.activation_bits != 8):
      raise ValueError('Only 8bit quantization is supported')

    # Quantize the graph by inserting quantize ops for weights and activations
    if is_training:
      contrib_quantize.experimental_create_training_graph(
          input_graph=tf.get_default_graph(),
          quant_delay=graph_rewriter_config.quantization.delay
      )
    else:
      contrib_quantize.experimental_create_eval_graph(
          input_graph=tf.get_default_graph()
      )
    slim.summarize_collection('quant_vars')

  return graph_rewrite_fn
