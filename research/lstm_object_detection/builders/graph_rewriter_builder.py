# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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

"""Custom version for quantized training and evaluation functions.

The main difference between this and the third_party graph_rewriter_builder.py
is that this version uses experimental_create_training_graph which allows the
customization of freeze_bn_delay.
"""

import re
import tensorflow.compat.v1 as tf
from tensorflow.contrib import layers as contrib_layers
from tensorflow.contrib import quantize as contrib_quantize
from tensorflow.contrib.quantize.python import common
from tensorflow.contrib.quantize.python import input_to_ops
from tensorflow.contrib.quantize.python import quant_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops


def build(graph_rewriter_config,
          quant_overrides_config=None,
          is_training=True,
          is_export=False):
  """Returns a function that modifies default graph based on options.

  Args:
    graph_rewriter_config: graph_rewriter_pb2.GraphRewriter proto.
    quant_overrides_config: quant_overrides_pb2.QuantOverrides proto.
    is_training: whether in training or eval mode.
    is_export: whether exporting the graph.
  """
  def graph_rewrite_fn():
    """Function to quantize weights and activation of the default graph."""
    if (graph_rewriter_config.quantization.weight_bits != 8 or
        graph_rewriter_config.quantization.activation_bits != 8):
      raise ValueError('Only 8bit quantization is supported')

    graph = tf.get_default_graph()

    # Insert custom quant ops.
    if quant_overrides_config is not None:
      input_to_ops_map = input_to_ops.InputToOps(graph)
      for q in quant_overrides_config.quant_configs:
        producer = graph.get_operation_by_name(q.op_name)
        if producer is None:
          raise ValueError('Op name does not exist in graph.')
        context = _get_context_from_op(producer)
        consumers = input_to_ops_map.ConsumerOperations(producer)
        if q.fixed_range:
          _insert_fixed_quant_op(
              context,
              q.quant_op_name,
              producer,
              consumers,
              init_min=q.min,
              init_max=q.max,
              quant_delay=q.delay if is_training else 0)
        else:
          raise ValueError('Learned ranges are not yet supported.')

    # Quantize the graph by inserting quantize ops for weights and activations
    if is_training:
      contrib_quantize.experimental_create_training_graph(
          input_graph=graph,
          quant_delay=graph_rewriter_config.quantization.delay,
          freeze_bn_delay=graph_rewriter_config.quantization.delay)
    else:
      contrib_quantize.experimental_create_eval_graph(
          input_graph=graph,
          quant_delay=graph_rewriter_config.quantization.delay
          if not is_export else 0)

    contrib_layers.summarize_collection('quant_vars')

  return graph_rewrite_fn


def _get_context_from_op(op):
  """Gets the root context name from the op name."""
  context_re = re.search(r'^(.*)/([^/]+)', op.name)
  if context_re:
    return context_re.group(1)
  return ''


def _insert_fixed_quant_op(context,
                           name,
                           producer,
                           consumers,
                           init_min=-6.0,
                           init_max=6.0,
                           quant_delay=None):
  """Adds a fake quant op with fixed ranges.

  Args:
    context: The parent scope of the op to be quantized.
    name: The name of the fake quant op.
    producer: The producer op to be quantized.
    consumers: The consumer ops to the producer op.
    init_min: The minimum range for the fake quant op.
    init_max: The maximum range for the fake quant op.
    quant_delay: Number of steps to wait before activating the fake quant op.

  Raises:
    ValueError: When producer operation is not directly connected to the
      consumer operation.
  """
  name_prefix = name if not context else context + '/' + name
  inputs = producer.outputs[0]
  quant = quant_ops.FixedQuantize(
      inputs, init_min=init_min, init_max=init_max, scope=name_prefix)

  if quant_delay and quant_delay > 0:
    activate_quant = math_ops.greater_equal(
        common.CreateOrGetQuantizationStep(),
        quant_delay,
        name=name_prefix + '/activate_quant')
    quant = control_flow_ops.cond(
        activate_quant,
        lambda: quant,
        lambda: inputs,
        name=name_prefix + '/delayed_quant')

  if consumers:
    tensors_modified_count = common.RerouteTensor(
        quant, inputs, can_modify=consumers)
    # Some operations can have multiple output tensors going to the same
    # consumer. Since consumers is a set, we need to ensure that
    # tensors_modified_count is greater than or equal to the length of the set
    # of consumers.
    if tensors_modified_count < len(consumers):
      raise ValueError('No inputs quantized for ops: [%s]' % ', '.join(
          [consumer.name for consumer in consumers]))
