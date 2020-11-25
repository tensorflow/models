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
# Lint as: python3
"""Utils to convert to a TFLite model."""
import tensorflow.compat.v1 as tf


def _dump_graph_in_text_format(filename, graph_def):
  """Dump a tensorflow graph in readable text format."""
  f = open(filename, 'w')

  for node in graph_def.node:
    f.write('Node: %s (%s)\n' % (node.name, node.op))
    for input_name in node.input:
      f.write('\tInput: %s\n' % input_name)
  f.close()


class InterpreterWithCustomOps(tf.lite.Interpreter):

  def __init__(self, model_content, custom_op_registerers):
    self._custom_op_registerers = custom_op_registerers
    super(InterpreterWithCustomOps, self).__init__(model_content=model_content)


def set_output_quantized_for_custom_ops(graph_def):
  """Set output types/quantized flag for custom/unsupported ops."""
  quantized_custom_ops = {
      'SequenceStringProjection': [tf.float32.as_datatype_enum],
      'SequenceStringProjectionV2': [tf.float32.as_datatype_enum],
      'PoolingOp': [tf.float32.as_datatype_enum],
      'ExpectedValueOp': [tf.float32.as_datatype_enum],
      'LayerNorm': [tf.float32.as_datatype_enum],
      'UniformCausalAttn': [tf.float32.as_datatype_enum],
  }
  custom_op_renames = {
      'SequenceStringProjection': 'SEQUENCE_STRING_PROJECTION',
      'SequenceStringProjectionV2': 'SEQUENCE_STRING_PROJECTION_V2',
  }

  for node in graph_def.node:
    if node.op in quantized_custom_ops:
      node.attr['_output_quantized'].b = True
      node.attr['_output_types'].list.type[:] = quantized_custom_ops[node.op]
    if node.op in custom_op_renames:
      node.op = custom_op_renames[node.op]


def generate_tflite(session, graph, input_tensors, output_tensors):
  """Generate TFLite model from a session, graph and input/output tensors."""
  output_nodes = [tensor.name.split(':')[0] for tensor in output_tensors]
  graph_def = tf.graph_util.convert_variables_to_constants(
      session, graph.as_graph_def(), output_nodes)

  set_output_quantized_for_custom_ops(graph_def)

  # TODO(b/171063452): Bug needs to be fixed to handle this correctly.
  #   def _node_name(tensor):
  #     return tensor.name.split(':')[0]

  #   input_arrays_with_shape = [
  #       (_node_name(tensor), None) for tensor in input_tensors
  #   ]
  #   output_arrays = [_node_name(tensor) for tensor in output_tensors]
  #   converter = tf.lite.TFLiteConverter(graph_def, None, None,
  #                                      input_arrays_with_shape, output_arrays)
  converter = tf.lite.TFLiteConverter(graph_def, input_tensors, output_tensors)
  converter.inference_type = tf.uint8
  converter.default_ranges_stats = (127.5, 127.5)
  converter.quantized_input_stats = {
      tensor.op.name: (127.5, 127.5) for tensor in input_tensors
  }
  converter.allow_custom_ops = True
  converter.experimental_new_converter = False
  return converter.convert()
