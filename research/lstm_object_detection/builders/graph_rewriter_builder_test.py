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

"""Tests for graph_rewriter_builder."""
import mock
import tensorflow.compat.v1 as tf
from tensorflow.contrib import layers as contrib_layers
from tensorflow.contrib import quantize as contrib_quantize
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from lstm_object_detection.builders import graph_rewriter_builder
from lstm_object_detection.protos import quant_overrides_pb2
from object_detection.protos import graph_rewriter_pb2


class QuantizationBuilderTest(tf.test.TestCase):

  def testQuantizationBuilderSetsUpCorrectTrainArguments(self):
    with mock.patch.object(
        contrib_quantize,
        'experimental_create_training_graph') as mock_quant_fn:
      with mock.patch.object(contrib_layers,
                             'summarize_collection') as mock_summarize_col:
        graph_rewriter_proto = graph_rewriter_pb2.GraphRewriter()
        graph_rewriter_proto.quantization.delay = 10
        graph_rewriter_proto.quantization.weight_bits = 8
        graph_rewriter_proto.quantization.activation_bits = 8
        graph_rewrite_fn = graph_rewriter_builder.build(
            graph_rewriter_proto, is_training=True)
        graph_rewrite_fn()
        _, kwargs = mock_quant_fn.call_args
        self.assertEqual(kwargs['input_graph'], tf.get_default_graph())
        self.assertEqual(kwargs['quant_delay'], 10)
        mock_summarize_col.assert_called_with('quant_vars')

  def testQuantizationBuilderSetsUpCorrectEvalArguments(self):
    with mock.patch.object(contrib_quantize,
                           'experimental_create_eval_graph') as mock_quant_fn:
      with mock.patch.object(contrib_layers,
                             'summarize_collection') as mock_summarize_col:
        graph_rewriter_proto = graph_rewriter_pb2.GraphRewriter()
        graph_rewriter_proto.quantization.delay = 10
        graph_rewrite_fn = graph_rewriter_builder.build(
            graph_rewriter_proto, is_training=False)
        graph_rewrite_fn()
        _, kwargs = mock_quant_fn.call_args
        self.assertEqual(kwargs['input_graph'], tf.get_default_graph())
        mock_summarize_col.assert_called_with('quant_vars')

  def testQuantizationBuilderAddsQuantOverride(self):
    graph = ops.Graph()
    with graph.as_default():
      self._buildGraph()

      quant_overrides_proto = quant_overrides_pb2.QuantOverrides()
      quant_config = quant_overrides_proto.quant_configs.add()
      quant_config.op_name = 'test_graph/add_ab'
      quant_config.quant_op_name = 'act_quant'
      quant_config.fixed_range = True
      quant_config.min = 0
      quant_config.max = 6
      quant_config.delay = 100

      graph_rewriter_proto = graph_rewriter_pb2.GraphRewriter()
      graph_rewriter_proto.quantization.delay = 10
      graph_rewriter_proto.quantization.weight_bits = 8
      graph_rewriter_proto.quantization.activation_bits = 8

      graph_rewrite_fn = graph_rewriter_builder.build(
          graph_rewriter_proto,
          quant_overrides_config=quant_overrides_proto,
          is_training=True)
      graph_rewrite_fn()

      act_quant_found = False
      quant_delay_found = False
      for op in graph.get_operations():
        if (quant_config.quant_op_name in op.name and
            op.type == 'FakeQuantWithMinMaxArgs'):
          act_quant_found = True
          min_val = op.get_attr('min')
          max_val = op.get_attr('max')
          self.assertEqual(min_val, quant_config.min)
          self.assertEqual(max_val, quant_config.max)
        if ('activate_quant' in op.name and
            quant_config.quant_op_name in op.name and op.type == 'Const'):
          tensor = op.get_attr('value')
          if tensor.int64_val[0] == quant_config.delay:
            quant_delay_found = True

      self.assertTrue(act_quant_found)
      self.assertTrue(quant_delay_found)

  def _buildGraph(self, scope='test_graph'):
    with ops.name_scope(scope):
      a = tf.constant(10, dtype=dtypes.float32, name='input_a')
      b = tf.constant(20, dtype=dtypes.float32, name='input_b')
      ab = tf.add(a, b, name='add_ab')
      c = tf.constant(30, dtype=dtypes.float32, name='input_c')
      abc = tf.multiply(ab, c, name='mul_ab_c')
      return abc


if __name__ == '__main__':
  tf.test.main()
