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
"""Tests for graph_rewriter_builder."""
import unittest
from unittest import mock  # pylint: disable=g-importing-member
import tensorflow.compat.v1 as tf
import tf_slim as slim

from object_detection.builders import graph_rewriter_builder
from object_detection.protos import graph_rewriter_pb2
from object_detection.utils import tf_version


if tf_version.is_tf1():
  from tensorflow.contrib import quantize as contrib_quantize  # pylint: disable=g-import-not-at-top


@unittest.skipIf(tf_version.is_tf2(), 'Skipping TF1.X only test.')
class QuantizationBuilderTest(tf.test.TestCase):

  def testQuantizationBuilderSetsUpCorrectTrainArguments(self):
    with mock.patch.object(
        contrib_quantize,
        'experimental_create_training_graph') as mock_quant_fn:
      with mock.patch.object(slim,
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
      with mock.patch.object(slim,
                             'summarize_collection') as mock_summarize_col:
        graph_rewriter_proto = graph_rewriter_pb2.GraphRewriter()
        graph_rewriter_proto.quantization.delay = 10
        graph_rewrite_fn = graph_rewriter_builder.build(
            graph_rewriter_proto, is_training=False)
        graph_rewrite_fn()
        _, kwargs = mock_quant_fn.call_args
        self.assertEqual(kwargs['input_graph'], tf.get_default_graph())
        mock_summarize_col.assert_called_with('quant_vars')


if __name__ == '__main__':
  tf.test.main()
