# -*- coding: utf-8 -*-
# Copyright 2017 Google Inc. All Rights Reserved.
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

"""Tests for dragnn.python.visualization."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.platform import googletest
from dragnn.protos import spec_pb2
from dragnn.protos import trace_pb2
from dragnn.python import visualization


def _get_trace_proto_string():
  trace = trace_pb2.MasterTrace()
  trace.component_trace.add(
      step_trace=[
          trace_pb2.ComponentStepTrace(fixed_feature_trace=[]),
      ],
      # Google Translate says this is "component" in Chinese. (To test UTF-8).
      name='零件',)
  return trace.SerializeToString()


def _get_master_spec():
  return spec_pb2.MasterSpec(
      component=[spec_pb2.ComponentSpec(name='jalapeño')])


class VisualizationTest(googletest.TestCase):

  def testCanFindScript(self):
    script = visualization._load_viz_script()
    self.assertIsInstance(script, str)
    self.assertTrue(10e3 < len(script) < 10e6,
                    'Script size should be between 10k and 10M')

  def testSampleTraceSerialization(self):
    json = visualization.parse_trace_json(_get_trace_proto_string())
    self.assertIsInstance(json, str)
    self.assertTrue('component_trace' in json)

  def testInteractiveVisualization(self):
    widget = visualization.InteractiveVisualization()
    widget.initial_html()
    widget.show_trace(_get_trace_proto_string())

  def testMasterSpecJson(self):
    visualization.trace_html(
        _get_trace_proto_string(), master_spec=_get_master_spec())
    widget = visualization.InteractiveVisualization()
    widget.initial_html()
    widget.show_trace(_get_trace_proto_string(), master_spec=_get_master_spec())


if __name__ == '__main__':
  googletest.main()
