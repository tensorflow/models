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

"""Renders DRAGNN specs with Graphviz."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import warnings

import pygraphviz
from dragnn.protos import spec_pb2


def _component_contents(component):
  """Generates the label on component boxes.

  Args:
    component: spec_pb2.ComponentSpec proto

  Returns:
    String label
  """
  return """<
  <B>{name}</B><BR />
  {transition_name}<BR />
  {network_name}<BR />
  {num_actions_str}<BR />
  hidden: {num_hidden}
  >""".format(
      name=component.name,
      transition_name=component.transition_system.registered_name,
      network_name=component.network_unit.registered_name,
      num_actions_str="{} action{}".format(component.num_actions, "s" if
                                           component.num_actions != 1 else ""),
      num_hidden=component.network_unit.parameters.get("hidden_layer_sizes",
                                                       "not specified"))


def _linked_feature_label(linked_feature):
  """Generates the label on edges between components.

  Args:
    linked_feature: spec_pb2.LinkedFeatureChannel proto

  Returns:
    String label
  """
  return """<
  <B>{name}</B><BR />
  F={num_features} D={projected_dim}<BR />
  {fml}<BR />
  <U>{source_translator}</U><BR />
  <I>{source_layer}</I>
  >""".format(
      name=linked_feature.name,
      num_features=linked_feature.size,
      projected_dim=linked_feature.embedding_dim,
      fml=linked_feature.fml,
      source_translator=linked_feature.source_translator,
      source_layer=linked_feature.source_layer)


def master_spec_graph(master_spec):
  """Constructs a master spec graph.

  Args:
    master_spec: MasterSpec proto.

  Raises:
    TypeError, if master_spec is not the right type. N.B. that this may be
    raised if you import proto classes in non-standard ways (e.g. dynamically).

  Returns:
    SVG graph contents as a string.
  """
  if not isinstance(master_spec, spec_pb2.MasterSpec):
    raise TypeError("master_spec_graph() expects a MasterSpec input.")

  graph = pygraphviz.AGraph(directed=True)

  graph.node_attr.update(
      shape="box",
      style="filled",
      fillcolor="white",
      fontname="roboto, helvetica, arial",
      fontsize=11)
  graph.edge_attr.update(fontname="roboto, helvetica, arial", fontsize=11)

  for component in master_spec.component:
    graph.add_node(component.name, label=_component_contents(component))

  for component in master_spec.component:
    for linked_feature in component.linked_feature:
      graph.add_edge(
          linked_feature.source_component,
          component.name,
          label=_linked_feature_label(linked_feature))

  with warnings.catch_warnings():
    # Fontconfig spews some warnings, suppress them for now. (Especially because
    # they can clutter IPython notebooks).
    warnings.simplefilter("ignore")
    return graph.draw(format="svg", prog="dot")
