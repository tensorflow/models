// Copyright 2017 Google Inc. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// =============================================================================

#include "dragnn/runtime/xla/xla_cell_converter.h"

#include <vector>

#include "dragnn/runtime/xla/xla_graph_utils.h"
#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/strings/numbers.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/logging.h"

namespace syntaxnet {
namespace dragnn {
namespace runtime {
namespace {

// Returns true if the |tensor_name| denotes a control dependency.
bool IsControlDependency(const string &tensor_name) {
  return tensor_name[0] == '^';
}

// Returns the name of the node that supplies the input called |input_name|.
// This strips off any prefix on control dependencies and any suffix
// for specifying tensor output.
const string GetNodeNameFromInput(const string &input_name) {
  return input_name.substr(IsControlDependency(input_name) ? 1 : 0,
                           input_name.rfind(':'));
}

// Returns true if the |node| is a TF variable.
bool IsVariableNode(const tensorflow::NodeDef &node) {
  return node.op() == "VariableV2";
}

// Returns true if the |node| is skippable and can be changed
// to an Identity node.
bool IsNodeConvertibleToIdentity(const tensorflow::NodeDef &node) {
  return node.op() == "Enter";
}

// Returns true if the node attribute with |name| is one that should always be
// retained, when a node is being simplified or frozen.
bool AlwaysKeepAttribute(const string &name) {
  return name == "_output_shapes" || name == "T" || name == "dtype";
}

// Generates the name of the node that contains the serialized CellSubgraphSpec
// given a particular |component_name|.
string MakeCellSubgraphSpecNodeName(const string &component_name) {
  return tensorflow::strings::StrCat(component_name,
                                     "/EXPORT/CellSubgraphSpec");
}

// Loads the CellSubgraphSpec for the component named |component_name| from the
// |trained_model| into the |spec|.  On error, returns non-OK.
tensorflow::Status LoadCellSubgraphSpec(const string &component_name,
                                        const TrainedModel &trained_model,
                                        CellSubgraphSpec *spec) {
  const string tensor_name = MakeCellSubgraphSpecNodeName(component_name);
  tensorflow::Tensor tensor;
  TF_RETURN_IF_ERROR(trained_model.EvaluateTensor(tensor_name, &tensor));

  if (!spec->ParseFromString(tensor.scalar<string>()())) {
    return tensorflow::errors::InvalidArgument(
        "Failed to parse CellSubgraphSpec for component ", component_name);
  }

  VLOG(1) << tensor_name << " = \n" << spec->DebugString();
  return tensorflow::Status::OK();
}

}  // namespace

tensorflow::Status XlaCellConverter::FillNode(
    const tensorflow::NodeDef &src_node, tensorflow::NodeDef *dest_node) const {
  dest_node->set_name(src_node.name());
  dest_node->set_device(src_node.device());

  if (IsNodeConvertibleToIdentity(src_node)) {
    dest_node->set_op("Identity");
    FillNodeAttributes(true, src_node, dest_node);
  } else {
    dest_node->set_op(src_node.op());
    FillNodeAttributes(false, src_node, dest_node);
  }

  for (const string &input : src_node.input()) {
    if (IsNodeInSubgraph(GetNodeNameFromInput(input))) {
      dest_node->add_input(input);
    }
  }

  return tensorflow::Status::OK();
}

tensorflow::Status XlaCellConverter::FreezeSpecNode(
    const tensorflow::NodeDef &src_node, tensorflow::NodeDef *dest_node) const {
  dest_node->set_name(kFrozenCellSubgraphSpecNodeName);
  dest_node->set_op("Const");
  FillNodeAttributes(true, src_node, dest_node);

  tensorflow::Tensor tensor;
  TF_RETURN_IF_ERROR(trained_model_->EvaluateTensor(
      AsVariableName(TensorId(src_node.name(), 0)), &tensor));

  // Leaves constants directly accessible, which allows for simple
  // extraction of the value.
  tensor.AsProtoField((*dest_node->mutable_attr())["value"].mutable_tensor());

  return tensorflow::Status::OK();
}

tensorflow::Status XlaCellConverter::FreezeNode(
    const tensorflow::NodeDef &src_node, tensorflow::NodeDef *dest_node) const {
  dest_node->set_name(src_node.name());
  dest_node->set_op("Const");
  FillNodeAttributes(true, src_node, dest_node);

  tensorflow::Tensor tensor;
  TF_RETURN_IF_ERROR(trained_model_->EvaluateTensor(
      AsVariableName(TensorId(src_node.name(), 0)), &tensor));

  // Compactly stores tensor constants.
  tensor.AsProtoTensorContent(
      (*dest_node->mutable_attr())["value"].mutable_tensor());

  return tensorflow::Status::OK();
}

void XlaCellConverter::FillNodeAttributes(bool restrict_attributes,
                                          const tensorflow::NodeDef &src_node,
                                          tensorflow::NodeDef *dest_node) {
  for (const auto &attr : src_node.attr()) {
    if (!restrict_attributes || AlwaysKeepAttribute(attr.first)) {
      (*dest_node->mutable_attr())[attr.first] = attr.second;
    }
  }
}

bool XlaCellConverter::IsNodeInSubgraph(const string &node_name) const {
  return operations_.find(node_name) != operations_.end();
}

tensorflow::Status XlaCellConverter::Convert(const string &component_name,
                                             const TrainedModel &trained_model,
                                             tensorflow::GraphDef *graph,
                                             CellSubgraphSpec *spec) {
  return XlaCellConverter().ConvertImpl(component_name, trained_model, graph,
                                        spec);
}

tensorflow::Status XlaCellConverter::ConvertImpl(
    const string &component_name, const TrainedModel &trained_model,
    tensorflow::GraphDef *graph, CellSubgraphSpec *spec) {
  component_name_ = component_name;
  trained_model_ = &trained_model;

  TF_RETURN_IF_ERROR(
      LoadCellSubgraphSpec(component_name_, *trained_model_, spec));
  TF_RETURN_IF_ERROR(BuildInputsAndOutputs(*spec));
  TF_RETURN_IF_ERROR(BuildOperations());

  graph->Clear();
  const tensorflow::GraphDef *input_graph;
  TF_RETURN_IF_ERROR(trained_model_->GraphDef(&input_graph));

  // Adds in the CellSubgraphSpec node for this component.
  const tensorflow::NodeDef *cell_subgraph_spec_node = nullptr;
  TF_RETURN_IF_ERROR(trained_model_->LookupNode(
      MakeCellSubgraphSpecNodeName(component_name_), &cell_subgraph_spec_node));
  TF_RETURN_IF_ERROR(
      FreezeSpecNode(*cell_subgraph_spec_node, graph->add_node()));

  // Adds in frozen versions of the nodes needed for this cell.
  for (const tensorflow::NodeDef &node : input_graph->node()) {
    if (IsNodeInSubgraph(node.name())) {
      if (IsVariableNode(node)) {
        TF_RETURN_IF_ERROR(FreezeNode(node, graph->add_node()));
      } else {
        TF_RETURN_IF_ERROR(FillNode(node, graph->add_node()));
      }
    }
  }

  return tensorflow::Status::OK();
}

tensorflow::Status XlaCellConverter::BuildInputsAndOutputs(
    const CellSubgraphSpec &spec) {
  std::set<string> unique_input_names;
  for (const CellSubgraphSpec::Input &input : spec.input()) {
    if (!unique_input_names.insert(input.name()).second) {
      return tensorflow::errors::InvalidArgument(
          "Duplicate input name { ", input.ShortDebugString(), " }");
    }

    TensorId tensor_id;
    TF_RETURN_IF_ERROR(ParseTensorId(input.tensor(), &tensor_id));
    if (!inputs_.insert(tensor_id).second) {
      return tensorflow::errors::InvalidArgument(
          "Duplicate input variable { ", input.ShortDebugString(), " }");
    }
  }

  std::set<string> unique_output_names;
  for (const CellSubgraphSpec::Output &output : spec.output()) {
    if (!unique_output_names.insert(output.name()).second) {
      return tensorflow::errors::InvalidArgument(
          "Duplicate output name { ", output.ShortDebugString(), " }");
    }

    TensorId tensor_id;
    TF_RETURN_IF_ERROR(ParseTensorId(output.tensor(), &tensor_id));
    outputs_.insert(tensor_id);
  }

  // Check that recurrent inputs match the name of an output.
  for (const CellSubgraphSpec::Input &input : spec.input()) {
    if (input.type() != CellSubgraphSpec::Input::TYPE_RECURRENT) continue;

    if (unique_output_names.find(input.name()) == unique_output_names.end()) {
      return tensorflow::errors::InvalidArgument(
          "Recurrent input does not match any output { ",
          input.ShortDebugString(), " }");
    }
  }

  return tensorflow::Status::OK();
}

tensorflow::Status XlaCellConverter::BuildOperations() {
  // Extract sets of input and output node names.
  std::set<string> input_node_names;
  std::set<string> output_node_names;
  for (const TensorId &id : inputs_) input_node_names.insert(id.first);
  for (const TensorId &id : outputs_) output_node_names.insert(id.first);

  // Set of nodes that have already been visited by the DFS.
  std::set<string> visited;

  // DFS backwards from output nodes to input nodes and collect operations.
  std::vector<string> stack(output_node_names.begin(), output_node_names.end());
  while (!stack.empty()) {
    const string name = stack.back();
    stack.pop_back();
    if (!visited.insert(name).second) continue;  // already visited; skip

    const tensorflow::NodeDef *node = nullptr;
    TF_RETURN_IF_ERROR(trained_model_->LookupNode(name, &node));

    Operation &operation = operations_[name];
    if (operation.node != nullptr && operation.node != node) {
      return tensorflow::errors::Internal("Inconsistent nodes for operation ",
                                          name, " (", operation.node->name(),
                                          " vs ", node->name());
    }
    operation.node = node;

    // Function inputs bound the search; don't expand them.
    if (input_node_names.find(name) != input_node_names.end()) continue;

    // Expand (non-control) inputs.
    for (const string &input_name : node->input()) {
      if (IsControlDependency(input_name)) continue;
      VLOG(1) << name << " has input " << input_name;

      TensorId tensor_id;
      TF_RETURN_IF_ERROR(ParseTensorId(input_name, &tensor_id));
      stack.push_back(tensor_id.first);
    }
  }

  return tensorflow::Status::OK();
}

tensorflow::Status XlaCellConverter::ParseTensorId(const string &tensor_name,
                                                   TensorId *tensor_id) {
  return ParseTensorName(tensor_name, &tensor_id->first, &tensor_id->second);
}

string XlaCellConverter::AsVariableName(const TensorId &tensor_id) {
  if (tensor_id.second == 0) return tensor_id.first;
  return tensorflow::strings::StrCat(tensor_id.first, ":", tensor_id.second);
}

}  // namespace runtime
}  // namespace dragnn
}  // namespace syntaxnet
