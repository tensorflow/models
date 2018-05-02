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

#include "dragnn/runtime/myelin/myelin_cell_converter.h"

#include <stddef.h>
#include <algorithm>
#include <limits>

#include "dragnn/runtime/myelin/attr_value_utils.h"
#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor.pb.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_shape.pb.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/lib/strings/numbers.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/cpu_info.h"
#include "tensorflow/core/platform/logging.h"

namespace syntaxnet {
namespace dragnn {
namespace runtime {
namespace {

// Returns true if the |tensor_name| denotes a control dependency.
bool IsControlDependency(const string &tensor_name) {
  return tensor_name[0] == '^';
}

// Returns true if the |node| is a TF variable.
bool IsVariableNode(const tensorflow::NodeDef &node) {
  return node.op() == "VariableV2";
}

// Returns true if the |node| is a tf.constant().
bool IsConstantNode(const tensorflow::NodeDef &node) {
  return node.op() == "Const";
}

// Returns true if the |node| is a tf.placeholder().
bool IsPlaceholderNode(const tensorflow::NodeDef &node) {
  return node.op() == "Placeholder";
}

// Sets |max_value| to |value| if it is lesser.
void UpdateMax(uint32 value, uint32 *max_value) {
  *max_value = std::max(*max_value, value);
}

// Loads the |tensor| from the constant |node|.  On error, returns non-OK.
tensorflow::Status GetConstantTensor(const tensorflow::NodeDef &node,
                                     tensorflow::Tensor *tensor) {
  DCHECK(IsConstantNode(node));
  return tensorflow::GetNodeAttr(node, "value", tensor);
}

// Loads the |shape| from the placeholder |node|.  On error, returns non-OK.
tensorflow::Status GetPlaceholderShape(const tensorflow::NodeDef &node,
                                       tensorflow::TensorShape *shape) {
  DCHECK(IsPlaceholderNode(node));
  return tensorflow::GetNodeAttr(node, "shape", shape);
}

// Returns the dtype string associated with the |node|, or an empty string if it
// cannot be inferred.
string GetDType(const tensorflow::NodeDef &node) {
  tensorflow::DataType dtype;
  tensorflow::Status status = tensorflow::GetNodeAttr(node, "T", &dtype);
  if (!status.ok()) status = tensorflow::GetNodeAttr(node, "dtype", &dtype);
  if (status.ok()) return tensorflow::DataTypeString(dtype);
  return string();
}

// Modifies the |dtype| into a reference type.
void MarkAsReferenceDType(string *dtype) {
  DCHECK_NE((*dtype)[0], '&');
  *dtype = tensorflow::strings::StrCat("&", *dtype);
}

// Loads the CellSubgraphSpec for the component named |component_name| from the
// |trained_model| into the |spec|.  On error, returns non-OK.
tensorflow::Status LoadCellSubgraphSpec(const string &component_name,
                                        const TrainedModel &trained_model,
                                        CellSubgraphSpec *spec) {
  const string tensor_name =
      tensorflow::strings::StrCat(component_name, "/EXPORT/CellSubgraphSpec");
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

// Writer for incrementally building a Flow file.
// https://github.com/google/sling/tree/master/myelin#flow-file-format

class MyelinCellConverter::Writer {
 public:
  // TODO(googleuser): Add templated Write() methods and coerce typed data into
  // little-endian format, so this doesn't need to run on a little-endian CPU.
  static_assert(tensorflow::port::kLittleEndian,
                "Flow files must be written in little-endian format");

  // Creates a writer that overwrites |flow|.
  explicit Writer(string *flow) : flow_(CHECK_NOTNULL(flow)) {
    flow_->clear();
    Write("flow", 4);  // magic number
    WriteInt32(4);  // version
  }

  // Appends [|data|,|data|+|size|) to the Flow file.
  void Write(const void *data, size_t size) {
    flow_->append(reinterpret_cast<const char *>(data), size);
  }

  // Appends the |value| to the Flow file.
  void WriteInt32(int32 value) { Write(&value, sizeof(int32)); }
  void WriteUint64(uint64 value) { Write(&value, sizeof(uint64)); }

  // Writes the |str| to the Flow file as a length-prefixed string.
  void WriteString(const string &str) {
    DCHECK_LE(str.size(), std::numeric_limits<int32>::max());
    WriteInt32(str.size());
    Write(str.data(), str.size());
  }

 private:
  // Flow file content.
  string *const flow_;
};

tensorflow::Status MyelinCellConverter::Convert(
    const string &component_name, const TrainedModel &trained_model,
    string *flow) {
  return MyelinCellConverter().ConvertImpl(component_name, trained_model, flow);
}

tensorflow::Status MyelinCellConverter::ConvertImpl(
    const string &component_name, const TrainedModel &trained_model,
    string *flow) {
  component_name_ = component_name;
  trained_model_ = &trained_model;

  CellSubgraphSpec spec;
  TF_RETURN_IF_ERROR(
      LoadCellSubgraphSpec(component_name_, *trained_model_, &spec));
  TF_RETURN_IF_ERROR(BuildInputsAndOutputs(spec));
  TF_RETURN_IF_ERROR(BuildOperations());

  Writer writer(flow);
  TF_RETURN_IF_ERROR(WriteVariables(&writer));
  WriteOperations(&writer);
  WriteFunctions(&writer);
  WriteConnectors(&writer);
  WriteBlobs(&writer);

  return tensorflow::Status::OK();
}

tensorflow::Status MyelinCellConverter::BuildInputsAndOutputs(
    const CellSubgraphSpec &spec) {
  std::set<string> unique_input_names;
  for (const CellSubgraphSpec::Input &input : spec.input()) {
    if (!unique_input_names.insert(input.name()).second) {
      return tensorflow::errors::InvalidArgument(
          "Duplicate input name { ", input.ShortDebugString(), " }");
    }

    TensorId tensor_id;
    TF_RETURN_IF_ERROR(ParseTensorId(input.tensor(), &tensor_id));

    if (inputs_.find(tensor_id) != inputs_.end()) {
      return tensorflow::errors::InvalidArgument(
          "Duplicate input variable { ", input.ShortDebugString(),
          " }; currently has name '", inputs_[tensor_id], "'");
    }

    inputs_[tensor_id] = input.name();
  }

  std::set<string> unique_output_names;
  for (const CellSubgraphSpec::Output &output : spec.output()) {
    if (!unique_output_names.insert(output.name()).second) {
      return tensorflow::errors::InvalidArgument(
          "Duplicate output name { ", output.ShortDebugString(), " }");
    }

    TensorId tensor_id;
    TF_RETURN_IF_ERROR(ParseTensorId(output.tensor(), &tensor_id));

    outputs_[tensor_id].insert(output.name());
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

tensorflow::Status MyelinCellConverter::BuildOperations() {
  // Extract sets of input and output node names.
  std::set<string> input_node_names;
  std::set<string> output_node_names;
  for (const auto &it : inputs_) input_node_names.insert(it.first.first);
  for (const auto &it : outputs_) output_node_names.insert(it.first.first);

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

      // Add the input tensor and register the output index on the input op.
      operation.inputs.push_back(AsVariableName(tensor_id));
      UpdateMax(tensor_id.second + 1,
                &operations_[tensor_id.first].num_outputs);
    }
  }

  // Register output indices for the |outputs_|; the DFS does not cover these.
  for (const auto &it : outputs_) {
    const TensorId &tensor_id = it.first;
    UpdateMax(tensor_id.second + 1, &operations_[tensor_id.first].num_outputs);
  }

  // Sanity check: All operations must have nodes and outputs.
  for (const auto &it : operations_) {
    const Operation &operation = it.second;
    DCHECK(operation.node != nullptr);
    DCHECK_GT(operation.num_outputs, 0);
  }

  return tensorflow::Status::OK();
}

tensorflow::Status MyelinCellConverter::WriteVariables(Writer *writer) const {
  int num_variables = 0;
  for (const auto &it : operations_) num_variables += it.second.num_outputs;
  writer->WriteInt32(num_variables);

  for (const auto &it : operations_) {
    const Operation &operation = it.second;
    for (uint32 output = 0; output < operation.num_outputs; ++output) {
      TF_RETURN_IF_ERROR(WriteVariable(*operation.node, output, writer));
    }
  }

  return tensorflow::Status::OK();
}

tensorflow::Status MyelinCellConverter::WriteVariable(
    const tensorflow::NodeDef &node, uint32 output_index,
    Writer *writer) const {
  const TensorId tensor_id(node.name(), output_index);
  const string name = AsVariableName(tensor_id);
  const std::set<string> aliases = GetAliases(tensor_id);

  // Only cell inputs and outputs have aliases.
  const bool is_cell_input_or_output = !aliases.empty();

  // Treat cell inputs and outputs as references, so they can be pointed at
  // pieces of memory managed by the DRAGNN runtime.
  string dtype = GetDType(node);
  if (is_cell_input_or_output) MarkAsReferenceDType(&dtype);

  // Extract variable data and shape, if available.  Myelin treats a 0-element
  // shape (e.g., [0], [1, 0, 2]) as undefined and will infer shapes for such
  // variables, so we ensure that the shape is undefined unless explicitly set.
  tensorflow::Tensor tensor;
  tensorflow::TensorShape shape({0});  // undefined by default
  if (IsConstantNode(node)) {
    TF_RETURN_IF_ERROR(GetConstantTensor(node, &tensor));
    shape = tensor.shape();
  } else if (IsVariableNode(node)) {
    TF_RETURN_IF_ERROR(trained_model_->EvaluateTensor(name, &tensor));
    shape = tensor.shape();
  } else if (IsPlaceholderNode(node)) {
    TF_RETURN_IF_ERROR(GetPlaceholderShape(node, &shape));
  }
  const tensorflow::StringPiece data = tensor.tensor_data();

  writer->WriteString(name);
  writer->WriteInt32(aliases.size());
  for (const string &alias : aliases) writer->WriteString(alias);
  writer->WriteString(dtype);

  writer->WriteInt32(shape.dims());
  for (int i = 0; i < shape.dims(); ++i) writer->WriteInt32(shape.dim_size(i));

  writer->WriteUint64(data.size());
  writer->Write(data.data(), data.size());

  return tensorflow::Status::OK();
}

std::set<string> MyelinCellConverter::GetAliases(
    const TensorId &tensor_id) const {
  std::set<string> aliases;

  const auto input_it = inputs_.find(tensor_id);
  if (input_it != inputs_.end()) {
    const string &name = input_it->second;
    aliases.insert(tensorflow::strings::StrCat("INPUT/", name));
  }

  const auto output_it = outputs_.find(tensor_id);
  if (output_it != outputs_.end()) {
    for (const string &name : output_it->second) {
      aliases.insert(tensorflow::strings::StrCat("OUTPUT/", name));
    }
  }

  return aliases;
}

void MyelinCellConverter::WriteOperations(Writer *writer) const {
  writer->WriteInt32(operations_.size());
  for (const auto &it : operations_) {
    const Operation &operation = it.second;
    WriteOperation(operation, writer);
  }
}

void MyelinCellConverter::WriteOperation(const Operation &operation,
                                         Writer *writer) const {
  const string &name = operation.node->name();
  const string &type = operation.node->op();

  // Create one output per possible output index, in order.
  std::vector<string> outputs;
  for (uint32 output = 0; output < operation.num_outputs; ++output) {
    outputs.push_back(AsVariableName(TensorId(name, output)));
  }

  // Copy the attrs to a sorted map for deterministic ordering.
  std::map<string, tensorflow::AttrValue> attrs(operation.node->attr().begin(),
                                                operation.node->attr().end());

  writer->WriteString(name);
  writer->WriteString(type);

  writer->WriteInt32(operation.inputs.size());
  for (const string &input : operation.inputs) writer->WriteString(input);

  writer->WriteInt32(outputs.size());
  for (const string &output : outputs) writer->WriteString(output);

  writer->WriteInt32(attrs.size());
  for (const auto &it : attrs) {
    writer->WriteString(it.first);
    writer->WriteString(AttrValueToString(it.second));
  }
}

void MyelinCellConverter::WriteFunctions(Writer *writer) const {
  writer->WriteInt32(1);
  writer->WriteString(component_name_);
  writer->WriteInt32(operations_.size());
  for (const auto &it : operations_) writer->WriteString(it.first);
}

void MyelinCellConverter::WriteConnectors(Writer *writer) const {
  writer->WriteInt32(0);
}

void MyelinCellConverter::WriteBlobs(Writer *writer) const {
  writer->WriteInt32(0);
}

tensorflow::Status MyelinCellConverter::ParseTensorId(const string &tensor_name,
                                                      TensorId *tensor_id) {
  if (IsControlDependency(tensor_name)) {
    return tensorflow::errors::InvalidArgument(
        "Cannot parse tensor ID from control dependency '", tensor_name, "'");
  }

  const auto colon_index = tensor_name.rfind(':');

  // NB: If |colon_index| is string::npos, takes the whole string as desired.
  tensor_id->first = tensor_name.substr(0, colon_index);

  if (colon_index == string::npos) {  // no colon; assume 0
    tensor_id->second = 0;
  } else {
    const string output_str = tensor_name.substr(colon_index + 1);
    if (!tensorflow::strings::safe_strtou32(output_str, &tensor_id->second)) {
      return tensorflow::errors::InvalidArgument("Malformed tensor name ",
                                                 tensor_name);
    }
  }

  return tensorflow::Status::OK();
}

string MyelinCellConverter::AsVariableName(const TensorId &tensor_id) {
  if (tensor_id.second == 0) return tensor_id.first;
  return tensorflow::strings::StrCat(tensor_id.first, ":", tensor_id.second);
}

}  // namespace runtime
}  // namespace dragnn
}  // namespace syntaxnet
