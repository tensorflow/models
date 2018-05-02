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

#ifndef DRAGNN_RUNTIME_MYELIN_MYELIN_CELL_CONVERTER_H_
#define DRAGNN_RUNTIME_MYELIN_MYELIN_CELL_CONVERTER_H_

#include <map>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "dragnn/protos/export.pb.h"
#include "dragnn/runtime/trained_model.h"
#include "syntaxnet/base.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/lib/core/status.h"

namespace syntaxnet {
namespace dragnn {
namespace runtime {

// Converter that extracts the cell computation from a DRAGNN component and
// writes it as a Myelin Flow.

//
// The trained model that contains the DRAGNN component must also contain a
// CellSubgraphSpec proto embedded into the TF graph as a specifically-named
// constant node (see runtime_support.py).  The CellSubgraphSpec defines the
// boundaries of the cell comptation.
//
// The converted Myelin Flow contains a single function that runs the cell and
// is named after the component.  The function inputs and outputs are reference
// variables, so they can be pointed at externally-managed pieces of memory,
// provided sufficient size and alignment.  The function inputs and outputs are
// marked with special aliases, namely:
//   INPUT/<CellSubgraphSpec.Input.name>
//   OUTPUT/<CellSubgraphSpec.Output.name>
class MyelinCellConverter {
 public:
  // Extracts the cell of the DRAGNN component named |component_name| from the
  // |trained_model| and overwrites the |flow| with an equivalent Myelin Flow.
  // The |flow| file output is deterministic given identical inputs.  On error,
  // returns non-OK.
  static tensorflow::Status Convert(const string &component_name,
                                    const TrainedModel &trained_model,
                                    string *flow);

 private:
  // A (node_name, output_index) pair denoting a tensor.
  using TensorId = std::pair<string, uint32>;

  // Flow file writer; defined in the .cc file.
  class Writer;

  // An operation that makes up the cell, convertible to a Myelin operation.
  struct Operation {
    // The TF graph node represented by this operation.
    const tensorflow::NodeDef *node = nullptr;

    // Myelin variable names of inputs to this operation.  Order matters.
    std::vector<string> inputs;

    // Number of outputs observed for this operation.  Some of the outputs in
    // [0,|num_outputs|) might not actually be used in the cell, but we must
    // create variables for all of them to match the expected output arity and
    // ordering of the operation.
    uint32 num_outputs = 0;
  };

  // Creates an empty converter.
  MyelinCellConverter() = default;

  // Implements the static Convert() method.
  tensorflow::Status ConvertImpl(const string &component_name,
                                 const TrainedModel &trained_model,
                                 string *flow);

  // Populates the |inputs_| and |outputs_| based on the |spec|.  On error,
  // returns non-OK.
  tensorflow::Status BuildInputsAndOutputs(const CellSubgraphSpec &spec);

  // Walks from the |outputs_| to the |inputs_| in the |trained_model_|, adding
  // to |operations_| along the way.  Requires that BuildInputsAndOutputs() was
  // called.  On error, returns non-OK.
  tensorflow::Status BuildOperations();

  // Writes each section of a flow file to the |writer|.
  tensorflow::Status WriteVariables(Writer *writer) const;
  void WriteOperations(Writer *writer) const;
  void WriteFunctions(Writer *writer) const;
  void WriteConnectors(Writer *writer) const;
  void WriteBlobs(Writer *writer) const;

  // Writes a variable for the |output_index|'th output of the |node| to the
  // |writer|.  Retrieves constant variable data from the |trained_model_| if
  // necessary.  On error, returns non-OK.
  tensorflow::Status WriteVariable(const tensorflow::NodeDef &node,
                                   uint32 output_index, Writer *writer) const;

  // Writes the |operation| to the |writer|.
  void WriteOperation(const Operation &operation, Writer *writer) const;

  // Returns the set of aliases associated with the |tensor_id|.
  std::set<string> GetAliases(const TensorId &tensor_id) const;

  // Parses a |tensor_name| into a |tensor_id|.  E.g.,
  //   "foo/bar:1" => ("foo/bar", 1)
  //   "baz"       => ("baz", 0)
  // On error, returns non-OK.  It is an error if the |tensor_name| denotes a
  // control dependency.
  static tensorflow::Status ParseTensorId(const string &tensor_name,
                                          TensorId *tensor_id);

  // Returns the canonically-formatted name of the Myelin variable associated
  // with the |tensor_id|.
  static string AsVariableName(const TensorId &tensor_id);

  // Name of the component being converted.
  string component_name_;

  // Trained model that contains the DRAGNN model.
  const TrainedModel *trained_model_ = nullptr;

  // Mapping from input tensor to logical input name.
  std::map<TensorId, string> inputs_;

  // Mapping from output tensor to logical output names.  There may be more than
  // one name due to layer aliases (e.g., "last_layer").
  std::map<TensorId, std::set<string>> outputs_;

  // Mapping from node name to Operation.
  std::map<string, Operation> operations_;
};

}  // namespace runtime
}  // namespace dragnn
}  // namespace syntaxnet

#endif  // DRAGNN_RUNTIME_MYELIN_MYELIN_CELL_CONVERTER_H_
