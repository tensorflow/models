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

#ifndef DRAGNN_RUNTIME_XLA_XLA_CELL_CONVERTER_H_
#define DRAGNN_RUNTIME_XLA_XLA_CELL_CONVERTER_H_

#include <map>
#include <set>
#include <string>
#include <utility>

#include "dragnn/protos/export.pb.h"
#include "dragnn/runtime/trained_model.h"
#include "syntaxnet/base.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/types.h"

namespace syntaxnet {
namespace dragnn {
namespace runtime {

// Converter that extracts the cell computation from a DRAGNN component and
// writes it as a frozen TF GraphDef.

//
// The trained model that contains the DRAGNN component must also contain a
// CellSubgraphSpec proto embedded into the TF graph as a specifically-named
// constant node (see runtime_support.py).  The CellSubgraphSpec defines the
// boundaries of the cell comptation.
//
// Each frozen GraphDef contains a single function that runs the cell and
// is named after the component.  The function inputs are reference
// variables, so they can be pointed at externally-managed pieces of memory,
// provided sufficient size and alignment. Output storage is managed by XLA.
// The function inputs and outputs are marked with special names, namely:
//   INPUT__<CellSubgraphSpec.Input.name>
//   OUTPUT__<CellSubgraphSpec.Output.name>
class XlaCellConverter {
 public:
  // Extracts the cell of the DRAGNN component named |component_name| from the
  // |trained_model| and overwrites the |graph| with an equivalent
  // TF GraphDef in |graph| which is frozen (it encapsulates Variables). The
  // CellSubgraphSpec stored in the graph is copied into |spec|. On error,
  // returns non-OK.
  static tensorflow::Status Convert(const string &component_name,
                                    const TrainedModel &trained_model,
                                    tensorflow::GraphDef *graph,
                                    CellSubgraphSpec *spec);

 private:
  // A (node_name, output_index) pair denoting a tensor.
  using TensorId = std::pair<string, uint32>;

  // A TF operation that makes up the cell.
  struct Operation {
    // The TF graph node represented by this operation.
    const tensorflow::NodeDef *node = nullptr;
  };

  // Creates an empty converter.
  XlaCellConverter() = default;

  // Populates |dest_node| with the contents of |src_node|. For most nodes
  // this is a complete copy. The exception is for nodes converted to Identity
  // ops (e.g. Enter nodes). In this case, the op is changed to "Identity" and
  // only critical attributes (for tensor type and shape) are retained.
  tensorflow::Status FillNode(const tensorflow::NodeDef &src_node,
                              tensorflow::NodeDef *dest_node) const;

  // Populates |dest_node| with the frozen contents of |src_node| which
  // evaluates to a CellSubgraphSpec. The serialized contents will be
  // stored in the value.tensor.string_val which makes extraction and
  // development cleaner.
  tensorflow::Status FreezeSpecNode(const tensorflow::NodeDef &src_node,
                                    tensorflow::NodeDef *dest_node) const;

  // Populates |dest_node| with the frozen contents of |src_node|. The
  // output tensor for |src_node| will be evaluated and included as a
  // constant in |dest_node|. On error, returns non-OK.
  tensorflow::Status FreezeNode(const tensorflow::NodeDef &src_node,
                                tensorflow::NodeDef *dest_node) const;

  // Copies over node attributes from |src_node| to |dest_node|, stripping out
  // those which don't apply generally when |restrict_attributes| is true.
  static void FillNodeAttributes(bool restrict_attributes,
                                 const tensorflow::NodeDef &src_node,
                                 tensorflow::NodeDef *dest_node);

  // Returns true if a node called |node_name| is in the subgraph required
  // for evaluating the cell.
  bool IsNodeInSubgraph(const string &node_name) const;

  // Implements the static Convert() method.
  tensorflow::Status ConvertImpl(const string &component_name,
                                 const TrainedModel &trained_model,
                                 tensorflow::GraphDef *graph,
                                 CellSubgraphSpec *spec);

  // Populates the |inputs_| and |outputs_| based on the |spec|.  On error,
  // returns non-OK.
  tensorflow::Status BuildInputsAndOutputs(const CellSubgraphSpec &spec);

  // Walks from the |outputs_| to the |inputs_| in the |trained_model_|, adding
  // to |operations_| along the way.  Requires that BuildInputsAndOutputs() was
  // called.  On error, returns non-OK.
  tensorflow::Status BuildOperations();

  // Parses a |tensor_name| into a |tensor_id|.  E.g.,
  //   "foo/bar:1" => ("foo/bar", 1)
  //   "baz"       => ("baz", 0)
  // On error, returns non-OK.  It is an error if the |tensor_name| denotes a
  // control dependency.
  static tensorflow::Status ParseTensorId(const string &tensor_name,
                                          TensorId *tensor_id);

  // Returns the canonically-formatted name of the graph variable associated
  // with the |tensor_id|.
  static string AsVariableName(const TensorId &tensor_id);

  // Name of the component being converted.
  string component_name_;

  // Trained model that contains the DRAGNN model.
  const TrainedModel *trained_model_ = nullptr;

  // Tensor ids that serve as inputs and outputs.
  std::set<TensorId> inputs_;
  std::set<TensorId> outputs_;

  // Mapping from node name to Operation.
  std::map<string, Operation> operations_;
};

}  // namespace runtime
}  // namespace dragnn
}  // namespace syntaxnet

#endif  // DRAGNN_RUNTIME_XLA_XLA_CELL_CONVERTER_H_
