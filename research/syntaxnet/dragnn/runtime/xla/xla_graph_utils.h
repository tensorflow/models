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

// Utils for working with specifications of XLA-based DRAGNN runtime models.

#ifndef DRAGNN_RUNTIME_XLA_XLA_GRAPH_UTILS_H_
#define DRAGNN_RUNTIME_XLA_XLA_GRAPH_UTILS_H_

#include <string>

#include "dragnn/protos/export.pb.h"
#include "syntaxnet/base.h"
#include "tensorflow/compiler/tf2xla/tf2xla.pb.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/lib/core/status.h"

namespace syntaxnet {
namespace dragnn {
namespace runtime {

// The name of the node in the frozen GraphSpec (for a particular component)
// that contains the serialized CellSubgraphSpec.
extern const char *const kFrozenCellSubgraphSpecNodeName;

// Loads a GraphDef file from the |frozen_graph_def_path| into the |graph_def|.
// Assumes binary proto unless |frozen_graph_def_path| ends with ".pbtxt", in
// which case it assumes text proto format. On error, returns non-OK.
tensorflow::Status LoadFrozenGraphDef(const string &frozen_graph_def_path,
                                      tensorflow::GraphDef *graph_def);

// Saves a GraphDef |graph_def| in the file |frozen_graph_def_path|. Uses
// deterministic serialization to avoid churn due to attr map order.
// Always writes in binary format. On error, returns non-OK.
tensorflow::Status SaveFrozenGraphDef(const string &frozen_graph_def_path,
                                      const tensorflow::GraphDef &graph_def);

// Fills in |name| and |index| given the |tensor_name| of the form
// "name" or "name:index". On error, changes nothing and returns non-OK.
tensorflow::Status ParseTensorName(const string &tensor_name, string *name,
                                   uint32 *index);

// Given a frozen |graph_def|, extracts the |cell_subgraph_spec| stored within
// it, and generates the |xla_config| proto. Whenever an output tensor is
// aliased, the output in |xla_config| is taken the first occurrence of the
// tensor in |cell_subgraph_spec| (aliases are resolved in the XLA component
// in InitializeOutputLayers). On error, returns non-OK.
tensorflow::Status GetSpecAndMakeXlaConfig(
    const tensorflow::GraphDef &graph_def, CellSubgraphSpec *cell_subgraph_spec,
    tensorflow::tf2xla::Config *xla_config);

}  // namespace runtime
}  // namespace dragnn
}  // namespace syntaxnet

#endif  // DRAGNN_RUNTIME_XLA_XLA_GRAPH_UTILS_H_
