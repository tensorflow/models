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

// Writes a file containing a text tf2xla::Config proto that is extracted
// from a frozen binary GraphDef file for a DRAGNN component.
//
// Usage: xla_extract_config input-graph-def output-config
//   input-graph-def: input frozen tensorflow.GraphDef binary proto
//   output-config:   extracted tensorflow.tf2xla.Config text proto

#include <string.h>

#include "dragnn/protos/export.pb.h"
#include "dragnn/runtime/xla/xla_graph_utils.h"
#include "tensorflow/compiler/tf2xla/tf2xla.pb.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/platform/logging.h"

namespace syntaxnet {
namespace dragnn {
namespace runtime {
namespace {

// Writes the Config extracted from |input_graph_def| to |output_config|.
// On error, returns non-OK.
tensorflow::Status XlaExtractConfig(const char *input_graph_def,
                                    const char *output_config) {
  tensorflow::GraphDef graph;
  TF_RETURN_IF_ERROR(LoadFrozenGraphDef(input_graph_def, &graph));

  CellSubgraphSpec cell_subgraph_spec;
  tensorflow::tf2xla::Config xla_config;
  TF_RETURN_IF_ERROR(
      GetSpecAndMakeXlaConfig(graph, &cell_subgraph_spec, &xla_config));

  return WriteTextProto(tensorflow::Env::Default(), output_config, xla_config);
}

}  // namespace
}  // namespace runtime
}  // namespace dragnn
}  // namespace syntaxnet

int main(int argc, char **argv) {
  tensorflow::port::InitMain(argv[0], &argc, &argv);
  if (argc != 3 || strlen(argv[1]) == 0 || strlen(argv[2]) == 0) {
    LOG(FATAL)
        << "Usage: xla_extract_config input-graph-def output-config\n"
           "  input-graph-def: input frozen tensorflow.GraphDef binary proto\n"
           "  output-config: extracted tensorflow.tf2xla.Config text proto\n";
  }
  TF_CHECK_OK(syntaxnet::dragnn::runtime::XlaExtractConfig(argv[1], argv[2]));
  return 0;
}
