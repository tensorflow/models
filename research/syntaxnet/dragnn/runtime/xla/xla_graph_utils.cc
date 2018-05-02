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

#include "dragnn/runtime/xla/xla_graph_utils.h"

#include <cstddef>
#include <map>
#include <set>
#include <utility>

#include "dragnn/runtime/xla/xla_spec_utils.h"
#include "syntaxnet/base.h"
#include "tensorflow/compiler/tf2xla/tf2xla.pb.h"
#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/framework/tensor_shape.pb.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/logging.h"

namespace syntaxnet {
namespace dragnn {
namespace runtime {

const char *const kFrozenCellSubgraphSpecNodeName = "CellSubgraphSpec";

namespace {

// Fills the TensorId fields given |tensor_name|.  On error, returns non-OK.
tensorflow::Status FillXlaTensorId(const string &tensor_name,
                                   tensorflow::tf2xla::TensorId *id) {
  string name;
  uint32 index;
  TF_RETURN_IF_ERROR(ParseTensorName(tensor_name, &name, &index));
  id->set_node_name(name);
  id->set_output_index(index);

  return tensorflow::Status::OK();
}

// Loads the |shape| proto from the placeholder |node|. On error, returns
// non-OK.
tensorflow::Status GetPlaceholderShape(
    const tensorflow::NodeDef &node,
    tensorflow::TensorShapeProto *shape_proto) {
  if (node.op() != "Placeholder") {
    return tensorflow::errors::InvalidArgument("Input node '", node.name(),
                                               "' is not a Placeholder");
  }
  return tensorflow::GetNodeAttr(node, "shape", shape_proto);
}

}  // namespace

tensorflow::Status LoadFrozenGraphDef(const string &frozen_graph_def_path,
                                      tensorflow::GraphDef *graph_def) {
  if (tensorflow::str_util::EndsWith(frozen_graph_def_path, ".pbtxt")) {
    return tensorflow::ReadTextProto(tensorflow::Env::Default(),
                                     frozen_graph_def_path, graph_def);
  }
  return tensorflow::ReadBinaryProto(tensorflow::Env::Default(),
                                     frozen_graph_def_path, graph_def);
}

tensorflow::Status SaveFrozenGraphDef(const string &frozen_graph_def_path,
                                      const tensorflow::GraphDef &graph_def) {
  const std::size_t size = graph_def.ByteSizeLong();
  string data(size, '\0');
  if (size > 0) {
    tensorflow::protobuf::io::ArrayOutputStream array_stream(&data[0], size);
    tensorflow::protobuf::io::CodedOutputStream output_stream(&array_stream);

    output_stream.SetSerializationDeterministic(true);
    graph_def.SerializeWithCachedSizes(&output_stream);
    if (output_stream.HadError() || size != output_stream.ByteCount()) {
      return tensorflow::errors::InvalidArgument("Cannot serialize GraphDef");
    }
  }
  return tensorflow::WriteStringToFile(tensorflow::Env::Default(),
                                       frozen_graph_def_path, data);
}

tensorflow::Status ParseTensorName(const string &tensor_name, string *name,
                                   uint32 *index) {
  if (tensor_name[0] == '^') {
    return tensorflow::errors::InvalidArgument(
        "Cannot parse name of control input '", tensor_name, "'");
  }

  const auto colon_index = tensor_name.rfind(':');

  if (colon_index == string::npos) {  // no colon; assume 0
    *index = 0;
  } else {
    const string output_str = tensor_name.substr(colon_index + 1);
    if (!tensorflow::strings::safe_strtou32(output_str, index)) {
      return tensorflow::errors::InvalidArgument("Malformed tensor name ",
                                                 tensor_name);
    }
  }

  // NB: If |colon_index| is string::npos, takes the whole string as desired.
  *name = tensor_name.substr(0, colon_index);

  return tensorflow::Status::OK();
}

tensorflow::Status GetSpecAndMakeXlaConfig(
    const tensorflow::GraphDef &graph_def, CellSubgraphSpec *cell_subgraph_spec,
    tensorflow::tf2xla::Config *xla_config) {
  // Maps the node name to its corresponding node in the GraphDef.
  std::map<string, const tensorflow::NodeDef *> node_name_map;
  for (const tensorflow::NodeDef &node : graph_def.node()) {
    node_name_map[node.name()] = &node;
  }

  // Looks for a node called |name| in |graph_def|. If present, returns OK
  // and fills in |*node|, otherwise returns non-OK.
  auto lookup_node = [&](const string &name, const tensorflow::NodeDef **node) {
    const auto it = node_name_map.find(name);
    if (it == node_name_map.end()) {
      return tensorflow::errors::NotFound("Cannot find node ", name);
    }
    *node = it->second;
    return tensorflow::Status::OK();
  };

  // Retrieves the CellSubgraphSpec from the frozen graph.
  const tensorflow::NodeDef *spec_node = nullptr;
  TF_RETURN_IF_ERROR(lookup_node("CellSubgraphSpec", &spec_node));
  const auto value_it = spec_node->attr().find("value");
  if (value_it == spec_node->attr().end()) {
    return tensorflow::errors::NotFound("Cannot find CellSubgraphSpec value");
  }
  if (!cell_subgraph_spec->ParseFromString(
          value_it->second.tensor().string_val(0))) {
    return tensorflow::errors::InvalidArgument(
        "Failed to parse CellSubgraphSpec");
  }

  VLOG(1) << "CellSubgraphSpec: " << cell_subgraph_spec->DebugString();

  // Builds the Config feeds.
  for (const auto &input : cell_subgraph_spec->input()) {
    auto *feed = xla_config->add_feed();
    feed->set_name(MakeXlaInputLayerName(input.name()));
    TF_RETURN_IF_ERROR(FillXlaTensorId(input.tensor(), feed->mutable_id()));

    const tensorflow::NodeDef *input_node;
    TF_RETURN_IF_ERROR(lookup_node(feed->id().node_name(), &input_node));
    TF_RETURN_IF_ERROR(GetPlaceholderShape(*input_node, feed->mutable_shape()));
  }

  // Builds the Config fetches and alias map.
  std::set<string> output_tensors;
  for (const auto &output : cell_subgraph_spec->output()) {
    if (output_tensors.insert(output.tensor()).second) {
      // The first time a tensor is encountered, this adds a fetch along with
      // its name. The remaining names associated with the same tensor (aliases)
      // are handled by InitializeOutputLayers.
      auto *fetch = xla_config->add_fetch();
      fetch->set_name(MakeXlaOutputLayerName(output.name()));
      TF_RETURN_IF_ERROR(FillXlaTensorId(output.tensor(), fetch->mutable_id()));
    }
  }

  VLOG(1) << "Config: " << xla_config->DebugString();

  return tensorflow::Status::OK();
}

}  // namespace runtime
}  // namespace dragnn
}  // namespace syntaxnet
