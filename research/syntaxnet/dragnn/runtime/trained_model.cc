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

#include "dragnn/runtime/trained_model.h"

#include <unordered_set>

#include "tensorflow/cc/saved_model/tag_constants.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/strings/strcat.h"

namespace syntaxnet {
namespace dragnn {
namespace runtime {

tensorflow::Status TrainedModel::Reset(const string &saved_model_dir) {
  const std::unordered_set<string> tags = {tensorflow::kSavedModelTagServe};
  tensorflow::SavedModelBundle saved_model;
  TF_RETURN_IF_ERROR(
      tensorflow::LoadSavedModel({}, {}, saved_model_dir, tags, &saved_model));

  // Success; make modifications.
  saved_model_.session = std::move(saved_model.session);
  saved_model_.meta_graph_def = std::move(saved_model.meta_graph_def);
  nodes_.clear();
  const tensorflow::GraphDef &graph = saved_model_.meta_graph_def.graph_def();
  for (const tensorflow::NodeDef &node : graph.node()) {
    nodes_[node.name()] = &node;
  }
  return tensorflow::Status::OK();
}

tensorflow::Status TrainedModel::EvaluateTensor(
    const string &name, tensorflow::Tensor *tensor) const {
  if (saved_model_.session == nullptr) {
    return tensorflow::errors::FailedPrecondition("TF Session is not active");
  }

  // For some reason, runtime hook nodes cannot be evaluated without feeding an
  // input batch.  An empty batch currently works, but if DRAGNN starts failing
  // on empty batches, a reasonable alternative is a batch of empty strings.
  const string input_name = "annotation/ComputeSession/InputBatch";
  const tensorflow::Tensor empty_batch(tensorflow::DT_STRING,
                                       tensorflow::TensorShape({0}));

  // Evaluate the variable in the session.
  std::vector<tensorflow::Tensor> outputs;
  tensorflow::Status status = saved_model_.session->Run(
      {{input_name, empty_batch}}, {name}, {}, &outputs);
  if (!status.ok()) {
    // Attach some extra information to the session error.
    return tensorflow::Status(
        status.code(),
        tensorflow::strings::StrCat("Failed to evaluate tensor '", name,
                                    "': ", status.error_message()));
  }

  if (outputs.size() != 1) {
    return tensorflow::errors::Unknown("Expected exactly one output, but got ",
                                       outputs.size(), " outputs");
  }

  *tensor = outputs[0];
  return tensorflow::Status::OK();
}

tensorflow::Status TrainedModel::LookupNode(
    const string &name, const tensorflow::NodeDef **node) const {
  if (saved_model_.session == nullptr) {
    return tensorflow::errors::FailedPrecondition("TF Session is not active");
  }

  const auto it = nodes_.find(name);
  if (it == nodes_.end()) {
    return tensorflow::errors::NotFound("Unknown node: '", name, "'");
  }
  *node = it->second;
  return tensorflow::Status::OK();
}

tensorflow::Status TrainedModel::GraphDef(
    const tensorflow::GraphDef **graph) const {
  if (saved_model_.session == nullptr) {
    return tensorflow::errors::FailedPrecondition("TF Session is not active");
  }
  *graph = &saved_model_.meta_graph_def.graph_def();
  return tensorflow::Status::OK();
}

tensorflow::Status TrainedModel::Close() {
  if (saved_model_.session == nullptr) {
    return tensorflow::errors::FailedPrecondition("TF Session is not active");
  }

  tensorflow::Status status = saved_model_.session->Close();
  saved_model_.session.reset();
  saved_model_.meta_graph_def.Clear();
  nodes_.clear();
  return status;
}

}  // namespace runtime
}  // namespace dragnn
}  // namespace syntaxnet
