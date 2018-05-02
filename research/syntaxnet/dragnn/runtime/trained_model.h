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

#ifndef DRAGNN_RUNTIME_TRAINED_MODEL_H_
#define DRAGNN_RUNTIME_TRAINED_MODEL_H_

#include <map>
#include <string>

#include "syntaxnet/base.h"
#include "tensorflow/cc/saved_model/loader.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/lib/core/status.h"

namespace syntaxnet {
namespace dragnn {
namespace runtime {

// A trained DRAGNN model, which can be queried for nodes and tensors.
class TrainedModel {
 public:
  // Creates an uninitialized model; call Reset() before use.
  TrainedModel() = default;

  // Loads the TF SavedModel at the |saved_model_dir|, replacing the current
  // model, if any.  On error, returns non-OK and modifies nothing.
  tensorflow::Status Reset(const string &saved_model_dir);

  // Evaluates the tensor with the |name| in the |session_| and sets |tensor| to
  // the result.  On error, returns non-OK and modifies nothing.
  //
  // NB: Tensors that are embedded inside a tf.while_loop() cannot be evaluated.
  // Such evaluations fail with errors like "Retval[0] does not have value".
  tensorflow::Status EvaluateTensor(const string &name,
                                    tensorflow::Tensor *tensor) const;

  // Finds the node with the |name| in the |graph_| and points the |node| at it.
  // On error, returns non-OK and modifies nothing.
  tensorflow::Status LookupNode(const string &name,
                                const tensorflow::NodeDef **node) const;

  // Points |graph| at the GraphDef for the current model. It is an error if
  // there is no current model.
  tensorflow::Status GraphDef(const tensorflow::GraphDef **graph) const;

  // Discards the current model.  It is an error if there is no current model.
  // On error, returns non-OK but still discards the model.
  tensorflow::Status Close();

 private:
  // TF SavedModel that contains the trained DRAGNN model.
  tensorflow::SavedModelBundle saved_model_;

  // Nodes in the TF graph, indexed by name.
  std::map<string, const tensorflow::NodeDef *> nodes_;
};

}  // namespace runtime
}  // namespace dragnn
}  // namespace syntaxnet

#endif  // DRAGNN_RUNTIME_TRAINED_MODEL_H_
