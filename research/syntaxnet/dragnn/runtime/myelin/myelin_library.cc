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

#include "dragnn/runtime/myelin/myelin_library.h"

#include <string>

#include "syntaxnet/base.h"
#include "tensorflow/core/lib/strings/strcat.h"

namespace syntaxnet {
namespace dragnn {
namespace runtime {

bool PreMultipliedEmbeddings::Transform(sling::myelin::Flow *flow) {
  bool transformed_something = false;
  for (sling::myelin::Flow::Operation *matmul :
       flow->Find({"Gather", "MatMul"})) {
    if (matmul->indegree() != 2) continue;
    sling::myelin::Flow::Variable *gathered = matmul->inputs[0];
    sling::myelin::Flow::Variable *weights = matmul->inputs[1];
    sling::myelin::Flow::Operation *gather = gathered->producer;
    if (gather->indegree() != 2) continue;
    sling::myelin::Flow::Variable *embeddings = gather->inputs[0];
    sling::myelin::Flow::Variable *indices = gather->inputs[1];

    if (gathered->out) continue;
    if (!weights->constant()) continue;
    if (weights->rank() != 2) continue;
    if (!embeddings->constant()) continue;
    if (embeddings->rank() != 2) continue;
    if (embeddings->type != weights->type) continue;

    // Add an operation to pre-multiply the embeddings and weights.
    const string product_name =
        tensorflow::strings::StrCat(embeddings->name, "/", weights->name);
    const string pre_multiply_name =
        tensorflow::strings::StrCat(product_name, "/PreMultiply");
    sling::myelin::Flow::Variable *product = flow->AddVariable(
        product_name, weights->type, {embeddings->dim(0), weights->dim(1)});
    flow->AddOperation(gather->func, pre_multiply_name, "MatMul",
                       {embeddings, weights}, {product});

    // Convert the MatMul into a Gather on the pre-multiplied embeddings.
    matmul->type = "Gather";
    matmul->ReplaceInput(gathered, product);
    matmul->ReplaceInput(weights, indices);

    // Remove the original Gather if it is no longer used.
    if (gathered->consumers.empty()) flow->RemoveOperation(gather);
    transformed_something = true;
  }
  return transformed_something;
}

}  // namespace runtime
}  // namespace dragnn
}  // namespace syntaxnet
