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

// Myelin typers, transformers, and kernels specific to the DRAGNN runtime.

#ifndef DRAGNN_RUNTIME_MYELIN_MYELIN_LIBRARY_H_
#define DRAGNN_RUNTIME_MYELIN_MYELIN_LIBRARY_H_

#include "sling/myelin/flow.h"

namespace syntaxnet {
namespace dragnn {
namespace runtime {

// Rearranges the flow to allow the "pre-multiplied embeddings" optimization.
// Specifically, performs the following transformation:
//
// tf.matmul(tf.gather(embeddings, indices), weights) =
//     tf.gather(tf.matmul(embeddings, weights), indices)
//
// The transformation only applies if the embeddings and weights are constants.
// Myelin has constant folding transformations that will trigger and pre-compute
// the multiplication of the embeddings and weights.
//
// NB: There is already a PrecomputedEmbeddings transformer in Myelin but that
// operates on the Lookup op and expects an intervening Reshape.
class PreMultipliedEmbeddings : public sling::myelin::Transformer {
 public:
  // Implements Transformer.
  bool Transform(sling::myelin::Flow *flow) override;
};

}  // namespace runtime
}  // namespace dragnn
}  // namespace syntaxnet

#endif  // DRAGNN_RUNTIME_MYELIN_MYELIN_LIBRARY_H_
