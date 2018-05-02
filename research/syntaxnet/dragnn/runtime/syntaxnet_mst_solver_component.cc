// Copyright 2018 Google Inc. All Rights Reserved.
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

#include <string>
#include <vector>

#include "dragnn/core/compute_session.h"
#include "dragnn/core/input_batch_cache.h"
#include "dragnn/io/sentence_input_batch.h"
#include "dragnn/io/syntaxnet_sentence.h"
#include "dragnn/protos/trace.pb.h"
#include "dragnn/runtime/mst_solver_component_base.h"
#include "dragnn/runtime/session_state.h"
#include "syntaxnet/base.h"
#include "syntaxnet/sentence.pb.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/gtl/array_slice.h"

namespace syntaxnet {
namespace dragnn {
namespace runtime {
namespace {

// Selects heads for SyntaxNetComponent batches.
class SyntaxNetMstSolverComponent : public MstSolverComponentBase {
 public:
  SyntaxNetMstSolverComponent()
      : MstSolverComponentBase("SyntaxNetMstSolverComponent",
                               "SyntaxNetComponent") {}

  // Implements Component.
  tensorflow::Status Evaluate(SessionState *session_state,
                              ComputeSession *compute_session,
                              ComponentTrace *component_trace) const override;
};

tensorflow::Status SyntaxNetMstSolverComponent::Evaluate(
    SessionState *session_state, ComputeSession *compute_session,
    ComponentTrace *component_trace) const {
  InputBatchCache *input = compute_session->GetInputBatchCache();
  if (input == nullptr) {
    return tensorflow::errors::InvalidArgument("Null input batch");
  }

  const std::vector<SyntaxNetSentence> &data =
      *input->GetAs<SentenceInputBatch>()->data();
  if (data.size() != 1) {
    return tensorflow::errors::InvalidArgument("Non-singleton batch: got ",
                                               data.size(), " elements");
  }

  tensorflow::gtl::ArraySlice<Index> heads;
  TF_RETURN_IF_ERROR(ComputeHeads(session_state, &heads));
  Sentence *sentence = data[0].sentence();
  if (heads.size() != sentence->token_size()) {
    return tensorflow::errors::InvalidArgument(
        "Sentence size mismatch: expected ", heads.size(), " tokens but got ",
        sentence->token_size());
  }

  const int num_tokens = heads.size();
  for (int modifier = 0; modifier < num_tokens; ++modifier) {
    Token *token = sentence->mutable_token(modifier);
    const int head = heads[modifier];
    if (head == modifier) {
      token->clear_head();
    } else {
      token->set_head(head);
    }
  }

  return tensorflow::Status::OK();
}

DRAGNN_RUNTIME_REGISTER_COMPONENT(SyntaxNetMstSolverComponent);

}  // namespace
}  // namespace runtime
}  // namespace dragnn
}  // namespace syntaxnet
