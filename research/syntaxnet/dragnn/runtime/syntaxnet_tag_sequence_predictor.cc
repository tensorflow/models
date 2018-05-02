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

#include <algorithm>
#include <string>
#include <vector>

#include "dragnn/core/input_batch_cache.h"
#include "dragnn/io/sentence_input_batch.h"
#include "dragnn/io/syntaxnet_sentence.h"
#include "dragnn/protos/spec.pb.h"
#include "dragnn/runtime/math/types.h"
#include "dragnn/runtime/sequence_predictor.h"
#include "dragnn/runtime/term_map_sequence_predictor.h"
#include "dragnn/runtime/transition_system_traits.h"
#include "syntaxnet/base.h"
#include "syntaxnet/sentence.pb.h"
#include "syntaxnet/term_frequency_map.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"

namespace syntaxnet {
namespace dragnn {
namespace runtime {
namespace {

// Predicts sequences of POS tags in SyntaxNetComponent batches.
class SyntaxNetTagSequencePredictor : public TermMapSequencePredictor {
 public:
  SyntaxNetTagSequencePredictor();

  // Implements SequencePredictor.
  bool Supports(const ComponentSpec &component_spec) const override;
  tensorflow::Status Initialize(const ComponentSpec &component_spec) override;
  tensorflow::Status Predict(Matrix<float> logits,
                             InputBatchCache *input) const override;

 private:
  // Whether to process sequences from left to right.
  bool left_to_right_ = true;
};

SyntaxNetTagSequencePredictor::SyntaxNetTagSequencePredictor()
    : TermMapSequencePredictor("tag-map") {}

bool SyntaxNetTagSequencePredictor::Supports(
    const ComponentSpec &component_spec) const {
  return TermMapSequencePredictor::SupportsTermMap(component_spec) &&
         component_spec.backend().registered_name() == "SyntaxNetComponent" &&
         component_spec.transition_system().registered_name() == "tagger";
}

tensorflow::Status SyntaxNetTagSequencePredictor::Initialize(
    const ComponentSpec &component_spec) {
  // Load all tags.
  constexpr int kMinFrequency = 0;
  constexpr int kMaxNumTerms = 0;
  TF_RETURN_IF_ERROR(TermMapSequencePredictor::InitializeTermMap(
      component_spec, kMinFrequency, kMaxNumTerms));

  if (term_map().Size() == 0) {
    return tensorflow::errors::InvalidArgument("Empty tag map");
  }

  const int map_num_tags = term_map().Size();
  const int spec_num_tags = component_spec.num_actions();
  if (map_num_tags != spec_num_tags) {
    return tensorflow::errors::InvalidArgument(
        "Tag count mismatch between term map (", map_num_tags,
        ") and ComponentSpec (", spec_num_tags, ")");
  }

  left_to_right_ = TransitionSystemTraits(component_spec).is_left_to_right;
  return tensorflow::Status::OK();
}

tensorflow::Status SyntaxNetTagSequencePredictor::Predict(
    Matrix<float> logits, InputBatchCache *input) const {
  if (logits.num_columns() != term_map().Size()) {
    return tensorflow::errors::InvalidArgument(
        "Logits shape mismatch: expected ", term_map().Size(),
        " columns but got ", logits.num_columns());
  }

  const std::vector<SyntaxNetSentence> &data =
      *input->GetAs<SentenceInputBatch>()->data();
  if (data.size() != 1) {
    return tensorflow::errors::InvalidArgument("Non-singleton batch: got ",
                                               data.size(), " elements");
  }

  Sentence *sentence = data[0].sentence();
  const int num_tokens = sentence->token_size();
  if (logits.num_rows() != num_tokens) {
    return tensorflow::errors::InvalidArgument(
        "Logits shape mismatch: expected ", num_tokens, " rows but got ",
        logits.num_rows());
  }

  int token_index = left_to_right_ ? 0 : num_tokens - 1;
  const int token_increment = left_to_right_ ? 1 : -1;
  for (int i = 0; i < num_tokens; ++i, token_index += token_increment) {
    const Vector<float> row = logits.row(i);
    Token *token = sentence->mutable_token(token_index);
    const float *const begin = row.begin();
    const float *const end = row.end();
    token->set_tag(term_map().GetTerm(std::max_element(begin, end) - begin));
  }

  return tensorflow::Status::OK();
}

DRAGNN_RUNTIME_REGISTER_SEQUENCE_PREDICTOR(SyntaxNetTagSequencePredictor);

}  // namespace
}  // namespace runtime
}  // namespace dragnn
}  // namespace syntaxnet
