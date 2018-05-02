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

#include <string>
#include <vector>

#include "dragnn/core/input_batch_cache.h"
#include "dragnn/io/sentence_input_batch.h"
#include "dragnn/io/syntaxnet_sentence.h"
#include "dragnn/protos/spec.pb.h"
#include "dragnn/runtime/sequence_extractor.h"
#include "dragnn/runtime/term_map_sequence_extractor.h"
#include "dragnn/runtime/term_map_utils.h"
#include "dragnn/runtime/transition_system_traits.h"
#include "syntaxnet/base.h"
#include "syntaxnet/term_frequency_map.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"

namespace syntaxnet {
namespace dragnn {
namespace runtime {
namespace {

// Sequence extractor that extracts words from a SyntaxNetComponent batch.
class SyntaxNetWordSequenceExtractor
    : public TermMapSequenceExtractor<TermFrequencyMap> {
 public:
  SyntaxNetWordSequenceExtractor();

  // Implements SequenceExtractor.
  bool Supports(const FixedFeatureChannel &channel,
                const ComponentSpec &component_spec) const override;
  tensorflow::Status Initialize(const FixedFeatureChannel &channel,
                                const ComponentSpec &component_spec) override;
  tensorflow::Status GetIds(InputBatchCache *input,
                            std::vector<int32> *ids) const override;

 private:
  // Parses |fml| and sets |min_frequency| and |max_num_terms| to the specified
  // values.  If the |fml| does not specify a supported feature, returns non-OK
  // and modifies nothing.
  static tensorflow::Status ParseFml(const string &fml, int *min_frequency,
                                     int *max_num_terms);

  // Feature ID for unknown words.
  int32 unknown_id_ = -1;
};

SyntaxNetWordSequenceExtractor::SyntaxNetWordSequenceExtractor()
    : TermMapSequenceExtractor("word-map") {}

tensorflow::Status SyntaxNetWordSequenceExtractor::ParseFml(
    const string &fml, int *min_frequency, int *max_num_terms) {
  return ParseTermMapFml(fml, {"input", "token", "word"}, min_frequency,
                         max_num_terms);
}

bool SyntaxNetWordSequenceExtractor::Supports(
    const FixedFeatureChannel &channel,
    const ComponentSpec &component_spec) const {
  TransitionSystemTraits traits(component_spec);
  int unused_min_frequency = 0;
  int unused_max_num_terms = 0;
  const tensorflow::Status parse_fml_status =
      ParseFml(channel.fml(), &unused_min_frequency, &unused_max_num_terms);

  return TermMapSequenceExtractor::SupportsTermMap(channel, component_spec) &&
         parse_fml_status.ok() &&
         component_spec.backend().registered_name() == "SyntaxNetComponent" &&
         traits.is_sequential && traits.is_token_scale;
}

tensorflow::Status SyntaxNetWordSequenceExtractor::Initialize(
    const FixedFeatureChannel &channel, const ComponentSpec &component_spec) {
  int min_frequency = 0;
  int max_num_terms = 0;
  TF_RETURN_IF_ERROR(ParseFml(channel.fml(), &min_frequency, &max_num_terms));
  TF_RETURN_IF_ERROR(TermMapSequenceExtractor::InitializeTermMap(
      channel, component_spec, min_frequency, max_num_terms));

  unknown_id_ = term_map().Size();
  const int outside_id = unknown_id_ + 1;

  const int map_vocab_size = outside_id + 1;
  const int spec_vocab_size = channel.vocabulary_size();
  if (map_vocab_size != spec_vocab_size) {
    return tensorflow::errors::InvalidArgument(
        "Word vocabulary size mismatch between term map (", map_vocab_size,
        ") and ComponentSpec (", spec_vocab_size, ")");
  }

  return tensorflow::Status::OK();
}

tensorflow::Status SyntaxNetWordSequenceExtractor::GetIds(
    InputBatchCache *input, std::vector<int32> *ids) const {
  ids->clear();

  const std::vector<SyntaxNetSentence> &data =
      *input->GetAs<SentenceInputBatch>()->data();
  if (data.size() != 1) {
    return tensorflow::errors::InvalidArgument("Non-singleton batch: got ",
                                               data.size(), " elements");
  }

  const Sentence &sentence = *data[0].sentence();
  for (const Token &token : sentence.token()) {
    ids->push_back(term_map().LookupIndex(token.word(), unknown_id_));
  }

  return tensorflow::Status::OK();
}

DRAGNN_RUNTIME_REGISTER_SEQUENCE_EXTRACTOR(SyntaxNetWordSequenceExtractor);

}  // namespace
}  // namespace runtime
}  // namespace dragnn
}  // namespace syntaxnet
