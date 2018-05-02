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
#include "dragnn/runtime/unicode_dictionary.h"
#include "syntaxnet/base.h"
#include "syntaxnet/segmenter_utils.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "util/utf8/unicodetext.h"

namespace syntaxnet {
namespace dragnn {
namespace runtime {
namespace {

// Sequence extractor that extracts characters from a SyntaxNetComponent batch.
class SyntaxNetCharacterSequenceExtractor
    : public TermMapSequenceExtractor<UnicodeDictionary> {
 public:
  SyntaxNetCharacterSequenceExtractor();

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

  // Feature IDs for break characters and unknown characters.
  int32 break_id_ = -1;
  int32 unknown_id_ = -1;
};

SyntaxNetCharacterSequenceExtractor::SyntaxNetCharacterSequenceExtractor()
    : TermMapSequenceExtractor("char-map") {}

tensorflow::Status SyntaxNetCharacterSequenceExtractor::ParseFml(
    const string &fml, int *min_frequency, int *max_num_terms) {
  return ParseTermMapFml(fml, {"char-input", "text-char"}, min_frequency,
                         max_num_terms);
}

bool SyntaxNetCharacterSequenceExtractor::Supports(
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
         traits.is_sequential && traits.is_character_scale;
}

tensorflow::Status SyntaxNetCharacterSequenceExtractor::Initialize(
    const FixedFeatureChannel &channel, const ComponentSpec &component_spec) {
  int min_frequency = 0;
  int max_num_terms = 0;
  TF_RETURN_IF_ERROR(ParseFml(channel.fml(), &min_frequency, &max_num_terms));
  TF_RETURN_IF_ERROR(TermMapSequenceExtractor::InitializeTermMap(
      channel, component_spec, min_frequency, max_num_terms));

  const int num_known = term_map().size();
  break_id_ = num_known;
  unknown_id_ = break_id_ + 1;

  const int map_vocab_size = unknown_id_ + 1;
  const int spec_vocab_size = channel.vocabulary_size();
  if (map_vocab_size != spec_vocab_size) {
    return tensorflow::errors::InvalidArgument(
        "Character vocabulary size mismatch between term map (", map_vocab_size,
        ") and ComponentSpec (", spec_vocab_size, ")");
  }

  return tensorflow::Status::OK();
}

tensorflow::Status SyntaxNetCharacterSequenceExtractor::GetIds(
    InputBatchCache *input, std::vector<int32> *ids) const {
  ids->clear();

  const std::vector<SyntaxNetSentence> &data =
      *input->GetAs<SentenceInputBatch>()->data();
  if (data.size() != 1) {
    return tensorflow::errors::InvalidArgument("Non-singleton batch: got ",
                                               data.size(), " elements");
  }

  const Sentence &sentence = *data[0].sentence();
  if (sentence.token_size() == 0) return tensorflow::Status::OK();

  const string &text = sentence.text();
  const int start_byte = sentence.token(0).start();
  const int end_byte = sentence.token(sentence.token_size() - 1).end();
  const int num_bytes = end_byte - start_byte + 1;

  string character;
  UnicodeText unicode_text;
  unicode_text.PointToUTF8(text.data() + start_byte, num_bytes);
  const auto end = unicode_text.end();
  for (auto it = unicode_text.begin(); it != end; ++it) {
    character.assign(it.utf8_data(), it.utf8_length());
    if (SegmenterUtils::IsBreakChar(character)) {
      ids->push_back(break_id_);
    } else {
      ids->push_back(
          term_map().Lookup(character.data(), character.size(), unknown_id_));
    }
  }

  return tensorflow::Status::OK();
}

DRAGNN_RUNTIME_REGISTER_SEQUENCE_EXTRACTOR(SyntaxNetCharacterSequenceExtractor);

}  // namespace
}  // namespace runtime
}  // namespace dragnn
}  // namespace syntaxnet
