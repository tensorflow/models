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

#include <stddef.h>
#include <string>
#include <vector>

#include "dragnn/core/input_batch_cache.h"
#include "dragnn/io/sentence_input_batch.h"
#include "dragnn/io/syntaxnet_sentence.h"
#include "dragnn/protos/spec.pb.h"
#include "dragnn/runtime/sequence_linker.h"
#include "dragnn/runtime/transition_system_traits.h"
#include "syntaxnet/base.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/logging.h"
#include "util/utf8/unilib_utf8_utils.h"

namespace syntaxnet {
namespace dragnn {
namespace runtime {
namespace {

// Focus character to link to in each token.
enum class Focus {
  kFirst,  // first character in token
  kLast,   // last character in token
};

// Translator to apply to the linked character index.
enum class Translator {
  kIdentity,  // direct identity link
  kReversed,  // reverse-order link
};

// Returns the LinkedFeatureChannel.fml for the |focus|.
string ChannelFml(Focus focus) {
  switch (focus) {
    case Focus::kFirst:
      return "input.first-char-focus";
    case Focus::kLast:
      return "input.last-char-focus";
  }
}

// Returns the LinkedFeatureChannel.source_translator for the |translator|.
string ChannelTranslator(Translator translator) {
  switch (translator) {
    case Translator::kIdentity:
      return "identity";
    case Translator::kReversed:
      return "reverse-char";
  }
}

// Returns the |focus| byte index for the |token|.  The returned index must be
// within the span of the |token|.
int32 GetFocusByte(Focus focus, const Token &token) {
  switch (focus) {
    case Focus::kFirst:
      return token.start();
    case Focus::kLast:
      return token.end();
  }
}

// Applies the |translator| to the character |index| w.r.t. the |last_index| and
// returns the result.
int32 Translate(Translator translator, int32 last_index, int32 index) {
  switch (translator) {
    case Translator::kIdentity:
      return index;
    case Translator::kReversed:
      return last_index - index;
  }
}

// Translates links from tokens in the target layer to UTF-8 characters in the
// source layer.  Templated on a |focus| and |translator| (see above).
template <Focus focus, Translator translator>
class SyntaxNetCharacterSequenceLinker : public SequenceLinker {
 public:
  // Implements SequenceLinker.
  bool Supports(const LinkedFeatureChannel &channel,
                const ComponentSpec &component_spec) const override;
  tensorflow::Status Initialize(const LinkedFeatureChannel &channel,
                                const ComponentSpec &component_spec) override;
  tensorflow::Status GetLinks(size_t source_num_steps, InputBatchCache *input,
                              std::vector<int32> *links) const override;
};

template <Focus focus, Translator translator>
bool SyntaxNetCharacterSequenceLinker<focus, translator>::Supports(
    const LinkedFeatureChannel &channel,
    const ComponentSpec &component_spec) const {
  TransitionSystemTraits traits(component_spec);
  return channel.fml() == ChannelFml(focus) &&
         channel.source_translator() == ChannelTranslator(translator) &&
         component_spec.backend().registered_name() == "SyntaxNetComponent" &&
         traits.is_sequential && traits.is_token_scale;
}

template <Focus focus, Translator translator>
tensorflow::Status
SyntaxNetCharacterSequenceLinker<focus, translator>::Initialize(
    const LinkedFeatureChannel &channel, const ComponentSpec &component_spec) {
  return tensorflow::Status::OK();
}

template <Focus focus, Translator translator>
tensorflow::Status
SyntaxNetCharacterSequenceLinker<focus, translator>::GetLinks(
    size_t source_num_steps, InputBatchCache *input,
    std::vector<int32> *links) const {
  const std::vector<SyntaxNetSentence> &batch =
      *input->GetAs<SentenceInputBatch>()->data();
  if (batch.size() != 1) {
    return tensorflow::errors::InvalidArgument("Non-singleton batch: got ",
                                               batch.size(), " elements");
  }

  const Sentence &sentence = *batch[0].sentence();
  const int32 num_tokens = sentence.token_size();
  links->resize(num_tokens);
  if (num_tokens == 0) return tensorflow::Status::OK();

  // Given the properties selected in Supports(), the number of source steps
  // must match the number of UTF-8 characters.  The last character index will
  // be used in Translate().
  const int32 last_char_index = static_cast<int32>(source_num_steps) - 1;

  // [start,end) byte range of the text spanned by the sentence tokens.
  const int32 start_byte = sentence.token(0).start();
  const int32 end_byte = sentence.token(num_tokens - 1).end() + 1;
  const char *const data = sentence.text().data();

  if (UniLib::IsTrailByte(data[start_byte])) {
    return tensorflow::errors::InvalidArgument(
        "First token starts in the middle of a UTF-8 character: ",
        sentence.token(0).ShortDebugString());
  }

  // Current character index and its past-the-end byte in the sentence.
  int32 char_index = 0;
  int32 char_end_byte = start_byte + UniLib::OneCharLen(data + start_byte);

  // Current token index and its byte index.
  int32 token_index = 0;
  int32 token_byte = GetFocusByte(focus, sentence.token(0));

  // Scan through the characters and tokens.  For each token, we assign it the
  // character whose byte range contains its focus byte.
  while (true) {
    // If the character ends after the token, then the token must lie within the
    // character, or we would have consumed the token in a previous iteration.
    if (char_end_byte > token_byte) {
      (*links)[token_index] =
          Translate(translator, last_char_index, char_index);
      if (++token_index >= num_tokens) break;
      token_byte = GetFocusByte(focus, sentence.token(token_index));
    } else if (char_end_byte < end_byte) {
      ++char_index;
      char_end_byte += UniLib::OneCharLen(data + char_end_byte);
    } else {
      break;
    }
  }

  if (char_end_byte > end_byte) {
    return tensorflow::errors::InvalidArgument(
        "Last token ends in the middle of a UTF-8 character: ",
        sentence.token(num_tokens - 1).ShortDebugString());
  }

  // Since GetFocusByte() always returns a byte index within the span of the
  // token, the loop above must consume all tokens.
  DCHECK_EQ(token_index, num_tokens);

  return tensorflow::Status::OK();
}

using SyntaxNetFirstCharacterIdentitySequenceLinker =
    SyntaxNetCharacterSequenceLinker<Focus::kFirst, Translator::kIdentity>;
using SyntaxNetFirstCharacterReversedSequenceLinker =
    SyntaxNetCharacterSequenceLinker<Focus::kFirst, Translator::kReversed>;
using SyntaxNetLastCharacterIdentitySequenceLinker =
    SyntaxNetCharacterSequenceLinker<Focus::kLast, Translator::kIdentity>;
using SyntaxNetLastCharacterReversedSequenceLinker =
    SyntaxNetCharacterSequenceLinker<Focus::kLast, Translator::kReversed>;

DRAGNN_RUNTIME_REGISTER_SEQUENCE_LINKER(
    SyntaxNetFirstCharacterIdentitySequenceLinker);
DRAGNN_RUNTIME_REGISTER_SEQUENCE_LINKER(
    SyntaxNetFirstCharacterReversedSequenceLinker);
DRAGNN_RUNTIME_REGISTER_SEQUENCE_LINKER(
    SyntaxNetLastCharacterIdentitySequenceLinker);
DRAGNN_RUNTIME_REGISTER_SEQUENCE_LINKER(
    SyntaxNetLastCharacterReversedSequenceLinker);

}  // namespace
}  // namespace runtime
}  // namespace dragnn
}  // namespace syntaxnet
