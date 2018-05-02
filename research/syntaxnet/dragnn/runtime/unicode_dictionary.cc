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

#include "dragnn/runtime/unicode_dictionary.h"

#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/strings/strcat.h"

namespace syntaxnet {
namespace dragnn {
namespace runtime {
namespace {

// Returns a string representation of the byte sequence of the |character|.
string CharacterDebugString(const string &character) {
  const auto *bytes = reinterpret_cast<const uint8 *>(character.data());
  string debug = "[";
  for (int i = 0; i < character.size(); ++i) {
    tensorflow::strings::StrAppend(&debug, i == 0 ? "" : " ", bytes[i]);
  }
  tensorflow::strings::StrAppend(&debug, "]");
  return debug;
}

}  // namespace

UnicodeDictionary::UnicodeDictionary() { Clear(); }

UnicodeDictionary::UnicodeDictionary(const string &character_map_path,
                                     int min_frequency, int max_num_terms) {
  TF_CHECK_OK(Reset(
      TermFrequencyMap(character_map_path, min_frequency, max_num_terms)));
}

void UnicodeDictionary::Clear() {
  size_ = 0;
  for (int32 &index : single_byte_indices_) index = -1;
  multi_byte_indices_.clear();
}

tensorflow::Status UnicodeDictionary::Reset(
    const TermFrequencyMap &character_map) {
  Clear();
  size_ = character_map.Size();

  for (int32 index = 0; index < character_map.Size(); ++index) {
    const string &character = character_map.GetTerm(index);
    if (character.empty()) {
      return tensorflow::errors::InvalidArgument("Term ", index, " is empty");
    }

    const size_t correct_size = UniLib::OneCharLen(character.data());
    if (character.size() != correct_size) {
      return tensorflow::errors::InvalidArgument(
          "Term ", index, " should have size ", correct_size, ": ",
          CharacterDebugString(character));
    }

    if (!UniLib::IsUTF8ValidCodepoint(character)) {
      return tensorflow::errors::InvalidArgument(
          "Term ", index,
          " is not valid UTF-8: ", CharacterDebugString(character));
    }

    const auto *bytes = reinterpret_cast<const uint8 *>(character.data());
    if (character.size() == 1) {
      DCHECK_EQ(single_byte_indices_[*bytes], -1);
      single_byte_indices_[*bytes] = index;
    } else {
      const uint32 key = MultiByteKey(bytes, character.size());
      DCHECK(multi_byte_indices_.find(key) == multi_byte_indices_.end());
      multi_byte_indices_[key] = index;
    }
  }

  return tensorflow::Status::OK();
}

}  // namespace runtime
}  // namespace dragnn
}  // namespace syntaxnet
