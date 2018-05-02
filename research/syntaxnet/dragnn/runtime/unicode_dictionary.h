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

#ifndef DRAGNN_RUNTIME_UNICODE_DICTIONARY_H_
#define DRAGNN_RUNTIME_UNICODE_DICTIONARY_H_

#include <stddef.h>

#include <unordered_map>
#include <string>

#include "syntaxnet/base.h"
#include "syntaxnet/term_frequency_map.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/macros.h"

#include "util/utf8/unilib.h"
#include "util/utf8/unilib_utf8_utils.h"

namespace syntaxnet {
namespace dragnn {
namespace runtime {

// A mapping from Unicode characters to indices.
//
// TODO(googleuser): Try integrating break chars into this mapping, maybe just for
// the ASCII break chars.  They could be mapped directly to the break ID, so all
// one-byte characters are handled directly.
class UnicodeDictionary {
 public:
  // Creates an empty mapping.
  UnicodeDictionary();

  // Loads a TermFrequencyMap from the |character_map_path| while applying the
  // |min_frequency| and |max_num_terms|, and Reset()s this from it.  On error,
  // dies.  This is for use in SharedStore; prefer Initialize() otherwise.
  UnicodeDictionary(const string &character_map_path, int min_frequency,
                    int max_num_terms);

  // Resets this to the |character_map|.  On error, returns non-OK.
  tensorflow::Status Reset(const TermFrequencyMap &character_map);

  // Returns the index of the UTF-8 character spanning [|data|,|data|+|size|),
  // or the |unknown_index| if not present in this.
  int32 Lookup(const char *data, size_t size, int32 unknown_index) const;

  // Accessors.
  size_t size() const { return size_; }

 private:
  // Removes all entries from this mapping.
  void Clear();

  // Returns an integer that uniquely identifies the multi-byte UTF-8 character
  // spanning [|bytes|,|bytes|+|size|).  Note that the returned value is not a
  // Unicode codepoint.
  static uint32 MultiByteKey(const uint8 *bytes, size_t size);

  // Number of entries in this mapping.
  size_t size_ = 0;

  // Dense mapping from single-byte UTF-8 (i.e., ASCII) character to index, or
  // -1 if unmapped.
  int32 single_byte_indices_[128];

  // Sparse mapping from multi-byte UTF-8 character to index.
  std::unordered_map<uint32, int32> multi_byte_indices_;

};

// Implementation details below.

inline int32 UnicodeDictionary::Lookup(const char *data, size_t size,
                                       int32 unknown_index) const {
  DCHECK_GE(size, 1);
  DCHECK_EQ(size, UniLib::OneCharLen(data));
  DCHECK(UniLib::IsUTF8ValidCodepoint(string(data, size)));
  const auto *bytes = reinterpret_cast<const uint8 *>(data);
  if (size == 1) {
    // Look up single-byte characters in the dense mapping.
    DCHECK_LT(*bytes, 128);
    const int32 index = single_byte_indices_[*bytes];
    return index >= 0 ? index : unknown_index;
  } else {
    // Look up multi-byte characters in the sparse mapping.
    const auto it = multi_byte_indices_.find(MultiByteKey(bytes, size));
    return it != multi_byte_indices_.end() ? it->second : unknown_index;
  }
}

inline uint32 UnicodeDictionary::MultiByteKey(const uint8 *bytes, size_t size) {
  DCHECK_GE(size, 2);
  DCHECK_LE(size, 4);
  uint32 value = static_cast<uint32>(bytes[0]) |  //
                 static_cast<uint32>(bytes[1]) << 8;
  switch (size) {
    case 4:
      value |= static_cast<uint32>(bytes[3]) << 24;
      TF_FALLTHROUGH_INTENDED;
    case 3:
      value |= static_cast<uint32>(bytes[2]) << 16;
  }
  return value;
}

}  // namespace runtime
}  // namespace dragnn
}  // namespace syntaxnet

#endif  // DRAGNN_RUNTIME_UNICODE_DICTIONARY_H_
