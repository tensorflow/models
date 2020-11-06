/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
#ifndef TENSORFLOW_MODELS_SEQUENCE_PROJECTION_TF_OPS_PROJECTION_UTIL_H_
#define TENSORFLOW_MODELS_SEQUENCE_PROJECTION_TF_OPS_PROJECTION_UTIL_H_

#include <cassert>
#include <string>
#include <unordered_map>
#include <vector>

#include "libutf/utf.h"

inline int charntorune(Rune* r, const char* s, int n) {
  const int bytes_read = chartorune(r, const_cast<char *>(s));
  if (bytes_read > n) {
    *r = Runeerror;
    return 0;
  }
  return bytes_read;
}

// A hashing wrapper class that can hash a string and generate a hash code with
// requested number of features (two bit values). Some of the implementations
// are copied from murmurhash.
class Hasher {
 public:
  explicit Hasher(int feature_size) : feature_size_(feature_size) {
    GetHashCodesInternal(empty_string_, &null_hash_codes_);
  }
  void GetHashCodes(const std::string& word,
                    std::vector<uint64_t>* hash_codes) {
    if (word.empty()) {
      *hash_codes = null_hash_codes_;
    } else {
      hash_codes->clear();
      GetHashCodesInternal(word, hash_codes);
    }
  }

 private:
  static constexpr uint64_t kMul = 0xc6a4a7935bd1e995ULL;
  static constexpr uint64_t kMul2 = 0x9e3779b97f4a7835ULL;
  inline uint64_t ShiftMix(uint64_t val) { return val ^ (val >> 47); }
  inline uint64_t MurmurStep(uint64_t hash, uint64_t data) {
    hash ^= ShiftMix(data * kMul) * kMul;
    hash *= kMul;
    return hash;
  }
  inline uint64_t Load64VariableLength(const void* p, int len) {
    assert(len >= 1 && len <= 8);
    const char* buf = static_cast<const char*>(p);
    uint64_t val = 0;
    --len;
    do {
      val = (val << 8) | buf[len];
      // (--len >= 0) is about 10 % faster than (len--) in some benchmarks.
    } while (--len >= 0);
    // No ToHost64(...) needed. The bytes are accessed in little-endian manner
    // on every architecture.
    return val;
  }
  void GetMoreBits(uint64_t hash, uint64_t hash2, uint64_t* rlow,
                   uint64_t* rhigh) {
    hash = ShiftMix(hash) * kMul;
    hash2 ^= hash;
    *rhigh = ShiftMix(hash);
    *rlow = ShiftMix(hash2 * kMul2) * kMul2;
  }
  std::pair<uint64_t, uint64_t> MurmurHash128(const char* buf,
                                              const size_t len) {
    // Initialize the hashing value.
    uint64_t hash = len * kMul;
    // hash2 will be xored by hash during the hash computation iterations.
    // In the end we use an alternative mixture multiplier for mixing
    // the bits in hash2.
    uint64_t hash2 = 0;
    // Let's remove the bytes not divisible by the sizeof(uint64_t).
    // This allows the inner loop to process the data as 64 bit integers.
    const size_t len_aligned = len & ~0x7;
    const char* end = buf + len_aligned;

    for (const char* p = buf; p != end; p += 8) {
      // Manually unrolling this loop 2x did not help on Intel Core 2.
      hash = MurmurStep(hash, Load64VariableLength(p, 8));
      hash2 ^= hash;
    }
    if ((len & 0x7) != 0) {
      const uint64_t data = Load64VariableLength(end, len & 0x7);
      hash ^= data;
      hash *= kMul;
      hash2 ^= hash;
    }
    hash = ShiftMix(hash) * kMul;
    hash2 ^= hash;
    hash = ShiftMix(hash);

    // mul2 is a prime just above golden ratio. mul2 is used to ensure that the
    // impact of the last few bytes is different to the upper and lower 64 bits.
    hash2 = ShiftMix(hash2 * kMul2) * kMul2;

    return std::make_pair(hash, hash2);
  }
  void GetHashCodesInternal(const std::string& word,
                            std::vector<uint64_t>* hash_codes) {
    uint64_t hash_low = 0;
    uint64_t hash_high = 0;
    for (int i = 0; i < feature_size_; i += 64) {
      if (i == 0) {
        auto hash = MurmurHash128(word.c_str(), word.size());
        hash_low = hash.first;
        hash_high = hash.second;
      } else {
        GetMoreBits(hash_low, hash_high, &hash_low, &hash_high);
      }
      hash_codes->push_back(hash_low);
      hash_codes->push_back(hash_high);
    }
  }
  const std::string empty_string_ = "<null>";
  const int feature_size_;
  std::vector<uint64_t> null_hash_codes_;
};

// Unicode processor for tensorflow and tflite string projection ops.
class ProjectionUnicodeHandler {
 public:
  // Takes an utf8 string which lists the unicodes that are supported and are
  // part of the vocabulary of this instance. When the utf8 string is empty,
  // all unicode segments are supported by this instance. The boolean
  // flag exclude_nonalphaspace_unicodes is used to indicate if nonalpha and
  // space unicode segments from the input should be stripped out.
  // Another way to analyse the filtering logic is as below.
  // Vocabulary acts as a allowlist when provided and all unicode set when
  // empty. The flag exclude_nonalphaspace_unicodes when true acts as a
  // allowlist on all alpha characters and space. It includes the entire unicode
  // set when false. Valid unicode segments are the intersection of these 2
  // sets.
  explicit ProjectionUnicodeHandler(const std::string& vocabulary,
                                    bool exclude_nonalphaspace_unicodes = false)
      : exclude_nonalphaspace_unicodes_(exclude_nonalphaspace_unicodes) {
    InitializeVocabulary(vocabulary);
  }

  // Performs language independent lower case and returns a string with
  // supported unicode segments.
  std::string LowerCaseUTF8WithSupportedUnicodes(
      const std::pair<const char*, size_t>& source) const;

  // Returns a boolean flag indicating if the unicode segment is part of the
  // vocabulary.
  bool IsValidUnicode(Rune rune) const {
    return valid_chars_.find(rune) != valid_chars_.end();
  }

  // Returns an index in [0, |vocabulary|), if the unicode is part of the
  // vocabulary and -1 if it's not.
  int UnicodeIndex(Rune rune) const {
    return IsValidUnicode(rune) ? valid_chars_.at(rune) : -1;
  }

  // Returns |vocabulary|.
  size_t NumberOfValidUnicodes() const { return valid_chars_.size(); }

  // Returns true if the vocabulary is empty which means all unicode segments
  // are supported.
  bool IsUnrestrictedVocabulary() const { return valid_chars_.empty(); }

  // Tokenizes input by space or unicode point segmentation. Limit to
  // max_tokens, when it is not -1.
  std::vector<std::pair<const char*, size_t>> Tokenize(const std::string& input,
                                                       bool by_space,
                                                       int max_tokens) const {
    return Tokenize(input.c_str(), input.size(), by_space, max_tokens);
  }
  std::vector<std::pair<const char*, size_t>> Tokenize(const char* str,
                                                       size_t len,
                                                       bool by_space,
                                                       int max_tokens) const;

 private:
  // Parses and extracts supported unicode segments from a utf8 string.
  void InitializeVocabulary(const std::string& vocabulary);
  std::unordered_map<Rune, int> valid_chars_;
  bool exclude_nonalphaspace_unicodes_;
};

static constexpr size_t kEntireString = SIZE_MAX;
static constexpr size_t kAllTokens = SIZE_MAX;

std::vector<std::string> SplitBySpace(const char* input_ptr, size_t len,
                                      size_t max_input, size_t max_tokens);

std::vector<std::string> SplitByChar(const char* input_ptr, size_t len,
                                     size_t max_tokens);

std::string JoinPairsBySpace(std::vector<std::pair<const char*, size_t>> words);

#endif  // TENSORFLOW_MODELS_SEQUENCE_PROJECTION_TF_OPS_PROJECTION_UTIL_H_
