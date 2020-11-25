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
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "libutf/utf.h"

constexpr int kFirstCapOffset = 3;
constexpr int kAllCapsOffset = 4;
constexpr int kWordNoveltyOffset = 1;
constexpr int kDocSizeOffset = 2;

const char kMurmurHash[] = "murmur";
const char kXfixHash8[] = "xfixhash8";
const char kXfixHash16[] = "xfixhash16";
const char kXfixHash32[] = "xfixhash32";
const char kUnicodeHash8[] = "unicodehash8";
const char kUnicodeHash16[] = "unicodehash16";

class HashEngine {
 public:
  virtual void GetHashCodes(const std::string& word,
                            std::vector<uint64_t>* hash_codes,
                            int feature_size) = 0;
  virtual ~HashEngine() {}
};

// A hashing wrapper class that can hash a string and generate a hash code with
// requested number of features (two bit values). Some of the implementations
// are copied from murmurhash.
class Hasher {
 public:
  static Hasher* CreateHasher(int feature_size,
                              const std::string& hash_type = kMurmurHash);
  static bool SupportedHashType(const std::string& hash_type);
  bool GetHashCodes(const std::string& word,
                    std::vector<uint64_t>* hash_codes) {
    if (!hash_engine_) return false;
    if (word.empty()) {
      *hash_codes = null_hash_codes_;
    } else {
      hash_codes->clear();
      hash_engine_->GetHashCodes(word, hash_codes, feature_size_);
    }
    return true;
  }

 private:
  explicit Hasher(int feature_size, HashEngine* hash_engine);
  const std::string empty_string_ = "<null>";
  const int feature_size_;
  std::unique_ptr<HashEngine> hash_engine_;
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
      const std::pair<const char*, size_t>& source, bool* first_cap = nullptr,
      bool* all_caps = nullptr) const;

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
  static std::vector<std::pair<const char*, size_t>> Tokenize(
      const std::string& input, bool by_space, int max_tokens) {
    return Tokenize(input.c_str(), input.size(), by_space, max_tokens);
  }
  static std::vector<std::pair<const char*, size_t>> Tokenize(const char* str,
                                                              size_t len,
                                                              bool by_space,
                                                              int max_tokens);

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
