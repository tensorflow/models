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
#include "tf_ops/projection_util.h"  // seq_flow_lite

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <memory>
#include <sstream>
#include <unordered_set>

namespace {
constexpr int kInvalid = -1;
constexpr char kSpace = ' ';

// A HashEngine that uses MurmurHash to convert text to hashcodes.
class MurmurHash : public HashEngine {
 public:
  std::vector<uint64_t> GetHashCodes(const std::string& word,
                                     int feature_size) override {
    std::vector<uint64_t> hash_codes;
    hash_codes.reserve(2 * (feature_size / 64 + 1));
    uint64_t hash_low = 0;
    uint64_t hash_high = 0;
    for (int i = 0; i < feature_size; i += 64) {
      if (i == 0) {
        auto hash = MurmurHash128(word.data(), word.size());
        hash_low = hash.first;
        hash_high = hash.second;
      } else {
        GetMoreBits(hash_low, hash_high, &hash_low, &hash_high);
      }
      hash_codes.push_back(hash_low);
      hash_codes.push_back(hash_high);
    }
    return hash_codes;
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
    uint64_t hash1 = len * kMul;
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
      hash1 = MurmurStep(hash1, Load64VariableLength(p, 8));
      hash2 ^= hash1;
    }
    if ((len & 0x7) != 0) {
      const uint64_t data = Load64VariableLength(end, len & 0x7);
      hash1 ^= data;
      hash1 *= kMul;
      hash2 ^= hash1;
    }
    hash1 = ShiftMix(hash1) * kMul;
    hash2 ^= hash1;
    hash1 = ShiftMix(hash1);

    // mul2 is a prime just above golden ratio. mul2 is used to ensure that the
    // impact of the last few bytes is different to the upper and lower 64 bits.
    hash2 = ShiftMix(hash2 * kMul2) * kMul2;

    return {hash1, hash2};
  }
};

// A HashEngine that uses a prefix and suffix preserving hash to convert text
// to hashcodes.
class XFixHash : public HashEngine {
 public:
  explicit XFixHash(int bits_per_char)
      : bits_per_char_(bits_per_char), bit_mask_((1ULL << bits_per_char) - 1) {}

  std::vector<uint64_t> GetHashCodes(const std::string& word,
                                     int feature_size) override {
    std::vector<uint64_t> hash_codes;
    hash_codes.reserve(2 * (feature_size / 64 + 1));
    auto token_ptr = reinterpret_cast<const uint8_t*>(word.c_str());
    size_t token_size = word.size();
    int token_idx = 0;
    uint64_t hash_low = token_size * kMul;
    uint64_t hash_high = token_size * kMul2;
    uint64_t frhash = kMul;
    uint64_t brhash = kMul2;
    for (int i = 0; i < feature_size; i += 64) {
      for (int j = i ? 0 : bits_per_char_; j < 64;
           j += bits_per_char_, token_idx = (token_idx + 1) % token_size) {
        frhash = ((frhash << 8) | token_ptr[token_idx]) * kMul;
        brhash =
            ((brhash << 8) | token_ptr[token_size - 1 - token_idx]) * kMul2;
        hash_low = (hash_low << bits_per_char_) | (frhash & bit_mask_);
        hash_high = (hash_high << bits_per_char_) | (brhash & bit_mask_);
      }
      hash_codes.push_back(hash_low);
      hash_codes.push_back(hash_high);
    }
    return hash_codes;
  }

 private:
  const uint64_t kMul = 0xc6a4a7935bd1e995ULL;
  const uint64_t kMul2 = 0x9e3779b97f4a7835ULL;
  const int bits_per_char_;
  const uint64_t bit_mask_;
};

// A HashEngine that performs a position preserving unicode level hashing to
// convert text to hashcodes.
class UnicodeHash : public HashEngine {
 public:
  // bits_per_unicode should be a divisor of 64.
  explicit UnicodeHash(int bits_per_unicode)
      : bits_per_unicode_(bits_per_unicode),
        bit_mask_(((1ULL << bits_per_unicode) - 1) << (64 - bits_per_unicode)) {
  }

  std::vector<uint64_t> GetHashCodes(const std::string& word,
                                     int feature_size) override {
    std::vector<uint64_t> hash_codes;
    hash_codes.reserve(2 * (feature_size / 64 + 1));
    auto word_ptr = word.c_str();
    int utflength = utflen(const_cast<char*>(word_ptr));
    // Both `feature_size` and `bits_per_unicode` are bit lengths.
    const int max_usable_runes = feature_size * 2 / bits_per_unicode_;
    if (max_usable_runes < utflength) {
      const int unicode_skip = (utflength - max_usable_runes) / 2;
      for (int i = 0; i < unicode_skip; ++i) {
        Rune rune;
        word_ptr += chartorune(&rune, const_cast<char*>(word_ptr));
      }
      utflength = max_usable_runes;
    }

    std::vector<uint64_t> unicode_hashes;
    unicode_hashes.reserve(utflength);
    for (int i = 0; i < utflength; ++i) {
      Rune rune;
      word_ptr += chartorune(&rune, const_cast<char*>(word_ptr));
      unicode_hashes.push_back((rune * kMul) & bit_mask_);
    }

    uint64_t hash = 0;
    int k = 0;
    for (int i = 0; i < feature_size * 2; i += 64) {
      for (int j = 0; j < 64; j += bits_per_unicode_) {
        if (k < unicode_hashes.size()) {
          hash = (hash >> bits_per_unicode_) | unicode_hashes[k++];
        } else {
          hash = hash >> bits_per_unicode_;
        }
      }
      hash_codes.push_back(hash);
    }
    return hash_codes;
  }

 private:
  const uint64_t kMul = 0xc6a4a7935bd1e995ULL;
  const int bits_per_unicode_;
  const uint64_t bit_mask_;
};

}  // namespace

bool Hasher::SupportedHashType(const std::string& hash_type) {
  std::unordered_set<std::string> supported({kMurmurHash, kUnicodeHash8,
                                             kUnicodeHash16, kXfixHash8,
                                             kXfixHash16, kXfixHash32});
  return supported.find(hash_type) != supported.end();
}

Hasher* Hasher::CreateHasher(int feature_size, const std::string& hash_type) {
  if (SupportedHashType(hash_type)) {
    if (hash_type == kMurmurHash) {
      return new Hasher(feature_size, new MurmurHash());
    } else if (hash_type == kUnicodeHash8) {
      return new Hasher(feature_size, new UnicodeHash(8));
    } else if (hash_type == kUnicodeHash16) {
      return new Hasher(feature_size, new UnicodeHash(16));
    } else if (hash_type == kXfixHash8) {
      return new Hasher(feature_size, new XFixHash(8));
    } else if (hash_type == kXfixHash16) {
      return new Hasher(feature_size, new XFixHash(16));
    } else {
      return new Hasher(feature_size, new XFixHash(32));
    }
  }
  return nullptr;
}

Hasher::Hasher(int feature_size, HashEngine* hash_engine)
    : feature_size_(feature_size), hash_engine_(hash_engine) {
  null_hash_codes_ = hash_engine_->GetHashCodes(empty_string_, feature_size_);
}

std::string ProjectionUnicodeHandler::LowerCaseUTF8WithSupportedUnicodes(
    const std::pair<const char*, size_t>& source, bool* first_cap,
    bool* all_caps) const {
  // Ideally the size of target should be less than or equal to source. But
  // when we do to_lower the number of bytes needed to encode a unicode
  // character could increase. To account for this 4 times the source length
  // is allocated for target.
  const char* csource = source.first;
  int len = source.second;
  auto target = std::unique_ptr<char[]>(new char[len * 4]);
  auto target_ptr = target.get();
  int i = 0;
  bool first_char = true;
  bool first_cap_value = false;
  bool all_caps_value = false;
  while (i < len) {
    Rune rune;
    const int bytes_read = chartorune(&rune, const_cast<char*>(csource + i));
    if (bytes_read == 0 || bytes_read > len - i) {
      break;
    }
    i += bytes_read;
    if (rune != Runeerror) {
      Rune lower = tolowerrune(rune);
      // Skip processing the unicode if exclude_nonalphaspace_unicodes_ is
      // true and the unicode is not alpha and not space.
      const Rune kSpaceRune = ' ';
      if (exclude_nonalphaspace_unicodes_ && !isalpharune(lower) &&
          lower != kSpaceRune) {
        continue;
      }
      if (IsUnrestrictedVocabulary() || IsValidUnicode(lower)) {
        const int bytes_written = runetochar(target_ptr, &lower);
        target_ptr += bytes_written;

        const bool lower_case = (lower == rune);
        if (first_char) {
          first_cap_value = !lower_case;
          all_caps_value = !lower_case;
        } else {
          first_cap_value &= lower_case;
          all_caps_value &= !lower_case;
        }
        first_char = false;
      }
    }
  }
  if (first_cap) {
    *first_cap = first_cap_value;
  }
  if (all_caps) {
    *all_caps = all_caps_value;
  }
  return std::string(target.get(), target_ptr);
}

void ProjectionUnicodeHandler::InitializeVocabulary(
    const std::string& vocabulary) {
  for (size_t i = 0, index = 0; i < vocabulary.length();) {
    Rune rune;
    const int bytes_read =
        chartorune(&rune, const_cast<char*>(vocabulary.c_str() + i));
    if (!bytes_read || bytes_read > (vocabulary.length() - i)) {
      break;
    }
    i += bytes_read;
    // Include novel lower case unicode segments as part of valid chars.
    if (rune == Runeerror) {
      std::clog << "Invalid rune in vocabulary.";
    } else if (IsValidUnicode(rune)) {
      std::clog << "Duplicate rune " << rune << " found in vocabulary.";
    } else if (rune != tolowerrune(rune)) {
      std::clog << "Upper case rune " << rune << " found in vocabulary.";
    } else {
      valid_chars_[rune] = index++;
    }
  }
}

// Starting from input_ptr[from], search for the next occurrence of ' ',
// Don't search beyond input_ptr[length](non-inclusive), return -1 if not
// found.
inline size_t FindNextSpace(const char* input_ptr, size_t from, size_t length) {
  size_t space_index;
  for (space_index = from; space_index < length; space_index++) {
    if (input_ptr[space_index] == kSpace) {
      break;
    }
  }
  return space_index == length ? kInvalid : space_index;
}

template <typename T>
void SplitBySpaceInternal(std::vector<T>* tokens, const char* input_ptr,
                          size_t len, size_t max_input, size_t max_tokens) {
  size_t last_index =
      max_input == kEntireString ? len : (len < max_input ? len : max_input);
  size_t start = 0;
  // skip leading spaces
  while (start < last_index && input_ptr[start] == kSpace) {
    start++;
  }
  auto end = FindNextSpace(input_ptr, start, last_index);
  while (end != kInvalid &&
         (max_tokens == kAllTokens || tokens->size() < max_tokens - 1)) {
    auto length = end - start;
    if (length > 0) {
      tokens->emplace_back(input_ptr + start, length);
    }

    start = end + 1;
    end = FindNextSpace(input_ptr, start, last_index);
  }
  auto length = end == kInvalid ? (last_index - start) : (end - start);
  if (length > 0) {
    tokens->emplace_back(input_ptr + start, length);
  }
}

std::vector<std::pair<const char*, size_t>> SplitBySpaceAsPairs(
    const char* input_ptr, size_t len, size_t max_tokens) {
  std::vector<std::pair<const char*, size_t>> tokens;
  SplitBySpaceInternal(&tokens, input_ptr, len, kEntireString, max_tokens);
  return tokens;
}

std::vector<std::string> SplitBySpace(const char* input_ptr, size_t len,
                                      size_t max_input, size_t max_tokens) {
  std::vector<std::string> tokens;
  SplitBySpaceInternal(&tokens, input_ptr, len, max_input, max_tokens);
  return tokens;
}

template <typename T>
void SplitByCharInternal(std::vector<T>* tokens, const char* input_ptr,
                         size_t len, size_t max_tokens) {
  Rune rune;
  for (size_t i = 0; i < len;) {
    auto bytes_read = chartorune(&rune, const_cast<char*>(input_ptr + i));
    if (bytes_read == 0 || bytes_read > (len - i)) break;
    tokens->emplace_back(input_ptr + i, bytes_read);
    if (max_tokens != kInvalid && tokens->size() == max_tokens) {
      break;
    }
    i += bytes_read;
  }
}

std::vector<std::pair<const char*, size_t>> SplitByCharAsPairs(
    const char* input_ptr, size_t len, size_t max_tokens) {
  std::vector<std::pair<const char*, size_t>> tokens;
  SplitByCharInternal(&tokens, input_ptr, len, max_tokens);
  return tokens;
}

std::vector<std::string> SplitByChar(const char* input_ptr, size_t len,
                                     size_t max_tokens) {
  std::vector<std::string> tokens;
  SplitByCharInternal(&tokens, input_ptr, len, max_tokens);
  return tokens;
}

std::string JoinPairsBySpace(
    std::vector<std::pair<const char*, size_t>> words) {
  std::stringstream ss;
  bool first = true;
  for (auto& str_pair : words) {
    if (first) {
      ss << std::string(str_pair.first, str_pair.second);
      first = false;
    } else {
      ss << kSpace << std::string(str_pair.first, str_pair.second);
    }
  }
  return ss.str();
}

std::vector<std::pair<const char*, size_t>> ProjectionUnicodeHandler::Tokenize(
    const char* str, size_t len, bool by_space, int max_tokens) {
  return by_space ? SplitBySpaceAsPairs(str, len, max_tokens)
                  : SplitByCharAsPairs(str, len, max_tokens);
}
