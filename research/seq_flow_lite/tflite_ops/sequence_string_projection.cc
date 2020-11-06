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
/**
 * Sequence String projection op used in PRADO.
 */
#include "tflite_ops/sequence_string_projection.h"  // seq_flow_lite

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <unordered_map>

#include "flatbuffers/flexbuffers.h"  // flatbuffer
#include "tensorflow/lite/string_util.h"
#include "tf_ops/projection_normalizer_util.h"  // seq_flow_lite
#include "tf_ops/projection_util.h"  // seq_flow_lite
#include "tflite_ops/quantization_util.h"  // seq_flow_lite

namespace tflite {
namespace ops {
namespace custom {

namespace sequence_string_projection {
/**
 * This op referred to as Ternary Sequence String Projection op (TSP), tokenizes
 * input text either on space or unicode boundary. Fingerprint for each token is
 * computed using murmur hash and bit features are extracted from fingerprint
 * that maps every 2 bits to the ternary output {-1, 0, 1}. This effectively
 * turns a text input into a ternary rank 3 tensor (in 8bit/float format) of
 * shape [1, max token length, requested number of features].
 *
 * Input:
 *   tensor[0]: Input message, string[num_batch]
 *   attribute[0]: feature size
 *   attribute[1]: vocabulary, a set of allowed characters in utf8 format.
 *   attribute[2]: split_on_space, a boolean specifying the tokenization method.
 *   attribute[3]: max_splits, maximum number of splits allowed during
 *                 tokenization. When max_splits is set to -1, no limit on
 *                 number of tokens is imposed. When it is set to a positive
 *                 integer, number of tokens is truncated beyond that integer.
 *                 An end of input token is always added after tokenization,
 *                 hence the number of tokens is one more than the true number
 *                 of tokens. As a result, the number of tokens returned by this
 *                 op is not the same as absl::StrSplit.
 *   attribute[4]: word_novelty_bits, when set to a positive value less than 8,
 *                 generates a word specific novelty feature in the last feature
 *                 index.
 *   attribute[5]: doc_size_levels, when set to a positive value less than 17,
 *                 generates a feature proportional to the logarithm of the
 *                 number of tokens in the second to last feature index.
 *   attribute[6]: add_eos_tag, add an end of sequence tag to the output when
 *                 true. Defaults to true.
 *   attribute[7]: add_bos_tag, add a begin of sequence tag to the output when
 *                 true. Defaults to false.
 * Output:
 * tensor[0]: computed projections.
 *            float32[true number of tokens][feature size]
 *            true number of tokens is number of tokens + 1. (for end of
 *            sequence).
 */

namespace {

constexpr char kBeginToken[] = "<BOS>";
constexpr char kEndToken[] = "<EOS>";
constexpr int kInputMessage = 0;
constexpr int kOutputLabel = 0;

enum class BosTag { kGenerate, kNone };
enum class EosTag { kGenerate, kNone };

class ProjectionParams {
 public:
  ProjectionParams(int feature_size, const std::string& vocabulary,
                   int max_splits, bool split_on_space, int word_novelty_bits,
                   int doc_size_levels, BosTag add_bos_tag, EosTag add_eos_tag,
                   bool exclude_nonalphaspace_unicodes,
                   const std::string& token_separators,
                   bool normalize_repetition)
      : feature_size_(feature_size),
        unicode_handler_(vocabulary, exclude_nonalphaspace_unicodes),
        hasher_(feature_size),
        max_splits_(max_splits),
        split_on_space_(split_on_space),
        word_novelty_bits_(word_novelty_bits),
        doc_size_levels_(doc_size_levels),
        add_bos_tag_(add_bos_tag == BosTag::kGenerate),
        add_eos_tag_(add_eos_tag == EosTag::kGenerate) {
    assert(max_splits_ == -1 || max_splits_ > 0);
    assert(word_novelty_bits >= 0 && word_novelty_bits <= 7);
    if (word_novelty_bits_ != 0) {
      assert(feature_size_ >= 1);
    }
    assert(doc_size_levels >= 0 && doc_size_levels <= 16);
    if (doc_size_levels_ != 0) {
      assert(feature_size_ >= 2);
    }
    word_novelty_offset_ = 2.0f / (1 << word_novelty_bits_);

    if (!token_separators.empty() || normalize_repetition) {
      projection_normalizer_.reset(
          new ProjectionNormalizer(token_separators, normalize_repetition));
    }
  }
  virtual ~ProjectionParams() {}
  int FeatureSize() const { return feature_size_; }
  bool WordNoveltyEnabled() const { return word_novelty_bits_ != 0; }
  void WordNoveltyFeature(float* data, int word_count) const {
    *data = std::min((word_count * word_novelty_offset_) - 1.0f, 1.0f);
  }
  void WordNoveltyFeature(uint8_t* data, int word_count) const {
    float word_novelty_feature;
    WordNoveltyFeature(&word_novelty_feature, word_count);
    *data = PodQuantize(word_novelty_feature, 127.0f, 127);
  }
  bool DocSizeFeatureEnabled() const { return (doc_size_levels_ != 0); }
  int BosToken() const { return add_bos_tag_ ? 1 : 0; }
  int EosToken() const { return add_eos_tag_ ? 1 : 0; }
  void DocSizeFeature(float* data, int num_tokens) {
    float doc_size_feature =
        (doc_size_levels_ != 0)
            ? std::log2(static_cast<float>(num_tokens)) / doc_size_levels_
            : 0.0f;
    *data = std::min(doc_size_feature, 1.0f) * 2.0f - 1.0f;
  }
  void DocSizeFeature(uint8_t* data, int num_tokens) {
    float doc_size_feature;
    DocSizeFeature(&doc_size_feature, num_tokens);
    *data = PodQuantize(doc_size_feature, 127.0f, 127);
  }
  void Hash(const std::string& word, std::vector<uint64_t>* hash_codes) {
    hasher_.GetHashCodes(word, hash_codes);
  }
  // Lower cases the input text and eliminates all unsupported
  // unicodes in it if a vocabulary is provided.
  std::string LowerCaseUTF8WithSupportedUnicodes(
      std::pair<const char*, size_t> source) const {
    return unicode_handler_.LowerCaseUTF8WithSupportedUnicodes(source);
  }
  // Splits the input text into a set of tokens. Uses space as the delimiter
  // when split_on_space is True and unicode boundaries as the delimiter
  // otherwise. When max_splits is set to -1, no limit on number of tokens is
  // imposed. When it is set to a positive integer, number of tokens is
  // truncated beyond that integer. An end of input token is always added after
  // tokenization, hence the number of tokens is one more than the true number
  // of tokens.
  virtual TfLiteStatus PreprocessInput(TfLiteTensor* input_t,
                                       TfLiteContext* context) {
    if (input_t->bytes == 0) {
      context->ReportError(context, "Empty input not supported.");
      return kTfLiteError;
    }
    tflite::StringRef inputref = tflite::GetString(input_t, /*string_index=*/0);
    if (projection_normalizer_ == nullptr) {
      tokens_ = unicode_handler_.Tokenize(inputref.str, inputref.len,
                                          split_on_space_, max_splits_);
    } else {
      normalized_input_ = projection_normalizer_->Normalize(
          inputref.str, inputref.len, SIZE_MAX);
      tokens_ = unicode_handler_.Tokenize(normalized_input_, split_on_space_,
                                          max_splits_);
    }
    if (GetNumTokens() == 0 && !add_bos_tag_ && !add_eos_tag_) {
      context->ReportError(context, "No tokens found.");
      return kTfLiteError;
    }
    return kTfLiteOk;
  }
  int GetNumTokens() const { return tokens_.size(); }
  const std::vector<std::pair<const char*, size_t>>& GetTokens() const {
    return tokens_;
  }
  virtual std::string PreprocessToken(const std::string& word) { return word; }

 private:
  int feature_size_;
  ProjectionUnicodeHandler unicode_handler_;
  Hasher hasher_;
  int max_splits_;
  bool split_on_space_;
  int word_novelty_bits_;
  int doc_size_levels_;
  bool add_bos_tag_;
  bool add_eos_tag_;
  float word_novelty_offset_;
  std::string normalized_input_;

 protected:
  std::unique_ptr<ProjectionNormalizer> projection_normalizer_;
  std::vector<std::pair<const char*, size_t>> tokens_;
};

class ProjectionParamsV2 : public ProjectionParams {
 public:
  ProjectionParamsV2(int feature_size, const std::string& vocabulary,
                     BosTag add_bos_tag, EosTag add_eos_tag,
                     bool normalize_repetition)
      : ProjectionParams(feature_size, vocabulary, /*max_splits = */ -1,
                         /* split_on_space = */ true,
                         /*word_novelty_bits = */ 0, /*doc_size_levels = */ 0,
                         add_bos_tag, add_eos_tag,
                         /*exclude_nonalphaspace_unicodes = */ false,
                         /*token_separators = */ "", normalize_repetition) {}
  ~ProjectionParamsV2() override {}

  TfLiteStatus PreprocessInput(TfLiteTensor* input_t,
                               TfLiteContext* context) override {
    const TfLiteIntArray* const dims = input_t->dims;
    const int num_tokens = tflite::GetStringCount(input_t);
    if (num_tokens == 0) {
      context->ReportError(context, "Empty input not supported.");
      return kTfLiteError;
    }
    if (dims->size != 2) {
      context->ReportError(
          context, "Input tensor is expected to be rank 2, got rank %d.",
          dims->size);
      return kTfLiteError;
    } else if (dims->data[0] != 1) {
      context->ReportError(context,
                           "Input tensor batch size should be 1, got %d.",
                           dims->data[0]);
      return kTfLiteError;
    } else if (num_tokens != dims->data[1]) {
      context->ReportError(context,
                           "Inconsistent number of input tokens %d != %d.",
                           num_tokens, dims->data[1]);
      return kTfLiteError;
    }
    for (int i = 0; i < num_tokens; ++i) {
      const tflite::StringRef strref = tflite::GetString(input_t, i);
      tokens_.push_back(std::pair<const char*, size_t>(strref.str, strref.len));
    }
    return kTfLiteOk;
  }
  std::string PreprocessToken(const std::string& word) override {
    return projection_normalizer_ ? projection_normalizer_->Normalize(
                                        word.data(), word.length(), SIZE_MAX)
                                  : word;
  }
};

inline void SetTensorToDynamic(TfLiteTensor* tensor) {
  if (tensor->allocation_type != kTfLiteDynamic) {
    tensor->allocation_type = kTfLiteDynamic;
    tensor->data.raw = nullptr;
  }
}

// Determines whether tensor is dynamic. Note that a tensor can be non-const and
// not dynamic. This function specifically checks for a dynamic tensor.
inline bool IsDynamicTensor(const TfLiteTensor* tensor) {
  return tensor->allocation_type == kTfLiteDynamic;
}

void* Init(TfLiteContext* context, const char* buffer, size_t length) {
  const uint8_t* buffer_t = reinterpret_cast<const uint8_t*>(buffer);
  const flexbuffers::Map& m = flexbuffers::GetRoot(buffer_t, length).AsMap();
  const int word_novelty_bits =
      m["word_novelty_bits"].IsNull() ? 0 : m["word_novelty_bits"].AsInt32();
  const int doc_size_levels =
      m["doc_size_levels"].IsNull() ? 0 : m["doc_size_levels"].AsInt32();
  const bool add_bos_tag =
      m["add_bos_tag"].IsNull() ? false : m["add_bos_tag"].AsBool();
  const bool add_eos_tag =
      m["add_eos_tag"].IsNull() ? true : m["add_eos_tag"].AsBool();
  // Old models that use the op may not have this attribute set, for those
  // models the default value of false will be used.
  const bool exclude_nonalphaspace_unicodes =
      m["exclude_nonalphaspace_unicodes"].IsNull()
          ? false
          : m["exclude_nonalphaspace_unicodes"].AsBool();
  const std::string token_separators =
      m["token_separators"].IsNull() ? "" : m["token_separators"].ToString();
  const bool normalize_repetition = m["normalize_repetition"].AsBool();

  return new ProjectionParams(
      m["feature_size"].AsInt32(), m["vocabulary"].AsString().str(),
      m["max_splits"].AsInt32(), m["split_on_space"].AsBool(),
      word_novelty_bits, doc_size_levels,
      add_bos_tag ? BosTag::kGenerate : BosTag::kNone,
      add_eos_tag ? EosTag::kGenerate : EosTag::kNone,
      exclude_nonalphaspace_unicodes, token_separators, normalize_repetition);
}

void* InitV2(TfLiteContext* context, const char* buffer, size_t length) {
  const uint8_t* buffer_t = reinterpret_cast<const uint8_t*>(buffer);
  const flexbuffers::Map& m = flexbuffers::GetRoot(buffer_t, length).AsMap();

  return new ProjectionParamsV2(
      m["feature_size"].AsInt32(), m["vocabulary"].AsString().str(),
      m["add_bos_tag"].AsBool() ? BosTag::kGenerate : BosTag::kNone,
      m["add_eos_tag"].AsBool() ? EosTag::kGenerate : EosTag::kNone,
      m["normalize_repetition"].AsBool());
}

void Free(TfLiteContext* context, void* buffer) {
  delete reinterpret_cast<ProjectionParams*>(buffer);
}

TfLiteStatus Resize(TfLiteContext* context, TfLiteNode* node) {
  TfLiteTensor* output = &context->tensors[node->outputs->data[kOutputLabel]];
  SetTensorToDynamic(output);
  return kTfLiteOk;
}

constexpr int kHashCodeBits = 64;
constexpr int kMapBits = 2;
constexpr int kIncrement = kHashCodeBits / kMapBits;

template <typename T>
void TypedEval(const T* mapping_table, ProjectionParams* params, T* data) {
  auto tokens = params->GetTokens();
  std::vector<uint64_t> hash_codes;
  std::unordered_map<uint64_t, int> word_counter;

  T doc_size_feature = T{0};
  if (params->DocSizeFeatureEnabled()) {
    params->DocSizeFeature(&doc_size_feature, tokens.size());
  }
  const int num_tokens = tokens.size() + params->EosToken();
  for (int j = -params->BosToken(), offset0 = 0; j < num_tokens; ++j) {
    std::string word;
    if (j < 0) {
      word = kBeginToken;
    } else if (j < tokens.size()) {
      word = params->LowerCaseUTF8WithSupportedUnicodes(tokens[j]);
      word = params->PreprocessToken(word);
    } else {
      word = kEndToken;
    }
    params->Hash(word, &hash_codes);
    for (int hindex = 0, k = 0; hindex < hash_codes.size(); hindex++) {
      auto hash = hash_codes[hindex];
      for (int kmax = std::min(k + kIncrement, params->FeatureSize());
           k < kmax;) {
        data[offset0 + k++] = mapping_table[hash & ((1 << kMapBits) - 1)];
        hash >>= kMapBits;
      }
    }
    offset0 += params->FeatureSize();
    if (params->WordNoveltyEnabled() && !hash_codes.empty()) {
      params->WordNoveltyFeature(&data[offset0 - 1],
                                 word_counter[hash_codes[0]]++);
    }
    if (params->DocSizeFeatureEnabled()) {
      data[offset0 - 2] = doc_size_feature;
    }
  }
}

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  auto* params = reinterpret_cast<ProjectionParams*>(node->user_data);
  TF_LITE_ENSURE_OK(
      context,
      params->PreprocessInput(
          &context->tensors[node->inputs->data[kInputMessage]], context));

  TfLiteTensor* output = &context->tensors[node->outputs->data[kOutputLabel]];
  if (IsDynamicTensor(output)) {
    TfLiteIntArray* output_size = TfLiteIntArrayCreate(3);
    output_size->data[0] = 1;
    output_size->data[1] =
        params->BosToken() + params->GetNumTokens() + params->EosToken();
    output_size->data[2] = params->FeatureSize();
    TF_LITE_ENSURE_OK(context,
                      context->ResizeTensor(context, output, output_size));
  } else {
    context->ReportError(context, "Output must by dynamic.");
    return kTfLiteError;
  }

  if (output->type == kTfLiteUInt8) {
    const uint8_t kMappingTable[1 << kMapBits] = {127, 255, 0, 127};
    TypedEval(kMappingTable, params, output->data.uint8);
  } else if (output->type == kTfLiteFloat32) {
    const float kMappingTable[1 << kMapBits] = {0.0, 1.0, -1.0, 0.0};
    TypedEval(kMappingTable, params, output->data.f);
  } else {
    context->ReportError(context, "Output type must be UInt8 or Float32.");
    return kTfLiteError;
  }

  return kTfLiteOk;
}

}  // namespace
}  // namespace sequence_string_projection

const char kSequenceStringProjection[] = "SEQUENCE_STRING_PROJECTION";

// This op converts a list of strings to a sequence of features using hashing.
TfLiteRegistration* Register_SEQUENCE_STRING_PROJECTION() {
  static TfLiteRegistration r = {
      sequence_string_projection::Init, sequence_string_projection::Free,
      sequence_string_projection::Resize, sequence_string_projection::Eval};
  return &r;
}

const char kSequenceStringProjectionV2[] = "SEQUENCE_STRING_PROJECTION_V2";

// This op converts a sequence of tokens to a sequence of projected features
// using hashing.
TfLiteRegistration* Register_SEQUENCE_STRING_PROJECTION_V2() {
  static TfLiteRegistration r = {
      sequence_string_projection::InitV2, sequence_string_projection::Free,
      sequence_string_projection::Resize, sequence_string_projection::Eval};
  return &r;
}

}  // namespace custom
}  // namespace ops
}  // namespace tflite
