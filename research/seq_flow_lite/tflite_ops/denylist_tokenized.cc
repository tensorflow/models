/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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
#include "absl/container/flat_hash_set.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "flatbuffers/flexbuffers.h"  // flatbuffer
#include "tensorflow/lite/context.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/string_util.h"
#include "tf_ops/skipgram_finder.h"  // seq_flow_lite
#include "tflite_ops/denylist.h"  // seq_flow_lite

namespace seq_flow_lite {
namespace ops {
namespace custom {

namespace denylist {

using ::tflite::GetString;
using ::tflite::GetTensorData;
using ::tflite::StringRef;

// Generates prediction vectors for tokenized input strings using a skipgram
// denylist.  This uses the framework in `denylist.h`, with the implementation
// detail that the input is a string tensor of tokens and the terms are
// skipgrams.
class TokenizedDenylistOp : public DenylistOp {
 public:
  explicit TokenizedDenylistOp(const flexbuffers::Map& custom_options)
      : DenylistOp(custom_options),
        skipgram_finder_(custom_options["max_skip_size"].AsInt32()) {
    auto denylist = custom_options["denylist"].AsTypedVector();
    auto denylist_category =
        custom_options["denylist_category"].AsTypedVector();
    if (denylist.size() != denylist_category.size()) {
      AddError(
          absl::StrFormat("denylist.size (%d) != denylist_category.size (%d)",
                          denylist.size(), denylist_category.size()));
      return;
    }

    for (int i = 0; i < denylist.size(); i++) {
      int category = denylist_category[i].AsInt32();
      if (category < 0 || category >= categories()) {
        AddError(absl::StrFormat(
            "denylist_category[%d] (%d) is out of range: [0, %d)", i, category,
            categories()));
        continue;
      }
      flexbuffers::String s = denylist[i].AsString();
      skipgram_finder_.AddSkipgram(absl::string_view(s.c_str(), s.length()),
                                   category);
    }
  }

  TfLiteStatus InitializeInput(TfLiteContext* context,
                               TfLiteNode* node) override {
    tokens_ = &context->tensors[node->inputs->data[kInputTokens]];
    token_counts_ = &context->tensors[node->inputs->data[kInputTokenCounts]];
    return kTfLiteOk;
  }

  TfLiteStatus GetCategories(
      TfLiteContext* context, int i,
      absl::flat_hash_set<int>& categories) const override {
    std::vector<absl::string_view> tokens;

    int token_count = 0;
    switch (token_counts_->type) {
      case kTfLiteInt32:
        token_count = GetTensorData<int32_t>(token_counts_)[i];
        break;

      case kTfLiteInt64:
        token_count = GetTensorData<int64_t>(token_counts_)[i];
        break;

      default:
        context->ReportError(
            context, "TOKENIZED_DENYLIST: Unrecognized token_counts type: %d",
            token_counts_->type);
        return kTfLiteError;
    }

    tokens.reserve(token_count);
    int max_tokens = tokens_->dims->data[tokens_->dims->size - 1];
    int start = i * max_tokens;
    for (int j = 0; j < token_count; j++) {
      StringRef token = GetString(tokens_, start + j);
      tokens.emplace_back(token.str, token.len);
    }

    categories = skipgram_finder_.FindSkipgrams(tokens);
    return kTfLiteOk;
  }

  void FinalizeInput() override {
    tokens_ = nullptr;
    token_counts_ = nullptr;
  }

  TfLiteIntArray* GetInputShape(TfLiteContext* context,
                                TfLiteNode* node) override {
    return context->tensors[node->inputs->data[kInputTokenCounts]].dims;
  }

 private:
  SkipgramFinder skipgram_finder_;
  TfLiteTensor* tokens_;
  TfLiteTensor* token_counts_;

  static constexpr int kInputTokens = 0;
  static constexpr int kInputTokenCounts = 1;
};

void* TokenizedDenylistOpInit(TfLiteContext* context, const char* buffer,
                              size_t length) {
  const uint8_t* buffer_t = reinterpret_cast<const uint8_t*>(buffer);
  return new TokenizedDenylistOp(
      flexbuffers::GetRoot(buffer_t, length).AsMap());
}

}  // namespace denylist

TfLiteRegistration* Register_TOKENIZED_DENYLIST() {
  static TfLiteRegistration r = {denylist::TokenizedDenylistOpInit,
                                 denylist::Free, denylist::Resize,
                                 denylist::Eval};
  return &r;
}

}  // namespace custom
}  // namespace ops
}  // namespace seq_flow_lite
