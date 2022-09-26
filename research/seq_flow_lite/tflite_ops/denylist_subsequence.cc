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
#include "tensorflow/lite/string_util.h"
#include "tf_ops/subsequence_finder.h"  // seq_flow_lite
#include "tflite_ops/denylist.h"  // seq_flow_lite

namespace seq_flow_lite {
namespace ops {
namespace custom {

namespace denylist {

using ::tflite::GetString;
using ::tflite::StringRef;

// Generates prediction vectors for input strings using a subsequence denylist.
// This uses the framework in `denylist.h`, with the implementation detail
// that the input is a string tensor of messages and the terms are subsequences.
class SubsequenceDenylistOp : public DenylistOp {
 public:
  explicit SubsequenceDenylistOp(const flexbuffers::Map& custom_options)
      : DenylistOp(custom_options),
        subsequence_finder_(custom_options["max_skip_size"].AsInt32()) {
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
      subsequence_finder_.AddSubsequence(
          absl::string_view(s.c_str(), s.length()), category);
    }
  }

  TfLiteStatus InitializeInput(TfLiteContext* context,
                               TfLiteNode* node) override {
    input_ = &context->tensors[node->inputs->data[kInputMessage]];
    return kTfLiteOk;
  }

  TfLiteStatus GetCategories(
      TfLiteContext* context, int i,
      absl::flat_hash_set<int>& categories) const override {
    StringRef input = GetString(input_, i);
    categories = subsequence_finder_.FindSubsequences(
        absl::string_view(input.str, input.len));
    return kTfLiteOk;
  }

  void FinalizeInput() override { input_ = nullptr; }

  TfLiteIntArray* GetInputShape(TfLiteContext* context,
                                TfLiteNode* node) override {
    return context->tensors[node->inputs->data[kInputMessage]].dims;
  }

 private:
  SubsequenceFinder subsequence_finder_;
  TfLiteTensor* input_;

  static constexpr int kInputMessage = 0;
};

void* SubsequenceDenylistOpInit(TfLiteContext* context, const char* buffer,
                                size_t length) {
  const uint8_t* buffer_t = reinterpret_cast<const uint8_t*>(buffer);
  return new SubsequenceDenylistOp(
      flexbuffers::GetRoot(buffer_t, length).AsMap());
}

}  // namespace denylist

TfLiteRegistration* Register_SUBSEQUENCE_DENYLIST() {
  static TfLiteRegistration r = {denylist::SubsequenceDenylistOpInit,
                                 denylist::Free, denylist::Resize,
                                 denylist::Eval};
  return &r;
}

}  // namespace custom
}  // namespace ops
}  // namespace seq_flow_lite
