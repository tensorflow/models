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
#include "tflite_ops/expected_value.h"  // sequence_projection

#include <cmath>

#include "tflite_ops/quantization_util.h"  // sequence_projection

namespace tflite {
namespace ops {
namespace custom {

namespace {

constexpr int kInputAttentionLogits = 0;
constexpr int kInputValues = 1;
constexpr int kOutputExpectedValue = 0;

class ExpectedValueParams {
 public:
  // Get precomputed exponential table for the quantization range of the tensor.
  // The table is precomputed during first lookup and used till the tflite
  // interpreter is destroyed.
  float* GetPrecomputedTable(const TfLiteTensor& tensor) {
    if (!initialized_) {
      initialized_ = true;
      const float scale = tensor.params.scale;
      for (int i = 0;
           i < sizeof(precomputed_table_) / sizeof(precomputed_table_[0]);
           ++i) {
        precomputed_table_[i] = expf(-i * scale);
      }
    }
    return precomputed_table_;
  }

 private:
  bool initialized_ = false;
  float precomputed_table_[256];
};

void* Init(TfLiteContext* context, const char* buffer, size_t length) {
  return new ExpectedValueParams();
}

void Free(TfLiteContext* context, void* buffer) {
  delete reinterpret_cast<ExpectedValueParams*>(buffer);
}

TfLiteStatus Resize(TfLiteContext* context, TfLiteNode* node) {
  TF_LITE_ENSURE_EQ(context, node->inputs->size, 2);
  TF_LITE_ENSURE_EQ(context, node->outputs->size, 1);

  TfLiteTensor* attention_logits =
      &context->tensors[node->inputs->data[kInputAttentionLogits]];
  TfLiteTensor* values = &context->tensors[node->inputs->data[kInputValues]];
  // Currently only 8-bit input tensors are supported.
  TF_LITE_ENSURE_EQ(context, attention_logits->type, kTfLiteUInt8);
  TF_LITE_ENSURE_EQ(context, values->type, kTfLiteUInt8);
  // Both the input tensors are expected to be rank 3.
  TF_LITE_ENSURE_EQ(context, attention_logits->dims->size, 3);
  TF_LITE_ENSURE_EQ(context, attention_logits->dims->size, values->dims->size);
  // Currently batch size is expected to be 1.
  TF_LITE_ENSURE_EQ(context, attention_logits->dims->data[0], 1);
  // Dimensions of both the input tensors should match.
  for (int i = 0; i < values->dims->size; ++i) {
    TF_LITE_ENSURE_EQ(context, attention_logits->dims->data[i],
                      values->dims->data[i]);
  }

  TfLiteTensor* output =
      &context->tensors[node->outputs->data[kOutputExpectedValue]];
  TF_LITE_ENSURE_EQ(context, output->type, kTfLiteUInt8);
  TfLiteIntArray* output_size = TfLiteIntArrayCreate(2);
  // Expectation is over dimension 2 leaving a rank 2 output tensor with first
  // and last dimension as the input.
  output_size->data[0] = values->dims->data[0];
  output_size->data[1] = values->dims->data[2];
  return context->ResizeTensor(context, output, output_size);
}

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  auto logits_t = &context->tensors[node->inputs->data[kInputAttentionLogits]];
  auto values_t = &context->tensors[node->inputs->data[kInputValues]];
  auto output_t = &context->tensors[node->outputs->data[kOutputExpectedValue]];
  const int out_channels = logits_t->dims->data[2];
  const int sequence_length = logits_t->dims->data[1];

  const float out_inverse_scale = 1.0f / output_t->params.scale;
  const int32_t out_zero_point = output_t->params.zero_point;
  uint8_t* output = output_t->data.uint8;
  auto* params = reinterpret_cast<ExpectedValueParams*>(node->user_data);
  const float* table = params->GetPrecomputedTable(*logits_t);
  // Memory layout of the input tensor is row-major, hence the inner loops have
  // a pitch of out_channels instead of 1. The inner loop runs over this array
  // two times for logits and once for values. If the out_channels increases
  // beyond a reasonable value, the entire content of logits/values won't fit in
  // L1 cache, which would make these loops very inefficient. If the last
  // dimension increases, this handler should be rewritten to do transpose first
  // in a cache efficient manner before performing the compute.
  for (int i = 0; i < out_channels; ++i) {
    // Find max logit, max logit is subtracted to ensure numerical stability
    // when computing softmax.
    auto slogits = &logits_t->data.uint8[i];
    auto elogits = slogits + (sequence_length * out_channels);
    int32_t maxval = 0;
    for (auto logits = slogits; logits < elogits; logits += out_channels) {
      maxval = std::max(static_cast<int32_t>(*logits), maxval);
    }
    // Find normalizer to compute softmax (sum of exponential over logits).
    // Compute the softmax output (attention), perform the elementwise
    // multiplication and reduce by summing in a single loop. This results in
    // the unnormalized expected value, which is normalized later.
    float normalizer = 0.0f;
    float unnormalized_expected_value = 0.0f;
    auto values = &values_t->data.uint8[i];
    for (auto logits = slogits; logits < elogits;
         logits += out_channels, values += out_channels) {
      const float unnormalized_attention = table[maxval - *logits];
      normalizer += unnormalized_attention;
      unnormalized_expected_value +=
          unnormalized_attention * PodDequantizeValue(*values_t, *values);
    }
    const float expected_value = unnormalized_expected_value / normalizer;
    // Quantize and set the expected value in the output buffer.
    output[i] = PodQuantize(expected_value, out_zero_point, out_inverse_scale);
  }
  return kTfLiteOk;
}

}  // namespace

// This tflite fused op takes two input tensors (logits and values), which are
// expected to be rank 3 tensors of the form [batch size, sequence, channels].
// The op performs softmax on the sequence dimension of logits input, performs
// an element-wise multiplication with the values tensor, reduces the sequence
// dimension to a scalar value using sum operation and returns a tensor of the
// form [batch size, channels]. Batch size is assumed to be 1 in the current
// implementation.
TfLiteRegistration* Register_EXPECTED_VALUE() {
  static TfLiteRegistration r = {Init, Free, Resize, Eval};
  return &r;
}

}  // namespace custom
}  // namespace ops
}  // namespace tflite
