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

#include "models/sgnn/sgnn_projection.h"  // seq_flow_lite

#include <cstdlib>
#include <iostream>

#include "flatbuffers/flexbuffers.h"  // flatbuffer
#include "farmhash.h"
#include "tensorflow/lite/context.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/string_util.h"

namespace tflite {
namespace ops {
namespace custom {
namespace sgnn {

// This TFLite op implements the SGNN Projection
//
// Input:
// * data: A ragged string tensor of rank 2 (a 1D string value tensor and
//     a 1D int64 row_split tensor).
//
// Attributes:
// * hash_seed:             list of integers
//     Hash seeds to project features
// * buckets:               scalar integer
//     Bucketize computed hash signatures.
//
// Output:
// * output: A 2D float tensor, 1st dimension is the batch of `data`,
//     2nd dimension is the size of `hash_seed`.

constexpr int kValues = 0;
constexpr int kRowSplits = 1;

struct SgnnProjectionAttributes {
  int buckets;
  std::vector<int32_t> hash_seed;

  explicit SgnnProjectionAttributes(const flexbuffers::Map& m)
      : buckets(m["buckets"].AsInt32()) {
    buckets = m["buckets"].AsInt32();
    auto hash_seed_attr = m["hash_seed"].AsTypedVector();
    hash_seed = std::vector<int32_t>(hash_seed_attr.size());
    for (int i = 0; i < hash_seed_attr.size(); ++i) {
      hash_seed[i] = hash_seed_attr[i].AsInt32();
    }
  }
};

void* Init(TfLiteContext* context, const char* buffer, size_t length) {
  const uint8_t* buffer_t = reinterpret_cast<const uint8_t*>(buffer);
  return new SgnnProjectionAttributes(
      flexbuffers::GetRoot(buffer_t, length).AsMap());
}

void Free(TfLiteContext* context, void* buffer) {
  delete reinterpret_cast<SgnnProjectionAttributes*>(buffer);
}

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  const auto& attributes =
      *reinterpret_cast<SgnnProjectionAttributes*>(node->user_data);
  const TfLiteTensor* input_row_splits = GetInput(context, node, kRowSplits);
  TfLiteTensor* output = GetOutput(context, node, 0);
  TfLiteIntArray* output_shape = TfLiteIntArrayCreate(2);
  output_shape->data[0] = SizeOfDimension(input_row_splits, 0) - 1;
  output_shape->data[1] = attributes.hash_seed.size();
  TF_LITE_ENSURE_OK(context,
                    context->ResizeTensor(context, output, output_shape));
  return kTfLiteOk;
}

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  const auto& attributes =
      *reinterpret_cast<SgnnProjectionAttributes*>(node->user_data);
  const TfLiteTensor* ngrams = GetInput(context, node, kValues);
  const TfLiteTensor* row_splits = GetInput(context, node, kRowSplits);

  auto row_splits_values = GetTensorData<int64_t>(row_splits);
  auto output_values = GetTensorData<float>(GetOutput(context, node, 0));
  int output_idx = 0;
  for (int i = 1; i < SizeOfDimension(row_splits, 0); ++i) {
    int len = row_splits_values[i] - row_splits_values[i - 1];
    std::vector<int64_t> hash_signature(len);

    // Follow the implementation from
    // tensorflow/core/kernels/string_to_hash_bucket_op.h
    for (int j = 0; j < len; ++j) {
      int index = row_splits->data.i64[i - 1] + j;
      StringRef str = GetString(ngrams, index);
      hash_signature[j] =
          util::Fingerprint64(str.str, str.len) % attributes.buckets;
    }
    for (int k = 0; k < attributes.hash_seed.size(); ++k) {
      double result = 0;
      for (int j = 0; j < len; ++j) {
        int64_t tmp = hash_signature[j] * attributes.hash_seed[k];
        int64_t value = abs(tmp) % attributes.buckets;
        if (value > attributes.buckets / 2) {
          value -= attributes.buckets;
        }
        result += value;
      }
      output_values[output_idx] =
          static_cast<float>(result) / (attributes.buckets / 2) / len;
      output_idx++;
    }
  }
  return kTfLiteOk;
}

}  // namespace sgnn

TfLiteRegistration* Register_tftext_SGNN_PROJECTION() {
  static TfLiteRegistration r = {sgnn::Init, sgnn::Free, sgnn::Prepare,
                                 sgnn::Eval};
  return &r;
}

}  // namespace custom
}  // namespace ops
}  // namespace tflite
