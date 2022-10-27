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
#include "tflite_ops/denylist.h"  // seq_flow_lite

#include "absl/container/flat_hash_set.h"
#include "tensorflow/lite/context.h"
#include "tflite_ops/quantization_util.h"  // seq_flow_lite

namespace seq_flow_lite {
namespace ops {
namespace custom {
namespace denylist {

static const int kOutputCategories = 0;

void Free(TfLiteContext* context, void* buffer) {
  delete reinterpret_cast<DenylistOp*>(buffer);
}

TfLiteStatus Resize(TfLiteContext* context, TfLiteNode* node) {
  auto* op = reinterpret_cast<DenylistOp*>(node->user_data);
  TF_LITE_ENSURE_STATUS(op->CheckErrors(context));

  TfLiteIntArray* input_dims = op->GetInputShape(context, node);
  TfLiteIntArray* output_dims = TfLiteIntArrayCreate(input_dims->size + 1);
  for (int i = 0; i < input_dims->size; i++) {
    output_dims->data[i] = input_dims->data[i];
  }
  output_dims->data[input_dims->size] = op->categories();
  return context->ResizeTensor(
      context, &context->tensors[node->outputs->data[kOutputCategories]],
      output_dims);
}

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  auto* op = reinterpret_cast<DenylistOp*>(node->user_data);
  TF_LITE_ENSURE_STATUS(op->CheckErrors(context));

  TfLiteTensor* output_categories =
      &context->tensors[node->outputs->data[kOutputCategories]];

  TfLiteIntArray* input_dims = op->GetInputShape(context, node);
  int input_size = 1;
  for (int i = 0; i < input_dims->size; i++) {
    input_size *= input_dims->data[i];
  }
  int n_categories = op->categories();

  TF_LITE_ENSURE_STATUS(op->InitializeInput(context, node));
  if (output_categories->type == kTfLiteFloat32) {
    for (int i = 0; i < input_size; i++) {
      absl::flat_hash_set<int> categories;
      TF_LITE_ENSURE_STATUS(op->GetCategories(context, i, categories));
      if (categories.empty()) {
        for (int j = 0; j < n_categories; j++) {
          output_categories->data.f[i * n_categories + j] =
              (j < op->negative_categories()) ? 1.0 : 0.0;
        }
      } else {
        for (int j = 0; j < n_categories; j++) {
          output_categories->data.f[i * n_categories + j] =
              (categories.find(j) != categories.end()) ? 1.0 : 0.0;
        }
      }
    }
  } else if (output_categories->type == kTfLiteUInt8) {
    const uint8_t one =
        PodQuantize<uint8_t>(1.0, output_categories->params.zero_point,
                             1.0 / output_categories->params.scale);
    const uint8_t zero =
        PodQuantize<uint8_t>(0.0, output_categories->params.zero_point,
                             1.0 / output_categories->params.scale);
    for (int i = 0; i < input_size; i++) {
      absl::flat_hash_set<int> categories;
      TF_LITE_ENSURE_STATUS(op->GetCategories(context, i, categories));
      if (categories.empty()) {
        for (int j = 0; j < n_categories; j++) {
          output_categories->data.uint8[i * n_categories + j] =
              (j < op->negative_categories()) ? one : zero;
        }
      } else {
        for (int j = 0; j < n_categories; j++) {
          output_categories->data.uint8[i * n_categories + j] =
              (categories.find(j) != categories.end()) ? one : zero;
        }
      }
    }
  }
  op->FinalizeInput();
  return kTfLiteOk;
}

}  // namespace denylist
}  // namespace custom
}  // namespace ops
}  // namespace seq_flow_lite
