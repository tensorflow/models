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
#include "tflite_ops/quantization_util.h"  // seq_flow_lite
#include "tflite_ops/tflite_qrnn_pooling.h"  // seq_flow_lite

namespace seq_flow_lite {
namespace ops {
namespace custom {

namespace {

const uint8_t kPoolingForward = 255;

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  TF_LITE_ENSURE_EQ(context, node->inputs->size, 3);
  if (node->outputs->size < 1 || node->outputs->size > 2) {
    return kTfLiteError;
  }

  TfLiteTensor* multiplier = &context->tensors[node->inputs->data[0]];
  TfLiteTensor* constant = &context->tensors[node->inputs->data[1]];
  TfLiteTensor* direction = &context->tensors[node->inputs->data[2]];

  TF_LITE_ENSURE_EQ(context, multiplier->type, kTfLiteUInt8);
  TF_LITE_ENSURE_EQ(context, constant->type, kTfLiteUInt8);
  TF_LITE_ENSURE_EQ(context, direction->type, kTfLiteUInt8);

  TF_LITE_ENSURE_EQ(context, multiplier->dims->size, 3);
  TF_LITE_ENSURE_EQ(context, multiplier->dims->data[0], 1);
  const int time_steps = multiplier->dims->data[1];
  const int state_size = multiplier->dims->data[2];

  TF_LITE_ENSURE_EQ(context, constant->dims->size, 3);
  TF_LITE_ENSURE_EQ(context, constant->dims->data[0], 1);
  TF_LITE_ENSURE_EQ(context, constant->dims->data[1], time_steps);
  TF_LITE_ENSURE_EQ(context, constant->dims->data[2], state_size);

  TF_LITE_ENSURE_EQ(context, direction->dims->size, 1);
  TF_LITE_ENSURE_EQ(context, direction->dims->data[0], 1);

  TfLiteTensor* outputs = &context->tensors[node->outputs->data[0]];
  if (outputs) {
    TF_LITE_ENSURE_OK(
        context, context->ResizeTensor(context, outputs,
                                       TfLiteIntArrayCopy(multiplier->dims)));
  }

  if (node->outputs->size == 2) {
    TfLiteTensor* final_state = &context->tensors[node->outputs->data[1]];
    if (final_state) {
      TfLiteIntArray* final_state_dims = TfLiteIntArrayCreate(2);
      final_state_dims->data[0] = 1;
      final_state_dims->data[1] = state_size;
      TF_LITE_ENSURE_OK(context, context->ResizeTensor(context, final_state,
                                                       final_state_dims));
    }
  }
  return kTfLiteOk;
}

TfLiteStatus QRNNPooling(TfLiteContext* context, TfLiteTensor* multiplier,
                         TfLiteTensor* constant, TfLiteTensor* outputs,
                         TfLiteTensor* final_state, bool forward) {
  const int time_steps = multiplier->dims->data[1];
  const int state_size = multiplier->dims->data[2];

  auto state = std::make_unique<float[]>(state_size);
  memset(state.get(), 0, sizeof(float) * state_size);

  const int32_t out_zero_point = outputs ? outputs->params.zero_point : 0;
  const float out_inverse_scale = outputs ? 1.0f / outputs->params.scale : 1.0f;
  uint8_t* out_ptr = outputs ? outputs->data.uint8 : nullptr;
  for (int i = 0; i < time_steps; ++i) {
    for (int j = 0; j < state_size; ++j) {
      const int time_index = forward ? i : time_steps - (i + 1);
      const int index = time_index * state_size + j;
      float multiplier_value = PodDequantize(*multiplier, index);
      float constant_vale = PodDequantize(*constant, index);
      state[j] = state[j] * multiplier_value + constant_vale;
      if (outputs) {
        out_ptr[index] =
            PodQuantize(state[j], out_zero_point, out_inverse_scale);
      }
    }
  }

  if (final_state) {
    uint8_t* final_state_ptr = final_state->data.uint8;
    const int32_t zero_point = final_state->params.zero_point;
    const float inverse_scale = 1.0f / final_state->params.scale;
    for (int j = 0; j < state_size; ++j) {
      final_state_ptr[j] = PodQuantize(state[j], zero_point, inverse_scale);
    }
  }

  return kTfLiteOk;
}

TfLiteStatus Invoke(TfLiteContext* context, TfLiteNode* node) {
  TF_LITE_ENSURE_EQ(context, node->inputs->size, 3);
  if (node->outputs->size < 1 || node->outputs->size > 2) {
    return kTfLiteError;
  }

  TfLiteTensor* multiplier = &context->tensors[node->inputs->data[0]];
  TfLiteTensor* constant = &context->tensors[node->inputs->data[1]];
  TfLiteTensor* direction = &context->tensors[node->inputs->data[2]];
  TfLiteTensor* outputs = &context->tensors[node->outputs->data[0]];
  TfLiteTensor* final_state = (node->outputs->size == 2)
                                  ? &context->tensors[node->outputs->data[1]]
                                  : nullptr;

  // When pooling forward the direction parameter is expected to be
  // kPoolingForward.
  return QRNNPooling(context, multiplier, constant, outputs, final_state,
                     (direction->data.uint8[0] == kPoolingForward));
}

}  // namespace

const char kPoolingOp[] = "PoolingOp";

void RegisterQRNNPooling(::tflite::ops::builtin::BuiltinOpResolver* resolver) {
  resolver->AddCustom(kPoolingOp, Register_QRNN_POOLING());
}

TfLiteRegistration* Register_QRNN_POOLING() {
  static TfLiteRegistration r = {nullptr, nullptr, Prepare, Invoke};
  return &r;
}

}  // namespace custom
}  // namespace ops
}  // namespace seq_flow_lite
