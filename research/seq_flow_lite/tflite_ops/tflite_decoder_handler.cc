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

#include "tflite_ops/tflite_decoder_handler.h"  // seq_flow_lite

#include <cstdint>

#include "flatbuffers/flexbuffers.h"  // flatbuffer
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tflite_ops/quantization_util.h"  // seq_flow_lite
#include "tflite_ops/tflite_decoder_cache.h"  // seq_flow_lite


namespace seq_flow_lite {
namespace ops {
namespace custom {

namespace {

static constexpr const char kFeatureSizeStr[] = "feature_size";
static constexpr const char kBeamSizeStr[] = "beam_size";
constexpr int kInputFeaturesIndex = 0;
constexpr int kTimestepIndex = 1;
constexpr int kSelectedBeamsIndex = 2;
constexpr int kOutputFeaturesIndex = 0;
}  // namespace

namespace tflite_decoder_uniform {
// Evaluates uniform average decoding operations.
class UniformDecoderOp : public tflite_decoder_base::BaseDecoderOp<float> {
 public:
  explicit UniformDecoderOp(int feature_size, int beam_size)
      : BaseDecoderOp(feature_size, beam_size) {}
  void Eval(int32_t step, const std::vector<int32_t>& selected_beams,
            const float* update, float* result);
  void EvalQuantized(int32_t step, const std::vector<int32_t>& selected_beams,
                     const TfLiteTensor* input, TfLiteTensor* output);
};

void UniformDecoderOp::Eval(int32_t step,
                            const std::vector<int32_t>& selected_beams,
                            const float* update, float* result) {
  const float normalizer = 1.0f / step;
  const float* cur_cache = CurrentCache(step);
  float* next_cache = NextCache(step);
  for (int i = 0, index = 0; i < BeamSize(); ++i) {
    const float* selected = cur_cache + (selected_beams[i] * FeatureSize());
    for (int j = 0; j < FeatureSize(); ++j, index++) {
      next_cache[index] = selected[j] + update[index];
      result[index] = next_cache[index] * normalizer;
    }
  }
}

void UniformDecoderOp::EvalQuantized(int32_t step,
                                     const std::vector<int32_t>& selected_beams,
                                     const TfLiteTensor* input,
                                     TfLiteTensor* output) {
  uint8_t* result = ::tflite::GetTensorData<uint8_t>(output);
  const float normalizer_and_inverse_scale =
      1.0f / (output->params.scale * step);
  const float* cur_cache = CurrentCache(step);
  float* next_cache = NextCache(step);
  for (int i = 0, index = 0; i < BeamSize(); ++i) {
    const float* selected = cur_cache + (selected_beams[i] * FeatureSize());
    for (int j = 0; j < FeatureSize(); ++j, index++) {
      next_cache[index] =
          selected[j] + ::seq_flow_lite::PodDequantize(*input, index);
      result[index] = ::seq_flow_lite::PodQuantize(
          next_cache[index], output->params.zero_point,
          normalizer_and_inverse_scale);
    }
  }
}

void* Init(TfLiteContext* context, const char* buffer, size_t length) {
  const uint8_t* buffer_t = reinterpret_cast<const uint8_t*>(buffer);
  const flexbuffers::Map& m = flexbuffers::GetRoot(buffer_t, length).AsMap();
  return new UniformDecoderOp(m[kFeatureSizeStr].AsInt32(),
                              m[kBeamSizeStr].AsInt32());
}

void Free(TfLiteContext* context, void* buffer) {
  delete reinterpret_cast<UniformDecoderOp*>(buffer);
}

TfLiteStatus Resize(TfLiteContext* context, TfLiteNode* node) {
  TF_LITE_ENSURE_EQ(context, ::tflite::NumInputs(node), 3);
  TF_LITE_ENSURE_EQ(context, ::tflite::NumOutputs(node), 1);
  const TfLiteTensor* input =
      ::tflite::GetInput(context, node, kInputFeaturesIndex);
  TfLiteTensor* output =
      ::tflite::GetOutput(context, node, kOutputFeaturesIndex);
  TF_LITE_ENSURE_TYPES_EQ(context, input->type, output->type);
  return context->ResizeTensor(context, output,
                               TfLiteIntArrayCopy(input->dims));
}

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  TF_LITE_ENSURE_EQ(context, ::tflite::NumInputs(node), 3);
  TF_LITE_ENSURE_EQ(context, ::tflite::NumOutputs(node), 1);
  auto* params = reinterpret_cast<UniformDecoderOp*>(node->user_data);
  const TfLiteTensor* input =
      ::tflite::GetInput(context, node, kInputFeaturesIndex);
  const TfLiteTensor* time_step =
      ::tflite::GetInput(context, node, kTimestepIndex);
  const TfLiteTensor* selected_beams =
      ::tflite::GetInput(context, node, kSelectedBeamsIndex);
  TF_LITE_ENSURE_EQ(context, time_step->type, kTfLiteInt32);
  TF_LITE_ENSURE_EQ(context, time_step->dims->size, 0);

  TF_LITE_ENSURE_EQ(context, selected_beams->type, kTfLiteInt32);
  TF_LITE_ENSURE_EQ(context, selected_beams->dims->size, 1);
  TF_LITE_ENSURE_EQ(context, selected_beams->dims->data[0], params->BeamSize());

  const int32_t time_step_value =
      ::tflite::GetTensorData<int32_t>(time_step)[0];
  const int32_t* selected_beams_ptr =
      ::tflite::GetTensorData<int32_t>(selected_beams);
  const std::vector<int32_t> selected_beams_value(
      selected_beams_ptr, selected_beams_ptr + params->BeamSize());
  for (auto value : selected_beams_value) {
    TF_LITE_ENSURE(context, value >= 0 && value < params->BeamSize());
  }
  TfLiteTensor* output =
      ::tflite::GetOutput(context, node, kOutputFeaturesIndex);
  TF_LITE_ENSURE_TYPES_EQ(context, input->type, output->type);
  // Time step is expected to be in [1, )
  TF_LITE_ENSURE(context, time_step_value >= 1);
  if (time_step_value == 1) {
    params->InitCache();
  }
  if (input->type == kTfLiteFloat32) {
    params->Eval(time_step_value, selected_beams_value,
                 ::tflite::GetTensorData<float>(input),
                 ::tflite::GetTensorData<float>(output));
  } else if (input->type == kTfLiteUInt8) {
    params->EvalQuantized(time_step_value, selected_beams_value, input, output);
  } else {
    context->ReportError(context, "Op type must be Float32 or UInt8.");
    return kTfLiteError;
  }
  return kTfLiteOk;
}

}  // namespace tflite_decoder_uniform

TfLiteRegistration* Register_UNIFORM_CAUSAL_ATTENTION() {
  static TfLiteRegistration r = {
      tflite_decoder_uniform::Init, tflite_decoder_uniform::Free,
      tflite_decoder_uniform::Resize, tflite_decoder_uniform::Eval};
  return &r;
}

}  // namespace custom
}  // namespace ops
}  // namespace seq_flow_lite
