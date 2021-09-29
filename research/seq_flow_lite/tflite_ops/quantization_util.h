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
#ifndef TENSORFLOW_MODELS_SEQUENCE_PROJECTION_TFLITE_OPS_QUANTIZATION_UTIL_H_
#define TENSORFLOW_MODELS_SEQUENCE_PROJECTION_TFLITE_OPS_QUANTIZATION_UTIL_H_

#include <algorithm>
#include <cmath>

#include "tensorflow/lite/context.h"

namespace seq_flow_lite {

// Returns the original (dequantized) value of 8bit value.
inline float PodDequantizeValue(const TfLiteTensor& tensor, uint8_t value) {
  const int32_t zero_point = tensor.params.zero_point;
  const float scale = tensor.params.scale;
  return (static_cast<int32_t>(value) - zero_point) * scale;
}

// Returns the original (dequantized) value of the 'index'-th element of
// 'tensor.
inline float PodDequantize(const TfLiteTensor& tensor, int index) {
  return PodDequantizeValue(tensor, tensor.data.uint8[index]);
}

// Quantizes 'value' to 8bit, given the quantization bias (zero_point) and
// factor (inverse_scale).
inline uint8_t PodQuantize(float value, int32_t zero_point,
                           float inverse_scale) {
  const float integer_value_in_float = value * inverse_scale;
  const float offset = (integer_value_in_float >= 0.0) ? 0.5f : -0.5f;
  // NOTE(sfeuz): This assumes value * inverse_scale is within [INT_MIN,
  // INT_MAX].
  int32_t integer_value =
      static_cast<int32_t>(integer_value_in_float + offset) + zero_point;
  return static_cast<uint8_t>(std::max(std::min(255, integer_value), 0));
}

}  // namespace seq_flow_lite

#endif  // TENSORFLOW_MODELS_SEQUENCE_PROJECTION_TFLITE_OPS_QUANTIZATION_UTIL_H_
