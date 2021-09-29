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
#include "tflite_ops/layer_norm.h"  // seq_flow_lite

#include <unordered_set>
#include <vector>

#include "tensorflow/lite/kernels/kernel_util.h"
#include "tflite_ops/quantization_util.h"  // seq_flow_lite

namespace seq_flow_lite {
namespace ops {
namespace custom {

namespace {

const int kInputIndex = 0;
const int kScaleIndex = 1;
const int kOffsetIndex = 2;
const int kAxisIndex = 3;
const int kOutputIndex = 0;

TfLiteStatus Resize(TfLiteContext* context, TfLiteNode* node) {
  if (node->outputs->size != 1) {
    return kTfLiteError;
  }

  TfLiteTensor* input = &context->tensors[node->inputs->data[kInputIndex]];
  TfLiteTensor* scale = &context->tensors[node->inputs->data[kScaleIndex]];
  TfLiteTensor* offset = &context->tensors[node->inputs->data[kOffsetIndex]];
  TF_LITE_ENSURE_EQ(context, input->type, kTfLiteUInt8);
  TF_LITE_ENSURE_EQ(context, offset->dims->data[0], 1);
  TF_LITE_ENSURE_EQ(context, offset->dims->size, 1);
  TF_LITE_ENSURE_EQ(context, offset->type, kTfLiteUInt8);
  TF_LITE_ENSURE_EQ(context, scale->dims->data[0], 1);
  TF_LITE_ENSURE_EQ(context, scale->dims->size, 1);
  TF_LITE_ENSURE_EQ(context, scale->type, kTfLiteUInt8);
  if (node->inputs->size == 4) {
    TfLiteTensor* axis = &context->tensors[node->inputs->data[kAxisIndex]];
    TF_LITE_ENSURE_EQ(context, axis->type, kTfLiteInt32);
  }

  TfLiteTensor* output = &context->tensors[node->outputs->data[kOutputIndex]];
  TF_LITE_ENSURE_EQ(context, output->type, kTfLiteUInt8);
  return context->ResizeTensor(context, output,
                               TfLiteIntArrayCopy(input->dims));
}

int GetNumberOfSteps(const TfLiteTensor* input) {
  int number_of_steps = 1;
  for (int i = 0; i < input->dims->size; ++i) {
    number_of_steps *= input->dims->data[i];
  }
  return number_of_steps;
}

inline int GetNumberOfFeatures(const TfLiteTensor* input, const int* axis,
                               const int num_axis) {
  int num_features = 1;
  for (int i = 0; i < num_axis; ++i) {
    num_features *= input->dims->data[axis[i]];
  }
  return num_features;
}

// Performs sanity checks on input axis and resolves into valid dimensions.
inline bool ResolveAxis(const int num_dims, const int* axis, const int num_axis,
                        int* out_axis, int* out_num_axis) {
  *out_num_axis = 0;
  // Short-circuit axis resolution for scalars; the axis will go unused.
  if (num_dims == 0) {
    return true;
  }

  // Using an unordered set to reduce complexity in looking up duplicates.
  std::unordered_set<int> unique_indices;
  for (int64_t idx = 0; idx < num_axis; ++idx) {
    // Handle negative index.
    int current = axis[idx] < 0 ? (axis[idx] + num_dims) : axis[idx];
    assert(current >= 0 && current < num_dims);
    // Only adding the axis if it wasn't added before.
    if (unique_indices.find(current) == unique_indices.end()) {
      unique_indices.insert(current);
      out_axis[*out_num_axis] = current;
      *out_num_axis += 1;
    }
  }
  return true;
}

// Given current position in the input array, the api computes the next valid
// index.
bool ValidIndex(const int* input_dims, const int input_dims_size,
                int* curr_pos) {
  if (input_dims_size == 0) {
    return false;
  }
  assert(input_dims != nullptr);
  assert(curr_pos != nullptr);
  for (int idx = input_dims_size - 1; idx >= 0; --idx) {
    int current_val = curr_pos[idx] + 1;
    assert(input_dims[idx] >= current_val);
    if (input_dims[idx] == current_val) {
      curr_pos[idx] = 0;
    } else {
      curr_pos[idx] = current_val;
      return true;
    }
  }
  return false;
}

// Gets next offset depending on reduction axis. Implementation borrowed from
// tflite reduce mean implementation.
int GetOffset(const int* input_dims, const int input_dims_size,
              const int* curr_pos, const int* axis, const int axis_size) {
  if (input_dims_size == 0) return 0;
  assert(input_dims != nullptr);
  assert(curr_pos != nullptr);
  int offset = 0;
  for (int idx = 0; idx < input_dims_size; ++idx) {
    // if idx is part of reduction axes, we skip offset calculation.
    bool is_axis = false;
    if (axis != nullptr) {
      for (int redux = 0; redux < axis_size; ++redux) {
        if (idx == axis[redux]) {
          is_axis = true;
          break;
        }
      }
    }
    if (!is_axis) offset = offset * input_dims[idx] + curr_pos[idx];
  }

  return offset;
}

// TODO(b/132896827): Current implementation needs further evaluation to reduce
// space time complexities.
TfLiteStatus FlexibleLayerNorm(const TfLiteTensor* input, const float scale,
                               const float offset, const int* axis,
                               const int num_axis, TfLiteTensor* output) {
  int num_features = GetNumberOfFeatures(input, &axis[0], num_axis);
  int time_steps = static_cast<int>(GetNumberOfSteps(input) / num_features);

  std::vector<float> sum_x(time_steps, 0.0f);
  std::vector<float> sum_xx(time_steps, 0.0f);
  std::vector<int> index_iter(input->dims->size, 0);

  // Computing sum and squared sum for features across the reduction axes.
  do {
    // Not passing reduction axes to get the input offset as we are simply
    // iterating through the multidimensional array.
    int input_offset = GetOffset(input->dims->data, input->dims->size,
                                 &index_iter[0], nullptr, 0);
    // Passing in the valid reduction axes as we would like to get the output
    // offset after reduction.
    int stats_offset = GetOffset(input->dims->data, input->dims->size,
                                 &index_iter[0], &axis[0], num_axis);
    float input_val = PodDequantize(*input, input_offset);
    sum_x[stats_offset] += input_val;
    sum_xx[stats_offset] += input_val * input_val;
  } while (ValidIndex(input->dims->data, input->dims->size, &index_iter[0]));

  std::vector<float> multiplier(time_steps, 1.0f);
  std::vector<float> bias(time_steps, 0.0f);

  // Computing stats for the reduction axes.
  for (int i = 0; i < time_steps; ++i) {
    sum_x[i] = sum_x[i] / num_features;
    sum_xx[i] = sum_xx[i] / num_features;
    const float variance = sum_xx[i] - sum_x[i] * sum_x[i];
    const float inverse_stddev = 1 / sqrt(variance + 1e-6);
    multiplier[i] = inverse_stddev * scale;
    bias[i] = offset - sum_x[i] * inverse_stddev * scale;
  }

  const float out_inverse_scale = 1.0f / output->params.scale;
  const int32_t out_zero_point = output->params.zero_point;
  uint8_t* out_ptr = output->data.uint8;
  std::fill(index_iter.begin(), index_iter.end(), 0);

  // Using the stats to fill the output pointer.
  do {
    // Not passing reduction axes to get the input offset as we are simply
    // iterating through the multidimensional array.
    int input_offset = GetOffset(input->dims->data, input->dims->size,
                                 &index_iter[0], nullptr, 0);
    // Passing in the valid reduction axes as we would like to get the output
    // offset after reduction.
    int stats_offset = GetOffset(input->dims->data, input->dims->size,
                                 &index_iter[0], &axis[0], num_axis);
    float input_val = PodDequantize(*input, input_offset);

    const float value =
        input_val * multiplier[stats_offset] + bias[stats_offset];
    out_ptr[input_offset] =
        PodQuantize(value, out_zero_point, out_inverse_scale);
  } while (ValidIndex(input->dims->data, input->dims->size, &index_iter[0]));

  return kTfLiteOk;
}

/*
 * Layer normalization is optimized as follows in integer arithmetic
 *
 * Algorithm
 * *********
 * Subscript i \in {1, ..., N}, Inputs q_i, Outputs oq_i.
 *
 * x_i = (q_i - input_zero_point) * input_scale
 * mean = sum_i x_i / N
 * var = sum_i (x_i * x_i / N) - mean * mean
 * std = sqrt(var + tolerance)
 * xni = (xi - mean) / std
 * yi = xni * scale + offset
 * o_i = round(y_i / output_scale + output_zero_point)
 * oq_i = clamp(o_i, 0, 255)
 *
 * Optimizations
 * *************
 * Applying linear expansion
 * x_i = q_i * input_scale - input_zero_point * input_scale
 * or x_i = m * qi + c
 * mean = m * mean_q + c
 * Variance is not affected by a constant shift to input
 * var = m^2 * var_q
 * std = m * sqrt(var_q + tolerance)
 * Expanding xi, mean, std in equation for xni
 * xni = (m * qi + c - m * mean_q - c) / m * sqrt(var_q + tolerance)
 * Simplifying
 * xni = (qi - mean_q) / sqrt(var_q + tolerance)
 * Setting inv_std_qi = 1 / sqrt(var_q + tolerance)
 * xni = qi * inv_std_qi - mean_q * inv_std_qi
 * yi = qi * inv_std_qi * scale - mean_q * inv_std_qi * scale + offset
 * o_i = round(qi * inv_std_qi * scale / output_scale
 *             - mean_q * inv_std_qi * scale / output_scale
 *             + offset / output_scale
 *             + output_zero_point)
 * Setting
 * static_bias = offset / output_scale + output_zero_point
 * static_scale = scale / output_scale
 * o_i = round(qi * inv_std_qi * static_scale
 *             - mean_q * inv_std_qi * static_scale
 *             + static_bias)
 * Setting
 * dynamic_scale = inv_std_qi * static_scale
 * dynamic_bias = static_bias - mean_q * dynamic_scale
 * o_i = round(qi * dynamic_scale + dynamic_bias)
 * oq_i = clamp(round(qi * dynamic_scale + dynamic_bias), 0, 255)
 *
 * This results in the below optimized implementation. The strategy is to first
 * compute first and second order summary statistics for qi in a loop,
 * then compute mean_q, var_q and then dynamic_scale/dynamic_bias. This
 * allows one to compute oqi quickly in a tight loop.
 * */
TfLiteStatus IntegerLayerNorm(const TfLiteTensor* input, const float scale,
                              const float offset, TfLiteTensor* output) {
  const int input_rank = input->dims->size;
  const int num_features = input->dims->data[input_rank - 1];
  const int time_steps =
      static_cast<int>(GetNumberOfSteps(input) / num_features);

  const float out_inverse_scale = 1.0f / output->params.scale;
  const float static_scale = scale * out_inverse_scale;
  const float static_bias = static_cast<float>(output->params.zero_point) +
                            offset * out_inverse_scale;
  const float inverse_num_features = 1.0f / num_features;
  const uint8_t* const in_ptr = input->data.uint8;
  uint8_t* out_ptr = output->data.uint8;
  for (int i = 0; i < time_steps; ++i) {
    int32_t i32_sum_q = 0;
    int32_t i32_sum_qq = 0;
    const int32_t index = i * num_features;
    for (int j = index; j < index + num_features; ++j) {
      const int32_t q_i = static_cast<int32_t>(in_ptr[j]);
      // Compute first and second order statistics for qi.
      i32_sum_q += q_i;
      i32_sum_qq += q_i * q_i;
    }
    const float second_moment_qq = i32_sum_qq * inverse_num_features;
    const float mean_q = i32_sum_q * inverse_num_features;
    const float var_q = second_moment_qq - mean_q * mean_q;
    const float inv_std_q = 1.0f / sqrt(var_q + 1e-6);
    const float dynamic_scale = inv_std_q * static_scale;
    const float dynamic_bias = static_bias - mean_q * dynamic_scale;
    for (int j = index; j < index + num_features; ++j) {
      const int32_t invalue = static_cast<int32_t>(in_ptr[j]);
      const float value = invalue * dynamic_scale + dynamic_bias;
      // Use an offseted cast to perform float round.
      const int32_t i32value =
          static_cast<int32_t>(value + ((value >= 0.0) ? 0.5f : -0.5f));
      // Clamp the result.
      out_ptr[j] = static_cast<uint8_t>(std::max(std::min(255, i32value), 0));
    }
  }
  return kTfLiteOk;
}

TfLiteStatus DefaultLayerNormFloat(const TfLiteTensor* input, const float scale,
                                   const float offset, TfLiteTensor* output) {
  const int input_rank = input->dims->size;
  const int num_features = input->dims->data[input_rank - 1];
  const int time_steps =
      static_cast<int>(GetNumberOfSteps(input) / num_features);
  float* out_ptr = output->data.f;
  for (int i = 0; i < time_steps; ++i) {
    float sum_x = 0;
    float sum_xx = 0;
    for (int j = 0, index = i * num_features; j < num_features; ++j, ++index) {
      sum_x += input->data.f[index];
      sum_xx += input->data.f[index] * input->data.f[index];
    }
    const float exp_xx = sum_xx / num_features;
    const float exp_x = sum_x / num_features;
    const float variance = exp_xx - exp_x * exp_x;
    const float inverse_stddev = 1 / sqrt(variance + 1e-6);
    const float multiplier = inverse_stddev * scale;

    const float bias = offset - exp_x * inverse_stddev * scale;
    for (int j = 0, index = i * num_features; j < num_features; ++j, ++index) {
      out_ptr[index] = input->data.f[index] * multiplier + bias;
    }
  }
  return kTfLiteOk;
}

TfLiteStatus DefaultLayerNorm(const TfLiteTensor* input, const float scale,
                              const float offset, TfLiteTensor* output) {
  const int input_rank = input->dims->size;
  const int num_features = input->dims->data[input_rank - 1];
  const int time_steps =
      static_cast<int>(GetNumberOfSteps(input) / num_features);

  std::vector<float> temp_buffer(num_features, 0.0f);
  const float out_inverse_scale = 1.0f / output->params.scale;
  const int32_t out_zero_point = output->params.zero_point;
  uint8_t* out_ptr = output->data.uint8;
  for (int i = 0; i < time_steps; ++i) {
    float sum_x = 0;
    float sum_xx = 0;
    for (int j = 0, index = i * num_features; j < num_features; ++j, ++index) {
      temp_buffer[j] = PodDequantize(*input, index);
      sum_x += temp_buffer[j];
      sum_xx += temp_buffer[j] * temp_buffer[j];
    }
    const float exp_xx = sum_xx / num_features;
    const float exp_x = sum_x / num_features;
    const float variance = exp_xx - exp_x * exp_x;
    const float inverse_stddev = 1 / sqrt(variance + 1e-6);
    const float multiplier = inverse_stddev * scale;
    const float bias = offset - exp_x * inverse_stddev * scale;
    for (int j = 0, index = i * num_features; j < num_features; ++j, ++index) {
      const float value = temp_buffer[j] * multiplier + bias;
      out_ptr[index] = PodQuantize(value, out_zero_point, out_inverse_scale);
    }
  }
  return kTfLiteOk;
}

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  const TfLiteTensor* input =
      &context->tensors[node->inputs->data[kInputIndex]];
  TfLiteTensor* output = &context->tensors[node->outputs->data[kOutputIndex]];
  TfLiteTensor scale_tensor = context->tensors[node->inputs->data[kScaleIndex]];
  TfLiteTensor offset_tensor =
      context->tensors[node->inputs->data[kOffsetIndex]];
  float scale = 1.0;
  float offset = 0.0;
  if (input->type == kTfLiteUInt8) {
    scale = PodDequantize(scale_tensor, 0);
    offset = PodDequantize(offset_tensor, 0);
  } else {
    scale = scale_tensor.data.f[0];
    offset = offset_tensor.data.f[0];
  }

  TfLiteTensor* axis = &context->tensors[node->inputs->data[kAxisIndex]];
  int num_axis = static_cast<int>(tflite::NumElements(axis));
  // For backward compatibility reasons, we handle the default layer norm for
  // last channel as below.
  if (num_axis == 1 && (axis->data.i32[0] == -1 ||
                        axis->data.i32[0] == (input->dims->size - 1))) {
    if (input->type == kTfLiteUInt8) {
      return IntegerLayerNorm(input, scale, offset, output);
    } else if (input->type == kTfLiteFloat32) {
      return DefaultLayerNormFloat(input, scale, offset, output);
    } else {
      TF_LITE_ENSURE_MSG(context, false,
                         "Input should be eith Uint8 or Float32.");
    }
  }

  std::vector<int> resolved_axis(num_axis);
  // Resolve axis.
  int num_resolved_axis = 0;
  if (!ResolveAxis(input->dims->size, axis->data.i32, num_axis,
                   &resolved_axis[0], &num_resolved_axis)) {
    return kTfLiteError;
  }

  return FlexibleLayerNorm(input, scale, offset, &resolved_axis[0],
                           num_resolved_axis, output);
}

}  // namespace

TfLiteRegistration* Register_LAYER_NORM() {
  static TfLiteRegistration r = {nullptr, nullptr, Resize, Eval};
  return &r;
}

}  // namespace custom
}  // namespace ops
}  // namespace seq_flow_lite
