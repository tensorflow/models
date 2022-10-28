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

#include <cstdint>
#include <vector>

#include <gtest/gtest.h>
#include "tensorflow/lite/kernels/test_util.h"

namespace seq_flow_lite {
namespace ops {
namespace custom {
namespace {

using ::testing::ElementsAreArray;
using ::tflite::ArrayFloatNear;
using ::tflite::Dequantize;
using ::tflite::TensorData;
using ::tflite::TensorType_INT32;
using ::tflite::TensorType_INT8;
using ::tflite::TensorType_UINT8;

template <typename T>
class LayerNormModel : public ::tflite::SingleOpModel {
 public:
  explicit LayerNormModel(const TensorData& input_t, const TensorData& output_t,
                          TensorData scale_t, TensorData offset_t, float scale,
                          float offset, std::initializer_list<int> axes)
      : scale_value_(scale), offset_value_(offset) {
    scale_t.min = std::min(scale, 0.0f);
    scale_t.max = std::max(scale, 0.0f);
    offset_t.min = std::min(offset, 0.0f);
    offset_t.max = std::max(offset, 0.0f);
    const int num_axes = axes.size();
    input_ = AddInput(input_t);
    scale_ = AddInput(scale_t);
    offset_ = AddInput(offset_t);
    axis_ = AddConstInput(TensorType_INT32, axes, {num_axes});
    output_ = AddOutput(output_t);
    SetCustomOp("LayerNorm", {}, Register_LAYER_NORM);
    BuildInterpreter({GetShape(input_)});
  }

  void SetInput(const std::vector<float>& data) {
    QuantizeAndPopulate<T>(input_, data);
    QuantizeAndPopulate<T>(scale_, {scale_value_});
    QuantizeAndPopulate<T>(offset_, {offset_value_});
  }

  std::vector<T> GetOutput() { return ExtractVector<T>(output_); }

  std::vector<float> GetDequantizedOutput() {
    return Dequantize<T>(ExtractVector<T>(output_), GetScale(output_),
                         GetZeroPoint(output_));
  }

 private:
  int input_;
  int scale_;
  int offset_;
  int axis_;
  float scale_value_;
  float offset_value_;
  int output_;
};

TEST(LayerNormModelTest, RegularInput) {
  const float kQuantizedTolerance = 20 * (1. / 256);
  const std::vector<float> input = {
      0, -6, 2, 4, 3, -2, 10, 1,
  };
  // Mean values are 0.0, 3.0
  // Standard deviation values are 3.74, 4.41
  const std::vector<float> expected_output = {0.0, -1.6,  0.53, 1.07,
                                              0.0, -1.13, 1.59, -0.45};

  LayerNormModel<uint8_t> m(
      {TensorType_UINT8, {1, 2, 4}, /*min=*/-10, /*max=*/10},
      {TensorType_UINT8, {}, /*min=*/-10.0f, /*max=*/10.0f},
      {TensorType_UINT8, {1}}, {TensorType_UINT8, {1}},
      /*scale=*/1.0, /*offset=*/0.0, /*axes=*/{2});

  m.SetInput(input);
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(
      m.GetDequantizedOutput(),
      ElementsAreArray(ArrayFloatNear(expected_output, kQuantizedTolerance)));
}

TEST(LayerNormModelTest, RegularInputInt8) {
  const float kQuantizedTolerance = 20 * (1. / 256);
  const std::vector<float> input = {
      0, -6, 2, 4, 3, -2, 10, 1,
  };
  // Mean values are 0.0, 3.0
  // Standard deviation values are 3.74, 4.41
  const std::vector<float> expected_output = {0.0, -1.6,  0.53, 1.07,
                                              0.0, -1.13, 1.59, -0.45};

  LayerNormModel<int8_t> m(
      {TensorType_INT8, {1, 2, 4}, /*min=*/-10, /*max=*/10},
      {TensorType_INT8, {}, /*min=*/-10.0f, /*max=*/10.0f},
      {TensorType_INT8, {1}}, {TensorType_INT8, {1}},
      /*scale=*/1.0, /*offset=*/0.0, /*axes=*/{2});
  m.SetInput(input);
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(
      m.GetDequantizedOutput(),
      ElementsAreArray(ArrayFloatNear(expected_output, kQuantizedTolerance)));
}

TEST(LayerNormModelTest, NegativeScale) {
  const float kQuantizedTolerance = 20 * (1. / 256);
  const std::vector<float> input = {
      0, -6, 2, 4, 3, -2, 10, 1,
  };
  // Mean values are 0.0, 3.0
  // Standard deviation values are 3.74, 4.41
  const std::vector<float> expected_output = {0.0, 1.6,  -0.53, -1.07,
                                              0.0, 1.13, -1.59, 0.45};

  LayerNormModel<uint8_t> m(
      {TensorType_UINT8, {1, 2, 4}, /*min=*/-10, /*max=*/10},
      {TensorType_UINT8, {}, /*min=*/-10.0f, /*max=*/10.0f},
      {TensorType_UINT8, {1}}, {TensorType_UINT8, {1}},
      /*scale=*/-1.0, /*offset=*/0.0, /*axes=*/{2});
  m.SetInput(input);
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(
      m.GetDequantizedOutput(),
      ElementsAreArray(ArrayFloatNear(expected_output, kQuantizedTolerance)));
}

TEST(LayerNormModelTest, NegativeScaleInt8) {
  const float kQuantizedTolerance = 20 * (1. / 256);
  const std::vector<float> input = {
      0, -6, 2, 4, 3, -2, 10, 1,
  };
  // Mean values are 0.0, 3.0
  // Standard deviation values are 3.74, 4.41
  const std::vector<float> expected_output = {0.0, 1.6,  -0.53, -1.07,
                                              0.0, 1.13, -1.59, 0.45};

  LayerNormModel<int8_t> m(
      {TensorType_INT8, {1, 2, 4}, /*min=*/-10, /*max=*/10},
      {TensorType_INT8, {}, /*min=*/-10.0f, /*max=*/10.0f},
      {TensorType_INT8, {1}}, {TensorType_INT8, {1}},
      /*scale=*/-1.0, /*offset=*/0.0, /*axes=*/{2});

  m.SetInput(input);
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(
      m.GetDequantizedOutput(),
      ElementsAreArray(ArrayFloatNear(expected_output, kQuantizedTolerance)));
}

TEST(LayerNormModelTest, NegativeOffset) {
  const float kQuantizedTolerance = 20 * (1. / 256);
  const std::vector<float> input = {
      0, -6, 2, 4, 3, -2, 10, 1,
  };
  // Mean values are 0.0, 3.0
  // Standard deviation values are 3.74, 4.41
  const std::vector<float> expected_output = {-1.0, -2.6,  -0.53, 0.07,
                                              -1.0, -2.13, 0.59,  -1.45};
  LayerNormModel<uint8_t> m(
      {TensorType_UINT8, {1, 2, 4}, /*min=*/-10, /*max=*/10},
      {TensorType_UINT8, {}, /*min=*/-10.0f, /*max=*/10.0f},
      {TensorType_UINT8, {1}}, {TensorType_UINT8, {1}},
      /*scale=*/1.0, /*offset=*/-1.0, /*axes=*/{2});
  m.SetInput(input);
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(
      m.GetDequantizedOutput(),
      ElementsAreArray(ArrayFloatNear(expected_output, kQuantizedTolerance)));
}

TEST(LayerNormModelTest, NegativeOffsetInt8) {
  const float kQuantizedTolerance = 20 * (1. / 256);
  const std::vector<float> input = {
      0, -6, 2, 4, 3, -2, 10, 1,
  };
  // Mean values are 0.0, 3.0
  // Standard deviation values are 3.74, 4.41
  const std::vector<float> expected_output = {-1.0, -2.6,  -0.53, 0.07,
                                              -1.0, -2.13, 0.59,  -1.45};
  LayerNormModel<int8_t> m(
      {TensorType_INT8, {1, 2, 4}, /*min=*/-10, /*max=*/10},
      {TensorType_INT8, {}, /*min=*/-10.0f, /*max=*/10.0f},
      {TensorType_INT8, {1}}, {TensorType_INT8, {1}},
      /*scale=*/1.0, /*offset=*/-1.0, /*axes=*/{2});
  m.SetInput(input);
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(
      m.GetDequantizedOutput(),
      ElementsAreArray(ArrayFloatNear(expected_output, kQuantizedTolerance)));
}

TEST(LayerNormModelTest, NegativeScaleAndOffset) {
  const float kQuantizedTolerance = 20 * (1. / 256);
  const std::vector<float> input = {
      0, -6, 2, 4, 3, -2, 10, 1,
  };
  // Mean values are 0.0, 3.0
  // Standard deviation values are 3.74, 4.41
  const std::vector<float> expected_output = {-1.0, 0.6,  -1.53, -2.07,
                                              -1.0, 0.13, -2.59, -0.55};
  LayerNormModel<uint8_t> m(
      {TensorType_UINT8, {1, 2, 4}, /*min=*/-10, /*max=*/10},
      {TensorType_UINT8, {}, /*min=*/-10.0f, /*max=*/10.0f},
      {TensorType_UINT8, {1}}, {TensorType_UINT8, {1}},
      /*scale=*/-1.0, /*offset=*/-1.0, /*axes=*/{2});

  m.SetInput(input);
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(
      m.GetDequantizedOutput(),
      ElementsAreArray(ArrayFloatNear(expected_output, kQuantizedTolerance)));
}

TEST(LayerNormModelTest, NegativeScaleAndOffsetInt8) {
  const float kQuantizedTolerance = 20 * (1. / 256);
  const std::vector<float> input = {
      0, -6, 2, 4, 3, -2, 10, 1,
  };
  // Mean values are 0.0, 3.0
  // Standard deviation values are 3.74, 4.41
  const std::vector<float> expected_output = {-1.0, 0.6,  -1.53, -2.07,
                                              -1.0, 0.13, -2.59, -0.55};
  LayerNormModel<int8_t> m(
      {TensorType_INT8, {1, 2, 4}, /*min=*/-10, /*max=*/10},
      {TensorType_INT8, {}, /*min=*/-10.0f, /*max=*/10.0f},
      {TensorType_INT8, {1}}, {TensorType_INT8, {1}},
      /*scale=*/-1.0, /*offset=*/-1.0, /*axes=*/{2});

  m.SetInput(input);
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(
      m.GetDequantizedOutput(),
      ElementsAreArray(ArrayFloatNear(expected_output, kQuantizedTolerance)));
}

TEST(LayerNormModelTest, MultipleAxis) {
  const float kQuantizedTolerance = 6 * (1. / 256);
  const std::vector<float> input = {0,  1,  2,  3, 2,  2, 3,  3,  2,  -3, 1, 0,
                                    -2, -3, -2, 0, -1, 0, -3, -2, -1, 0,  1, 2};
  const std::vector<float> expected_output = {
      0.06,  0.57,  1.08,  1.59,  0.69,  0.69,  1.15,  1.15,
      1.12,  -2.08, 0.48,  -0.16, -0.95, -1.46, -0.95, 0.06,
      -0.69, -0.23, -1.60, -1.15, -0.80, -0.16, 0.48,  1.12};

  LayerNormModel<uint8_t> m(
      {TensorType_UINT8, {1, 2, 3, 4}, /*min=*/-3, /*max=*/3},
      {TensorType_UINT8, {}, /*min=*/-3, /*max=*/3}, {TensorType_UINT8, {1}},
      {TensorType_UINT8, {1}},
      /*scale=*/1.0, /*offset=*/0.0, /*axes=*/{1, 3});

  m.SetInput(input);
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(
      m.GetDequantizedOutput(),
      ElementsAreArray(ArrayFloatNear(expected_output, kQuantizedTolerance)));
}

TEST(LayerNormModelTest, MultipleAxisInt8) {
  const float kQuantizedTolerance = 6 * (1. / 256);
  const std::vector<float> input = {0,  1,  2,  3, 2,  2, 3,  3,  2,  -3, 1, 0,
                                    -2, -3, -2, 0, -1, 0, -3, -2, -1, 0,  1, 2};
  const std::vector<float> expected_output = {
      0.06,  0.57,  1.08,  1.59,  0.69,  0.69,  1.15,  1.15,
      1.12,  -2.08, 0.48,  -0.16, -0.95, -1.46, -0.95, 0.06,
      -0.69, -0.23, -1.60, -1.15, -0.80, -0.16, 0.48,  1.12};

  LayerNormModel<int8_t> m(
      {TensorType_INT8, {1, 2, 3, 4}, /*min=*/-3, /*max=*/3},
      {TensorType_INT8, {}, /*min=*/-3, /*max=*/3}, {TensorType_INT8, {1}},
      {TensorType_INT8, {1}},
      /*scale=*/1.0, /*offset=*/0.0, /*axes=*/{1, 3});

  m.SetInput(input);
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(
      m.GetDequantizedOutput(),
      ElementsAreArray(ArrayFloatNear(expected_output, kQuantizedTolerance)));
}

TEST(LayerNormModelTest, MultipleNegativeAxis) {
  const float kQuantizedTolerance = 20 * (1. / 256);
  const std::vector<float> input = {
      0, -6, 2, 4, 3, -2, 10, 1,
  };
  // Mean values are 0.0, 3.0
  // Standard deviation values are 3.74, 4.41
  const std::vector<float> expected_output = {0.0, 1.6,  -0.53, -1.07,
                                              0.0, 1.13, -1.59, 0.45};
  LayerNormModel<uint8_t> m(
      {TensorType_UINT8, /*shape=*/{1, 2, 4}, /*min=*/-10, /*max=*/10},
      {TensorType_UINT8, {}, /*min=*/-10, /*max=*/10}, {TensorType_UINT8, {1}},
      {TensorType_UINT8, {1}},
      /*scale=*/-1.0, /*offset=*/0.0, /*axes=*/{2});

  m.SetInput(input);
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(
      m.GetDequantizedOutput(),
      ElementsAreArray(ArrayFloatNear(expected_output, kQuantizedTolerance)));
}

TEST(LayerNormModelTest, MultipleNegativeAxisInt8) {
  const float kQuantizedTolerance = 20 * (1. / 256);
  const std::vector<float> input = {
      0, -6, 2, 4, 3, -2, 10, 1,
  };
  // Mean values are 0.0, 3.0
  // Standard deviation values are 3.74, 4.41
  const std::vector<float> expected_output = {0.0, 1.6,  -0.53, -1.07,
                                              0.0, 1.13, -1.59, 0.45};
  LayerNormModel<int8_t> m(
      {TensorType_INT8, /*shape=*/{1, 2, 4}, /*min=*/-10, /*max=*/10},
      {TensorType_INT8, {}, /*min=*/-10, /*max=*/10}, {TensorType_INT8, {1}},
      {TensorType_INT8, {1}},
      /*scale=*/-1.0, /*offset=*/0.0, /*axes=*/{2});

  m.SetInput(input);
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(
      m.GetDequantizedOutput(),
      ElementsAreArray(ArrayFloatNear(expected_output, kQuantizedTolerance)));
}

TEST(LayerNormModelTest, MultipleAxisWithLargeDepth) {
  const float kQuantizedTolerance = 7 * (1. / 256);
  const std::vector<float> input = {
      0.1, 0.2, 0.3, 0.4, 0.2, 0.3, 0.4, 0.5, 0.1, 0.1, 0.1, 0.1,
      0.4, 0.2, 0.2, 0.2, 0.9, 0.9, 0.9, 0.9, 0.2, 0.3, 0.7, 0.7,
      0.1, 0.1, 0.3, 0.3, 0.1, 0.2, 0.3, 0.4, 0.1, 0.2, 0.3, 0.4};
  const std::vector<float> expected_output = {
      -1.06, -0.67, -0.28, 0.11,  -0.67, -0.28, 0.11,  0.50,  -1.06,
      -0.85, -0.85, -0.85, 0.42,  -0.42, -0.42, -0.42, 2.55,  2.55,
      2.05,  2.05,  -0.67, -0.28, 1.27,  1.27,  -1.06, -1.06, -0.28,
      0.,    -0.85, -0.42, 0.,    0.42,  -0.85, -0.42, 0.,    0.42};

  LayerNormModel<uint8_t> m({TensorType_UINT8, /*shape=*/{1, 2, 2, 9},
                             /*min=*/-1.0, /*max=*/1.0},
                            {TensorType_UINT8, {}, /*min=*/-3, /*max=*/3},
                            {TensorType_UINT8, {1}}, {TensorType_UINT8, {1}},
                            /*scale=*/1.0, /*offset=*/0.0, /*axes=*/{1, 3});

  m.SetInput(input);
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(
      m.GetDequantizedOutput(),
      ElementsAreArray(ArrayFloatNear(expected_output, kQuantizedTolerance)));
}

TEST(LayerNormModelTest, MultipleAxisWithLargeDepthInt8) {
  const float kQuantizedTolerance = 7 * (1. / 256);
  const std::vector<float> input = {
      0.1, 0.2, 0.3, 0.4, 0.2, 0.3, 0.4, 0.5, 0.1, 0.1, 0.1, 0.1,
      0.4, 0.2, 0.2, 0.2, 0.9, 0.9, 0.9, 0.9, 0.2, 0.3, 0.7, 0.7,
      0.1, 0.1, 0.3, 0.3, 0.1, 0.2, 0.3, 0.4, 0.1, 0.2, 0.3, 0.4};
  const std::vector<float> expected_output = {
      -1.06, -0.67, -0.28, 0.11,  -0.67, -0.28, 0.11,  0.50,  -1.06,
      -0.85, -0.85, -0.85, 0.42,  -0.42, -0.42, -0.42, 2.55,  2.55,
      2.05,  2.05,  -0.67, -0.28, 1.27,  1.27,  -1.06, -1.06, -0.28,
      0.,    -0.85, -0.42, 0.,    0.42,  -0.85, -0.42, 0.,    0.42};

  LayerNormModel<int8_t> m({TensorType_INT8, /*shape=*/{1, 2, 2, 9},
                            /*min=*/-1.0, /*max=*/1.0},
                           {TensorType_INT8, {}, /*min=*/-3, /*max=*/3},
                           {TensorType_INT8, {1}}, {TensorType_INT8, {1}},
                           /*scale=*/1.0, /*offset=*/0.0, /*axes=*/{1, 3});

  m.SetInput(input);
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(
      m.GetDequantizedOutput(),
      ElementsAreArray(ArrayFloatNear(expected_output, kQuantizedTolerance)));
}

}  // namespace
}  // namespace custom
}  // namespace ops
}  // namespace seq_flow_lite
