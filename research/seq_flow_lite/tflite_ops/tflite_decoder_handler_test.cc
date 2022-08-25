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
#include <cstdlib>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "flatbuffers/flexbuffers.h"  // flatbuffer
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/test_util.h"

namespace {

constexpr char kUniformAverageAttention[] = "UniformAverageAttentionDecoder";

class AverageAttentionDecoder : public tflite::SingleOpModel {
 public:
  explicit AverageAttentionDecoder(int feature_size, int beam_size,
                                   bool quantized = false)
      : quantized_(quantized) {
    flexbuffers::Builder fbb;
    fbb.Map([&] {
      fbb.Int("feature_size", feature_size);
      fbb.Int("beam_size", beam_size);
    });
    fbb.Finish();
    if (!quantized) {
      input_ =
          AddInput({tflite::TensorType_FLOAT32, {beam_size, 1, feature_size}});
      output_ =
          AddOutput({tflite::TensorType_FLOAT32, {beam_size, 1, feature_size}});
    } else {
      input_ = AddInput(
          {tflite::TensorType_UINT8, {beam_size, 1, feature_size}, 0.0f, 4.0f});
      output_ = AddOutput(
          {tflite::TensorType_UINT8, {beam_size, 1, feature_size}, 0.0f, 4.0f});
    }
    timestep_ = AddInput({tflite::TensorType_INT32, {}});
    beam_ = AddInput({tflite::TensorType_INT32, {beam_size}});

    SetCustomOp(
        kUniformAverageAttention, fbb.GetBuffer(),
        ::seq_flow_lite::ops::custom::Register_UNIFORM_CAUSAL_ATTENTION);
    BuildInterpreter({GetShape(input_), GetShape(timestep_), GetShape(beam_)});
    CHECK(interpreter_->AllocateTensors() == kTfLiteOk)
        << "Cannot allocate tensors";
  }
  TfLiteStatus Invoke(int timestep, const std::vector<int32_t>& beams,
                      const std::vector<float>& input_val) {
    PopulateTensor<int32_t>(timestep_, {timestep});
    PopulateTensor<int32_t>(beam_, beams);
    if (!quantized_) {
      PopulateTensor<float>(input_, input_val);
    } else {
      QuantizeAndPopulate<uint8_t>(input_, input_val);
    }
    return SingleOpModel::Invoke();
  }
  std::vector<float> GetOutput() {
    if (!quantized_) {
      return ExtractVector<float>(output_);
    } else {
      return tflite::Dequantize<uint8_t>(ExtractVector<uint8_t>(output_),
                                         GetScale(output_),
                                         GetZeroPoint(output_));
    }
  }

 private:
  int input_;
  int output_;
  int timestep_;
  int beam_;
  bool quantized_;
};

TEST(AverageAttentionDecoder, RegularInput) {
  AverageAttentionDecoder m(4, 4);
  auto status = m.Invoke(1, {0, 0, 0, 0},
                         {1.f, 1.f, 1.f, 1.f,  //
                          2.f, 2.f, 2.f, 2.f,  //
                          3.f, 3.f, 3.f, 3.f,  //
                          4.f, 4.f, 4.f, 4.f});
  EXPECT_EQ(status, kTfLiteOk);

  EXPECT_THAT(m.GetOutput(), testing::ElementsAreArray({1.f, 1.f, 1.f, 1.f,  //
                                                        2.f, 2.f, 2.f, 2.f,  //
                                                        3.f, 3.f, 3.f, 3.f,  //
                                                        4.f, 4.f, 4.f, 4.f}));
  status = m.Invoke(2, {2, 3, 1, 1},
                    {1.f, 1.f, 1.f, 1.f,  //
                     2.f, 2.f, 2.f, 2.f,  //
                     3.f, 3.f, 3.f, 3.f,  //
                     4.f, 4.f, 4.f, 4.f});
  EXPECT_EQ(status, kTfLiteOk);
  EXPECT_THAT(m.GetOutput(),
              testing::ElementsAreArray({2.f, 2.f, 2.f, 2.f,      //
                                         3.f, 3.f, 3.f, 3.f,      //
                                         2.5f, 2.5f, 2.5f, 2.5f,  //
                                         3.f, 3.f, 3.f, 3.f}));
}

TEST(AverageAttentionDecoder, RegularInputQuantized) {
  AverageAttentionDecoder m(4, 4, true);
  auto status = m.Invoke(1, {0, 0, 0, 0},
                         {1.f, 1.f, 1.f, 1.f,  //
                          2.f, 2.f, 2.f, 2.f,  //
                          3.f, 3.f, 3.f, 3.f,  //
                          4.f, 4.f, 4.f, 4.f});
  EXPECT_EQ(status, kTfLiteOk);
  EXPECT_THAT(m.GetOutput(),
              ElementsAreArray(tflite::ArrayFloatNear({1.f, 1.f, 1.f, 1.f,  //
                                                       2.f, 2.f, 2.f, 2.f,  //
                                                       3.f, 3.f, 3.f, 3.f,  //
                                                       4.f, 4.f, 4.f, 4.f},
                                                      1e-2)));
  EXPECT_EQ(status, kTfLiteOk);

  status = m.Invoke(2, {2, 3, 1, 1},
                    {1.f, 1.f, 1.f, 1.f,  //
                     2.f, 2.f, 2.f, 2.f,  //
                     3.f, 3.f, 3.f, 3.f,  //
                     4.f, 4.f, 4.f, 4.f});
  EXPECT_EQ(status, kTfLiteOk);
  EXPECT_THAT(m.GetOutput(), ElementsAreArray(tflite::ArrayFloatNear(
                                 {2.f, 2.f, 2.f, 2.f,      //
                                  3.f, 3.f, 3.f, 3.f,      //
                                  2.5f, 2.5f, 2.5f, 2.5f,  //
                                  3.f, 3.f, 3.f, 3.f},
                                 1e-2)));
}

TEST(AverageAttentionDecoder, RandomInput) {
  AverageAttentionDecoder m(4, 4);
  std::vector<float> input = {2.1,  3.1,  -1.6, 11.3,   //
                              22.6, 20.8, 32.2, -12.9,  //
                              13.2, 3.3,  -3.0, 33.3,   //
                              24.3, 14.9, -4.9, 4.7};

  auto status = m.Invoke(1, {0, 0, 0, 0}, input);
  EXPECT_EQ(status, kTfLiteOk);
  EXPECT_THAT(m.GetOutput(), testing::ElementsAreArray(input));

  status = m.Invoke(2, {2, 3, 1, 1}, input);
  EXPECT_EQ(status, kTfLiteOk);
  EXPECT_THAT(m.GetOutput(), testing::ElementsAreArray(tflite::ArrayFloatNear(
                                 {7.65, 3.2, -2.3, 22.3,      //
                                  23.45, 17.85, 13.65, -4.1,  //
                                  17.9, 12.05, 14.6, 10.2,    //
                                  23.45, 17.85, 13.65, -4.1},
                                 1e-2)));
}

TEST(AverageAttentionDecoder, IrrregularInput) {
  AverageAttentionDecoder m(4, 4, false);
  auto status = m.Invoke(1, {20, 3, 2, 0},
                         {1.f, 1.f, 1.f, 1.f,  //
                          2.f, 2.f, 2.f, 2.f,  //
                          3.f, 3.f, 3.f, 3.f,  //
                          4.f, 4.f});
  EXPECT_EQ(status, kTfLiteError);

  status = m.Invoke(-10, {0, 3, 2, 0},
                    {1.f, 1.f, 1.f, 1.f,  //
                     2.f, 2.f, 2.f, 2.f,  //
                     3.f, 3.f, 3.f, 3.f,  //
                     4.f, 4.f});
  EXPECT_EQ(status, kTfLiteError);

  status = m.Invoke(1, {0, 3, 2, 0},
                    {1.f, 1.f, 1.f, 1.f,  //
                     2.f, 2.f, 2.f, 2.f,  //
                     3.f, 3.f, 3.f, 3.f,  //
                     4.f, 4.f});
  EXPECT_EQ(status, kTfLiteOk);
}

}  // namespace

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
