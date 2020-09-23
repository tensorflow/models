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

#include "sgnn/sgnn_projection.h"  // sequence_projection

#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "flatbuffers/flexbuffers.h"  // flatbuffer
#include "tensorflow/lite/kernels/test_util.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/string_util.h"

namespace tflite {
namespace ops {
namespace custom {
namespace sgnn_projection {
namespace test {
namespace {

using ::testing::ElementsAre;
using ::testing::ElementsAreArray;

}  // namespace

class SgnnProjectionModel : public SingleOpModel {
 public:
  // Constructor for testing the op with a tf.Tensor
  SgnnProjectionModel(const std::vector<std::string>& input_values,
                      const std::vector<int64_t>& input_row_splits,
                      const std::vector<int64_t>& hash_seed, int64_t buckets) {
    input_values_index_ = AddInput(TensorType_STRING);
    input_row_splits_index_ = AddInput(TensorType_INT64);
    output_values_index_ = AddOutput(TensorType_FLOAT32);
    BuildCustomOp(hash_seed, buckets);
    BuildInterpreter({{static_cast<int>(input_values.size())},
                      {static_cast<int>(input_row_splits.size())}});
    PopulateStringTensor(input_values_index_, input_values);
    PopulateTensor(input_row_splits_index_, input_row_splits);
    Invoke();
  }

  std::vector<int> GetOutputShape() {
    return GetTensorShape(output_values_index_);
  }

  std::vector<float> ExtractOutputValue() {
    return ExtractVector<float>(output_values_index_);
  }

 private:
  void BuildCustomOp(const std::vector<int64_t>& hash_seed, int64_t buckets) {
    flexbuffers::Builder fbb;
    size_t start_map = fbb.StartMap();
    auto vector_start = fbb.StartVector("hash_seed");
    for (int i = 0; i < hash_seed.size(); i++) {
      fbb.Add(hash_seed[i]);
    }
    fbb.EndVector(vector_start, /*typed=*/true, /*fixed=*/false);
    fbb.Int("buckets", buckets);
    fbb.EndMap(start_map);
    fbb.Finish();
    SetCustomOp("tftext:custom:SgnnProjection", fbb.GetBuffer(),
                Register_tftext_SGNN_PROJECTION);
  }

  int input_values_index_;
  int input_row_splits_index_;
  int output_values_index_;
};

// Keep same result of test_projection in sgnn_test.py
TEST(SgnnProjectionTest, TensorSgnnProjection) {
  SgnnProjectionModel m({"^h", "he", "el", "ll", "lo", "o$", "^h", "hi", "i$"},
                        /*input_row_splits=*/{0, 6, 9}, /*hash_seed=*/{5, 7},
                        /*buckets=*/0x7FFFFFFF);
  EXPECT_THAT(m.GetOutputShape(), ElementsAre(2, 2));
  EXPECT_THAT(m.ExtractOutputValue(),
              ElementsAreArray(ArrayFloatNear(
                  { 0.448691, -0.238499, -0.037561,  0.080748})));
}

}  // namespace test
}  // namespace sgnn_projection
}  // namespace custom
}  // namespace ops
}  // namespace tflite
