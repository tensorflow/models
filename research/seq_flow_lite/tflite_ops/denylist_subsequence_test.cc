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
#include "tflite_ops/denylist_subsequence.h"  // seq_flow_lite

#include <functional>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include "flatbuffers/flexbuffers.h"  // flatbuffer
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/kernels/ops_testutil.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/kernels/test_util.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tflite_ops/tf_tflite_diff_test_util.h"  // seq_flow_lite

namespace seq_flow_lite {
namespace ops {
namespace custom {
namespace {

using ::seq_flow_lite::testing::AttrValue;
using ::seq_flow_lite::testing::FloatTensor;
using ::seq_flow_lite::testing::OpEquivTestCase;
using ::seq_flow_lite::testing::StringTensor;
using ::seq_flow_lite::testing::TensorflowTfLiteOpTest;
using ::testing::ElementsAreArray;
using ::tflite::SingleOpModel;
using ::tflite::TensorData;
using ::tflite::TensorType;
using ::tflite::TensorType_FLOAT32;
using ::tflite::TensorType_STRING;
using ::tflite::TensorType_UINT8;

class SubsequenceDenylistModel : public SingleOpModel {
 public:
  SubsequenceDenylistModel(const std::vector<std::string>& denylist,
                           const std::vector<int>& denylist_category,
                           int categories, int negative_categories,
                           int max_skip_size, TensorType output_type,
                           const std::vector<int> input_shape) {
    input_ = AddInput(TensorType_STRING);

    TensorData output_data = {output_type};
    if (output_type == TensorType_UINT8) {
      output_data.zero_point = 0;
      output_data.scale = 0.25;
    }
    output_ = AddOutput(output_data);

    flexbuffers::Builder fbb;
    size_t start = fbb.StartMap();
    {
      size_t start = fbb.StartVector("denylist");
      for (const std::string& skipgram : denylist) {
        fbb.String(skipgram);
      }
      fbb.EndVector(start, /*typed=*/true, /*fixed=*/false);
    }
    {
      size_t start = fbb.StartVector("denylist_category");
      for (int category : denylist_category) {
        fbb.Int(category);
      }
      fbb.EndVector(start, /*typed=*/true, /*fixed=*/false);
    }
    fbb.Int("categories", categories);
    fbb.Int("negative_categories", negative_categories);
    fbb.Int("max_skip_size", max_skip_size);
    fbb.EndMap(start);
    fbb.Finish();
    SetCustomOp("SUBSEQUENCE_DENYLIST", fbb.GetBuffer(),
                Register_SUBSEQUENCE_DENYLIST);

    input_shape_ = input_shape;
    categories_ = categories;
    BuildInterpreter({input_shape});
  }

  template <typename T>
  std::vector<T> Invoke(const std::vector<std::string>& input) {
    PopulateStringTensor(input_, input);
    CHECK_EQ(SingleOpModel::Invoke(), kTfLiteOk);

    // Check that the output tensor has the correct shape:
    //   [<input_shape_>, categories_]
    std::vector<int> output_shape = GetTensorShape(output_);
    EXPECT_EQ(output_shape.size(), input_shape_.size() + 1);
    for (int i = 0; i < input_shape_.size(); i++) {
      EXPECT_EQ(output_shape[i], input_shape_[i]);
    }
    EXPECT_EQ(output_shape[input_shape_.size()], categories_);
    return ExtractVector<T>(output_);
  }

 private:
  int input_;
  int output_;
  int categories_;
  std::vector<int> input_shape_;
};

TEST(SubsequenceDenylistTest, Unquantized) {
  SubsequenceDenylistModel m({"a b c"}, {1}, 2, 1, 1, TensorType_FLOAT32,
                             {2, 1});
  EXPECT_THAT(m.Invoke<float>({"qaqbqcq", "qabqqc"}),
              ElementsAreArray({0.0, 1.0, 1.0, 0.0}));
}

TEST(SubsequenceDenylistTest, Quantized) {
  SubsequenceDenylistModel m({"a b c"}, {1}, 2, 1, 1, TensorType_UINT8, {1, 2});
  EXPECT_THAT(m.Invoke<uint8_t>({"qaqbqcq", "qabqqc"}),
              ElementsAreArray({0, 4, 4, 0}));
}

TEST(SubsequenceDenylistDeathTest, ZeroCategories) {
  EXPECT_DEATH(SubsequenceDenylistModel m({"a b c"}, {1}, 0, -1, 1,
                                          TensorType_UINT8, {1, 2}),
               "categories \\(0\\) <= 0");
}

TEST(SubsequenceDenylistDeathTest, NegativeCategoriesLessThanZero) {
  EXPECT_DEATH(SubsequenceDenylistModel m({"a b c"}, {1}, 1, -1, 1,
                                          TensorType_UINT8, {1, 2}),
               "negative_categories \\(-1\\) <= 0");
}

TEST(SubsequenceDenylistDeathTest, AllNegativeCategories) {
  EXPECT_DEATH(SubsequenceDenylistModel m({"a b c"}, {1}, 1, 1, 1,
                                          TensorType_UINT8, {1, 2}),
               "negative_categories \\(1\\) >= categories \\(1\\)");
}

class SubsequenceDenylistEquivTest : public TensorflowTfLiteOpTest {
  std::function<TfLiteRegistration*()> TfLiteOpRegistration() override {
    return Register_SUBSEQUENCE_DENYLIST;
  }
  std::string TensorflowOpName() override { return "SubsequenceDenylist"; }
};

TEST_P(SubsequenceDenylistEquivTest, Compare) {
  RunTensorflowOp();
  RunTfLiteOp();
  CompareOpOutput();
}

std::vector<OpEquivTestCase> SubsequenceDenylistEquivTestCases() {
  std::vector<OpEquivTestCase> test_cases;
  {
    // Check TF and TFLite op equivalence with a simple denylist.
    OpEquivTestCase test_case;
    test_case.test_name = "Simple";
    test_case.attributes["max_skip_size"] = AttrValue(1);
    test_case.attributes["denylist"] =
        AttrValue(std::vector<std::string>({"a b c"}));
    test_case.attributes["denylist_category"] =
        AttrValue(std::vector<int>({1}));
    test_case.attributes["categories"] = AttrValue(2);
    test_case.attributes["negative_categories"] = AttrValue(1);
    test_case.input_tensors.push_back(StringTensor({2}, {"qaqbqcq", "qabqqc"}));
    test_case.output_tensors.emplace_back(FloatTensor({2}, {}));
    test_cases.push_back(test_case);
  }
  return test_cases;
}

INSTANTIATE_TEST_SUITE_P(
    SubsequenceDenylistEquivTest, SubsequenceDenylistEquivTest,
    ::testing::ValuesIn(SubsequenceDenylistEquivTestCases()));

}  // namespace
}  // namespace custom
}  // namespace ops
}  // namespace seq_flow_lite
