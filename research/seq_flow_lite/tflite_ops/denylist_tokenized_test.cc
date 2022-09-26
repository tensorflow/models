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
#include "tflite_ops/denylist_tokenized.h"  // seq_flow_lite

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
using ::seq_flow_lite::testing::Int64Tensor;
using ::seq_flow_lite::testing::OpEquivTestCase;
using ::seq_flow_lite::testing::StringTensor;
using ::seq_flow_lite::testing::TensorflowTfLiteOpTest;
using ::testing::ElementsAreArray;
using ::tflite::GetTensorType;
using ::tflite::SingleOpModel;
using ::tflite::TensorData;
using ::tflite::TensorType_STRING;
using ::tflite::TensorType_UINT8;

template <typename Toutput, typename Tcount>
class TokenizedDenylistModel : public SingleOpModel {
 public:
  TokenizedDenylistModel(const std::vector<std::string>& denylist,
                         const std::vector<int>& denylist_category,
                         int categories, int negative_categories,
                         int max_skip_size,
                         const std::vector<int> input_shape) {
    input_ = AddInput(TensorType_STRING);
    token_count_ = AddInput(GetTensorType<Tcount>());

    TensorData output_data = {GetTensorType<Toutput>()};
    if (output_data.type == TensorType_UINT8) {
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
    SetCustomOp("TOKENIZED_DENYLIST", fbb.GetBuffer(),
                Register_TOKENIZED_DENYLIST);

    input_shape_ = input_shape;
    categories_ = categories;
    std::vector<int> token_count_shape = input_shape;
    token_count_shape.pop_back();
    BuildInterpreter({input_shape, token_count_shape});
  }

  std::vector<Toutput> Invoke(const std::vector<std::string>& input) {
    PopulateStringTensor(input_, input);

    std::vector<Tcount> token_count;
    int max_tokens = input_shape_.back();
    for (int i = 0; i < input.size(); i += max_tokens) {
      int j = max_tokens;
      while (input[i + j - 1].empty() && j > 0) {
        j--;
      }
      token_count.push_back(j);
    }
    PopulateTensor(token_count_, token_count);

    CHECK_EQ(SingleOpModel::Invoke(), kTfLiteOk);

    // Check that the output tensor has the correct shape:
    //   [<input_shape_ - token_dimension>, categories_]
    std::vector<int> output_shape = GetTensorShape(output_);
    std::vector<int> expected_output_shape = input_shape_;
    expected_output_shape.pop_back();
    expected_output_shape.push_back(categories_);
    EXPECT_THAT(output_shape, ElementsAreArray(expected_output_shape));

    return ExtractVector<Toutput>(output_);
  }

 private:
  int input_;
  int token_count_;
  int output_;
  int categories_;
  std::vector<int> input_shape_;
};

TEST(TokenizedDenylistTest, Unquantized) {
  TokenizedDenylistModel<float, int64_t> m({"a b c"}, {1}, 2, 1, 1, {2, 1, 7});
  EXPECT_THAT(m.Invoke({"q", "a", "q", "b", "q", "c", "q",  //
                        "q", "a", "b", "q", "q", "c", ""}),
              ElementsAreArray({0.0, 1.0, 1.0, 0.0}));
}

TEST(TokenizedDenylistTest, Quantized) {
  TokenizedDenylistModel<uint8_t, int64_t> m({"a b c"}, {1}, 2, 1, 1,
                                             {1, 2, 7});
  EXPECT_THAT(m.Invoke({"q", "a", "q", "b", "q", "c", "q",  //
                        "q", "a", "b", "q", "q", "c", ""}),
              ElementsAreArray({0, 4, 4, 0}));
}

TEST(TokenizedDenylistTest, Prefix) {
  TokenizedDenylistModel<float, int64_t> m({"a b.* c"}, {1}, 2, 1, 1,
                                           {2, 1, 7});
  EXPECT_THAT(m.Invoke({"q", "a", "q", "bq", "q", "c", "q",  //
                        "q", "a", "bq", "q", "q", "c", ""}),
              ElementsAreArray({0.0, 1.0, 1.0, 0.0}));
}

TEST(TokenizedDenylistTest, Int32TokenCount) {
  TokenizedDenylistModel<float, int32_t> m({"a b c"}, {1}, 2, 1, 1, {2, 1, 7});
  EXPECT_THAT(m.Invoke({"q", "a", "q", "b", "q", "c", "q",  //
                        "q", "a", "b", "q", "q", "c", ""}),
              ElementsAreArray({0.0, 1.0, 1.0, 0.0}));
}

using Model = TokenizedDenylistModel<uint8_t, int64_t>;

TEST(TokenizedDenylistDeathTest, ZeroCategories) {
  EXPECT_DEATH(Model m({"a b c"}, {1}, 0, -1, 1, {1, 2}),
               "categories \\(0\\) <= 0");
}

TEST(TokenizedDenylistDeathTest, NegativeCategoriesLessThanZero) {
  EXPECT_DEATH(Model m({"a b c"}, {1}, 1, -1, 1, {1, 2}),
               "negative_categories \\(-1\\) <= 0");
}

TEST(TokenizedDenylistDeathTest, AllNegativeCategories) {
  EXPECT_DEATH(Model m({"a b c"}, {1}, 1, 1, 1, {1, 2}),
               "negative_categories \\(1\\) >= categories \\(1\\)");
}

class TokenizedDenylistEquivTest : public TensorflowTfLiteOpTest {
  std::function<TfLiteRegistration*()> TfLiteOpRegistration() override {
    return Register_TOKENIZED_DENYLIST;
  }
  std::string TensorflowOpName() override { return "TokenizedDenylist"; }
};

TEST_P(TokenizedDenylistEquivTest, Compare) {
  RunTensorflowOp();
  RunTfLiteOp();
  CompareOpOutput();
}

std::vector<OpEquivTestCase> TokenizedDenylistEquivTestCases() {
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
    test_case.input_tensors.push_back(
        StringTensor({2, 7}, {"q", "a", "q", "b", "q", "c", "q",  //
                              "q", "a", "b", "q", "q", "c", ""}));
    test_case.input_tensors.push_back(Int64Tensor({2}, {7, 6}));
    test_case.output_tensors.emplace_back(FloatTensor({2, 2}, {}));
    test_cases.push_back(test_case);
  }
  return test_cases;
}

INSTANTIATE_TEST_SUITE_P(
    TokenizedDenylistEquivTest, TokenizedDenylistEquivTest,
    ::testing::ValuesIn(TokenizedDenylistEquivTestCases()));

}  // namespace
}  // namespace custom
}  // namespace ops
}  // namespace seq_flow_lite
