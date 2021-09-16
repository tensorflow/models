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
// Tests equivalence between TF and TFLite versions of an op.

#ifndef TENSORFLOW_MODELS_SEQUENCE_PROJECTION_TFLITE_OPS_TF_TFLITE_DIFF_TEST_UTIL_H_
#define TENSORFLOW_MODELS_SEQUENCE_PROJECTION_TFLITE_OPS_TF_TFLITE_DIFF_TEST_UTIL_H_

#include <string>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "tensorflow/core/framework/attr_value_util.h"
#include "tensorflow/core/kernels/ops_testutil.h"
#include "tensorflow/lite/kernels/test_util.h"

namespace seq_flow_lite {
namespace testing {

// Convenience constructors.
template <typename T>
::tensorflow::AttrValue AttrValue(T value) {
  ::tensorflow::AttrValue attr_value;
  ::tensorflow::SetAttrValue(value, &attr_value);
  return attr_value;
}
::tensorflow::TensorProto BoolTensor(const std::vector<int>& shape,
                                     const std::vector<bool>& values);
::tensorflow::TensorProto IntTensor(const std::vector<int>& shape,
                                    const std::vector<int>& values);
::tensorflow::TensorProto FloatTensor(const std::vector<int>& shape,
                                      const std::vector<float>& values);
::tensorflow::TensorProto StringTensor(const std::vector<int>& shape,
                                       const std::vector<std::string>& values);

struct OutputTensor {
  explicit OutputTensor(const ::tensorflow::TensorProto& tensor)
      : tensor(tensor) {
    quantization_params.scale = 0.0;
  }
  OutputTensor(const ::tensorflow::TensorProto& tensor, float scale,
               int zero_point)
      : tensor(tensor) {
    quantization_params.scale = scale;
    quantization_params.zero_point = zero_point;
  }

  ::tensorflow::TensorProto tensor;
  TfLiteQuantizationParams quantization_params;
};

struct OpEquivTestCase {
  std::string test_name;
  absl::flat_hash_map<std::string, ::tensorflow::AttrValue> attributes;
  std::vector<::tensorflow::TensorProto> input_tensors;
  std::vector<OutputTensor> output_tensors;
};

// Convert Tensorflow attributes into an equivalent TFLite flatbuffer.  Adds the
// default attribute values from `tensorflow_op`, if they are not set in
// `attributes`.
std::vector<uint8_t> ConstructTfLiteCustomOptions(
    absl::flat_hash_map<std::string, ::tensorflow::AttrValue> attributes,
    const std::string& tensorflow_op);

// A test class that can be used to compare that a Tensorflow op and a
// TFLite op are producing the same output.
//
// To use:
// 1) Sub-class TensorflowTfLiteOpTest.
//    Define TfLiteOpRegistration() and TensorflowOpName().
//
//    class NewOpEquivTest : public TensorflowTfLiteOpTest {
//      std::function<TfLiteRegistration*()> TfLiteOpRegistration() override {
//        return ::tflite::custom::Register_NEW_OP;
//      }
//      std::string TensorflowOpName() override { return "NewOp"; }
//    };
//
// 2) Declare a TEST_P (parameterized test) to perform the comparison.
//
//    TEST_P(NewOpEquivTest, Compare) {
//      RunTensorflowOp();
//      RunTfLiteOp();
//      CompareOpOutput();
//    }
//
// 3) Define your test cases.
//
//    std::vector<OpEquivTestCase> NewEquivOpTestCases() {
//      std::vector<OpEquivTestCase> test_cases;
//      {
//        OpEquivTestCase test_case;
//        test_case.test_name = "Simple";
//        test_case.attributes["int_attr"] = AttrValue(1);
//        test_case.attributes["bool_attr"] = AttrValue(true);
//        test_case.input_tensor.push_back(StringTensor({1, 2}, {"a", "b"}));
//        test_case.output_tensors.emplace_back(FloatTensor({}, {}));
//        test_cases.push_back(test_case);
//      }
//      return test_cases;
//    }
//
// 4) Instantiate your tests.
//
//    INSTANTIATE_TEST_SUITE_P(
//        NewOpEquivTest,
//        NewOpEquivTest,
//        ::testing::ValuesIn(NewOpEquivTestCases()),
//        ::expander::GetTestName());
class TensorflowTfLiteOpTest
    : public ::tensorflow::OpsTestBase,
      public ::testing::WithParamInterface<OpEquivTestCase> {
 protected:
  void SetUp() override;

  virtual void ConstructTensorflowOp();
  virtual void RunTensorflowOp();

  virtual void ConstructTfLiteOp();
  virtual void RunTfLiteOp();

  virtual void CompareOpOutput();

  virtual std::function<TfLiteRegistration*()> TfLiteOpRegistration() = 0;
  virtual std::string TfLiteOpName() { return "TestOp"; }
  virtual std::string TensorflowOpName() = 0;

 private:
  ::tflite::SingleOpModel tflite_op_;
  std::vector<int> tflite_inputs_;
  std::vector<int> tflite_outputs_;
};

}  // namespace testing
}  // namespace seq_flow_lite

#endif  // TENSORFLOW_MODELS_SEQUENCE_PROJECTION_TFLITE_OPS_TF_TFLITE_DIFF_TEST_UTIL_H_
