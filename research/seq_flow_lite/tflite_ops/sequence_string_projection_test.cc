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
#include "tflite_ops/sequence_string_projection.h"  // seq_flow_lite

#include <vector>

#include "flatbuffers/flexbuffers.h"  // flatbuffer
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/kernels/test_util.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/string_util.h"
#include "tf_ops/projection_util.h"  // seq_flow_lite
#include "tflite_ops/tf_tflite_diff_test_util.h"  // seq_flow_lite

namespace seq_flow_lite {

namespace ops {
namespace custom {

namespace {

using ::seq_flow_lite::testing::AttrValue;
using ::seq_flow_lite::testing::FloatTensor;
using ::seq_flow_lite::testing::IntTensor;
using ::seq_flow_lite::testing::OpEquivTestCase;
using ::seq_flow_lite::testing::StringTensor;
using ::seq_flow_lite::testing::TensorflowTfLiteOpTest;
using ::testing::ElementsAreArray;
using ::testing::Not;
using ::tflite::TensorType_FLOAT32;
using ::tflite::TensorType_STRING;
using ::tflite::TensorType_UINT8;

class SequenceStringProjectionModel : public ::tflite::SingleOpModel {
 public:
  explicit SequenceStringProjectionModel(
      bool split_on_space, int max_splits, int word_novelty_bits,
      int doc_size_levels, bool add_eos_tag, ::tflite::TensorType output_type,
      const std::string& token_separators = "",
      bool normalize_repetition = false, float add_first_cap = 0.0,
      float add_all_caps = 0.0, const std::string& hashtype = kMurmurHash,
      bool normalize_spaces = false) {
    flexbuffers::Builder fbb;
    fbb.Map([&] {
      fbb.Int("feature_size", 4);
      fbb.String("vocabulary", "abcdefghijklmnopqrstuvwxyz");
      fbb.Int("word_novelty_bits", word_novelty_bits);
      fbb.Int("doc_size_levels", doc_size_levels);
      fbb.Int("max_splits", max_splits);
      fbb.Bool("split_on_space", split_on_space);
      fbb.Bool("add_eos_tag", add_eos_tag);
      fbb.String("token_separators", token_separators);
      fbb.String("hashtype", hashtype);
      fbb.Bool("normalize_repetition", normalize_repetition);
      fbb.Float("add_first_cap_feature", add_first_cap);
      fbb.Float("add_all_caps_feature", add_all_caps);
      fbb.Bool("normalize_spaces", normalize_spaces);
    });
    fbb.Finish();
    output_ = AddOutput({output_type, {}});
    SetCustomOp(kSequenceStringProjection, fbb.GetBuffer(),
                Register_SEQUENCE_STRING_PROJECTION);
    BuildInterpreter({GetShape(input_)});
  }
  void Invoke(const std::string& input) {
    PopulateStringTensor(input_, {input});
    CHECK(interpreter_->AllocateTensors() == kTfLiteOk)
        << "Cannot allocate tensors";
    CHECK_EQ(SingleOpModel::Invoke(), kTfLiteOk);
  }
  TfLiteStatus InvokeFailable(const std::string& input) {
    PopulateStringTensor(input_, {input});
    CHECK(interpreter_->AllocateTensors() == kTfLiteOk)
        << "Cannot allocate tensors";
    return SingleOpModel::Invoke();
  }

  template <typename T>
  std::vector<T> GetOutput() {
    return ExtractVector<T>(output_);
  }

  void CheckOutputTensorShape(const std::vector<int>& expected_shape) {
    EXPECT_EQ(GetTensorShape(output_), expected_shape);
  }

 private:
  int input_ = AddInput(TensorType_STRING);
  int output_;
};

TEST(SequenceStringProjectionTest, IncorrectHashtype) {
  SequenceStringProjectionModel m(true, -1, 0, 0, true, TensorType_UINT8, "",
                                  false, 0.0, 0.0, "unsupported");
  EXPECT_EQ(m.InvokeFailable(" "), kTfLiteError);
}

TEST(SequenceStringProjectionTest, RegularInputUint8) {
  std::vector<std::pair<std::string, std::vector<uint8_t>>> testcase = {
      {"hello", {127, 255, 255, 127, 127, 255, 127, 127}},
      {"world", {127, 255, 127, 127, 127, 255, 127, 127}},
  };
  for (const auto& test : testcase) {
    SequenceStringProjectionModel m(true, -1, 0, 0, true, TensorType_UINT8);
    m.Invoke(test.first);
    EXPECT_THAT(m.GetOutput<uint8_t>(), ElementsAreArray(test.second));
  }
}

TEST(SequenceStringProjectionTest, RegularInputUint8NoEOSTag) {
  std::vector<std::pair<std::string, std::vector<uint8_t>>> testcase = {
      {"hello", {127, 255, 255, 127}},
      {"world", {127, 255, 127, 127}},
  };
  for (const auto& test : testcase) {
    SequenceStringProjectionModel m(true, -1, 0, 0, false, TensorType_UINT8);
    m.Invoke(test.first);
    EXPECT_THAT(m.GetOutput<uint8_t>(), ElementsAreArray(test.second));
  }
}

TEST(SequenceStringProjectionTest, RegularInputUint8DocSize) {
  std::vector<std::pair<std::string, std::vector<uint8_t>>> testcase = {
      {"hello", {127, 255, 0, 127, 127, 255, 0, 127}},
      {"world", {127, 255, 0, 127, 127, 255, 0, 127}},
  };
  for (const auto& test : testcase) {
    SequenceStringProjectionModel m(true, -1, 0, 8, true, TensorType_UINT8);
    m.Invoke(test.first);
    EXPECT_THAT(m.GetOutput<uint8_t>(), ElementsAreArray(test.second));
  }
}

TEST(SequenceStringProjectionTest, RegularInputUint8DocSizeWordNovelty) {
  std::vector<std::pair<std::string, std::vector<uint8_t>>> testcase = {
      {"hello", {127, 255, 0, 0, 127, 255, 0, 0}},
      {"world", {127, 255, 0, 0, 127, 255, 0, 0}},
  };
  for (const auto& test : testcase) {
    SequenceStringProjectionModel m(true, -1, 4, 8, true, TensorType_UINT8);
    m.Invoke(test.first);
    EXPECT_THAT(m.GetOutput<uint8_t>(), ElementsAreArray(test.second));
  }
}

TEST(SequenceStringProjectionTest, RegularInputUint8WordNovelty) {
  std::vector<std::pair<std::string, std::vector<uint8_t>>> testcase = {
      {"hello", {127, 255, 255, 0, 127, 255, 127, 0}},
      {"world", {127, 255, 127, 0, 127, 255, 127, 0}},
  };
  for (const auto& test : testcase) {
    SequenceStringProjectionModel m(true, -1, 3, 0, true, TensorType_UINT8);
    m.Invoke(test.first);
    EXPECT_THAT(m.GetOutput<uint8_t>(), ElementsAreArray(test.second));
  }
}

TEST(SequenceStringProjectionTest, RegularInputFloat) {
  std::vector<std::pair<std::string, std::vector<float>>> testcase = {
      {"hello", {0, 1, 1, 0, 0, 1, 0, 0}},
      {"world", {0, 1, 0, 0, 0, 1, 0, 0}},
  };
  for (const auto& test : testcase) {
    SequenceStringProjectionModel m(true, -1, 0, 0, true, TensorType_FLOAT32);
    m.Invoke(test.first);
    EXPECT_THAT(m.GetOutput<float>(), ElementsAreArray(test.second));
  }
}

TEST(SequenceStringProjectionTest, RegularInputFloatNoEOSTag) {
  std::vector<std::pair<std::string, std::vector<float>>> testcase = {
      {"hello", {0, 1, 1, 0}},
      {"world", {0, 1, 0, 0}},
  };
  for (const auto& test : testcase) {
    SequenceStringProjectionModel m(true, -1, 0, 0, false, TensorType_FLOAT32);
    m.Invoke(test.first);
    EXPECT_THAT(m.GetOutput<float>(), ElementsAreArray(test.second));
  }
}

TEST(SequenceStringProjectionTest, RegularInputWithoutSplitOnSpace) {
  std::vector<std::pair<std::string, std::vector<uint8_t>>> testcase = {
      {"h", {127, 127, 255, 127, 127, 255, 127, 127}},
      {"w", {255, 127, 255, 127, 127, 255, 127, 127}},
  };
  for (const auto& test : testcase) {
    SequenceStringProjectionModel m(false, -1, 0, 0, true, TensorType_UINT8);
    m.Invoke(test.first);
    EXPECT_THAT(m.GetOutput<uint8_t>(), ElementsAreArray(test.second));
  }
}

TEST(SequenceStringProjectionTest, CheckSequenceLimit) {
  std::string input;
  for (int i = 0; i < 600; ++i) {
    input += "hello world ";
  }
  SequenceStringProjectionModel m(true, 511, 0, 0, true, TensorType_UINT8);
  m.Invoke(input);
  const std::vector<int> expected_shape = {1, 512, 4};
  m.CheckOutputTensorShape(expected_shape);
}

TEST(SequenceStringProjectionTest, CheckSequenceLimitBoundary) {
  std::vector<std::pair<std::string, std::vector<int>>> testcase = {
      {"hello", {1, 2, 4}},
      {"hello ", {1, 2, 4}},
      {"hello world", {1, 3, 4}},
      {"hellow world ", {1, 3, 4}},
  };
  for (const auto& test : testcase) {
    SequenceStringProjectionModel m(true, 2, 0, 0, true, TensorType_FLOAT32);
    m.Invoke(test.first);
    m.CheckOutputTensorShape(test.second);
  }
}

TEST(SequenceStringProjectionTest, CheckSequenceLimitBoundaryWithoutSpace) {
  std::vector<std::pair<std::string, std::vector<int>>> testcase = {
      {"h", {1, 2, 4}},
      {"he", {1, 3, 4}},
      {"hel", {1, 3, 4}},
      {"hello ", {1, 3, 4}},
  };
  for (const auto& test : testcase) {
    SequenceStringProjectionModel m(false, 2, 0, 0, true, TensorType_UINT8);
    m.Invoke(test.first);
    m.CheckOutputTensorShape(test.second);
  }
}

TEST(SequenceStringProjectionTest,
     CheckSequenceLimitBoundaryWithoutSpaceNoEOS) {
  std::vector<std::pair<std::string, std::vector<int>>> testcase = {
      {"h", {1, 1, 4}},
      {"he", {1, 2, 4}},
      {"hel", {1, 2, 4}},
      {"hello ", {1, 2, 4}},
  };
  for (const auto& test : testcase) {
    SequenceStringProjectionModel m(false, 2, 0, 0, false, TensorType_UINT8);
    m.Invoke(test.first);
    m.CheckOutputTensorShape(test.second);
  }
}

TEST(SequenceStringProjectionTest, TokenSeparators) {
  // Separate the input using "!".
  SequenceStringProjectionModel m1(true, -1, 0, 0, true, TensorType_UINT8, "!",
                                   false);
  m1.Invoke("great!!!");
  auto output1 = m1.GetOutput<uint8_t>();

  SequenceStringProjectionModel m2(true, -1, 0, 0, true, TensorType_UINT8, "!",
                                   false);
  m2.Invoke("great ! ! !");
  auto output2 = m2.GetOutput<uint8_t>();

  EXPECT_THAT(output1, ElementsAreArray(output2));
}

TEST(SequenceStringProjectionTest, EmptyInput) {
  // Separate the input using "!".
  SequenceStringProjectionModel no_eos(true, -1, 0, 0, false, TensorType_UINT8,
                                       " ", false);
  EXPECT_EQ(no_eos.InvokeFailable(" "), kTfLiteError);
  EXPECT_EQ(no_eos.InvokeFailable("   "), kTfLiteError);
  EXPECT_EQ(no_eos.InvokeFailable(""), kTfLiteError);
  EXPECT_EQ(no_eos.InvokeFailable("hello"), kTfLiteOk);

  SequenceStringProjectionModel with_eos(true, -1, 0, 0, true, TensorType_UINT8,
                                         " ", false);
  EXPECT_EQ(with_eos.InvokeFailable(" "), kTfLiteOk);
  EXPECT_EQ(with_eos.InvokeFailable("   "), kTfLiteOk);
  EXPECT_EQ(with_eos.InvokeFailable(""), kTfLiteOk);
  EXPECT_EQ(with_eos.InvokeFailable("hello"), kTfLiteOk);
}

TEST(SequenceStringProjectionTest, FirstCap) {
  SequenceStringProjectionModel op(/*split_on_space=*/true, /*max_splits=*/-1,
                                   /*word_novelty_bits=*/0,
                                   /*doc_size_levels=*/0, /*add_eos_tag=*/false,
                                   /*output_type=*/TensorType_UINT8,
                                   /*token_separators=*/" ",
                                   /*normalize_repetition=*/false,
                                   /*add_first_cap=*/0.5);
  op.Invoke("hello");
  auto output1 = op.GetOutput<uint8_t>();

  op.Invoke("Hello");
  auto output2 = op.GetOutput<uint8_t>();

  EXPECT_NE(output1[1], output2[1]);
}

TEST(SequenceStringProjectionTest, AllCaps) {
  SequenceStringProjectionModel op(
      /*split_on_space=*/true, /*max_splits=*/-1, /*word_novelty_bits=*/0,
      /*doc_size_levels=*/0, /*add_eos_tag=*/false,
      /*output_type=*/TensorType_UINT8, /*token_separators=*/" ",
      /*normalize_repetition=*/false, /*add_first_cap=*/0.0,
      /*add_all_caps=*/0.5);
  op.Invoke("hello");
  auto output1 = op.GetOutput<uint8_t>();

  op.Invoke("HELLO");
  auto output2 = op.GetOutput<uint8_t>();

  EXPECT_NE(output1[0], output2[0]);
}

TEST(SequenceStringProjectionTest, NormalizeRepetition) {
  // Normalize the repeated special tokens. Used for the emotion models.
  SequenceStringProjectionModel m1(true, -1, 0, 0, true, TensorType_UINT8, "",
                                   true);
  m1.Invoke("hello..");
  auto output1 = m1.GetOutput<uint8_t>();

  SequenceStringProjectionModel m2(true, -1, 0, 0, true, TensorType_UINT8, "",
                                   true);
  m2.Invoke("hello.....");
  auto output2 = m2.GetOutput<uint8_t>();

  EXPECT_THAT(output1, ElementsAreArray(output2));
}

TEST(SequenceStringProjectionTest, NormalizeSpaces) {
  SequenceStringProjectionModel model_nonormalize(false, -1, 0, 0, false,
                                                  TensorType_UINT8, "", false,
                                                  0.0, 0.0, kMurmurHash, false);
  SequenceStringProjectionModel model_normalize(false, -1, 0, 0, false,
                                                TensorType_UINT8, "", false,
                                                0.0, 0.0, kMurmurHash, true);

  const char kNoExtraSpaces[] = "Hello there.";
  const char kExtraSpaces[] = " Hello   there.  ";

  model_nonormalize.Invoke(kNoExtraSpaces);
  auto output_noextra_nonorm = model_nonormalize.GetOutput<uint8_t>();
  model_nonormalize.Invoke(kExtraSpaces);
  auto output_extra_nonorm = model_nonormalize.GetOutput<uint8_t>();
  model_normalize.Invoke(kNoExtraSpaces);
  auto output_noextra_norm = model_normalize.GetOutput<uint8_t>();
  model_normalize.Invoke(kExtraSpaces);
  auto output_extra_norm = model_normalize.GetOutput<uint8_t>();

  EXPECT_THAT(output_noextra_nonorm, ElementsAreArray(output_noextra_norm));
  EXPECT_THAT(output_noextra_nonorm, ElementsAreArray(output_extra_norm));
  EXPECT_THAT(output_noextra_nonorm,
              Not(ElementsAreArray(output_extra_nonorm)));
}

class SequenceStringProjectionTest : public TensorflowTfLiteOpTest {
  std::function<TfLiteRegistration*()> TfLiteOpRegistration() override {
    return ops::custom::Register_SEQUENCE_STRING_PROJECTION;
  }

  std::string TensorflowOpName() override { return "SequenceStringProjection"; }
};

TEST_P(SequenceStringProjectionTest, TensorflowTfLiteSame) {
  RunTensorflowOp();
  RunTfLiteOp();
  CompareOpOutput();
}

std::vector<OpEquivTestCase> SequenceStringProjectionTestCases() {
  std::vector<OpEquivTestCase> test_cases;
  constexpr float kScale = 2.0 / 255;
  constexpr int kZero = 127;

  {
    OpEquivTestCase test_case;
    test_case.test_name = "CheckEqualityNoBoSNoEoS";
    test_case.attributes["vocabulary"] = AttrValue("");
    test_case.attributes["split_on_space"] = AttrValue(true);
    test_case.attributes["feature_size"] = AttrValue(8);
    test_case.attributes["add_eos_tag"] = AttrValue(false);
    test_case.attributes["add_bos_tag"] = AttrValue(false);
    test_case.input_tensors.push_back(
        StringTensor({1}, {"Hello World 7153845&^$&^$&"}));
    test_case.output_tensors.emplace_back(FloatTensor({}, {}), kScale, kZero);
    test_cases.push_back(test_case);
  }

  {
    OpEquivTestCase test_case;
    test_case.test_name = "CheckEqualityNoBoS";
    test_case.attributes["vocabulary"] = AttrValue("");
    test_case.attributes["split_on_space"] = AttrValue(true);
    test_case.attributes["feature_size"] = AttrValue(8);
    test_case.attributes["add_eos_tag"] = AttrValue(true);
    test_case.attributes["add_bos_tag"] = AttrValue(false);
    test_case.input_tensors.push_back(
        StringTensor({1}, {"Hello World 7153845&^$&^$&"}));
    test_case.output_tensors.emplace_back(FloatTensor({}, {}), kScale, kZero);
    test_cases.push_back(test_case);
  }

  {
    OpEquivTestCase test_case;
    test_case.test_name = "CheckEqualityNoEoS";
    test_case.attributes["vocabulary"] = AttrValue("");
    test_case.attributes["split_on_space"] = AttrValue(true);
    test_case.attributes["feature_size"] = AttrValue(8);
    test_case.attributes["add_eos_tag"] = AttrValue(false);
    test_case.attributes["add_bos_tag"] = AttrValue(true);
    test_case.input_tensors.push_back(
        StringTensor({1}, {"Hello World 7153845&^$&^$&"}));
    test_case.output_tensors.emplace_back(FloatTensor({}, {}), kScale, kZero);
    test_cases.push_back(test_case);
  }

  {
    OpEquivTestCase test_case;
    test_case.test_name = "CheckEquality";
    test_case.attributes["vocabulary"] = AttrValue("");
    test_case.attributes["split_on_space"] = AttrValue(true);
    test_case.attributes["feature_size"] = AttrValue(8);
    test_case.attributes["add_eos_tag"] = AttrValue(true);
    test_case.attributes["add_bos_tag"] = AttrValue(true);
    test_case.input_tensors.push_back(
        StringTensor({1}, {"Hello World 7153845&^$&^$&"}));
    test_case.output_tensors.emplace_back(FloatTensor({}, {}), kScale, kZero);
    test_cases.push_back(test_case);
  }

  {
    OpEquivTestCase test_case;
    test_case.test_name = "SplitOnSpace";
    test_case.attributes["vocabulary"] =
        AttrValue("abcdefghijklmnopqrstuvwxyz");
    test_case.attributes["split_on_space"] = AttrValue(false);
    test_case.attributes["max_splits"] = AttrValue(-1);
    test_case.attributes["word_novelty_bits"] = AttrValue(0);
    test_case.attributes["doc_size_levels"] = AttrValue(0);
    test_case.attributes["feature_size"] = AttrValue(8);
    test_case.attributes["exclude_nonalphaspace_unicodes"] = AttrValue(false);
    test_case.input_tensors.push_back(
        StringTensor({1}, {"Hello World hello world"}));
    test_case.output_tensors.emplace_back(FloatTensor({}, {}), kScale, kZero);
    test_cases.push_back(test_case);
  }

  {
    OpEquivTestCase test_case;
    test_case.test_name = "NoSplitOnSpace";
    test_case.attributes["vocabulary"] =
        AttrValue("abcdefghijklmnopqrstuvwxyz");
    test_case.attributes["split_on_space"] = AttrValue(true);
    test_case.attributes["max_splits"] = AttrValue(-1);
    test_case.attributes["word_novelty_bits"] = AttrValue(0);
    test_case.attributes["doc_size_levels"] = AttrValue(0);
    test_case.attributes["feature_size"] = AttrValue(8);
    test_case.attributes["exclude_nonalphaspace_unicodes"] = AttrValue(false);
    test_case.input_tensors.push_back(
        StringTensor({1}, {"Hello World hello world"}));
    test_case.output_tensors.emplace_back(FloatTensor({}, {}), kScale, kZero);
    test_cases.push_back(test_case);
  }

  {
    OpEquivTestCase test_case;
    test_case.test_name = "SplitOnSpaceWithMax";
    test_case.attributes["vocabulary"] =
        AttrValue("abcdefghijklmnopqrstuvwxyz");
    test_case.attributes["split_on_space"] = AttrValue(true);
    test_case.attributes["max_splits"] = AttrValue(2);
    test_case.attributes["word_novelty_bits"] = AttrValue(0);
    test_case.attributes["doc_size_levels"] = AttrValue(0);
    test_case.attributes["feature_size"] = AttrValue(8);
    test_case.attributes["exclude_nonalphaspace_unicodes"] = AttrValue(false);
    test_case.input_tensors.push_back(
        StringTensor({1}, {"Hello World hello world"}));
    test_case.output_tensors.emplace_back(FloatTensor({}, {}), kScale, kZero);
    test_cases.push_back(test_case);
  }

  {
    OpEquivTestCase test_case;
    test_case.test_name = "NoSplitOnSpaceWithMax";
    test_case.attributes["vocabulary"] =
        AttrValue("abcdefghijklmnopqrstuvwxyz");
    test_case.attributes["split_on_space"] = AttrValue(false);
    test_case.attributes["max_splits"] = AttrValue(4);
    test_case.attributes["word_novelty_bits"] = AttrValue(0);
    test_case.attributes["doc_size_levels"] = AttrValue(0);
    test_case.attributes["feature_size"] = AttrValue(8);
    test_case.attributes["exclude_nonalphaspace_unicodes"] = AttrValue(false);
    test_case.input_tensors.push_back(StringTensor({1}, {"Hello World"}));
    test_case.output_tensors.emplace_back(FloatTensor({}, {}), kScale, kZero);
    test_cases.push_back(test_case);
  }

  {
    OpEquivTestCase test_case;
    test_case.test_name = "NoSplitOnSpaceWithDocSize";
    test_case.attributes["vocabulary"] =
        AttrValue("abcdefghijklmnopqrstuvwxyz");
    test_case.attributes["split_on_space"] = AttrValue(false);
    test_case.attributes["max_splits"] = AttrValue(-1);
    test_case.attributes["word_novelty_bits"] = AttrValue(0);
    test_case.attributes["doc_size_levels"] = AttrValue(6);
    test_case.attributes["feature_size"] = AttrValue(8);
    test_case.attributes["exclude_nonalphaspace_unicodes"] = AttrValue(false);
    test_case.input_tensors.push_back(
        StringTensor({1}, {"Hello World hello world"}));
    test_case.output_tensors.emplace_back(FloatTensor({}, {}), kScale, kZero);
    test_cases.push_back(test_case);
  }

  {
    OpEquivTestCase test_case;
    test_case.test_name = "SplitOnSpaceWithDocSize";
    test_case.attributes["vocabulary"] =
        AttrValue("abcdefghijklmnopqrstuvwxyz");
    test_case.attributes["split_on_space"] = AttrValue(true);
    test_case.attributes["max_splits"] = AttrValue(-1);
    test_case.attributes["word_novelty_bits"] = AttrValue(0);
    test_case.attributes["doc_size_levels"] = AttrValue(7);
    test_case.attributes["feature_size"] = AttrValue(8);
    test_case.attributes["exclude_nonalphaspace_unicodes"] = AttrValue(false);
    test_case.input_tensors.push_back(
        StringTensor({1}, {"Hello World hello world"}));
    test_case.output_tensors.emplace_back(FloatTensor({}, {}), kScale, kZero);
    test_cases.push_back(test_case);
  }

  {
    OpEquivTestCase test_case;
    test_case.test_name = "SplitOnSpaceWithMaxSplitsAndDocSize";
    test_case.attributes["vocabulary"] =
        AttrValue("abcdefghijklmnopqrstuvwxyz");
    test_case.attributes["split_on_space"] = AttrValue(true);
    test_case.attributes["max_splits"] = AttrValue(2);
    test_case.attributes["word_novelty_bits"] = AttrValue(0);
    test_case.attributes["doc_size_levels"] = AttrValue(8);
    test_case.attributes["feature_size"] = AttrValue(8);
    test_case.attributes["exclude_nonalphaspace_unicodes"] = AttrValue(false);
    test_case.input_tensors.push_back(
        StringTensor({1}, {"Hello World hello world"}));
    test_case.output_tensors.emplace_back(FloatTensor({}, {}), kScale, kZero);
    test_cases.push_back(test_case);
  }

  {
    OpEquivTestCase test_case;
    test_case.test_name = "NoSplitOnSpaceWithMaxSplitsAndDocSize";
    test_case.attributes["vocabulary"] =
        AttrValue("abcdefghijklmnopqrstuvwxyz");
    test_case.attributes["split_on_space"] = AttrValue(false);
    test_case.attributes["max_splits"] = AttrValue(4);
    test_case.attributes["word_novelty_bits"] = AttrValue(0);
    test_case.attributes["doc_size_levels"] = AttrValue(4);
    test_case.attributes["feature_size"] = AttrValue(8);
    test_case.attributes["exclude_nonalphaspace_unicodes"] = AttrValue(false);
    test_case.input_tensors.push_back(
        StringTensor({1}, {"Hello World hello world"}));
    test_case.output_tensors.emplace_back(FloatTensor({}, {}), kScale, kZero);
    test_cases.push_back(test_case);
  }

  {
    OpEquivTestCase test_case;
    test_case.test_name = "NoSplitOnSpaceWithWordNovelty";
    test_case.attributes["vocabulary"] =
        AttrValue("abcdefghijklmnopqrstuvwxyz");
    test_case.attributes["split_on_space"] = AttrValue(false);
    test_case.attributes["max_splits"] = AttrValue(-1);
    test_case.attributes["word_novelty_bits"] = AttrValue(2);
    test_case.attributes["doc_size_levels"] = AttrValue(0);
    test_case.attributes["feature_size"] = AttrValue(8);
    test_case.attributes["exclude_nonalphaspace_unicodes"] = AttrValue(false);
    test_case.input_tensors.push_back(
        StringTensor({1}, {"Hello World hello world"}));
    test_case.output_tensors.emplace_back(FloatTensor({}, {}), kScale, kZero);
    test_cases.push_back(test_case);
  }

  {
    OpEquivTestCase test_case;
    test_case.test_name = "SplitOnSpaceWithWordNovelty";
    test_case.attributes["vocabulary"] =
        AttrValue("abcdefghijklmnopqrstuvwxyz");
    test_case.attributes["split_on_space"] = AttrValue(true);
    test_case.attributes["max_splits"] = AttrValue(-1);
    test_case.attributes["word_novelty_bits"] = AttrValue(3);
    test_case.attributes["doc_size_levels"] = AttrValue(0);
    test_case.attributes["feature_size"] = AttrValue(8);
    test_case.attributes["exclude_nonalphaspace_unicodes"] = AttrValue(false);
    test_case.input_tensors.push_back(
        StringTensor({1}, {"Hello World hello world"}));
    test_case.output_tensors.emplace_back(FloatTensor({}, {}), kScale, kZero);
    test_cases.push_back(test_case);
  }

  {
    OpEquivTestCase test_case;
    test_case.test_name = "SplitOnSpaceWithMaxSplitsAndWordNovelty";
    test_case.attributes["vocabulary"] =
        AttrValue("abcdefghijklmnopqrstuvwxyz");
    test_case.attributes["split_on_space"] = AttrValue(true);
    test_case.attributes["max_splits"] = AttrValue(2);
    test_case.attributes["word_novelty_bits"] = AttrValue(4);
    test_case.attributes["doc_size_levels"] = AttrValue(0);
    test_case.attributes["feature_size"] = AttrValue(8);
    test_case.attributes["exclude_nonalphaspace_unicodes"] = AttrValue(false);
    test_case.input_tensors.push_back(
        StringTensor({1}, {"Hello World hello world"}));
    test_case.output_tensors.emplace_back(FloatTensor({}, {}), kScale, kZero);
    test_cases.push_back(test_case);
  }

  {
    OpEquivTestCase test_case;
    test_case.test_name = "NoSplitOnSpaceWithMaxSplitsAndWordNovelty";
    test_case.attributes["vocabulary"] =
        AttrValue("abcdefghijklmnopqrstuvwxyz");
    test_case.attributes["split_on_space"] = AttrValue(false);
    test_case.attributes["max_splits"] = AttrValue(4);
    test_case.attributes["word_novelty_bits"] = AttrValue(5);
    test_case.attributes["doc_size_levels"] = AttrValue(0);
    test_case.attributes["feature_size"] = AttrValue(8);
    test_case.attributes["exclude_nonalphaspace_unicodes"] = AttrValue(false);
    test_case.input_tensors.push_back(StringTensor({1}, {"Hello World"}));
    test_case.output_tensors.emplace_back(FloatTensor({}, {}), kScale, kZero);
    test_cases.push_back(test_case);
  }

  {
    OpEquivTestCase test_case;
    test_case.test_name = "NoSplitOnSpaceWithWordNoveltyAndDocSize";
    test_case.attributes["vocabulary"] =
        AttrValue("abcdefghijklmnopqrstuvwxyz");
    test_case.attributes["split_on_space"] = AttrValue(false);
    test_case.attributes["max_splits"] = AttrValue(-1);
    test_case.attributes["word_novelty_bits"] = AttrValue(2);
    test_case.attributes["doc_size_levels"] = AttrValue(8);
    test_case.attributes["feature_size"] = AttrValue(8);
    test_case.attributes["exclude_nonalphaspace_unicodes"] = AttrValue(false);
    test_case.input_tensors.push_back(
        StringTensor({1}, {"Hello World hello world"}));
    test_case.output_tensors.emplace_back(FloatTensor({}, {}), kScale, kZero);
    test_cases.push_back(test_case);
  }

  {
    OpEquivTestCase test_case;
    test_case.test_name = "SplitOnSpaceWithWordNoveltyAndDocSize";
    test_case.attributes["vocabulary"] =
        AttrValue("abcdefghijklmnopqrstuvwxyz");
    test_case.attributes["split_on_space"] = AttrValue(true);
    test_case.attributes["max_splits"] = AttrValue(-1);
    test_case.attributes["word_novelty_bits"] = AttrValue(3);
    test_case.attributes["doc_size_levels"] = AttrValue(6);
    test_case.attributes["feature_size"] = AttrValue(8);
    test_case.attributes["exclude_nonalphaspace_unicodes"] = AttrValue(false);
    test_case.input_tensors.push_back(
        StringTensor({1}, {"Hello World hello world"}));
    test_case.output_tensors.emplace_back(FloatTensor({}, {}), kScale, kZero);
    test_cases.push_back(test_case);
  }

  {
    OpEquivTestCase test_case;
    test_case.test_name = "SplitOnSpaceWithEverything";
    test_case.attributes["vocabulary"] =
        AttrValue("abcdefghijklmnopqrstuvwxyz");
    test_case.attributes["split_on_space"] = AttrValue(true);
    test_case.attributes["max_splits"] = AttrValue(2);
    test_case.attributes["word_novelty_bits"] = AttrValue(5);
    test_case.attributes["doc_size_levels"] = AttrValue(8);
    test_case.attributes["feature_size"] = AttrValue(8);
    test_case.attributes["exclude_nonalphaspace_unicodes"] = AttrValue(false);
    test_case.input_tensors.push_back(
        StringTensor({1}, {"Hello World hello world"}));
    test_case.output_tensors.emplace_back(FloatTensor({}, {}), kScale, kZero);
    test_cases.push_back(test_case);
  }

  {
    OpEquivTestCase test_case;
    test_case.test_name = "NoSplitOnSpaceWithEverything";
    test_case.attributes["vocabulary"] =
        AttrValue("abcdefghijklmnopqrstuvwxyz");
    test_case.attributes["split_on_space"] = AttrValue(false);
    test_case.attributes["max_splits"] = AttrValue(4);
    test_case.attributes["word_novelty_bits"] = AttrValue(7);
    test_case.attributes["doc_size_levels"] = AttrValue(9);
    test_case.attributes["feature_size"] = AttrValue(8);
    test_case.attributes["exclude_nonalphaspace_unicodes"] = AttrValue(false);
    test_case.input_tensors.push_back(StringTensor({1}, {"Hello World"}));
    test_case.output_tensors.emplace_back(FloatTensor({}, {}), kScale, kZero);
    test_cases.push_back(test_case);
  }

  {
    OpEquivTestCase test_case;
    test_case.test_name = "SplitOnSpaceWithEverythingAndExclude";
    test_case.attributes["vocabulary"] = AttrValue("");
    test_case.attributes["split_on_space"] = AttrValue(true);
    test_case.attributes["max_splits"] = AttrValue(2);
    test_case.attributes["word_novelty_bits"] = AttrValue(5);
    test_case.attributes["doc_size_levels"] = AttrValue(8);
    test_case.attributes["feature_size"] = AttrValue(8);
    test_case.attributes["exclude_nonalphaspace_unicodes"] = AttrValue(true);
    test_case.input_tensors.push_back(
        StringTensor({1}, {"Hello World 7153845&^$&^$&"}));
    test_case.output_tensors.emplace_back(FloatTensor({}, {}), kScale, kZero);
    test_cases.push_back(test_case);
  }

  {
    OpEquivTestCase test_case;
    test_case.test_name = "NoSplitOnSpaceWithEverythingAndExclude";
    test_case.attributes["vocabulary"] = AttrValue("");
    test_case.attributes["split_on_space"] = AttrValue(false);
    test_case.attributes["max_splits"] = AttrValue(2);
    test_case.attributes["word_novelty_bits"] = AttrValue(5);
    test_case.attributes["doc_size_levels"] = AttrValue(8);
    test_case.attributes["feature_size"] = AttrValue(8);
    test_case.attributes["exclude_nonalphaspace_unicodes"] = AttrValue(true);
    test_case.input_tensors.push_back(
        StringTensor({1}, {"Hello World 7153845&^$&^$&"}));
    test_case.output_tensors.emplace_back(FloatTensor({}, {}), kScale, kZero);
    test_cases.push_back(test_case);
  }

  {
    OpEquivTestCase test_case;
    test_case.test_name = "NormalizeRepetition";
    test_case.attributes["vocabulary"] = AttrValue("");
    test_case.attributes["split_on_space"] = AttrValue(true);
    test_case.attributes["feature_size"] = AttrValue(8);
    test_case.attributes["add_eos_tag"] = AttrValue(false);
    test_case.attributes["add_bos_tag"] = AttrValue(false);
    test_case.attributes["normalize_repetition"] = AttrValue(true);
    test_case.input_tensors.push_back(StringTensor({1}, {"Hello World ..."}));
    test_case.output_tensors.emplace_back(FloatTensor({}, {}), kScale, kZero);
    test_cases.push_back(test_case);
  }

  {
    OpEquivTestCase test_case;
    test_case.test_name = "TokenSeparator";
    test_case.attributes["vocabulary"] = AttrValue("");
    test_case.attributes["split_on_space"] = AttrValue(true);
    test_case.attributes["feature_size"] = AttrValue(8);
    test_case.attributes["add_eos_tag"] = AttrValue(false);
    test_case.attributes["add_bos_tag"] = AttrValue(false);
    test_case.attributes["token_separators"] = AttrValue("-");
    test_case.input_tensors.push_back(StringTensor({1}, {"Hello-World"}));
    test_case.output_tensors.emplace_back(FloatTensor({}, {}), kScale, kZero);
    test_cases.push_back(test_case);
  }

  {
    OpEquivTestCase test_case;
    test_case.test_name = "CapBaseline";
    test_case.attributes["vocabulary"] = AttrValue("");
    test_case.attributes["split_on_space"] = AttrValue(true);
    test_case.attributes["feature_size"] = AttrValue(8);
    test_case.attributes["add_eos_tag"] = AttrValue(false);
    test_case.attributes["add_bos_tag"] = AttrValue(false);
    test_case.input_tensors.push_back(StringTensor({1}, {"Hello hello HELLO"}));
    test_case.output_tensors.emplace_back(FloatTensor({}, {}), kScale, kZero);
    test_cases.push_back(test_case);
  }

  {
    OpEquivTestCase test_case;
    test_case.test_name = "FirstCap";
    test_case.attributes["vocabulary"] = AttrValue("");
    test_case.attributes["split_on_space"] = AttrValue(true);
    test_case.attributes["feature_size"] = AttrValue(8);
    test_case.attributes["add_eos_tag"] = AttrValue(false);
    test_case.attributes["add_bos_tag"] = AttrValue(false);
    test_case.attributes["add_first_cap_feature"] = AttrValue(1.0);
    test_case.input_tensors.push_back(StringTensor({1}, {"Hello hello HELLO"}));
    test_case.output_tensors.emplace_back(FloatTensor({}, {}), kScale, kZero);
    test_cases.push_back(test_case);
  }

  {
    OpEquivTestCase test_case;
    test_case.test_name = "AllCaps";
    test_case.attributes["vocabulary"] = AttrValue("");
    test_case.attributes["split_on_space"] = AttrValue(true);
    test_case.attributes["feature_size"] = AttrValue(8);
    test_case.attributes["add_eos_tag"] = AttrValue(false);
    test_case.attributes["add_bos_tag"] = AttrValue(false);
    test_case.attributes["add_all_caps_feature"] = AttrValue(1.0);
    test_case.input_tensors.push_back(StringTensor({1}, {"Hello hello HELLO"}));
    test_case.output_tensors.emplace_back(FloatTensor({}, {}), kScale, kZero);
    test_cases.push_back(test_case);
  }

  {
    OpEquivTestCase test_case;
    test_case.test_name = "FirstCapAllCaps";
    test_case.attributes["vocabulary"] = AttrValue("");
    test_case.attributes["split_on_space"] = AttrValue(true);
    test_case.attributes["feature_size"] = AttrValue(8);
    test_case.attributes["add_eos_tag"] = AttrValue(false);
    test_case.attributes["add_bos_tag"] = AttrValue(false);
    test_case.attributes["add_first_cap_feature"] = AttrValue(1.0);
    test_case.attributes["add_all_caps_feature"] = AttrValue(1.0);
    test_case.input_tensors.push_back(StringTensor({1}, {"Hello hello HELLO"}));
    test_case.output_tensors.emplace_back(FloatTensor({}, {}), kScale, kZero);
    test_cases.push_back(test_case);
  }

  {
    OpEquivTestCase test_case;
    test_case.test_name = "NormalizeSpaces";
    test_case.attributes["vocabulary"] = AttrValue("");
    test_case.attributes["split_on_space"] = AttrValue(true);
    test_case.attributes["feature_size"] = AttrValue(8);
    test_case.attributes["add_eos_tag"] = AttrValue(false);
    test_case.attributes["add_bos_tag"] = AttrValue(false);
    test_case.attributes["normalize_spaces"] = AttrValue(true);
    test_case.input_tensors.push_back(StringTensor({1}, {" Hello   there.  "}));
    test_case.output_tensors.emplace_back(FloatTensor({}, {}), kScale, kZero);
    test_cases.push_back(test_case);
  }

  return test_cases;
}

INSTANTIATE_TEST_SUITE_P(
    SequenceStringProjectionTests, SequenceStringProjectionTest,
    ::testing::ValuesIn(SequenceStringProjectionTestCases()));

class SequenceStringProjectionV2Model : public ::tflite::SingleOpModel {
 public:
  explicit SequenceStringProjectionV2Model(
      std::vector<std::vector<int>> input_shapes,
      const std::string& hashtype = kMurmurHash) {
    flexbuffers::Builder fbb;
    fbb.Map([&] {
      fbb.Int("feature_size", 4);
      fbb.String("hashtype", hashtype);
    });
    fbb.Finish();
    input_ = AddInput(TensorType_STRING);
    output_ = AddOutput({TensorType_UINT8, {}});
    SetCustomOp(kSequenceStringProjectionV2, fbb.GetBuffer(),
                Register_SEQUENCE_STRING_PROJECTION_V2);
    BuildInterpreter(input_shapes);
  }
  void Invoke(const std::vector<std::string>& input, TfLiteStatus expected) {
    PopulateStringTensor(input_, input);
    CHECK(interpreter_->AllocateTensors() == kTfLiteOk)
        << "Cannot allocate tensors";
    ASSERT_EQ(SingleOpModel::Invoke(), expected);
  }
  TfLiteStatus InvokeFailable(const std::string& input) {
    PopulateStringTensor(input_, {input});
    CHECK(interpreter_->AllocateTensors() == kTfLiteOk)
        << "Cannot allocate tensors";
    return SingleOpModel::Invoke();
  }
  std::vector<int> GetOutputShape() { return GetTensorShape(output_); }

 private:
  int input_;
  int output_;
};

TEST(SequenceStringProjectionV2Test, IncorrectHashtype) {
  SequenceStringProjectionV2Model m({{1, 0}}, "unsupported");
  EXPECT_EQ(m.InvokeFailable(" "), kTfLiteError);
}

TEST(SequenceStringProjectionV2Test, RegularInputUint8EmptyNotSupported) {
  // TFLite test infratructure currently does not let the error message to be
  // extracted on failure. As a result just the return error code is tested
  // as all other TFLite op handler tests. The error message each test invokes
  // is captured in a comment though.
  // ERROR: Empty input not supported.
  SequenceStringProjectionV2Model m({{1, 0}});
  m.Invoke({}, kTfLiteError);
}

TEST(SequenceStringProjectionV2Test, RegularInputUint8BatchNotSupported) {
  // TFLite test infratructure currently does not let the error message to be
  // extracted on failure. As a result just the return error code is tested
  // as all other TFLite op handler tests. The error message each test invokes
  // is captured in a comment though.
  // ERROR: Input tensor batch size should be 1, got 2.
  SequenceStringProjectionV2Model m({{2, 1}});
  m.Invoke({"hello", "world"}, kTfLiteError);
}

TEST(SequenceStringProjectionV2Test, RegularInputUint8RankNot2NotSupported) {
  // TFLite test infratructure currently does not let the error message to be
  // extracted on failure. As a result just the return error code is tested
  // as all other TFLite op handler tests. The error message each test invokes
  // is captured in a comment though.
  // ERROR: Input tensor is expected to be rank 2, got rank 3.
  SequenceStringProjectionV2Model m({{2, 1, 1}});
  m.Invoke({"hello", "world"}, kTfLiteError);
}

TEST(SequenceStringProjectionV2Test, RegularInputUint8InconsistentInput) {
  // TFLite test infratructure currently does not let the error message to be
  // extracted on failure. As a result just the return error code is tested
  // as all other TFLite op handler tests. The error message each test invokes
  // is captured in a comment though.
  // ERROR: Inconsistent number of input tokens 3 != 2.
  SequenceStringProjectionV2Model m({{1, 2}});
  m.Invoke({"hello", "world", "goodbye"}, kTfLiteError);
}

TEST(SequenceStringProjectionV2Test, RegularInputUint8) {
  // OK
  SequenceStringProjectionV2Model m({{1, 2}});
  m.Invoke({"hello", "world"}, kTfLiteOk);
}

TEST(SequenceStringProjectionV2Test, NumberProjectionsForMultipleInputs) {
  SequenceStringProjectionV2Model m({{1, 2}});
  std::vector<std::string> input = {"hello", "world"};
  m.Invoke(input, kTfLiteOk);
  EXPECT_EQ(m.GetOutputShape()[1], input.size());
  m.Invoke(input, kTfLiteOk);
  EXPECT_EQ(m.GetOutputShape()[1], input.size());
}

class SequenceStringProjectionV2Test : public TensorflowTfLiteOpTest {
  std::function<TfLiteRegistration*()> TfLiteOpRegistration() override {
    return ops::custom::Register_SEQUENCE_STRING_PROJECTION_V2;
  }

  std::string TensorflowOpName() override {
    return "SequenceStringProjectionV2";
  }
};

TEST_P(SequenceStringProjectionV2Test, TensorflowTfLiteSame) {
  RunTensorflowOp();
  RunTfLiteOp();
  CompareOpOutput();
}

std::vector<OpEquivTestCase> SequenceStringProjectionV2TestCases() {
  std::vector<OpEquivTestCase> test_cases;
  constexpr float kScale = 2.0 / 255;
  constexpr int kZero = 127;

  {
    OpEquivTestCase test_case;
    test_case.test_name = "CheckEqualityNoBoSNoEoS";
    test_case.attributes["vocabulary"] = AttrValue("");
    test_case.attributes["feature_size"] = AttrValue(8);
    test_case.attributes["add_eos_tag"] = AttrValue(false);
    test_case.attributes["add_bos_tag"] = AttrValue(false);
    test_case.input_tensors.push_back(
        StringTensor({1, 5}, {"Hello", "World", "7153845", "&^$&", "^$&"}));
    test_case.input_tensors.push_back(IntTensor({1}, {5}));
    test_case.output_tensors.emplace_back(FloatTensor({}, {}), kScale, kZero);
    test_cases.push_back(test_case);
  }

  {
    OpEquivTestCase test_case;
    test_case.test_name = "CheckEqualityNoBoS";
    test_case.attributes["vocabulary"] = AttrValue("");
    test_case.attributes["feature_size"] = AttrValue(8);
    test_case.attributes["add_eos_tag"] = AttrValue(true);
    test_case.attributes["add_bos_tag"] = AttrValue(false);
    test_case.input_tensors.push_back(
        StringTensor({1, 4}, {"Hello", "World", "7153845", "&^$&^$&"}));
    test_case.input_tensors.push_back(IntTensor({1}, {4}));
    test_case.output_tensors.emplace_back(FloatTensor({}, {}), kScale, kZero);
    test_cases.push_back(test_case);
  }

  {
    OpEquivTestCase test_case;
    test_case.test_name = "CheckEqualityNoEoS";
    test_case.attributes["vocabulary"] = AttrValue("");
    test_case.attributes["feature_size"] = AttrValue(8);
    test_case.attributes["add_eos_tag"] = AttrValue(false);
    test_case.attributes["add_bos_tag"] = AttrValue(true);
    test_case.input_tensors.push_back(
        StringTensor({1, 3}, {"Hello", "World", "7153845&^$&^$&"}));
    test_case.input_tensors.push_back(IntTensor({1}, {3}));
    test_case.output_tensors.emplace_back(FloatTensor({}, {}), kScale, kZero);
    test_cases.push_back(test_case);
  }

  {
    OpEquivTestCase test_case;
    test_case.test_name = "CheckEquality";
    test_case.attributes["vocabulary"] = AttrValue("");
    test_case.attributes["feature_size"] = AttrValue(8);
    test_case.attributes["add_eos_tag"] = AttrValue(true);
    test_case.attributes["add_bos_tag"] = AttrValue(true);
    test_case.input_tensors.push_back(
        StringTensor({1, 3}, {"Hello", "Worldddd", "7153845&^$&^$&"}));
    test_case.input_tensors.push_back(IntTensor({1}, {3}));
    test_case.output_tensors.emplace_back(FloatTensor({}, {}), kScale, kZero);
    test_cases.push_back(test_case);
  }

  {
    OpEquivTestCase test_case;
    test_case.test_name = "NormalizeRepetition";
    test_case.attributes["vocabulary"] = AttrValue("");
    test_case.attributes["feature_size"] = AttrValue(8);
    test_case.attributes["add_eos_tag"] = AttrValue(false);
    test_case.attributes["add_bos_tag"] = AttrValue(false);
    test_case.attributes["normalize_repetition"] = AttrValue(true);
    test_case.input_tensors.push_back(
        StringTensor({1, 6}, {"Hello", "World", "...", "..", ".", "...."}));
    test_case.input_tensors.push_back(IntTensor({1}, {6}));
    test_case.output_tensors.emplace_back(FloatTensor({}, {}), kScale, kZero);
    test_cases.push_back(test_case);
  }

  return test_cases;
}

INSTANTIATE_TEST_SUITE_P(
    SequenceStringProjectionV2Tests, SequenceStringProjectionV2Test,
    ::testing::ValuesIn(SequenceStringProjectionV2TestCases()));

}  // namespace
}  // namespace custom
}  // namespace ops
}  // namespace seq_flow_lite

int main(int argc, char** argv) {
  // On Linux, add: absl::SetFlag(&FLAGS_logtostderr, true);
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
