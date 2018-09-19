// Copyright 2017 Google Inc. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// =============================================================================

#include "dragnn/core/input_batch_cache.h"

#include "dragnn/core/interfaces/input_batch.h"
#include <gmock/gmock.h>
#include "tensorflow/core/platform/test.h"

namespace syntaxnet {
namespace dragnn {

class StringData : public InputBatch {
 public:
  StringData() {}

  void SetData(const std::vector<string> &data) override {
    for (const auto &element : data) {
      data_.push_back(element + "_converted");
    }
  }

  int GetSize() const override { return data_.size(); }

  const std::vector<string> GetSerializedData() const override { return data_; }

  std::vector<string> *data() { return &data_; }

 private:
  std::vector<string> data_;
};

class DifferentStringData : public InputBatch {
 public:
  DifferentStringData() {}

  void SetData(const std::vector<string> &data) override {
    for (const auto &element : data) {
      data_.push_back(element + "_also_converted");
    }
  }

  int GetSize() const override { return data_.size(); }

  const std::vector<string> GetSerializedData() const override { return data_; }

  std::vector<string> *data() { return &data_; }

 private:
  std::vector<string> data_;
};

// Expects that two pointers have the same address.
void ExpectSameAddress(const void *pointer1, const void *pointer2) {
  EXPECT_EQ(pointer1, pointer2);
}

TEST(InputBatchCacheTest, ConvertsSingleInput) {
  string test_string = "Foo";
  InputBatchCache generic_set(test_string);
  auto data = generic_set.GetAs<StringData>();
  EXPECT_EQ(data->data()->size(), 1);
  EXPECT_EQ(data->data()->at(0), "Foo_converted");
}

TEST(InputBatchCacheTest, ConvertsAddedInput) {
  string test_string = "Foo";
  InputBatchCache generic_set;
  generic_set.AddData(test_string);
  auto data = generic_set.GetAs<StringData>();
  EXPECT_EQ(data->data()->size(), 1);
  EXPECT_EQ(data->data()->at(0), "Foo_converted");
}

TEST(InputBatchCacheTest, ConvertsVectorOfInputs) {
  std::vector<string> test_inputs;
  test_inputs.push_back("Foo");
  test_inputs.push_back("Bar");
  test_inputs.push_back("Baz");
  InputBatchCache generic_set(test_inputs);
  auto data = generic_set.GetAs<StringData>();
  EXPECT_EQ(data->data()->size(), test_inputs.size());
  EXPECT_EQ(data->data()->at(0), "Foo_converted");
  EXPECT_EQ(data->data()->at(1), "Bar_converted");
  EXPECT_EQ(data->data()->at(2), "Baz_converted");
}

TEST(InputBatchCacheTest, ConvertingMultipleDataTypesCausesCheck) {
  string test_string = "Foo";
  InputBatchCache generic_set(test_string);
  auto data = generic_set.GetAs<StringData>();
  EXPECT_EQ(data->data()->at(0), "Foo_converted");
  ASSERT_DEATH(generic_set.GetAs<DifferentStringData>(),
               "Attempted to convert to two object types!.*");
}

TEST(InputBatchCacheTest, ReturnsSingleInput) {
  string test_string = "Foo";
  InputBatchCache generic_set(test_string);
  auto data = generic_set.GetAs<StringData>();
  EXPECT_NE(nullptr, data);
  auto returned = generic_set.SerializedData();
  EXPECT_EQ(returned.size(), 1);
  EXPECT_EQ(returned.at(0), "Foo_converted");
}

TEST(InputBatchCacheTest, ConvertsAddedInputDiesAfterGetAs) {
  string test_string = "Foo";
  InputBatchCache generic_set;
  generic_set.AddData(test_string);
  auto data = generic_set.GetAs<StringData>();
  EXPECT_EQ(data->data()->size(), 1);
  EXPECT_EQ(data->data()->at(0), "Foo_converted");
  EXPECT_DEATH(generic_set.AddData("YOU MAY NOT DO THIS AND IT WILL DIE."),
               "after the cache has been converted");
}

TEST(InputBatchCacheTest, SerializedDataAndSize) {
  InputBatchCache generic_set;
  generic_set.AddData("Foo");
  generic_set.AddData("Bar");
  generic_set.GetAs<StringData>();

  const std::vector<string> expected_data = {"Foo_converted", "Bar_converted"};
  EXPECT_EQ(expected_data, generic_set.SerializedData());
  EXPECT_EQ(2, generic_set.Size());
}

TEST(InputBatchCacheTest, InitializeFromInputBatch) {
  const std::vector<string> kInputData = {"foo", "bar", "baz"};
  const std::vector<string> kExpectedData = {"foo_converted",  //
                                             "bar_converted",  //
                                             "baz_converted"};

  std::unique_ptr<StringData> string_data(new StringData());
  string_data->SetData(kInputData);
  const StringData *string_data_ptr = string_data.get();

  InputBatchCache generic_set(std::move(string_data));
  auto data = generic_set.GetAs<StringData>();

  ExpectSameAddress(string_data_ptr, data);
  EXPECT_EQ(data->GetSize(), 3);
  EXPECT_EQ(data->GetSerializedData(), kExpectedData);
  EXPECT_EQ(*data->data(), kExpectedData);

  // AddData() shouldn't work since the cache is already populated.
  EXPECT_DEATH(generic_set.AddData("YOU MAY NOT DO THIS AND IT WILL DIE."),
               "after the cache has been converted");

  // GetAs() shouldn't work with a different type.
  EXPECT_DEATH(generic_set.GetAs<DifferentStringData>(),
               "Attempted to convert to two object types!");
}

TEST(InputBatchCacheTest, CannotInitializeFromNullInputBatch) {
  EXPECT_DEATH(InputBatchCache(std::unique_ptr<StringData>()),
               "Cannot initialize from a null InputBatch");
}

}  // namespace dragnn
}  // namespace syntaxnet
