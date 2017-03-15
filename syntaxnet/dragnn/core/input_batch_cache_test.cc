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

  const std::vector<string> GetSerializedData() const override { return data_; }

  std::vector<string> *data() { return &data_; }

 private:
  std::vector<string> data_;
};

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

}  // namespace dragnn
}  // namespace syntaxnet
