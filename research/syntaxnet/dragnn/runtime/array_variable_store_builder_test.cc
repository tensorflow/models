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

#include "dragnn/runtime/array_variable_store_builder.h"

#include <stddef.h>
#include <map>
#include <string>
#include <vector>


#include "dragnn/core/test/generic.h"
#include "dragnn/protos/runtime.pb.h"
#include "dragnn/runtime/alignment.h"
#include "dragnn/runtime/test/helpers.h"
#include "syntaxnet/base.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/test.h"


namespace syntaxnet {
namespace dragnn {
namespace runtime {
namespace {

// Tests that the builder rejects invalid flat variables.
TEST(ArrayVariableStoreBuilderTest, InvalidFlatVariable) {
  AlignedView view;
  ArrayVariableStoreBuilder::Variables variables;
  ArrayVariableStoreSpec spec;
  string data;

  TF_ASSERT_OK(view.Reset(nullptr, 2 * internal::kAlignmentBytes));

  // Try an empty area.
  std::pair<string, VariableSpec::Format> foo_key("foo",
                                                  VariableSpec::FORMAT_FLAT);
  AlignedArea area;
  TF_ASSERT_OK(area.Reset(view, 0, 0));
  std::pair<std::vector<size_t>, AlignedArea> foo_value({1}, area);
  variables.push_back(std::make_pair(foo_key, foo_value));
  EXPECT_THAT(ArrayVariableStoreBuilder::Build(variables, &spec, &data),
              test::IsErrorWithSubstr(
                  "Flat variables must have 1 view, but 'foo' has 0"));

  // Try an area with more than 1 sub-view.
  TF_ASSERT_OK(area.Reset(view, 2, 0));
  variables[0].second.second = area;
  EXPECT_THAT(ArrayVariableStoreBuilder::Build(variables, &spec, &data),
              test::IsErrorWithSubstr(
                  "Flat variables must have 1 view, but 'foo' has 2"));
}

// Tests that the builder succeeds on good inputs and reproduces an expected
// byte array.
//
// NB: Since this test directly compares the byte array, it implicitly requires
// that the builder lays out the variables in a particular order.  If that order
// changes, the test expectations must be updated.
TEST(ArrayVariableStoreBuilderTest, RegressionTest) {
  const string kLocalSpecPath =
      "dragnn/runtime/testdata/array_variable_store_spec";
  const string kLocalDataPath =
      "dragnn/runtime/testdata/array_variable_store_data";

  const string kExpectedSpecPath = tensorflow::io::JoinPath(
      test::GetTestDataPrefix(),
      "dragnn/runtime/testdata/array_variable_store_spec");
  const string kExpectedDataPath = tensorflow::io::JoinPath(
      test::GetTestDataPrefix(),
      "dragnn/runtime/testdata/array_variable_store_data");

  // If these values are changed, make sure to rewrite the test data and update
  // array_variable_store_test.cc.
  UniqueMatrix<float> foo({{0.0, 0.5, 1.0},  //
                           {1.5, 2.0, 2.5},  //
                           {3.0, 3.5, 4.0},  //
                           {4.5, 5.0, 5.5}});
  UniqueMatrix<double> baz_data({{1.0, 2.0, 2.0, 2.0},  //
                                 {3.0, 4.0, 4.0, 4.0},  //
                                 {5.0, 6.0, 6.0, 6.0},  //
                                 {7.0, 8.0, 8.0, 8.0}});

  ArrayVariableStoreBuilder::Variables variables;
  std::pair<string, VariableSpec::Format> foo_key(
      "foo", VariableSpec::FORMAT_ROW_MAJOR_MATRIX);
  std::pair<std::vector<size_t>, AlignedArea> foo_value(
      {foo->num_rows(), foo->num_columns()}, AlignedArea(foo.area()));
  variables.push_back(std::make_pair(foo_key, foo_value));
  std::pair<string, VariableSpec::Format> baz_key(
      "baz", VariableSpec::FORMAT_COLUMN_BLOCKED_ROW_MAJOR_MATRIX);
  std::pair<std::vector<size_t>, AlignedArea> baz_value(
      {2, 8, 4}, AlignedArea(baz_data.area()));
  variables.push_back(std::make_pair(baz_key, baz_value));

  ArrayVariableStoreSpec actual_spec;
  actual_spec.set_version(999);
  string actual_data = "garbage to be overwritten";
  TF_ASSERT_OK(
      ArrayVariableStoreBuilder::Build(variables, &actual_spec, &actual_data));

  if (false) {

    // Rewrite the test data.
    TF_CHECK_OK(tensorflow::WriteTextProto(tensorflow::Env::Default(),
                                           kLocalSpecPath, actual_spec));
    TF_CHECK_OK(tensorflow::WriteStringToFile(tensorflow::Env::Default(),
                                              kLocalDataPath, actual_data));
  } else {
    // Compare to the test data.
    ArrayVariableStoreSpec expected_spec;
    string expected_data;
    TF_CHECK_OK(tensorflow::ReadTextProto(tensorflow::Env::Default(),
                                          kExpectedSpecPath, &expected_spec));
    TF_CHECK_OK(tensorflow::ReadFileToString(
        tensorflow::Env::Default(), kExpectedDataPath, &expected_data));

    EXPECT_THAT(actual_spec, test::EqualsProto(expected_spec));
    EXPECT_EQ(actual_data, expected_data);
  }
}

}  // namespace
}  // namespace runtime
}  // namespace dragnn
}  // namespace syntaxnet
