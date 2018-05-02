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

#include "dragnn/runtime/test/fake_variable_store.h"

#include <stddef.h>
#include <memory>
#include <string>
#include <vector>

#include "dragnn/core/test/generic.h"
#include "dragnn/runtime/alignment.h"
#include "dragnn/runtime/math/types.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"

namespace syntaxnet {
namespace dragnn {
namespace runtime {
namespace {

// Returns a data matrix that has no alignment padding.  This is required for
// BlockedMatrix, which does not tolerate alignment padding.  The contents of
// the returned matrix are [0.0, 1.0, 2.0, ...] in the natural order.
std::vector<std::vector<float>> MakeBlockedData() {
  const size_t kNumRows = 18;
  const size_t kNumColumns = internal::kAlignmentBytes / sizeof(float);
  std::vector<std::vector<float>> data(kNumRows);

  float counter = 0.0;
  for (std::vector<float> &row : data) {
    row.resize(kNumColumns);
    for (float &value : row) value = counter++;
  }
  return data;
}

// Tests that Lookup*() behaves properly w.r.t. AddOrDie().
TEST(FakeVariableStoreTest, Lookup) {
  FakeVariableStore store;
  AlignedView view;
  Vector<float> vector;
  Matrix<float> matrix;
  BlockedMatrix<float> blocked_matrix;

  // Fail to look up an unknown name.
  EXPECT_THAT(store.Lookup("foo", &vector),
              test::IsErrorWithSubstr("Unknown variable"));
  EXPECT_TRUE(view.empty());  // not modified

  // Add some data and try looking it up.
  store.AddOrDie("foo", {{1.0, 2.0, 3.0}});

  TF_EXPECT_OK(store.Lookup("foo", &vector));
  ASSERT_EQ(vector.size(), 3);
  EXPECT_EQ(vector[0], 1.0);
  EXPECT_EQ(vector[1], 2.0);
  EXPECT_EQ(vector[2], 3.0);

  TF_EXPECT_OK(store.Lookup("foo", &matrix));
  ASSERT_EQ(matrix.num_rows(), 1);
  ASSERT_EQ(matrix.num_columns(), 3);
  EXPECT_EQ(matrix.row(0)[0], 1.0);
  EXPECT_EQ(matrix.row(0)[1], 2.0);
  EXPECT_EQ(matrix.row(0)[2], 3.0);

  // Try a funny name.
  store.AddOrDie("", {{5.0, 7.0}, {11.0, 13.0}});
  TF_EXPECT_OK(store.Lookup("", &vector));
  ASSERT_EQ(vector.size(), 4);
  EXPECT_EQ(vector[0], 5.0);
  EXPECT_EQ(vector[1], 7.0);
  EXPECT_EQ(vector[2], 11.0);
  EXPECT_EQ(vector[3], 13.0);

  TF_EXPECT_OK(store.Lookup("", &matrix));
  ASSERT_EQ(matrix.num_rows(), 2);
  ASSERT_EQ(matrix.num_columns(), 2);
  EXPECT_EQ(matrix.row(0)[0], 5.0);
  EXPECT_EQ(matrix.row(0)[1], 7.0);
  EXPECT_EQ(matrix.row(1)[0], 11.0);
  EXPECT_EQ(matrix.row(1)[1], 13.0);

  // Try blocked matrices.  These must not have alignment padding.
  const auto blocked_data = MakeBlockedData();
  store.AddOrDie("blocked", blocked_data);
  TF_ASSERT_OK(store.Lookup("blocked", &blocked_matrix));
  ASSERT_EQ(blocked_matrix.num_rows(), blocked_data.size());
  ASSERT_EQ(blocked_matrix.num_columns(), blocked_data[0].size());
  ASSERT_EQ(blocked_matrix.block_size(), blocked_data[0].size());
  for (size_t vector = 0; vector < blocked_matrix.num_vectors(); ++vector) {
    for (size_t i = 0; i < blocked_matrix.block_size(); ++i) {
      EXPECT_EQ(blocked_matrix.vector(vector)[i],
                vector * blocked_matrix.block_size() + i);
    }
  }

  // Check that overriding dimensions is OK. Instead of a matrix that has every
  // row as a block, every row is now has two blocks, so there are half as many
  // rows and each row (number of columns) is twice as long.
  const size_t kNumColumns = internal::kAlignmentBytes / sizeof(float);
  store.SetBlockedDimensionOverride("blocked",
                                    {9, 2 * kNumColumns, kNumColumns});
  TF_ASSERT_OK(store.Lookup("blocked", &blocked_matrix));
  ASSERT_EQ(blocked_matrix.num_rows(), blocked_data.size() / 2);
  ASSERT_EQ(blocked_matrix.num_columns(), 2 * blocked_data[0].size());
  ASSERT_EQ(blocked_matrix.block_size(), blocked_data[0].size());
}

// Tests that the fake variable never contains variables with unknown format.
TEST(FakeVariableStoreTest, NeverContainsUnknownFormat) {
  FakeVariableStore store;
  store.AddOrDie("foo", {{0.0}});

  std::vector<size_t> dimensions;
  AlignedArea area;
  EXPECT_THAT(
      store.Lookup("foo", VariableSpec::FORMAT_UNKNOWN, &dimensions, &area),
      test::IsErrorWithSubstr("Unknown variable"));
}

// Tests that the fake variable store can create a variable that only appears in
// one format.
TEST(FakeVariableStoreTest, AddWithSpecificFormat) {
  const auto data = MakeBlockedData();

  FakeVariableStore store;
  store.AddOrDie("flat", data, VariableSpec::FORMAT_FLAT);
  store.AddOrDie("matrix", data, VariableSpec::FORMAT_ROW_MAJOR_MATRIX);
  store.AddOrDie("blocked", data,
                 VariableSpec::FORMAT_COLUMN_BLOCKED_ROW_MAJOR_MATRIX);

  // Vector lookups should only work for "flat".
  Vector<float> vector;
  TF_ASSERT_OK(store.Lookup("flat", &vector));
  EXPECT_THAT(store.Lookup("matrix", &vector),
              test::IsErrorWithSubstr("Unknown variable"));
  EXPECT_THAT(store.Lookup("blocked", &vector),
              test::IsErrorWithSubstr("Unknown variable"));

  // Matrix lookups should only work for "matrix".
  Matrix<float> matrix;
  EXPECT_THAT(store.Lookup("flat", &matrix),
              test::IsErrorWithSubstr("Unknown variable"));
  TF_ASSERT_OK(store.Lookup("matrix", &matrix));
  EXPECT_THAT(store.Lookup("blocked", &matrix),
              test::IsErrorWithSubstr("Unknown variable"));

  // Blocked matrix lookups should only work for "blocked".
  BlockedMatrix<float> blocked_matrix;
  EXPECT_THAT(store.Lookup("flat", &blocked_matrix),
              test::IsErrorWithSubstr("Unknown variable"));
  EXPECT_THAT(store.Lookup("matrix", &blocked_matrix),
              test::IsErrorWithSubstr("Unknown variable"));
  TF_ASSERT_OK(store.Lookup("blocked", &blocked_matrix));
}

// Tests that Close() always succeeds.
TEST(FakeVariableStoreTest, Close) {
  FakeVariableStore store;
  TF_EXPECT_OK(store.Close());
  store.AddOrDie("foo", {{1.0, 2.0, 3.0}});
  TF_EXPECT_OK(store.Close());
  store.AddOrDie("bar", {{1.0, 2.0}, {3.0, 4.0}});
  TF_EXPECT_OK(store.Close());
}

// Tests that SimpleFakeVariableStore returns the user-specified mock values.
TEST(SimpleFakeVariableStoreTest, ReturnsMockedValues) {
  SimpleFakeVariableStore store;
  store.MockLookup<float>({1, 2}, {{1.0, 2.0}});

  Matrix<float> matrix;
  TF_ASSERT_OK(store.Lookup("name_doesnt_matter", &matrix));
  ASSERT_EQ(matrix.num_rows(), 1);
  ASSERT_EQ(matrix.num_columns(), 2);
  EXPECT_EQ(matrix.row(0)[0], 1.0);
  EXPECT_EQ(matrix.row(0)[1], 2.0);

  TF_ASSERT_OK(store.Close());
}

}  // namespace
}  // namespace runtime
}  // namespace dragnn
}  // namespace syntaxnet
