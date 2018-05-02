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

#include "dragnn/runtime/variable_store_wrappers.h"

#include <stddef.h>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "dragnn/core/test/generic.h"
#include "dragnn/protos/runtime.pb.h"
#include "dragnn/runtime/flexible_matrix_kernel.h"
#include "dragnn/runtime/math/transformations.h"
#include "dragnn/runtime/math/types.h"
#include "dragnn/runtime/test/fake_variable_store.h"
#include "syntaxnet/base.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/test.h"

namespace syntaxnet {
namespace dragnn {
namespace runtime {
namespace {

// Returns a variable store with some default entries for tests.  Specifically,
// "foo" has an averaged version while "bar" does not.
std::unique_ptr<VariableStore> NewVariableStore() {
  std::unique_ptr<FakeVariableStore> store(new FakeVariableStore());
  store->AddOrDie("foo", {{1.0, 2.0},  //
                          {3.0, 4.0}});
  store->AddOrDie("foo/ExponentialMovingAverage", {{10.0, 20.0},  //
                                                   {30.0, 40.0}});
  store->AddOrDie("bar", {{10.0, 9.0, 8.0},  //
                          {7.0, 6.0, 5.0}});
  return std::move(store);
}

// Expects that the |vector| contains the |data|.
template <typename T>
void ExpectVector(Vector<T> vector, const std::vector<T> &data) {
  ASSERT_EQ(vector.size(), data.size());
  for (size_t i = 0; i < data.size(); ++i) EXPECT_EQ(vector[i], data[i]);
}

// Expects that the |matrix| contains the |data|.
void ExpectMatrix(Matrix<float> matrix,
                  const std::vector<std::vector<float>> &data) {
  ASSERT_EQ(matrix.num_rows(), data.size());
  if (data.empty()) return;
  ASSERT_EQ(matrix.num_columns(), data[0].size());
  for (size_t i = 0; i < data.size(); ++i) ExpectVector(matrix.row(i), data[i]);
}

// Tests that the averaging wrapper uses the averaged version of a variable if
// available, the non-averaged version failing that, and errors out otherwise.
TEST(TryAveragedVariableStoreWrapperTest, FallbackAllowed) {
  TryAveragedVariableStoreWrapper store(NewVariableStore(),
                                        /*allow_fallback=*/true);
  Matrix<float> foo_averaged;
  Matrix<float> bar_non_averaged;
  Matrix<float> unused_matrix;

  TF_ASSERT_OK(store.Lookup("foo", &foo_averaged));
  TF_ASSERT_OK(store.Lookup("bar", &bar_non_averaged));
  EXPECT_THAT(store.Lookup("missing", &unused_matrix),
              test::IsErrorWithSubstr("Unknown variable"));
  TF_EXPECT_OK(store.Close());

  ExpectMatrix(foo_averaged, {{10.0, 20.0},  //
                              {30.0, 40.0}});
  ExpectMatrix(bar_non_averaged, {{10.0, 9.0, 8.0},  //
                                  {7.0, 6.0, 5.0}});
}

// As above, but with fallback disabled (the default behavior).
TEST(TryAveragedVariableStoreWrapperTest, FallbackForbidden) {
  TryAveragedVariableStoreWrapper store(NewVariableStore());
  Matrix<float> foo_averaged;
  Matrix<float> bar_non_averaged;
  Matrix<float> unused_matrix;

  TF_ASSERT_OK(store.Lookup("foo", &foo_averaged));
  EXPECT_THAT(store.Lookup("bar", &bar_non_averaged),
              test::IsErrorWithSubstr("Failed to retrieve averaged variable "
                                      "'bar/ExponentialMovingAverage' for "
                                      "variable 'bar'"));
  EXPECT_THAT(store.Lookup("missing", &unused_matrix),
              test::IsErrorWithSubstr("Failed to retrieve averaged variable "
                                      "'missing/ExponentialMovingAverage' for "
                                      "variable 'missing'"));
  TF_EXPECT_OK(store.Close());

  ExpectMatrix(foo_averaged, {{10.0, 20.0},  //
                              {30.0, 40.0}});
}

// Tests that the capturing wrapper correctly records the set of variables that
// have been looked up.
TEST(CaptureUsedVariableStoreWrapperTest, Capturing) {
  CaptureUsedVariableStoreWrapper store(NewVariableStore());
  Vector<float> unused_vector;
  Matrix<float> unused_row_major_matrix;

  // Try a completely missing variable.  As a failed lookup, this should not
  // appear among the captured variables.
  EXPECT_THAT(store.Lookup("missing", &unused_vector),
              test::IsErrorWithSubstr("Unknown variable"));

  // Look up one variable of each type.
  TF_ASSERT_OK(store.Lookup("foo", &unused_vector));
  TF_ASSERT_OK(store.Lookup("bar", &unused_row_major_matrix));
  TF_EXPECT_OK(store.Close());

  // Check the names and formats of the captured variables.
  const auto &variables = store.variables();
  ASSERT_EQ(variables.size(), 2);

  // The variables must be returned in order. Check their names and format
  // first.
  EXPECT_EQ(variables[0].first.first, "foo");
  EXPECT_EQ(variables[0].first.second, VariableSpec::FORMAT_FLAT);
  EXPECT_EQ(variables[1].first.first, "bar");
  EXPECT_EQ(variables[1].first.second, VariableSpec::FORMAT_ROW_MAJOR_MATRIX);

  // Check the content of 'foo'.
  EXPECT_EQ(variables[0].second.first, std::vector<size_t>{4});
  ExpectVector(Vector<float>(variables[0].second.second.view(0)),
               {1.0, 2.0, 3.0, 4.0});

  // Check the content of 'bar'.
  EXPECT_EQ(variables[1].second.first, std::vector<size_t>({2, 3}));
  ExpectMatrix(Matrix<float>(variables[1].second.second), {{10.0, 9.0, 8.0},  //
                                                           {7.0, 6.0, 5.0}});
}

// Returns a variable store with some blocked and transposed matrices, for
// testing the flexible matrix wrapper.
std::unique_ptr<VariableStore> NewBlockedAndTransposedStore() {
  std::unique_ptr<FakeVariableStore> store(new FakeVariableStore());

  // A tiny matrix, which favors the non-blocked format.
  store->AddOrDie("1x1", {{1.0}});
  store->AddOrDie("1x1/transposed", {{1.0}});
  store->AddOrDie("1x1/matrix/blocked32", {{1.0}});
  store->AddOrDie("1x1/matrix/blocked48", {{1.0}});

  // A matrix that is a multiple of 32, which should favor block size 32.
  const std::vector<float> row32(32, 32.0);
  const std::vector<std::vector<float>> data32(16, row32);
  store->AddOrDie("16x32", data32);
  store->AddOrDie("16x32/transposed", data32);
  store->AddOrDie("16x32/matrix/blocked32", data32);
  store->AddOrDie("16x32/matrix/blocked48", data32);

  // A matrix that is a multiple of 48, which should favor block size 48.
  const std::vector<float> row48(48, 48.0);
  const std::vector<std::vector<float>> data48(24, row48);
  store->AddOrDie("24x48", data48);
  store->AddOrDie("24x48/transposed", data48);
  store->AddOrDie("24x48/matrix/blocked32", data48);
  store->AddOrDie("24x48/matrix/blocked48", data48);

  return std::move(store);
}

// Expects that the |blocked_matrix| matches the |num_rows|, |num_columns|, and
// |block_size| and is filled with the |value|.
void ExpectBlockedMatrix(BlockedMatrix<float> blocked_matrix, size_t num_rows,
                         size_t num_columns, size_t block_size, float value) {
  ASSERT_EQ(blocked_matrix.num_rows(), num_rows);
  ASSERT_EQ(blocked_matrix.num_columns(), num_columns);
  ASSERT_EQ(blocked_matrix.block_size(), block_size);

  const std::vector<float> expected_vector(block_size, value);
  for (size_t i = 0; i < blocked_matrix.num_vectors(); ++i) {
    ExpectVector(blocked_matrix.vector(i), expected_vector);
  }
}

// Tests that the flexible matrix wrapper passes through variables that don't
// end in the right suffix.
TEST(FlexibleMatrixVariableStoreWrapperTest, PassThroughIrrelevantVariables) {
  FlexibleMatrixVariableStoreWrapper store(NewBlockedAndTransposedStore());
  Vector<float> vector;

  EXPECT_THAT(store.Lookup("missing", &vector),
              test::IsErrorWithSubstr("Unknown variable"));

  TF_ASSERT_OK(store.Lookup("1x1", &vector));
  ExpectVector(vector, {1.0});

  TF_EXPECT_OK(store.Close());
}

// Tests that the flexible matrix wrapper selects the plain matrix format for
// tiny matrices.
TEST(FlexibleMatrixVariableStoreWrapperTest, SelectPlainMatrixFormat) {
  FlexibleMatrixVariableStoreWrapper store(NewBlockedAndTransposedStore());
  Matrix<float> plain_matrix;
  BlockedMatrix<float> blocked_matrix;
  const string name =
      tensorflow::strings::StrCat("1x1", FlexibleMatrixKernel::kSuffix);

  EXPECT_THAT(store.Lookup(name, &blocked_matrix),
              test::IsErrorWithSubstr("Sub-optimal matrix format"));

  TF_ASSERT_OK(store.Lookup(name, &plain_matrix));
  ExpectMatrix(plain_matrix, {{1.0}});

  TF_EXPECT_OK(store.Close());
}

// Tests that the flexible matrix wrapper selects block size 32 for a matrix
// whose size is a multiple of 32.
TEST(FlexibleMatrixVariableStoreWrapperTest, SelectBlocked32MatrixFormat) {
  FlexibleMatrixVariableStoreWrapper store(NewBlockedAndTransposedStore());
  Matrix<float> plain_matrix;
  BlockedMatrix<float> blocked_matrix;
  const string name =
      tensorflow::strings::StrCat("16x32", FlexibleMatrixKernel::kSuffix);

  EXPECT_THAT(store.Lookup(name, &plain_matrix),
              test::IsErrorWithSubstr("Sub-optimal matrix format"));

  TF_ASSERT_OK(store.Lookup(name, &blocked_matrix));
  ExpectBlockedMatrix(blocked_matrix, 16, 32, 32, 32.0);

  TF_EXPECT_OK(store.Close());
}

// Tests that the flexible matrix wrapper selects block size 48 for a matrix
// whose size is a multiple of 48.
TEST(FlexibleMatrixVariableStoreWrapperTest, SelectBlocked48MatrixFormat) {
  FlexibleMatrixVariableStoreWrapper store(NewBlockedAndTransposedStore());
  Matrix<float> plain_matrix;
  BlockedMatrix<float> blocked_matrix;
  const string name =
      tensorflow::strings::StrCat("24x48", FlexibleMatrixKernel::kSuffix);

  EXPECT_THAT(store.Lookup(name, &plain_matrix),
              test::IsErrorWithSubstr("Sub-optimal matrix format"));

  TF_ASSERT_OK(store.Lookup(name, &blocked_matrix));
  ExpectBlockedMatrix(blocked_matrix, 24, 48, 48, 48.0);

  TF_EXPECT_OK(store.Close());
}

}  // namespace
}  // namespace runtime
}  // namespace dragnn
}  // namespace syntaxnet
