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

#include "dragnn/runtime/variable_store.h"

#include <stddef.h>

#include <algorithm>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "dragnn/core/test/generic.h"
#include "dragnn/protos/runtime.pb.h"
#include "dragnn/runtime/test/fake_variable_store.h"
#include "dragnn/runtime/test/helpers.h"
#include "syntaxnet/base.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/platform/test.h"

namespace syntaxnet {
namespace dragnn {
namespace runtime {
namespace {

// Tests that VariableStore::Lookup() fails to retrieve a vector if the
// underlying area does not have exactly one sub-view.
TEST(VariableStoreTest, LookupEmptyVector) {
  SimpleFakeVariableStore store;
  Vector<uint32> vector32;

  store.MockLookup<uint32>({0}, {});
  EXPECT_THAT(store.Lookup("empty", &vector32),
              test::IsErrorWithSubstr(
                  "Vector variable 'empty' should have 1 sub-view but has 0"));
}

TEST(VariableStoreTest, LookupVectorWrongDimensions) {
  SimpleFakeVariableStore store;
  Vector<float> vector;

  // Dimensions should indicate number of logical elements (1), not bytes (4).
  store.MockLookup<char>({4}, {{'1', '2', '3', '4'}});
  EXPECT_THAT(store.Lookup("wrongdim_1", &vector),
              test::IsErrorWithSubstr(
                  "Vector size (1) disagrees with dimensions[0] (4)"));

  // Missing dimensions raise errors.
  store.MockLookup<char>({}, {{'1', '2', '3', '4'}});
  EXPECT_THAT(store.Lookup("nodims", &vector),
              test::IsErrorWithSubstr("Expected 1 dimensions, got 0"));
}

// Tests that VariableStore::Lookup() fails to retrieve a vector if the
// underlying area is not divisible into elements of sizeof(T) bytes.
TEST(VariableStoreTest, LookupVector) {
  SimpleFakeVariableStore store;
  Vector<uint32> vector32;
  Vector<uint64> vector64;

  store.MockLookup<char>({6}, {{'1', '2', '3', '4', '5', '6'}});
  EXPECT_THAT(
      store.Lookup("123456", &vector32),
      test::IsErrorWithSubstr(
          "Vector variable '123456' does not divide into elements of size 4"));

  store.MockLookup<char>({6}, {{'1', '2', '3', '4', '5', '6'}});
  EXPECT_THAT(
      store.Lookup("123456", &vector64),
      test::IsErrorWithSubstr(
          "Vector variable '123456' does not divide into elements of size 8"));

  store.MockLookup<char>({2}, {{'1', '2', '3', '4', '5', '6', '7', '8'}});
  TF_EXPECT_OK(store.Lookup("12345678", &vector32));
  EXPECT_EQ(vector32.size(), 2);
  const string bytes32(reinterpret_cast<const char *>(vector32.data()), 8);
  EXPECT_EQ(bytes32, "12345678");

  store.MockLookup<uint64>({1}, {{7777}});
  TF_EXPECT_OK(store.Lookup("12345678", &vector64));
  EXPECT_EQ(vector64.size(), 1);
  EXPECT_EQ(vector64[0], 7777);
}

// Tests that the VariableStore fails to lookup a matrix if its dimensions are
// mismatched.
TEST(VariableStoreTest, LookupMatrixWrongDimensions) {
  SimpleFakeVariableStore store;
  Matrix<float> matrix;

  // Missing dimensions raise errors.
  store.MockLookup<char>({}, {{'1', '2', '3', '4'}});
  EXPECT_THAT(store.Lookup("nodims", &matrix),
              test::IsErrorWithSubstr("Expected 2 dimensions, got 0"));

  // Wrong number of columns returned.
  store.MockLookup<char>({1, 2}, {{'1', '2', '3', '4'}});
  EXPECT_THAT(store.Lookup("wrongcols", &matrix),
              test::IsErrorWithSubstr(
                  "Matrix columns (1) disagrees with dimensions[1] (2)"));

  // Wrong number of rows returned.
  store.MockLookup<char>({3, 1}, {{'1', '2', '3', '4'}});
  EXPECT_THAT(store.Lookup("wrongrows", &matrix),
              test::IsErrorWithSubstr(
                  "Matrix rows (1) disagrees with dimensions[0] (3)"));
}

// Tests that VariableStore::Lookup() fails to retrieve a row-major matrix if
// the underlying area is not divisible into elements of sizeof(T) bytes.
TEST(VariableStoreTest, LookupRowMajorMatrix) {
  SimpleFakeVariableStore store;
  Matrix<uint32> matrix32;
  Matrix<uint64> matrix64;

  store.MockLookup<char>(
      {6, 2}, ReplicateRows<char>({'1', '2', '3', '4', '5', '6'}, 6));
  EXPECT_THAT(
      store.Lookup("123456", &matrix32),
      test::IsErrorWithSubstr(
          "Matrix variable '123456' does not divide into elements of size 4"));

  store.MockLookup<char>(
      {6, 2}, ReplicateRows<char>({'1', '2', '3', '4', '5', '6'}, 6));
  EXPECT_THAT(
      store.Lookup("123456", &matrix64),
      test::IsErrorWithSubstr(
          "Matrix variable '123456' does not divide into elements of size 8"));

  store.MockLookup<char>(
      {8, 2}, ReplicateRows<char>({'1', '2', '3', '4', '5', '6', '7', '8'}, 8));
  TF_EXPECT_OK(store.Lookup("12345678", &matrix32));
  EXPECT_EQ(matrix32.num_rows(), 8);
  EXPECT_EQ(matrix32.num_columns(), 2);
  for (size_t i = 0; i < matrix32.num_rows(); ++i) {
    const string bytes32(reinterpret_cast<const char *>(matrix32.row(i).data()),
                         8);
    EXPECT_EQ(bytes32, "12345678");
  }

  store.MockLookup({8, 1}, ReplicateRows<uint64>({7777}, 8));
  TF_EXPECT_OK(store.Lookup("12345678", &matrix64));
  EXPECT_EQ(matrix64.num_rows(), 8);
  EXPECT_EQ(matrix64.num_columns(), 1);
  for (size_t i = 0; i < matrix64.num_rows(); ++i) {
    EXPECT_EQ(matrix64.row(i)[0], 7777);
  }
}

// Tests that the VariableStore fails to lookup a blocked matrix if its
// dimensions are mismatched.
TEST(VariableStoreTest, BlockedLookupWrongDimensions) {
  SimpleFakeVariableStore store;
  BlockedMatrix<float> matrix;

  // Missing dimensions raise errors.
  store.MockLookup<char>({}, {{'1', '2', '3', '4'}});
  EXPECT_THAT(store.Lookup("nodims", &matrix),
              test::IsErrorWithSubstr("Expected 3 dimensions, got 0"));

  // Wrong number of columns returned.
  store.MockLookup<char>({1, 2, 1}, {{'1', '2', '3', '4'}});
  EXPECT_THAT(store.Lookup("wrongcols", &matrix),
              test::IsErrorWithSubstr("Rows * cols (2) != area view size (1)"));

  // Wrong number of rows returned.
  store.MockLookup<char>({3, 1, 1}, {{'1', '2', '3', '4'}});
  EXPECT_THAT(store.Lookup("wrongrows", &matrix),
              test::IsErrorWithSubstr("Rows * cols (3) != area view size (1)"));

  // Wrong area view size.
  store.MockLookup<float>({1, 1, 1}, {{1.0f, 2.0f}});
  EXPECT_THAT(
      store.Lookup("wrongviewsize", &matrix),
      test::IsErrorWithSubstr("Area view size (8) doesn't correspond to block "
                              "size (1) times data type size (4)"));
}

TEST(VariableStoreTest, DoubleBlockedLookup) {
  // BlockedMatrix::Reset() will fail if there is any alignment padding, so we
  // construct an appropriate block size.
  static_assert(internal::kAlignmentBytes % sizeof(double) == 0,
                "Alignment requirement is too small");
  constexpr int kBlockSize = internal::kAlignmentBytes / sizeof(double);
  constexpr int kNumSubMatrices = 3;
  constexpr int kNumRows = 10;
  constexpr int kNumColumns = kNumSubMatrices * kBlockSize;
  constexpr int kNumBlocks = kNumSubMatrices * kNumRows;

  // Fill a data matrix with consecutively increasing values.
  std::vector<std::vector<double>> data;
  double value = 0.0;
  for (int block = 0; block < kNumBlocks; ++block) {
    data.emplace_back();
    for (int i = 0; i < kBlockSize; ++i) data.back().push_back(value++);
  }

  SimpleFakeVariableStore store;
  BlockedMatrix<double> matrix;

  store.MockLookup<double>({kNumRows, kNumColumns, kBlockSize}, data);
  TF_EXPECT_OK(store.Lookup("small_matrix_lookup", &matrix));

  EXPECT_EQ(matrix.num_rows(), kNumRows);
  EXPECT_EQ(matrix.num_columns(), kNumColumns);
  EXPECT_EQ(matrix.block_size(), kBlockSize);
  EXPECT_EQ(matrix.num_vectors(), kNumBlocks);

  double expected = 0.0;
  for (int i = 0; i < kNumBlocks; ++i) {
    for (double value : matrix.vector(i)) EXPECT_EQ(value, expected++);
  }
}

}  // namespace
}  // namespace runtime
}  // namespace dragnn
}  // namespace syntaxnet
