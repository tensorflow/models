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

#include "dragnn/runtime/math/transformations.h"

#include "dragnn/runtime/test/helpers.h"
#include <gmock/gmock.h>
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"

namespace syntaxnet {
namespace dragnn {
namespace runtime {
namespace {

// Generates a matrix where each value is of the form `aa.bb`, where `aa` is the
// column index and `bb` is the row index.
TEST(TransformationsTest, GenerateRowColIdxMatrix) {
  UniqueMatrix<float> row_col_matrix(5, 5);
  GenerateMatrix(
      5, 5,
      [](int row, int col) { return static_cast<float>(row) + (col / 100.0f); },
      row_col_matrix.get());

  ExpectMatrix(Matrix<float>(*row_col_matrix),
               {{0.0f, 0.01f, 0.02f, 0.03f, 0.04f},
                {1.0f, 1.01f, 1.02f, 1.03f, 1.04f},
                {2.0f, 2.01f, 2.02f, 2.03f, 2.04f},
                {3.0f, 3.01f, 3.02f, 3.03f, 3.04f},
                {4.0f, 4.01f, 4.02f, 4.03f, 4.04f}});
}

TEST(TransformationsTest, CopiesMatrix) {
  UniqueMatrix<float> a({{1, 2}}), b({{3, 4}});
  TF_EXPECT_OK(CopyMatrix(*a, b.get()));

  EXPECT_EQ(b->row(0)[0], 1);
  EXPECT_EQ(b->row(0)[1], 2);
}

TEST(TransformationsTest, CopiesRowBlockedMatrix) {
  UniqueMatrix<double> source({{1, 2, 3},     //
                               {4, 5, 6},     //
                               {7, 8, 9},     //
                               {10, 11, 12},  //
                               {13, 14, 15},  //
                               {16, 17, 18},  //
                               {19, 20, 21},  //
                               {22, 23, 24}});
  UniqueMatrix<double> dst_mem(6, 4);
  MutableBlockedMatrix<double, BlockedMatrixFormat::kRowBlockedColumnMajor>
      blocked;
  TF_EXPECT_OK(blocked.Reset(dst_mem.area(), 8, 3));

  TF_EXPECT_OK(CopyMatrix(*source, &blocked));

  ExpectMatrix(Matrix<double>(*dst_mem), {{1, 4, 7, 10},     //
                                          {2, 5, 8, 11},     //
                                          {3, 6, 9, 12},     //
                                          {13, 16, 19, 22},  //
                                          {14, 17, 20, 23},  //
                                          {15, 18, 21, 24}});
}

// This test is the same as the above, except everything is transposed.
TEST(TransformationsTest, CopiesColumnBlockedMatrix) {
  UniqueMatrix<double> source(         //
      {{1, 4, 7, 10, 13, 16, 19, 22},  //
       {2, 5, 8, 11, 14, 17, 20, 23},  //
       {3, 6, 9, 12, 15, 18, 21, 24}});
  UniqueMatrix<double> dst_mem(6, 4);
  MutableBlockedMatrix<double> blocked;
  TF_EXPECT_OK(blocked.Reset(dst_mem.area(), 3, 8));

  TF_EXPECT_OK(CopyMatrix(*source, &blocked));

  ExpectMatrix(Matrix<double>(*dst_mem), {{1, 4, 7, 10},     //
                                          {2, 5, 8, 11},     //
                                          {3, 6, 9, 12},     //
                                          {13, 16, 19, 22},  //
                                          {14, 17, 20, 23},  //
                                          {15, 18, 21, 24}});
}

}  // namespace
}  // namespace runtime
}  // namespace dragnn
}  // namespace syntaxnet
