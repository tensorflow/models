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

#include "dragnn/runtime/test/helpers.h"

#include <string>

#include "dragnn/runtime/alignment.h"
#include "dragnn/runtime/math/types.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"

namespace syntaxnet {
namespace dragnn {
namespace runtime {
namespace {

// Fills the |slice| with the |value|.  Slice must have .data() and .size().
template <class Slice, class T>
void Fill(Slice slice, T value) {
  for (size_t i = 0; i < slice.size(); ++i) slice.data()[i] = value;
}

// Returns the sum of all elements in the |slice|, casted to double.  Slice must
// have .data() and .size().
template <class Slice>
double Sum(Slice slice) {
  double sum = 0.0;
  for (size_t i = 0; i < slice.size(); ++i) {
    sum += static_cast<double>(slice.data()[i]);
  }
  return sum;
}

// Expects that the two pointers have the same address.
void ExpectSameAddress(const void *pointer1, const void *pointer2) {
  EXPECT_EQ(pointer1, pointer2);
}

// Tests that each byte of a UniqueView is usable.
TEST(UniqueViewTest, Usable) {
  UniqueView view(100);
  EXPECT_EQ(view->size(), 100);
  Fill(*view, 'x');
  LOG(INFO) << "Prevents elision by optimizer: " << Sum(*view);
  EXPECT_EQ(view->data()[0], 'x');
}

// Tests that each byte of a UniqueArea is usable.
TEST(UniqueAreaTest, Usable) {
  UniqueArea area(10, 100);
  EXPECT_EQ(area->num_views(), 10);
  EXPECT_EQ(area->view_size(), 100);
  for (size_t i = 0; i < 10; ++i) {
    Fill(area->view(i), 'y');
    LOG(INFO) << "Prevents elision by optimizer: " << Sum(area->view(i));
    EXPECT_EQ(area->view(i).data()[0], 'y');
  }
}

// Tests that UniqueVector is empty by default.
TEST(UniqueVectorTest, EmptyByDefault) {
  UniqueVector<float> vector;
  EXPECT_EQ(vector->size(), 0);
}

// Tests that each element of a UniqueVector is usable.
TEST(UniqueVectorTest, Usable) {
  UniqueVector<float> vector(100);
  EXPECT_EQ(vector->size(), 100);
  Fill(*vector, 1.5);
  LOG(INFO) << "Prevents elision by optimizer: " << Sum(*vector);
  EXPECT_EQ((*vector)[0], 1.5);
}

// Tests that UniqueVector also exports a view.
TEST(UniqueVectorTest, View) {
  UniqueVector<float> vector(123);
  ExpectSameAddress(vector.view().data(), vector->data());
  EXPECT_EQ(vector.view().size(), 123 * sizeof(float));
}

// Tests that a UniqueVector can be constructed with an initial value.
TEST(UniqueVectorTest, Initialization) {
  UniqueVector<int> vector({2, 3, 5, 7});
  EXPECT_EQ(vector->size(), 4);
  EXPECT_EQ((*vector)[0], 2);
  EXPECT_EQ((*vector)[1], 3);
  EXPECT_EQ((*vector)[2], 5);
  EXPECT_EQ((*vector)[3], 7);
}

// Tests that UniqueMatrix is empty by default.
TEST(UniqueMatrixTest, EmptyByDefault) {
  UniqueMatrix<float> row_major_matrix;
  EXPECT_EQ(row_major_matrix->num_rows(), 0);
  EXPECT_EQ(row_major_matrix->num_columns(), 0);
}

// Tests that each element of a UniqueMatrix is usable.
TEST(UniqueMatrixTest, Usable) {
  UniqueMatrix<float> row_major_matrix(10, 100);
  EXPECT_EQ(row_major_matrix->num_rows(), 10);
  EXPECT_EQ(row_major_matrix->num_columns(), 100);
  for (size_t i = 0; i < 10; ++i) {
    Fill(row_major_matrix->row(i), 1.75);
    LOG(INFO) << "Prevents elision by optimizer: "
              << Sum(row_major_matrix->row(i));
    EXPECT_EQ(row_major_matrix->row(i)[0], 1.75);
  }
}

// Tests that UniqueMatrix also exports an area.
TEST(UniqueMatrixTest, Area) {
  UniqueMatrix<float> row_major_matrix(12, 34);
  ExpectSameAddress(row_major_matrix.area().view(0).data(),
                    row_major_matrix->row(0).data());
  EXPECT_EQ(row_major_matrix.area().num_views(), 12);
  EXPECT_EQ(row_major_matrix.area().view_size(), 34 * sizeof(float));
}

// Tests that a UniqueMatrix can be constructed with an initial value.
TEST(UniqueMatrixTest, Initialization) {
  UniqueMatrix<int> row_major_matrix({{2, 3, 5}, {7, 11, 13}});
  EXPECT_EQ(row_major_matrix->num_rows(), 2);
  EXPECT_EQ(row_major_matrix->num_columns(), 3);
  EXPECT_EQ(row_major_matrix->row(0)[0], 2);
  EXPECT_EQ(row_major_matrix->row(0)[1], 3);
  EXPECT_EQ(row_major_matrix->row(0)[2], 5);
  EXPECT_EQ(row_major_matrix->row(1)[0], 7);
  EXPECT_EQ(row_major_matrix->row(1)[1], 11);
  EXPECT_EQ(row_major_matrix->row(1)[2], 13);
}

}  // namespace
}  // namespace runtime
}  // namespace dragnn
}  // namespace syntaxnet
