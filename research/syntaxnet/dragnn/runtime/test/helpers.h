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

// Helpers to make it less painful to create instances of aligned values.
// Intended for testing or benchmarking; production code should use managed
// memory allocation, for example Operands.

#ifndef DRAGNN_RUNTIME_TEST_HELPERS_H_
#define DRAGNN_RUNTIME_TEST_HELPERS_H_

#include <stddef.h>
#include <algorithm>
#include <functional>
#include <vector>

#include "dragnn/runtime/alignment.h"
#include "dragnn/runtime/math/avx_vector_array.h"
#include "dragnn/runtime/math/types.h"
#include <gmock/gmock.h>
#include "tensorflow/core/platform/logging.h"

namespace syntaxnet {
namespace dragnn {
namespace runtime {

// An aligned view and its uniquely-owned underlying storage.  Can be used like
// a std::unique_ptr<MutableAlignedView>.
class UniqueView {
 public:
  // Creates a view of |size| uninitialized bytes.
  explicit UniqueView(size_t size);

  // Provides std::unique_ptr-like access.
  MutableAlignedView *get() { return &view_; }
  MutableAlignedView &operator*() { return view_; }
  MutableAlignedView *operator->() { return &view_; }

 private:
  // View and its underlying storage.
  UniqueAlignedArray array_;
  MutableAlignedView view_;
};

// An aligned area and its uniquely-owned underlying storage.  Can be used like
// a std::unique_ptr<MutableAlignedArea>.
class UniqueArea {
 public:
  // Creates an area with |num_views| sub-views, each of which has |view_size|
  // uninitialized bytes.  Check-fails on error.
  UniqueArea(size_t num_views, size_t view_size);

  // Provides std::unique_ptr-like access.
  MutableAlignedArea *get() { return &area_; }
  MutableAlignedArea &operator*() { return area_; }
  MutableAlignedArea *operator->() { return &area_; }

 private:
  // Area and its underlying storage.
  UniqueAlignedArray array_;
  MutableAlignedArea area_;
};

// A vector and its uniquely-owned underlying storage.  Can be used like a
// std::unique_ptr<MutableVector<T>>.
template <class T>
class UniqueVector {
 public:
  // Creates an empty vector.
  UniqueVector() : UniqueVector(0) {}

  // Creates a vector with |dimension| uninitialized Ts.
  explicit UniqueVector(size_t dimension)
      : view_(dimension * sizeof(T)), vector_(*view_) {}

  // Creates a vector initialized to hold the |values|.
  explicit UniqueVector(const std::vector<T> &values);

  // Provides std::unique_ptr-like access.
  MutableVector<T> *get() { return &vector_; }
  MutableVector<T> &operator*() { return vector_; }
  MutableVector<T> *operator->() { return &vector_; }

  // Returns a view pointing to the same memory.
  MutableAlignedView view() { return *view_; }

 private:
  // Vector and its underlying view.
  UniqueView view_;
  MutableVector<T> vector_;
};

// A matrix and its uniquely-owned underlying storage.  Can be used like a
// std::unique_ptr<MutableMatrix<T>>>.
template <class T>
class UniqueMatrix {
 public:
  // Creates an empty matrix.
  UniqueMatrix() : UniqueMatrix(0, 0) {}

  // Creates a matrix with |num_rows| x |num_columns| uninitialized Ts.
  UniqueMatrix(size_t num_rows, size_t num_columns)
      : area_(num_rows, num_columns * sizeof(T)), matrix_(*area_) {}

  // Creates a matrix initialized to hold the |values|.
  explicit UniqueMatrix(const std::vector<std::vector<T>> &values);

  // Provides std::unique_ptr-like access.
  MutableMatrix<T> *get() { return &matrix_; }
  MutableMatrix<T> &operator*() { return matrix_; }
  MutableMatrix<T> *operator->() { return &matrix_; }

  // Returns an area pointing to the same memory.
  MutableAlignedArea area() { return *area_; }

 private:
  // Matrix and its underlying area.
  UniqueArea area_;
  MutableMatrix<T> matrix_;
};

// Implementation details below.

template <class T>
UniqueVector<T>::UniqueVector(const std::vector<T> &values)
    : UniqueVector(values.size()) {
  std::copy(values.begin(), values.end(), vector_.begin());
}

template <class T>
UniqueMatrix<T>::UniqueMatrix(const std::vector<std::vector<T>> &values)
    : UniqueMatrix(values.size(), values.empty() ? 0 : values[0].size()) {
  for (size_t i = 0; i < values.size(); ++i) {
    CHECK_EQ(values[0].size(), values[i].size());
    std::copy(values[i].begin(), values[i].end(), matrix_.row(i).begin());
  }
}

// Expects that the |matrix| contains the |data|.
template <class T>
void ExpectMatrix(Matrix<T> matrix, const std::vector<std::vector<T>> &data) {
  ASSERT_EQ(matrix.num_rows(), data.size());
  if (data.empty()) return;
  ASSERT_EQ(matrix.num_columns(), data[0].size());
  for (size_t row = 0; row < data.size(); ++row) {
    for (size_t column = 0; column < data[row].size(); ++column) {
      EXPECT_EQ(matrix.row(row)[column], data[row][column]);
    }
  }
}

// Initializes a floating-point vector with random values, using a normal
// distribution centered at 0 with standard deviation 1.
void InitRandomVector(MutableVector<float> vector);

void InitRandomMatrix(MutableMatrix<float> matrix);

// Fuzz test using AVX vectors.
// If this file gets too big, move into something like math/test_helpers.h.
void AvxVectorFuzzTest(
    const std::function<void(AvxFloatVec *vec)> &run,
    const std::function<void(float input_value, float output_value)> &check);

}  // namespace runtime
}  // namespace dragnn
}  // namespace syntaxnet

#endif  // DRAGNN_RUNTIME_TEST_HELPERS_H_
