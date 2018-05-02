// Copyright 2018 Google Inc. All Rights Reserved.
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

#include "dragnn/runtime/math/eigen.h"

#include <vector>

#include "dragnn/core/test/generic.h"
#include "dragnn/runtime/math/types.h"
#include "dragnn/runtime/test/helpers.h"
#include "tensorflow/core/platform/test.h"

namespace syntaxnet {
namespace dragnn {
namespace runtime {
namespace {

// Expects that two pointers point to the same address.
void ExpectSameAddress(const void *ptr1, const void *ptr2) {
  EXPECT_EQ(ptr1, ptr2);
}

// Expects that the |vector| has the |values|.
void ExpectValues(MutableVector<float> vector,
                  const std::vector<float> &values) {
  ASSERT_EQ(vector.size(), values.size());
  for (int i = 0; i < values.size(); ++i) {
    EXPECT_EQ(vector[i], values[i]);
  }
}

// Expects that the Eigen |matrix| has the |values|.
template <class EigenMatrix>
void ExpectValues(const EigenMatrix &matrix,
                  const std::vector<std::vector<float>> &values) {
  ASSERT_EQ(matrix.rows(), values.size());
  for (int row = 0; row < matrix.rows(); ++row) {
    ASSERT_EQ(matrix.cols(), values[row].size());
    for (int column = 0; column < matrix.cols(); ++column) {
      EXPECT_EQ(matrix(row, column), values[row][column]);
    }
  }
}

// Tests that an Eigen vector map references the same memory as the underlying
// runtime vector.
TEST(EigenTest, Vector) {
  UniqueVector<float> vector({1.0, 2.0, 3.0, 4.0});

  EigenVectorMap<float> const_eigen_vector = AsEigenMap(Vector<float>(*vector));
  ExpectSameAddress(const_eigen_vector.data(), vector->data());
  ExpectValues(const_eigen_vector, {{1.0, 2.0, 3.0, 4.0}});

  MutableEigenVectorMap<float> mutable_eigen_vector = AsEigenMap(*vector);
  ExpectSameAddress(mutable_eigen_vector.data(), vector->data());
  ExpectValues(mutable_eigen_vector, {{1.0, 2.0, 3.0, 4.0}});

  // Write into the runtime vector and read from the other views.
  (*vector)[0] = 10.0;
  (*vector)[1] = 20.0;
  (*vector)[2] = 30.0;
  (*vector)[3] = 40.0;
  ExpectValues(const_eigen_vector, {{10.0, 20.0, 30.0, 40.0}});
  ExpectValues(mutable_eigen_vector, {{10.0, 20.0, 30.0, 40.0}});

  // Write into the mutable Eigen vector and read from the other views.
  mutable_eigen_vector << 100.0, 200.0, 300.0, 400.0;
  ExpectValues(const_eigen_vector, {{100.0, 200.0, 300.0, 400.0}});
  ExpectValues(*vector, {100.0, 200.0, 300.0, 400.0});
}

// Tests that an Eigen matrix map references the same memory as the underlying
// runtime vector.
TEST(EigenTest, Matrix) {
  UniqueMatrix<float> matrix({{1.0, 2.0, 3.0},  //
                              {4.0, 5.0, 6.0},  //
                              {7.0, 8.0, 9.0}});

  EigenMatrixMap<float> const_eigen_matrix = AsEigenMap(Matrix<float>(*matrix));
  ExpectSameAddress(const_eigen_matrix.data(), matrix->row(0).data());
  ExpectValues(const_eigen_matrix, {{1.0, 2.0, 3.0},  //
                                    {4.0, 5.0, 6.0},  //
                                    {7.0, 8.0, 9.0}});

  MutableEigenMatrixMap<float> mutable_eigen_matrix = AsEigenMap(*matrix);
  ExpectSameAddress(mutable_eigen_matrix.data(), matrix->row(0).data());
  ExpectValues(mutable_eigen_matrix, {{1.0, 2.0, 3.0},  //
                                      {4.0, 5.0, 6.0},  //
                                      {7.0, 8.0, 9.0}});

  // Write into the runtime matrix and read from the other views.
  matrix->row(0)[0] = 10.0;
  matrix->row(0)[1] = 20.0;
  matrix->row(0)[2] = 30.0;
  matrix->row(1)[0] = 40.0;
  matrix->row(1)[1] = 50.0;
  matrix->row(1)[2] = 60.0;
  matrix->row(2)[0] = 70.0;
  matrix->row(2)[1] = 80.0;
  matrix->row(2)[2] = 90.0;
  ExpectValues(const_eigen_matrix, {{10.0, 20.0, 30.0},  //
                                    {40.0, 50.0, 60.0},  //
                                    {70.0, 80.0, 90.0}});
  ExpectValues(mutable_eigen_matrix, {{10.0, 20.0, 30.0},  //
                                      {40.0, 50.0, 60.0},  //
                                      {70.0, 80.0, 90.0}});

  // Write into the mutable Eigen matrix and read from the other views.
  mutable_eigen_matrix << 100.0, 200.0, 300.0,
                          400.0, 500.0, 600.0,
                          700.0, 800.0, 900.0;
  ExpectValues(const_eigen_matrix, {{100.0, 200.0, 300.0},  //
                                    {400.0, 500.0, 600.0},  //
                                    {700.0, 800.0, 900.0}});
  ExpectValues(matrix->row(0), {100.0, 200.0, 300.0});
  ExpectValues(matrix->row(1), {400.0, 500.0, 600.0});
  ExpectValues(matrix->row(2), {700.0, 800.0, 900.0});
}

}  // namespace
}  // namespace runtime
}  // namespace dragnn
}  // namespace syntaxnet
