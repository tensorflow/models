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

#include <vector>

#include "dragnn/runtime/flexible_matrix_kernel.h"

#include "dragnn/core/test/generic.h"
#include "dragnn/protos/runtime.pb.h"
#include "dragnn/runtime/math/transformations.h"
#include "dragnn/runtime/test/fake_variable_store.h"
#include "dragnn/runtime/test/helpers.h"

#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/test.h"

namespace syntaxnet {
namespace dragnn {
namespace runtime {

std::vector<std::vector<float>> TestValues(int inner_dimension) {
  std::vector<std::vector<float>> values;
  for (int block = 0; block < 32; ++block) {
    std::vector<float> row_values;
    for (int value = 0; value < inner_dimension; ++value) {
      row_values.push_back(0.1f);
    }
    values.push_back(row_values);
  }
  return values;
}

// Tests that the FlexibleMatrixKernel will use a blocked matrix if that is the
// only available format.
TEST(FlexibleMatrixKernelTest, UseBlockedMatrix) {
  std::vector<std::vector<float>> values = TestValues(32);

  for (int actual_rows : {24, 30, 32}) {
    // Add the variable using a blocked format.
    FakeVariableStore store;
    store.AddOrDie(
        tensorflow::strings::StrCat("weights", FlexibleMatrixKernel::kSuffix),
        values, VariableSpec::FORMAT_COLUMN_BLOCKED_ROW_MAJOR_MATRIX);

    FlexibleMatrixKernel kernel;
    TF_EXPECT_OK(
        kernel.Initialize("test_weights", "weights", actual_rows, &store));
    EXPECT_EQ(kernel.NumPaddedRows(), 32);

    UniqueVector<float> vector(values.back());
    UniqueVector<float> output(actual_rows);

    kernel.MatrixVectorProduct(Vector<float>(*vector), *output);

    // Every value in `output` should be 32 * 0.1 * 0.1 = 0.32.
    for (int i = 0; i < actual_rows; ++i) {
      EXPECT_NEAR((*output)[i], 0.32f, 1e-6f);
    }

    kernel.MatrixVectorProduct(Vector<float>(*vector), Vector<float>(*output),
                               *output);

    // Every value in `output` should be 2 * 32 * 0.1 * 0.1 = 0.64.
    for (int i = 0; i < actual_rows; ++i) {
      EXPECT_NEAR((*output)[i], 0.64f, 1e-6f);
    }
  }
}

// Tests that the FlexibleMatrixKernel will use a non-blocked matrix if that is
// the only available format.
TEST(FlexibleMatrixKernelTest, UseNonBlockedMatrix) {
  const int kOutputDim = 32;
  std::vector<std::vector<float>> values = TestValues(kOutputDim);

  // Add the variable using a non-blocked format.
  FakeVariableStore store;
  store.AddOrDie(
      tensorflow::strings::StrCat("weights", FlexibleMatrixKernel::kSuffix),
      values, VariableSpec::FORMAT_ROW_MAJOR_MATRIX);

  FlexibleMatrixKernel kernel;
  TF_EXPECT_OK(
      kernel.Initialize("test_weights", "weights", kOutputDim, &store));

  EXPECT_EQ(kernel.NumPaddedRows(), 32);
  EXPECT_EQ(kernel.NumColumns(), kOutputDim);

  UniqueVector<float> vector(values.back());
  UniqueVector<float> output(kOutputDim);

  kernel.MatrixVectorProduct(Vector<float>(*vector), *output);

  const float kExpectedFirstResult = kOutputDim * 0.1 * 0.1;
  for (int i = 0; i < kOutputDim; ++i) {
    EXPECT_NEAR((*output)[i], kExpectedFirstResult, 1e-6f);
  }

  kernel.MatrixVectorProduct(Vector<float>(*vector), Vector<float>(*output),
                             *output);

  const float kExpectedSecondResult = 2.0 * kExpectedFirstResult;
  for (int i = 0; i < kOutputDim; ++i) {
    EXPECT_NEAR((*output)[i], kExpectedSecondResult, 1e-6f);
  }
}

TEST(FlexibleMatrixKernelTest, MissingVariableIsFailure) {
  FakeVariableStore store;

  FlexibleMatrixKernel kernel;
  EXPECT_THAT(kernel.Initialize("test_weights", "weights", 30, &store),
              test::IsErrorWithSubstr("Unknown variable: weights"));
}


}  // namespace runtime
}  // namespace dragnn
}  // namespace syntaxnet
