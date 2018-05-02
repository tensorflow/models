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

#include "dragnn/runtime/lstm_cell/cell_function.h"

#include <cmath>

#include <chrono>
#include <iostream>
#include <random>
#include <tuple>

#include "dragnn/core/test/generic.h"
#include "dragnn/runtime/lstm_cell/test_helpers.h"
#include "dragnn/runtime/math/arithmetic.h"
#include "dragnn/runtime/math/transformations.h"
#include "dragnn/runtime/test/helpers.h"
#include "syntaxnet/base.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"

namespace syntaxnet {
namespace dragnn {
namespace runtime {
namespace {

// Covenience aliases, since we always use the same batch size.
constexpr int kBatchSize = LstmCellFunction<>::kBatchSize;
using CellMatrix = SgemvMatrix<kBatchSize>;

// Un-optimized version of the LSTM cell. Practically the same API except the
// constructor takes size arguments.
class UnoptimizedCellFunction {
 public:
  UnoptimizedCellFunction(int hidden_size, int input_size)
      : hidden_size_(hidden_size),
        input_to_cell_input_state_output_(3 * hidden_size, input_size),
        last_hidden_to_cell_input_state_output_(3 * hidden_size, hidden_size),
        last_cell_state_to_cell_input_(hidden_size, hidden_size),
        cell_state_to_cell_output_(hidden_size, hidden_size) {}

  tensorflow::Status Initialize(
      int hidden_size, Vector<float> cell_input_state_output_bias,
      CellMatrix input_to_cell_input_state_output,
      CellMatrix last_hidden_to_cell_input_state_output,
      CellMatrix last_cell_state_to_cell_input,
      CellMatrix cell_state_to_cell_output) {
    cell_input_state_output_bias_ = cell_input_state_output_bias;

    // Copies a padded SGEMV matrix to a non-padded regular matrix.
    auto copy_matrix_to_unpadded = [&](CellMatrix input,
                                       MutableMatrix<float> output) {
      CopyMatrixPrefix(input.matrix(), output.num_rows(), output.num_columns(),
                       &output);
    };
    copy_matrix_to_unpadded(input_to_cell_input_state_output,
                            *input_to_cell_input_state_output_);
    copy_matrix_to_unpadded(last_hidden_to_cell_input_state_output,
                            *last_hidden_to_cell_input_state_output_);
    copy_matrix_to_unpadded(last_cell_state_to_cell_input,
                            *last_cell_state_to_cell_input_);
    copy_matrix_to_unpadded(cell_state_to_cell_output,
                            *cell_state_to_cell_output_);
    return tensorflow::Status::OK();
  }

  tensorflow::Status Run(bool is_initial, Vector<float> input,
                         Vector<float> last_hidden,
                         Vector<float> last_cell_state,
                         MutableVector<float> cell_input_temp,
                         MutableVector<float> cell_state,
                         MutableVector<float> cell_output,
                         MutableVector<float> next_hidden) {
    MutableVector<float> cell_input =
        cell_input_temp.Subsequence(0, hidden_size_);
    MultiplyMatrixAndVectorWithBias(
        Matrix<float>(*input_to_cell_input_state_output_),
        cell_input_state_output_bias_, input, cell_input_temp);
    if (!is_initial) {
      MultiplyMatrixAndVectorWithBias(
          Matrix<float>(*last_hidden_to_cell_input_state_output_),
          Vector<float>(cell_input_temp), last_hidden, cell_input_temp);

      MultiplyMatrixAndVectorWithBias(
          Matrix<float>(*last_cell_state_to_cell_input_),
          Vector<float>(cell_input), last_cell_state, cell_input);
    }

    // Apply sigmoid (using cmath).
    for (int i = 0; i < hidden_size_; ++i) {
      cell_input[i] = 1.0 / (1.0 + exp(-cell_input[i]));
    }

    // Cell state.
    for (int i = 0; i < hidden_size_; ++i) {
      if (is_initial) {
        cell_state[i] = cell_input[i] * tanh(cell_input_temp[hidden_size_ + i]);
      } else {
        float forget = 1.0f - cell_input[i];

        // Recall cell_input_temp[hidden_size_ + i] is the i'th value of
        // the partial sum [x2c] * x_t + [h2c] * h_{t-1} + b_c.
        cell_state[i] =
            (forget * last_cell_state[i]) +
            (cell_input[i] * tanh(cell_input_temp[hidden_size_ + i]));
      }
    }

    // Cell output.
    auto cell_output_partial_sum =
        cell_input_temp.Subsequence(2 * hidden_size_, hidden_size_);
    MultiplyMatrixAndVectorWithBias(Matrix<float>(*cell_state_to_cell_output_),
                                    Vector<float>(cell_output_partial_sum),
                                    Vector<float>(cell_state), cell_output);
    for (int i = 0; i < hidden_size_; ++i) {
      cell_output[i] = 1.0 / (1.0 + exp(-cell_output[i]));
    }

    // Hidden state.
    for (int i = 0; i < hidden_size_; ++i) {
      next_hidden[i] = cell_output[i] * tanh(cell_state[i]);
    }

    return tensorflow::Status::OK();
  }

 private:
  int hidden_size_;
  Vector<float> cell_input_state_output_bias_;
  UniqueMatrix<float> input_to_cell_input_state_output_;
  UniqueMatrix<float> last_hidden_to_cell_input_state_output_;
  UniqueMatrix<float> last_cell_state_to_cell_input_;
  UniqueMatrix<float> cell_state_to_cell_output_;
};

TEST(CellFunctionTest, TestInitializeErrors) {
  int hidden_size = 128;
  int input_dimension = 32;

  // RAII storage for vectors and matrices.
  VectorMatrixStorage storage;

  // LSTM cell.
  Vector<float> cell_input_state_output_bias(
      storage.RandomVector(3 * hidden_size));
  CellMatrix input_to_cell_input_state_output =
      storage.RandomMatrix<kBatchSize>(3 * hidden_size, input_dimension);
  CellMatrix last_hidden_to_cell_input_state_output =
      storage.RandomMatrix<kBatchSize>(3 * hidden_size, hidden_size);
  CellMatrix last_cell_state_to_cell_input =
      storage.RandomMatrix<kBatchSize>(hidden_size, hidden_size);
  CellMatrix cell_state_to_cell_output =
      storage.RandomMatrix<kBatchSize>(hidden_size, hidden_size);

  LstmCellFunction<float> cell;
  EXPECT_THAT(cell.Initialize(
                  hidden_size, Vector<float>(storage.RandomVector(hidden_size)),
                  input_to_cell_input_state_output,
                  last_hidden_to_cell_input_state_output,
                  last_cell_state_to_cell_input, cell_state_to_cell_output),
              test::IsErrorWithSubstr(
                  "Vector/matrix size cell_input_state_output_bias.size() (128)"
                  " does not match expected size 3 * "
                  "hidden_size (384)"));

  EXPECT_THAT(
      cell.Initialize(
          hidden_size, cell_input_state_output_bias,
          storage.RandomMatrix<kBatchSize>(hidden_size, input_dimension),
          last_hidden_to_cell_input_state_output, last_cell_state_to_cell_input,
          cell_state_to_cell_output),
      test::IsErrorWithSubstr("Vector/matrix size "
                              "input_to_cell_input_state_output.matrix().num_"
                              "rows() "
                              "(144) does not match expected size pad_rows(3 * "
                              "hidden_size) (384)"));

  EXPECT_THAT(cell.Initialize(
                  hidden_size, cell_input_state_output_bias,
                  input_to_cell_input_state_output,
                  storage.RandomMatrix<kBatchSize>(hidden_size, hidden_size),
                  last_cell_state_to_cell_input, cell_state_to_cell_output),
              test::IsErrorWithSubstr(
                  "Vector/matrix size "
                  "last_hidden_to_cell_input_state_output.matrix().num_rows() "
                  "(144) does "
                  "not match expected size pad_rows(3 * hidden_size) (384)"));

  EXPECT_THAT(
      cell.Initialize(
          hidden_size, cell_input_state_output_bias,
          input_to_cell_input_state_output,
          storage.RandomMatrix<kBatchSize>(3 * hidden_size, 2 * hidden_size),
          last_cell_state_to_cell_input, cell_state_to_cell_output),
      test::IsErrorWithSubstr("Vector/matrix size "
                              "last_hidden_to_cell_input_state_output.matrix()."
                              "num_columns() (256) does not "
                              "match expected size hidden_size (128)"));

  EXPECT_THAT(
      cell.Initialize(
          hidden_size, cell_input_state_output_bias,
          input_to_cell_input_state_output,
          last_hidden_to_cell_input_state_output,
          storage.RandomMatrix<kBatchSize>(2 * hidden_size, hidden_size),
          cell_state_to_cell_output),
      test::IsErrorWithSubstr(
          "Vector/matrix size "
          "last_cell_state_to_cell_input.matrix().num_rows() (288) does "
          "not match expected size pad_rows(hidden_size) (144)"));

  EXPECT_THAT(
      cell.Initialize(
          hidden_size, cell_input_state_output_bias,
          input_to_cell_input_state_output,
          last_hidden_to_cell_input_state_output,
          storage.RandomMatrix<kBatchSize>(hidden_size, 2 * hidden_size),
          cell_state_to_cell_output),
      test::IsErrorWithSubstr("Vector/matrix size "
                              "last_cell_state_to_cell_input.matrix().num_"
                              "columns() (256) does "
                              "not match expected size hidden_size (128)"));

  EXPECT_THAT(
      cell.Initialize(
          hidden_size, cell_input_state_output_bias,
          input_to_cell_input_state_output,
          last_hidden_to_cell_input_state_output, last_cell_state_to_cell_input,
          storage.RandomMatrix<kBatchSize>(2 * hidden_size, hidden_size)),
      test::IsErrorWithSubstr(
          "Vector/matrix size cell_state_to_cell_output.matrix().num_rows() "
          "(288) does not match expected size "
          "pad_rows(hidden_size) (144)"));

  EXPECT_THAT(
      cell.Initialize(
          hidden_size, cell_input_state_output_bias,
          input_to_cell_input_state_output,
          last_hidden_to_cell_input_state_output, last_cell_state_to_cell_input,
          storage.RandomMatrix<kBatchSize>(hidden_size, 2 * hidden_size)),
      test::IsErrorWithSubstr(
          "Vector/matrix size "
          "cell_state_to_cell_output.matrix().num_columns() (256) does not "
          "match expected size hidden_size (128)"));
}

TEST(CellFunctionTest, TestRunErrors) {
  int hidden_size = 128;
  int input_dimension = 32;

  // RAII storage for vectors and matrices.
  VectorMatrixStorage storage;

  // LSTM cell.
  Vector<float> cell_input_state_output_bias(
      storage.RandomVector(3 * hidden_size));
  CellMatrix input_to_cell_input_state_output =
      storage.RandomMatrix<kBatchSize>(3 * hidden_size, input_dimension);
  CellMatrix last_hidden_to_cell_input_state_output =
      storage.RandomMatrix<kBatchSize>(3 * hidden_size, hidden_size);
  CellMatrix last_cell_state_to_cell_input =
      storage.RandomMatrix<kBatchSize>(hidden_size, hidden_size);
  CellMatrix cell_state_to_cell_output =
      storage.RandomMatrix<kBatchSize>(hidden_size, hidden_size);

  // Per-run inputs.
  Vector<float> input(storage.RandomVector(input_dimension));
  Vector<float> last_hidden(storage.RandomVector(hidden_size));
  Vector<float> last_cell_state(storage.RandomVector(hidden_size));
  MutableVector<float> cell_input_temp = storage.RandomVector(3 * hidden_size);
  MutableVector<float> cell_state = storage.RandomVector(hidden_size);
  MutableVector<float> cell_output = storage.RandomVector(hidden_size);
  MutableVector<float> next_hidden = storage.RandomVector(hidden_size);

  LstmCellFunction<float> cell;
  TF_EXPECT_OK(cell.Initialize(
      hidden_size, cell_input_state_output_bias,
      input_to_cell_input_state_output, last_hidden_to_cell_input_state_output,
      last_cell_state_to_cell_input, cell_state_to_cell_output));
  EXPECT_THAT(
      cell.Run(true, Vector<float>(storage.RandomVector(input_dimension / 2)),
               last_hidden, last_cell_state, cell_input_temp, cell_state,
               cell_output, next_hidden),
      test::IsErrorWithSubstr("Vector/matrix size inputs.num_columns() (16) "
                              "does not match expected size "
                              "input_to_cell_input_state_output_.matrix().num_"
                              "columns() (32)"));

  EXPECT_THAT(cell.Run(false, input,
                       Vector<float>(storage.RandomVector(2 * hidden_size)),
                       last_cell_state, cell_input_temp, cell_state,
                       cell_output, next_hidden),
              test::IsErrorWithSubstr("Vector/matrix size last_hidden.size() "
                                      "(256) does not match expected size "
                                      "hidden_size_ (128)"));

  EXPECT_THAT(cell.Run(false, input, last_hidden,
                       Vector<float>(storage.RandomVector(2 * hidden_size)),
                       cell_input_temp, cell_state, cell_output, next_hidden),
              test::IsErrorWithSubstr(
                  "Vector/matrix size last_cell_state.size() (256) does not "
                  "match expected size hidden_size_ (128)"));

  EXPECT_THAT(cell.Run(true, input, last_hidden, last_cell_state,
                       storage.RandomVector(hidden_size), cell_state,
                       cell_output, next_hidden),
              test::IsErrorWithSubstr(
                  "Vector/matrix size cell_input_temps.num_columns() (128) "
                  "does not match expected size 3 * hidden_size_ (384)"));

  EXPECT_THAT(
      cell.Run(true, input, last_hidden, last_cell_state, cell_input_temp,
               storage.RandomVector(2 * hidden_size), cell_output, next_hidden),
      test::IsErrorWithSubstr("Vector/matrix size cell_state.size() (256) does "
                              "not match expected size hidden_size_ (128)"));

  EXPECT_THAT(
      cell.Run(true, input, last_hidden, last_cell_state, cell_input_temp,
               cell_state, storage.RandomVector(2 * hidden_size), next_hidden),
      test::IsErrorWithSubstr("Vector/matrix size cell_output.size() (256) "
                              "does not match expected size hidden_size_ "
                              "(128)"));

  EXPECT_THAT(
      cell.Run(true, input, last_hidden, last_cell_state, cell_input_temp,
               cell_state, cell_output, storage.RandomVector(2 * hidden_size)),
      test::IsErrorWithSubstr("Vector/matrix size next_hidden.size() (256) "
                              "does not match expected size hidden_size_ "
                              "(128)"));
}

// Test harness, with parameters hidden_size, input_dimension, and is_initial.
class CellFuzzTest
    : public ::testing::TestWithParam<std::tuple<int, int, bool>> {};

TEST_P(CellFuzzTest, TestMatchesNaiveAlgorithm) {
  int hidden_size;
  int input_dimension;
  bool is_initial;
  std::tie(hidden_size, input_dimension, is_initial) = GetParam();

  for (int iter = 0; iter < 100; ++iter) {
    // RAII storage for vectors and matrices.
    VectorMatrixStorage storage;

    // Parameters for the LSTM cell, and for one run. We allocate them together
    // so that it's easy to experiment with more coherent initialization
    // schemes.
    MatrixParameters parameters;
    parameters.Init(hidden_size, input_dimension, &storage);

    // Per-run inputs.
    Vector<float> input(storage.RandomVector(input_dimension));
    Vector<float> last_hidden(storage.RandomVector(hidden_size));
    MutableVector<float> last_cell_state_mutable =
        storage.RandomVector(hidden_size);
    Vector<float> last_cell_state(last_cell_state_mutable);
    MutableVector<float> cell_input_temp =
        storage.RandomVector(3 * hidden_size);
    MutableVector<float> cell_state = storage.RandomVector(hidden_size);
    MutableVector<float> cell_output = storage.RandomVector(hidden_size);
    MutableVector<float> next_hidden = storage.RandomVector(hidden_size);

    // Outputs for un-optimized algorithm.
    MutableVector<float> expected_cell_input_temp =
        storage.RandomVector(3 * hidden_size);
    MutableVector<float> expected_cell_state =
        storage.RandomVector(hidden_size);
    MutableVector<float> expected_cell_output =
        storage.RandomVector(hidden_size);
    MutableVector<float> expected_next_hidden =
        storage.RandomVector(hidden_size);

    UnoptimizedCellFunction unoptimized(hidden_size, input_dimension);
    TF_EXPECT_OK(parameters.InitializeCell(&unoptimized));
    TF_EXPECT_OK(unoptimized.Run(is_initial, input, last_hidden,
                                 last_cell_state, expected_cell_input_temp,
                                 expected_cell_state, expected_cell_output,
                                 expected_next_hidden));

    LstmCellFunction<float> cell;
    TF_EXPECT_OK(parameters.InitializeCell(&cell));
    TF_EXPECT_OK(cell.Run(is_initial, input, last_hidden, last_cell_state,
                          cell_input_temp, cell_state, cell_output,
                          next_hidden));

    // Both this and `bfloat16_tol` below could trip EXPECTs because we are
    // using random values. Adjust judiciously.
    float tol = 1e-6 * hidden_size;
    float bfloat16_tol = 7e-3 * hidden_size;

    // Compare the first values of the cell input state.
    for (int i = 0; i < hidden_size; ++i) {
      EXPECT_NEAR(cell_input_temp[i], expected_cell_input_temp[i], tol);
    }

    // Compare the cell state, cell output, and hidden vectors.
    for (int i = 0; i < hidden_size; ++i) {
      EXPECT_NEAR(cell_state[i], expected_cell_state[i], tol) << " i=" << i;
      EXPECT_NEAR(cell_output[i], expected_cell_output[i], tol) << " i=" << i;
      EXPECT_NEAR(next_hidden[i], expected_next_hidden[i], tol) << " i=" << i;
    }

    // Test float16 version.
    LstmCellFunction<TruncatedFloat16> bfloat16_cell;
    TF_EXPECT_OK(parameters.InitializeHalfFloatCell(&storage, &bfloat16_cell));
    TF_EXPECT_OK(bfloat16_cell.Run(is_initial, input, last_hidden,
                                   last_cell_state, cell_input_temp, cell_state,
                                   cell_output, next_hidden));
    for (int i = 0; i < hidden_size; ++i) {
      EXPECT_NEAR(cell_input_temp[i], expected_cell_input_temp[i],
                  bfloat16_tol);
      EXPECT_NEAR(cell_state[i], expected_cell_state[i], bfloat16_tol);
      EXPECT_NEAR(cell_output[i], expected_cell_output[i], bfloat16_tol);
      EXPECT_NEAR(next_hidden[i], expected_next_hidden[i], bfloat16_tol);
    }

    // Check that it is OK if the cell state vector is consumed (overwritten).
    TF_EXPECT_OK(cell.Run(is_initial, input, last_hidden, last_cell_state,
                          cell_input_temp, last_cell_state_mutable, cell_output,
                          next_hidden));
    for (int i = 0; i < hidden_size; ++i) {
      EXPECT_NEAR(last_cell_state_mutable[i], expected_cell_state[i], tol)
          << " i=" << i;
      EXPECT_NEAR(cell_output[i], expected_cell_output[i], tol) << " i=" << i;
      EXPECT_NEAR(next_hidden[i], expected_next_hidden[i], tol) << " i=" << i;
    }
  }
}

INSTANTIATE_TEST_CASE_P(CellFuzzTestInstance, CellFuzzTest,
                        ::testing::Values(std::make_tuple(8, 32, true),
                                          std::make_tuple(8, 32, false),
                                          std::make_tuple(8, 17, true),
                                          std::make_tuple(8, 17, false),
                                          std::make_tuple(96, 32, true),
                                          std::make_tuple(96, 32, false),
                                          std::make_tuple(128, 32, true),
                                          std::make_tuple(128, 32, false),
                                          std::make_tuple(128, 173, true),
                                          std::make_tuple(128, 173, false)));

// Test harness, with parameters hidden_size, input_dimension.
class CellInputFuzzTest
    : public ::testing::TestWithParam<std::tuple<int, int>> {};

TEST_P(CellInputFuzzTest, TestBulkInputMatches) {
  int hidden_size;
  int input_dimension;
  bool is_initial = true;
  std::tie(hidden_size, input_dimension) = GetParam();

  // RAII storage for vectors and matrices.
  VectorMatrixStorage storage;

  // Parameters for the LSTM cell, and for one run. We allocate them together
  // so that it's easy to experiment with more coherent initialization
  // schemes.
  MatrixParameters parameters;
  parameters.Init(hidden_size, input_dimension, &storage);

  // Per-run inputs.
  UniqueMatrix<float> inputs(2, input_dimension);
  UniqueMatrix<float> cell_input_temps(2, 3 * hidden_size);
  InitRandomMatrix(*inputs);
  InitRandomMatrix(*cell_input_temps);

  // Extra parameters for the naive algorithm, which should run everything.
  Vector<float> last_hidden;
  Vector<float> last_cell_state;
  std::vector<MutableVector<float>> expected_cell_input_temps = {
      storage.RandomVector(3 * hidden_size),
      storage.RandomVector(3 * hidden_size)};
  MutableVector<float> expected_cell_state = storage.RandomVector(hidden_size);
  MutableVector<float> expected_cell_output = storage.RandomVector(hidden_size);
  MutableVector<float> expected_next_hidden = storage.RandomVector(hidden_size);

  UnoptimizedCellFunction unoptimized(hidden_size, input_dimension);
  TF_EXPECT_OK(parameters.InitializeCell(&unoptimized));
  TF_EXPECT_OK(unoptimized.Run(
      is_initial, Vector<float>(inputs->row(0)), last_hidden, last_cell_state,
      expected_cell_input_temps[0], expected_cell_state, expected_cell_output,
      expected_next_hidden));
  TF_EXPECT_OK(unoptimized.Run(
      is_initial, Vector<float>(inputs->row(1)), last_hidden, last_cell_state,
      expected_cell_input_temps[1], expected_cell_state, expected_cell_output,
      expected_next_hidden));

  LstmCellFunction<float> cell;
  TF_EXPECT_OK(parameters.InitializeCell(&cell));
  TF_EXPECT_OK(
      cell.RunInputComputations(Matrix<float>(*inputs), *cell_input_temps));

  // Both this and `bfloat16_tol` below could trip EXPECTs because we are using
  // random values. Adjust judiciously.
  float tol = 1e-7 * hidden_size;
  float bfloat16_tol = 5e-3 * hidden_size;

  // Compare the first values of the cell input state. If we pass
  // RunInputComputation results through the sigmoid function, we should get the
  // same result as calling unoptimized.Run() with is_initial = true.
  for (int i = 0; i < hidden_size; ++i) {
    auto sigmoid = [](float input) { return 1.0 / (1.0 + exp(-input)); };
    EXPECT_NEAR(sigmoid(cell_input_temps->row(0)[i]),
                expected_cell_input_temps[0][i], tol);
    EXPECT_NEAR(sigmoid(cell_input_temps->row(1)[i]),
                expected_cell_input_temps[1][i], tol);
  }

  // Test float16 version.
  LstmCellFunction<TruncatedFloat16> bfloat16_cell;
  TF_EXPECT_OK(parameters.InitializeHalfFloatCell(&storage, &bfloat16_cell));
  TF_EXPECT_OK(bfloat16_cell.RunInputComputations(Matrix<float>(*inputs),
                                                  *cell_input_temps));
  for (int i = 0; i < hidden_size; ++i) {
    auto sigmoid = [](float input) { return 1.0 / (1.0 + exp(-input)); };
    EXPECT_NEAR(sigmoid(cell_input_temps->row(0)[i]),
                expected_cell_input_temps[0][i], bfloat16_tol);
    EXPECT_NEAR(sigmoid(cell_input_temps->row(1)[i]),
                expected_cell_input_temps[1][i], bfloat16_tol);
  }
}

INSTANTIATE_TEST_CASE_P(CellInputFuzzTestInstance, CellInputFuzzTest,
                        ::testing::Values(std::make_tuple(8, 32),
                                          std::make_tuple(8, 17),
                                          std::make_tuple(96, 32),
                                          std::make_tuple(128, 32),
                                          std::make_tuple(128, 173)));

}  // namespace
}  // namespace runtime
}  // namespace dragnn
}  // namespace syntaxnet
