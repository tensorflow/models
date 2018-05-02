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

#include "dragnn/runtime/lstm_cell/test_helpers.h"
#include "dragnn/runtime/math/transformations.h"
#include "dragnn/runtime/test/helpers.h"
#include "syntaxnet/base.h"

namespace syntaxnet {
namespace dragnn {
namespace runtime {
namespace benchmark {

// Covenience aliases, since we always use the same batch size.
using CellMatrix = SgemvMatrix<LstmCellFunction<>::kBatchSize>;

// This class allocates matrices and vectors contiguously, in the order they
// were requested. It estimates the storage necessary from the beginning, and
// CHECK-fails if this is insufficient. Ergo it should only be used for
// benchmarking.
class CoherentStorage : public VectorMatrixStorage {
 public:
  CoherentStorage() {
    constexpr int kMaxHiddenSize = 256;

    // This should be enough, though could be improved by factoring in input
    // size. Please adjust this class if it is not.
    array_.Resize(10 * sizeof(float) *
                  ComputeAlignedAreaSize(kMaxHiddenSize, kMaxHiddenSize));
  }

  MutableVector<float> RandomVector(int size) override {
    auto view = GetNextView(size);
    return MutableVector<float>(view, size);
  }

 protected:
  MutableBlockedMatrix<float, BlockedMatrixFormat::kRowBlockedColumnMajor>
  RandomBlockedMatrix(int rows, int columns, int batch_size) override {
    int rows_padded = batch_size * ((rows + batch_size - 1) / batch_size);
    int num_views = rows_padded * columns / batch_size;

    auto view = GetNextView(num_views * batch_size);
    MutableAlignedArea area;
    TF_CHECK_OK(area.Reset(view, num_views, batch_size * sizeof(float)));

    // Set random values. It doesn't matter that the rows/cols aren't what we
    // output.
    InitRandomMatrix(MutableMatrix<float>(area));

    // Construct SGEMV matrix types.
    MutableBlockedMatrix<float, BlockedMatrixFormat::kRowBlockedColumnMajor>
        blocked;
    TF_CHECK_OK(blocked.Reset(area, rows_padded, columns));
    return blocked;
  }

 private:
  // Gets the next view, where |size| is the number of floats desired.
  MutableAlignedView GetNextView(size_t size) {
    size_t size_bytes = PadToAlignment(size * sizeof(float));
    MutableAlignedView view;
    TF_CHECK_OK(view.Reset(&array_.view().data()[next_offset_], size_bytes));
    next_offset_ += size_bytes;
    CHECK_LE(next_offset_, array_.view().size());
    return view;
  }

  UniqueAlignedArray array_;

  // Next offset to return.
  int next_offset_ = 0;
};

template <class StorageClass = VectorMatrixStorage>
void LstmCellBenchmark(int hidden_size, int input_dimension, bool is_initial) {
  // RAII storage for vectors and matrices.
  StorageClass storage;

  // Helper function. Because StorageClass is template, we need to call
  // templated member functions with the `template` keyword as well, which gets
  // verbose.
  auto random_matrix = [&storage](int rows, int columns) {
    return storage.template RandomMatrix<LstmCellFunction<>::kBatchSize>(
        rows, columns);
  };

  // Parameters for the LSTM cell, and for one run. We allocate them together
  // so that it's easy to experiment with more coherent initialization schemes.
  MutableVector<float> cell_input_state_output_bias =
      storage.RandomVector(3 * hidden_size);
  CellMatrix input_to_cell_input_state_output =
      random_matrix(3 * hidden_size, input_dimension);
  Vector<float> input(storage.RandomVector(input_dimension));
  MutableVector<float> cell_input_temp = storage.RandomVector(3 * hidden_size);
  CellMatrix last_hidden_to_cell_input_state_output =
      random_matrix(3 * hidden_size, hidden_size);
  Vector<float> last_hidden(storage.RandomVector(hidden_size));
  CellMatrix last_cell_state_to_cell_input =
      random_matrix(hidden_size, hidden_size);
  Vector<float> last_cell_state(storage.RandomVector(hidden_size));
  MutableVector<float> cell_state = storage.RandomVector(hidden_size);
  CellMatrix cell_state_to_cell_output =
      random_matrix(hidden_size, hidden_size);
  MutableVector<float> cell_output = storage.RandomVector(hidden_size);
  MutableVector<float> next_hidden = storage.RandomVector(hidden_size);

  // TODO(googleuser): Benchmark with different matrix element types.
  LstmCellFunction<float> cell;
  TF_CHECK_OK(cell.Initialize(
      hidden_size, Vector<float>(cell_input_state_output_bias),
      input_to_cell_input_state_output, last_hidden_to_cell_input_state_output,
      last_cell_state_to_cell_input, cell_state_to_cell_output));

  double flops_per_run = cell.FlopsPerRun(is_initial);

  auto start_time = std::chrono::system_clock::now();
  int kIterations = static_cast<int>(10e9 / flops_per_run);
  for (int i = 0; i < kIterations; ++i) {
    TF_CHECK_OK(cell.Run(is_initial, input, last_hidden, last_cell_state,
                         cell_input_temp, cell_state, cell_output,
                         next_hidden));
  }
  auto end_time = std::chrono::system_clock::now();
  std::chrono::duration<double> elapsed_seconds = end_time - start_time;
  double elapsed = elapsed_seconds.count();

  double flops = flops_per_run * kIterations;
  std::cerr << "Cell with hidden=" << hidden_size
            << ", input_dimension=" << input_dimension
            << ", is_initial=" << is_initial
            << " kflops/run=" << std::round(flops_per_run / 1e3)
            << ", average GFLOPS=" << flops / 1e9 / elapsed << std::endl;
}

enum class Subcomputation { kAll, kInputOnly, kRecurrentOnly };

template <class StorageClass = VectorMatrixStorage,
          Subcomputation computation = Subcomputation::kAll>
void LstmCellMultiTokenBenchmark(int hidden_size, int input_dimension,
                                 int tokens_per_sentence) {
  std::cerr << "Document benchmark with hidden=" << hidden_size
            << ", input_dimension=" << input_dimension
            << ", tokens_per_sentence=" << tokens_per_sentence;

  // RAII storage for vectors and matrices.
  StorageClass storage;
  MatrixParameters parameters;
  parameters.Init(hidden_size, input_dimension, &storage);

  // Parameters for one run of the LSTM cell.
  UniqueMatrix<float> inputs(tokens_per_sentence, input_dimension);
  UniqueMatrix<float> cell_input_temps(tokens_per_sentence, 3 * hidden_size);
  UniqueMatrix<float> hiddens(tokens_per_sentence, hidden_size);
  InitRandomMatrix(*inputs);
  InitRandomMatrix(*cell_input_temps);
  InitRandomMatrix(*hiddens);

  MutableVector<float> cell_state = storage.RandomVector(hidden_size);
  MutableVector<float> cell_output = storage.RandomVector(hidden_size);

  // TODO(googleuser): Benchmark with different matrix element types.
  LstmCellFunction<float> cell;
  TF_CHECK_OK(parameters.InitializeCell(&cell));

  // There is 1 iniital state and n-1 non-initial states.
  double input_flops =
      tokens_per_sentence * 2.0 * (3 * hidden_size) * input_dimension;
  double flops_per_run = cell.FlopsPerRun(true) +
                         (tokens_per_sentence - 1) * cell.FlopsPerRun(false);
  if (computation == Subcomputation::kInputOnly) {
    flops_per_run = input_flops;
  } else if (computation == Subcomputation::kRecurrentOnly) {
    flops_per_run -= input_flops;
  }

  auto start_time = std::chrono::system_clock::now();
  int kIterations = static_cast<int>(10e9 / flops_per_run);
  for (int iter = 0; iter < kIterations; ++iter) {
    // SGEMVV input to [cell input, cell state, cell output] computation.
    if (computation != Subcomputation::kRecurrentOnly) {
      TF_CHECK_OK(
          cell.RunInputComputations(Matrix<float>(*inputs), *cell_input_temps));
    }

    // Run recurrent parts of the network.
    if (computation != Subcomputation::kInputOnly) {
      for (int i = 0; i < tokens_per_sentence; ++i) {
        Vector<float> last_cell_state;
        Vector<float> last_hidden;
        if (i != 0) {
          last_cell_state = Vector<float>(cell_state);
          last_hidden = Vector<float>(hiddens->row(i - 1));
        }
        TF_CHECK_OK(cell.RunRecurrentComputation(
            i == 0, last_hidden, last_cell_state, cell_input_temps->row(i),
            cell_state, cell_output, hiddens->row(i)));
      }
    }
  }
  auto end_time = std::chrono::system_clock::now();
  std::chrono::duration<double> elapsed_seconds = end_time - start_time;
  double elapsed = elapsed_seconds.count();

  double flops = flops_per_run * kIterations;
  std::cerr << " kflops/run=" << std::round(flops_per_run / 1e3)
            << ", average GFLOPS=" << flops / 1e9 / elapsed << std::endl;
}

}  // namespace benchmark
}  // namespace runtime
}  // namespace dragnn
}  // namespace syntaxnet

using syntaxnet::dragnn::runtime::VectorMatrixStorage;
using syntaxnet::dragnn::runtime::benchmark::CoherentStorage;
using syntaxnet::dragnn::runtime::benchmark::LstmCellBenchmark;
using syntaxnet::dragnn::runtime::benchmark::LstmCellMultiTokenBenchmark;
using syntaxnet::dragnn::runtime::benchmark::Subcomputation;

int main(int argc, char **argv) {
  LstmCellBenchmark(64, 32, false);
  LstmCellBenchmark(96, 32, false);
  LstmCellBenchmark(128, 32, false);
  LstmCellBenchmark(256, 32, false);

  std::cerr << std::endl << "With coherent memory:" << std::endl;
  LstmCellBenchmark<CoherentStorage>(64, 32, false);
  LstmCellBenchmark<CoherentStorage>(96, 32, false);
  LstmCellBenchmark<CoherentStorage>(128, 32, false);
  LstmCellBenchmark<CoherentStorage>(256, 32, false);

  // These are used for tuning coefficients in cell_function.cc.
  std::cerr << std::endl;
  LstmCellMultiTokenBenchmark(48, 32, 10);
  LstmCellMultiTokenBenchmark(64, 32, 5);
  LstmCellMultiTokenBenchmark(64, 32, 10);
  LstmCellMultiTokenBenchmark(96, 96, 2);
  LstmCellMultiTokenBenchmark(96, 96, 5);
  LstmCellMultiTokenBenchmark(96, 96, 10);
  LstmCellMultiTokenBenchmark(96, 96, 20);
  LstmCellMultiTokenBenchmark(128, 32, 2);
  LstmCellMultiTokenBenchmark(128, 32, 5);
  LstmCellMultiTokenBenchmark(128, 32, 10);
  LstmCellMultiTokenBenchmark(128, 32, 20);
  LstmCellMultiTokenBenchmark(128, 128, 10);
  LstmCellMultiTokenBenchmark(144, 32, 10);
  LstmCellMultiTokenBenchmark(256, 32, 10);

  std::cerr << std::endl
            << "Input computation only (similar to sgemvv_test):" << std::endl;
  LstmCellMultiTokenBenchmark<VectorMatrixStorage, Subcomputation::kInputOnly>(
      96, 96, 2);
  LstmCellMultiTokenBenchmark<VectorMatrixStorage, Subcomputation::kInputOnly>(
      96, 96, 10);
  LstmCellMultiTokenBenchmark<VectorMatrixStorage, Subcomputation::kInputOnly>(
      96, 96, 20);

  std::cerr << std::endl << "Recurrent computation only:" << std::endl;
  LstmCellMultiTokenBenchmark<VectorMatrixStorage,
                              Subcomputation::kRecurrentOnly>(96, 96, 2);
  LstmCellMultiTokenBenchmark<VectorMatrixStorage,
                              Subcomputation::kRecurrentOnly>(96, 96, 10);
  LstmCellMultiTokenBenchmark<VectorMatrixStorage,
                              Subcomputation::kRecurrentOnly>(96, 96, 20);

  std::cerr << std::endl << "With coherent memory:" << std::endl;
  LstmCellMultiTokenBenchmark<CoherentStorage>(48, 32, 10);
  LstmCellMultiTokenBenchmark<CoherentStorage>(64, 32, 10);
  LstmCellMultiTokenBenchmark<CoherentStorage>(96, 32, 10);
  LstmCellMultiTokenBenchmark<CoherentStorage>(128, 32, 10);
  LstmCellMultiTokenBenchmark<CoherentStorage>(144, 32, 10);

  return 0;
}
