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

#ifndef DRAGNN_RUNTIME_LSTM_CELL_TEST_HELPERS_H_
#define DRAGNN_RUNTIME_LSTM_CELL_TEST_HELPERS_H_

#include "dragnn/runtime/lstm_cell/cell_function.h"
#include "dragnn/runtime/math/float16_types.h"
#include "dragnn/runtime/math/sgemvv.h"
#include "dragnn/runtime/test/helpers.h"

namespace syntaxnet {
namespace dragnn {
namespace runtime {

// Contains storage for multiple arrays during the test. This one is
// simple/naive: it just allocates new objects, whereever malloc places them.
// A more advanced version is in cell_function_benchmark.cc, but doesn't seem
// to have much benefit yet.
class VectorMatrixStorage {
 public:
  VectorMatrixStorage() {}
  virtual ~VectorMatrixStorage() {}

  // Allocates a vector and fills it with random values.
  virtual MutableVector<float> RandomVector(int size) {
    vectors_.emplace_back(size);
    InitRandomVector(*vectors_.back());
    return *vectors_.back();
  }

  // Allocates a SGEMV matrix and fills it with random values. Subclasses can
  // implement RandomBlockedMatrix(), which doesn't rely on a template
  // parameter.
  template <int batch_size>
  SgemvMatrix<batch_size> RandomMatrix(int rows, int columns) {
    auto blocked = RandomBlockedMatrix(rows, columns, batch_size);
    SgemvMatrix<batch_size> sgemv_matrix;
    TF_CHECK_OK(sgemv_matrix.Initialize(blocked.AsConst()));
    return sgemv_matrix;
  }

  // Allocates a bfloat16 version of a matrix.
  template <int batch_size>
  SgemvMatrix<batch_size, TruncatedFloat16> ConvertToHalfFloat(
      const SgemvMatrix<batch_size> &matrix) {
    auto blocked = ConvertBlockedMatrix(matrix.matrix());
    SgemvMatrix<batch_size, TruncatedFloat16> sgemv_matrix;
    TF_CHECK_OK(sgemv_matrix.Initialize(blocked.AsConst()));
    return sgemv_matrix;
  }

 protected:
  virtual MutableBlockedMatrix<float,
                               BlockedMatrixFormat::kRowBlockedColumnMajor>
  RandomBlockedMatrix(int rows, int columns, int batch_size);

  virtual MutableBlockedMatrix<TruncatedFloat16,
                               BlockedMatrixFormat::kRowBlockedColumnMajor>
  ConvertBlockedMatrix(
      const BlockedMatrix<float, BlockedMatrixFormat::kRowBlockedColumnMajor>
          &uncompressed);

 private:
  std::vector<UniqueVector<float>> vectors_;
  std::vector<UniqueMatrix<float>> matrices_;
  std::vector<UniqueMatrix<TruncatedFloat16>> converted_matrices_;
};

// Pulls out matrix parameters, makes them usable for multiple LSTM cell
// implementations (namely, unoptimized and normal).
struct MatrixParameters {
  // Convenience aliases, since we always use the same batch size.
  static constexpr int kBatchSize = LstmCellFunction<>::kBatchSize;
  using CellMatrix = typename LstmCellFunction<float>::MatrixType;

  void Init(int hidden_size, int input_dimension, VectorMatrixStorage *storage);

  template <class CellFunction>
  tensorflow::Status InitializeCell(CellFunction *cell) {
    return cell->Initialize(
        hidden_size, Vector<float>(cell_input_state_output_bias),
        input_to_cell_input_state_output,
        last_hidden_to_cell_input_state_output, last_cell_state_to_cell_input,
        cell_state_to_cell_output);
  }

  template <class CellFunction>
  tensorflow::Status InitializeHalfFloatCell(VectorMatrixStorage *storage,
                                             CellFunction *cell) {
    return cell->Initialize(
        hidden_size, Vector<float>(cell_input_state_output_bias),
        storage->ConvertToHalfFloat(input_to_cell_input_state_output),
        storage->ConvertToHalfFloat(last_hidden_to_cell_input_state_output),
        storage->ConvertToHalfFloat(last_cell_state_to_cell_input),
        storage->ConvertToHalfFloat(cell_state_to_cell_output));
  }

  int hidden_size;
  MutableVector<float> cell_input_state_output_bias;
  CellMatrix input_to_cell_input_state_output;
  CellMatrix last_hidden_to_cell_input_state_output;
  CellMatrix last_cell_state_to_cell_input;
  CellMatrix cell_state_to_cell_output;
};

// Implementation details.
inline MutableBlockedMatrix<float, BlockedMatrixFormat::kRowBlockedColumnMajor>
VectorMatrixStorage::RandomBlockedMatrix(int rows, int columns,
                                         int batch_size) {
  int rows_padded = batch_size * ((rows + batch_size - 1) / batch_size);
  int num_views = rows_padded * columns / batch_size;
  matrices_.emplace_back(num_views, batch_size);
  auto &sgemv_storage = matrices_.back();

  // Set random values. It doesn't matter that the rows/cols aren't what we
  // output.
  InitRandomMatrix(*sgemv_storage);

  // Construct SGEMV matrix types.
  MutableBlockedMatrix<float, BlockedMatrixFormat::kRowBlockedColumnMajor>
      blocked;
  TF_CHECK_OK(blocked.Reset(sgemv_storage.area(), rows_padded, columns));
  return blocked;
}

inline void ConvertRow(Vector<float> input,
                       MutableVector<TruncatedFloat16> output) {
  CHECK_EQ(input.size() % 16, 0);
  CHECK_EQ(input.size(), output.size());

  for (int i = 0; i < input.size(); ++i) {
    int i_permuted = (i / 16) * 16 + FastUnpackPermutation(i % 16);
    output[i] = TruncatedFloat16::DebugFromFloat(input[i_permuted]);
  }
}

inline MutableBlockedMatrix<TruncatedFloat16,
                            BlockedMatrixFormat::kRowBlockedColumnMajor>
VectorMatrixStorage::ConvertBlockedMatrix(
    const BlockedMatrix<float, BlockedMatrixFormat::kRowBlockedColumnMajor>
        &uncompressed) {
  converted_matrices_.emplace_back(uncompressed.num_vectors(),
                                   uncompressed.block_size());
  auto &compressed_storage = converted_matrices_.back();
  MutableBlockedMatrix<TruncatedFloat16,
                       BlockedMatrixFormat::kRowBlockedColumnMajor>
      compressed;
  TF_CHECK_OK(compressed.Reset(compressed_storage.area(),
                               uncompressed.num_rows(),
                               uncompressed.num_columns()));

  for (int i = 0; i < uncompressed.num_vectors(); ++i) {
    ConvertRow(uncompressed.vector(i), compressed.vector(i));
  }
  return compressed;
}

inline void MatrixParameters::Init(int hidden_size, int input_dimension,
                                   VectorMatrixStorage *storage) {
  this->hidden_size = hidden_size;
  cell_input_state_output_bias = storage->RandomVector(3 * hidden_size);
  input_to_cell_input_state_output =
      storage->RandomMatrix<kBatchSize>(3 * hidden_size, input_dimension);
  last_hidden_to_cell_input_state_output =
      storage->RandomMatrix<kBatchSize>(3 * hidden_size, hidden_size);
  last_cell_state_to_cell_input =
      storage->RandomMatrix<kBatchSize>(hidden_size, hidden_size);
  cell_state_to_cell_output =
      storage->RandomMatrix<kBatchSize>(hidden_size, hidden_size);
}

}  // namespace runtime
}  // namespace dragnn
}  // namespace syntaxnet

#endif  // DRAGNN_RUNTIME_LSTM_CELL_TEST_HELPERS_H_
