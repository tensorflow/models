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

#ifndef DRAGNN_RUNTIME_LSTM_CELL_CELL_FUNCTION_H_
#define DRAGNN_RUNTIME_LSTM_CELL_CELL_FUNCTION_H_

#include "dragnn/runtime/math/avx_vector_array.h"
#include "dragnn/runtime/math/sgemvv.h"
#include "dragnn/runtime/math/types.h"
#include "tensorflow/core/lib/core/status.h"

namespace syntaxnet {
namespace dragnn {
namespace runtime {

// Interface for either type of LSTM cell function. Initialization is
// type-dependent, so not included in the shared interface.
class LstmCellFunctionBase {
 public:
  virtual ~LstmCellFunctionBase() {}

  // Runs the LSTM cell. |is_initial| indicates whether this is the first run.
  // |input| is the embedded feature vector (sometimes denoted "x"),
  // |last_hidden| is the last hidden state, denoted h_{t-1} (null/invalid when
  // |is_initial| is True), and similarly |last_cell_state| is the previous cell
  // state, denoted c_{t-1}.
  //
  // The caller must allocate the temporary |cell_input_temp|, which must be
  // 3 * hidden_size; the first |hidden_size| values will be the cell input
  // vector (which is typically not used externally). |cell_state|,
  // |cell_output|, and |next_hidden| must be |hidden_size|-length vectors.
  //
  // Returns InvalidArgument errors if any of the vector sizes are not expected.
  tensorflow::Status Run(bool is_initial, Vector<float> input,
                         Vector<float> last_hidden,
                         Vector<float> last_cell_state,
                         MutableVector<float> cell_input_temp,
                         MutableVector<float> cell_state,
                         MutableVector<float> cell_output,
                         MutableVector<float> next_hidden) const {
    TF_RETURN_IF_ERROR(RunInputComputations(
        Matrix<float>(input), MutableMatrix<float>(cell_input_temp)));
    return RunRecurrentComputation(is_initial, last_hidden, last_cell_state,
                                   cell_input_temp, cell_state, cell_output,
                                   next_hidden);
  }

  // Runs the LSTM cell input computations.
  //
  // |inputs| constains vectors of embedded feature vectors (sometimes denoted
  // "x"). The caller must allocate the temporary |cell_input_temps|, each of
  // which must be 3 * hidden_size.
  //
  // Returns InvalidArgument errors if any of the vector sizes are not expected.
  virtual tensorflow::Status RunInputComputations(
      Matrix<float> inputs, MutableMatrix<float> cell_input_temps) const = 0;

  // Runs the recurrent part of the LSTM cell.
  //
  // |is_initial| indicates whether this is the first run. The temporary
  // |cell_input_temp| must be from RunInputComputation(), |last_hidden| is the
  // last hidden state, denoted h_{t-1} (null/invalid when |is_initial| is
  // True), and similarly |last_cell_state| is the previous cell state, denoted
  // c_{t-1}.
  //
  // |cell_state|, |cell_output|, and |next_hidden| must be |hidden_size|-length
  // vectors.
  //
  // Returns InvalidArgument errors if any of the vector sizes are not expected.
  virtual tensorflow::Status RunRecurrentComputation(
      bool is_initial, Vector<float> last_hidden, Vector<float> last_cell_state,
      MutableVector<float> cell_input_temp, MutableVector<float> cell_state,
      MutableVector<float> cell_output,
      MutableVector<float> next_hidden) const = 0;

  // Returns the number of floating-point operations necessary for one run. This
  // is typically dominated by matrix-vector-multiply operations, which use 2 *
  // width * height floating point operations.
  virtual double FlopsPerRun(bool is_initial) const = 0;
};

// Helper class which computes the LSTM function. This is a separate class from
// the network unit so that its performance can be tested and tuned separately.
template <typename MatrixElementType = float>
class LstmCellFunction : public LstmCellFunctionBase {
 public:
  // Batch size for SGEMV matrices. It's probably OK to use one batch size,
  // because we concatenate [x2i, x2c, x2o], etc. matrices so there is less
  // inefficiency from batching.
  static constexpr int kBatchSize = 48;

  // Public type alias for the underlying matrix type.
  using MatrixType = SgemvMatrix<kBatchSize, MatrixElementType>;

  LstmCellFunction() = default;

  // Instantiates a LSTM cell function.
  //
  // Pass the following vectors and matrices,
  //
  //  * |cell_input_state_output_bias| - Concatenated bias terms for cell input
  //    (typically denoted `i`), cell state (denoted `c`), and cell output
  //    (denoted `o`).
  //  * |input_to_cell_input_state_output| - A matrix which will compute partial
  //    sums of cell input, state, and output expressions, given the input
  //    vector `x`. This is the concatenation of [x2i], [x2c], and [x2o]
  //    matrices in the Python network builder code.
  //  * |last_hidden_to_cell_input_state_output| - Likewise, computes partial
  //    sums given the last hidden state.
  //  * |last_cell_state_to_cell_input| - Used to compute partial sum of cell
  //    input, given *previous* cell state.
  //  * |cell_state_to_cell_output| - Used to compute partial sum of cell
  //    output, given *current* cell state.
  //
  // Returns an InvalidArgument error if hidden_size is not a multiple of the
  // AVX width (currently 8). This is used to reduce copying slightly, but is
  // not an essential optimization.
  tensorflow::Status Initialize(
      int hidden_size, Vector<float> cell_input_state_output_bias,
      MatrixType input_to_cell_input_state_output,
      MatrixType last_hidden_to_cell_input_state_output,
      MatrixType last_cell_state_to_cell_input,
      MatrixType cell_state_to_cell_output);

  // Implements LstmCellFunctionBase.
  tensorflow::Status RunInputComputations(
      Matrix<float> inputs,
      MutableMatrix<float> cell_input_temps) const override;
  tensorflow::Status RunRecurrentComputation(
      bool is_initial, Vector<float> last_hidden, Vector<float> last_cell_state,
      MutableVector<float> cell_input_temp, MutableVector<float> cell_state,
      MutableVector<float> cell_output,
      MutableVector<float> next_hidden) const override;
  double FlopsPerRun(bool is_initial) const override;

 private:
  // Hidden layer size.
  int hidden_size_;

  // Concatenated bias terms for cell input (typically denoted `i`), cell state
  // (denoted `c`), and cell output (denoted `o`).
  Vector<float> cell_input_state_output_bias_;

  // A matrix which will compute partial sums of cell input, state, and output
  // expressions, given the input vector `x`. This is the concatenation of
  // [x2i], [x2c], and [x2o] matrices in the Python network builder code.
  MatrixType input_to_cell_input_state_output_;

  // Likewise, computes partial sums given the last hidden state.
  MatrixType last_hidden_to_cell_input_state_output_;

  // Used to compute partial sum of cell input, given *previous* cell state.
  MatrixType last_cell_state_to_cell_input_;

  // Used to compute partial sum of cell output, given *current* cell state.
  MatrixType cell_state_to_cell_output_;
};

}  // namespace runtime
}  // namespace dragnn
}  // namespace syntaxnet

#endif  // DRAGNN_RUNTIME_LSTM_CELL_CELL_FUNCTION_H_
