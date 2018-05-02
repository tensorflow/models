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

#if defined(__SSE2__)
#include <xmmintrin.h>
#endif

#include "dragnn/runtime/math/avx_activation_functions.h"
#include "tensorflow/core/lib/core/errors.h"

namespace syntaxnet {
namespace dragnn {
namespace runtime {
namespace {

template <class T>
void PrefetchVector(Vector<T> vector) {
#if defined(__SSE2__)
  constexpr size_t kPrefetchStride = 64 / sizeof(T);
  for (int i = 0; i < vector.size(); i += kPrefetchStride) {
    _mm_prefetch(vector.data() + i, _MM_HINT_T1);
  }
#endif
}

// Calls the single-vector instance of SGEMV with output masking. (See SGEMVV
// documentation for |lookahead_1| and |lookahead_2| semantics.
template <int lookahead_1, int lookahead_2, class MatrixType>
void CellMatrixVector(const MatrixType &matrix, Vector<float> input,
                      Vector<float> initial, MutableVector<float> output) {
  SgemvInputBatch<1> inputs{{input.data()}, {initial.data()}};
  SgemvOutputBatch<1> outputs{{output.data()}};

  const bool use_optimized =
      output.size() % LstmCellFunction<>::kBatchSize == 0;
  if (use_optimized) {
    matrix.template MatrixMultiVectorProduct<1, lookahead_1, lookahead_2>(
        inputs, &outputs);
  } else {
    matrix.template MaskedMatrixMultiVectorProduct<1, lookahead_1, lookahead_2>(
        inputs, output.size(), &outputs);
  }
}

// Calls the single-vector instance of SGEMV with output masking, adding to an
// existing vector (partial sum). (See SGEMVV documentation for |lookahead_1|
// and |lookahead_2| semantics.
template <int lookahead_1, int lookahead_2, typename MatrixType>
void CellMatrixVector(const MatrixType &matrix, Vector<float> input,
                      MutableVector<float> initial_and_output) {
  CellMatrixVector<lookahead_1, lookahead_2>(
      matrix, input, Vector<float>(initial_and_output), initial_and_output);
}

// Internal helper function for applying an n-ary function element-wise to
// vectors. We could make it more user-friendly by using a special type
// generator for `indices`, but by taking it explicitly the implementation is
// simpler. Also, public API helpers are easier to interact with.
template <int batch_size, class Function, int... indices>
void ApplyVariadic(const Function &fcn, int size,
                   Vector<float> inputs[sizeof...(indices)],
                   MutableVector<float> output) {
  for (int start = 0; start < size; start += batch_size) {
    const int load_store_max_idx = (size - start) / kAvxWidth;
    AvxFloatVecArray<batch_size / kAvxWidth> arrays[sizeof...(indices)];
    for (int i = 0; i < sizeof...(indices); ++i) {
      // NOTE: This calls .data() to skip debug size checks; it is generally
      // OK to prefetch a bit too far ahead.
      _mm_prefetch(&inputs[i].data()[start + batch_size], _MM_HINT_T0);
      arrays[i].Load(&inputs[i][start], load_store_max_idx);
    }
    for (int i = 0; i < batch_size / kAvxWidth; i++) {
      // We store the result to a random input cell. The choice of the first is
      // actually inconsequential; all we're going to do is write it out later.
      arrays[0].vectors[i] = fcn(arrays[indices].vectors[i]...);
    }
    arrays[0].Store(&output[start], load_store_max_idx);
  }
}

// Apply a unary function on one vector, modifying its contents.
template <int batch_size, class Function>
void ApplyUnary(const Function &fcn, MutableVector<float> vector) {
  Vector<float> inputs[] = {Vector<float>(vector)};
  ApplyVariadic<batch_size, Function, 0>(fcn, vector.size(), inputs, vector);
}

// Apply a binary function on two vectors, storing the result in a (possibly
// separate) output.
template <int batch_size, class Function>
void ApplyBinary(const Function &fcn, Vector<float> arg_1, Vector<float> arg_2,
                 MutableVector<float> result) {
  Vector<float> inputs[] = {arg_1, arg_2};
  ApplyVariadic<batch_size, Function, 0, 1>(fcn, result.size(), inputs, result);
}

template <int batch_size, class Function>
void ApplyTrinary(const Function &fcn, Vector<float> arg_1, Vector<float> arg_2,
                  Vector<float> arg_3, MutableVector<float> result) {
  Vector<float> inputs[] = {arg_1, arg_2, arg_3};
  ApplyVariadic<batch_size, Function, 0, 1, 2>(fcn, result.size(), inputs,
                                               result);
}

AvxFloatVec InitialCellStateFunction(AvxFloatVec cell_input,
                                     AvxFloatVec cell_state_partial_sum) {
  return AvxFloatVec(cell_input * activations::Tanh(cell_state_partial_sum));
}

AvxFloatVec CellStateFunction(AvxFloatVec cell_input,
                              AvxFloatVec last_cell_state,
                              AvxFloatVec cell_state_partial_sum) {
  AvxFloatVec dot_tanh(cell_input * activations::Tanh(cell_state_partial_sum));
  return (AvxFloatVec::Const(1.0) - cell_input) * last_cell_state + dot_tanh;
}

AvxFloatVec HiddenStateFunction(AvxFloatVec cell_output,
                                AvxFloatVec cell_state) {
  return AvxFloatVec(cell_output * activations::Tanh(cell_state));
}

}  // namespace

#define DRAGNN_RETURN_IF_NOT_EQUAL(actual_size, expected_size)  \
  if ((actual_size) != (expected_size)) {                       \
    return tensorflow::errors::InvalidArgument(                 \
        "Vector/matrix size " #actual_size " (", (actual_size), \
        ") does not "                                           \
        "match expected size " #expected_size " (",             \
        (expected_size), ")");                                  \
  }

template <typename MatrixElementType>
tensorflow::Status LstmCellFunction<MatrixElementType>::Initialize(
    int hidden_size, Vector<float> cell_input_state_output_bias,
    SgemvMatrix<kBatchSize, MatrixElementType> input_to_cell_input_state_output,
    SgemvMatrix<kBatchSize, MatrixElementType>
        last_hidden_to_cell_input_state_output,
    SgemvMatrix<kBatchSize, MatrixElementType> last_cell_state_to_cell_input,
    SgemvMatrix<kBatchSize, MatrixElementType> cell_state_to_cell_output) {
  if (hidden_size % kAvxWidth != 0) {
    return tensorflow::errors::InvalidArgument(
        "Expected hidden size (", hidden_size,
        ") to be a multiple of the AVX width (", kAvxWidth, ")");
  }
  auto pad_rows = [](size_t size) {
    return kBatchSize * ((size + kBatchSize - 1) / kBatchSize);
  };
  DRAGNN_RETURN_IF_NOT_EQUAL(cell_input_state_output_bias.size(),
                             3 * hidden_size);
  DRAGNN_RETURN_IF_NOT_EQUAL(
      input_to_cell_input_state_output.matrix().num_rows(),
      pad_rows(3 * hidden_size));
  DRAGNN_RETURN_IF_NOT_EQUAL(
      last_hidden_to_cell_input_state_output.matrix().num_rows(),
      pad_rows(3 * hidden_size));
  DRAGNN_RETURN_IF_NOT_EQUAL(
      last_hidden_to_cell_input_state_output.matrix().num_columns(),
      hidden_size);
  DRAGNN_RETURN_IF_NOT_EQUAL(last_cell_state_to_cell_input.matrix().num_rows(),
                             pad_rows(hidden_size));
  DRAGNN_RETURN_IF_NOT_EQUAL(
      last_cell_state_to_cell_input.matrix().num_columns(), hidden_size);
  DRAGNN_RETURN_IF_NOT_EQUAL(cell_state_to_cell_output.matrix().num_rows(),
                             pad_rows(hidden_size));
  DRAGNN_RETURN_IF_NOT_EQUAL(cell_state_to_cell_output.matrix().num_columns(),
                             hidden_size);
  hidden_size_ = hidden_size;
  cell_input_state_output_bias_ = cell_input_state_output_bias;
  input_to_cell_input_state_output_ = input_to_cell_input_state_output;
  last_hidden_to_cell_input_state_output_ =
      last_hidden_to_cell_input_state_output;
  last_cell_state_to_cell_input_ = last_cell_state_to_cell_input;
  cell_state_to_cell_output_ = cell_state_to_cell_output;
  return tensorflow::Status::OK();
}

template <typename MatrixElementType>
tensorflow::Status LstmCellFunction<MatrixElementType>::RunInputComputations(
    const Matrix<float> inputs,
    const MutableMatrix<float> cell_input_temps) const {
  DRAGNN_RETURN_IF_NOT_EQUAL(inputs.num_rows(), cell_input_temps.num_rows());
  DRAGNN_RETURN_IF_NOT_EQUAL(
      inputs.num_columns(),
      input_to_cell_input_state_output_.matrix().num_columns());
  DRAGNN_RETURN_IF_NOT_EQUAL(cell_input_temps.num_columns(), 3 * hidden_size_);

  const bool use_optimized = (3 * hidden_size_) % kBatchSize == 0;

  // Pair each input with its neighbor, and run SGEMVV.
  SgemvInputBatch<2> sgemvv_inputs;
  SgemvOutputBatch<2> sgemvv_outputs;
  for (int i = 0; i + 1 < inputs.num_rows(); i += 2) {
    for (int op = 0; op < 2; ++op) {
      sgemvv_inputs.input[op] = inputs.row(i + op).data();
      sgemvv_inputs.initial[op] = cell_input_state_output_bias_.data();
      sgemvv_outputs.output[op] = cell_input_temps.row(i + op).data();
    }

    if (use_optimized) {
      input_to_cell_input_state_output_
          .template MatrixMultiVectorProduct<2, 8, 8>(sgemvv_inputs,
                                                      &sgemvv_outputs);
    } else {
      input_to_cell_input_state_output_
          .template MaskedMatrixMultiVectorProduct<2, 8, 8>(
              sgemvv_inputs, 3 * hidden_size_, &sgemvv_outputs);
    }
  }

  // Odd-sized inputs need an additional SGEMV operation.
  if (inputs.num_rows() % 2 != 0) {
    const int i = inputs.num_rows() - 1;
    SgemvInputBatch<1> sgemvv_inputs;
    SgemvOutputBatch<1> sgemvv_outputs;
    sgemvv_inputs.input[0] = inputs.row(i).data();
    sgemvv_inputs.initial[0] = cell_input_state_output_bias_.data();
    sgemvv_outputs.output[0] = cell_input_temps.row(i).data();
    if (use_optimized) {
      input_to_cell_input_state_output_
          .template MatrixMultiVectorProduct<1, 8, 8>(sgemvv_inputs,
                                                      &sgemvv_outputs);
    } else {
      input_to_cell_input_state_output_
          .template MaskedMatrixMultiVectorProduct<1, 8, 8>(
              sgemvv_inputs, 3 * hidden_size_, &sgemvv_outputs);
    }
  }

  return tensorflow::Status::OK();
}

template <typename MatrixElementType>
tensorflow::Status LstmCellFunction<MatrixElementType>::RunRecurrentComputation(
    bool is_initial, Vector<float> last_hidden, Vector<float> last_cell_state,
    MutableVector<float> cell_input_temp, MutableVector<float> cell_state,
    MutableVector<float> cell_output, MutableVector<float> next_hidden) const {
  // Check input sizes.
  if (!is_initial) {
    DRAGNN_RETURN_IF_NOT_EQUAL(last_hidden.size(), hidden_size_);
    DRAGNN_RETURN_IF_NOT_EQUAL(last_cell_state.size(), hidden_size_);
  }
  DRAGNN_RETURN_IF_NOT_EQUAL(cell_input_temp.size(), 3 * hidden_size_);
  DRAGNN_RETURN_IF_NOT_EQUAL(cell_state.size(), hidden_size_);
  DRAGNN_RETURN_IF_NOT_EQUAL(cell_output.size(), hidden_size_);
  DRAGNN_RETURN_IF_NOT_EQUAL(next_hidden.size(), hidden_size_);
#undef DRAGNN_RETURN_IF_NOT_EQUAL

  MutableVector<float> cell_input =
      cell_input_temp.Subsequence(0, hidden_size_);
  Vector<float> cell_state_partial_sum(
      cell_input_temp.Subsequence(hidden_size_, hidden_size_));
  Vector<float> cell_output_partial_sum(
      cell_input_temp.Subsequence(2 * hidden_size_, hidden_size_));

  if (!is_initial) {
    PrefetchVector(last_cell_state);
    CellMatrixVector<16, 0>(last_hidden_to_cell_input_state_output_,
                            last_hidden, cell_input_temp);
    CellMatrixVector<1, 0>(last_cell_state_to_cell_input_, last_cell_state,
                           cell_input);
  }
  ApplyUnary<24>(activations::Sigmoid, cell_input);

  // Computes cell state,
  //
  // $c_t = f_t \cdot c_{t-1} + i_t \cdot tanh([x2c] x_t + [h2c] h_{t-1} + b_c)$
  //
  // where $f_t = 1 - i_t$.
  if (is_initial) {
    ApplyBinary<32>(InitialCellStateFunction, Vector<float>(cell_input),
                    cell_state_partial_sum, cell_state);
  } else {
    ApplyTrinary<16>(CellStateFunction, Vector<float>(cell_input),
                     last_cell_state, cell_state_partial_sum, cell_state);
  }

  // Computes cell output,
  //
  // $o_t = \sigma([x2o] x_t + [h2o] h_{t-1} + [c2o] c_t + b_o)$
  //
  // where all but the $c_t$ component of the affine transformation have already
  // been computed by the composite "ico" matrices above.
  CellMatrixVector<0, 0>(cell_state_to_cell_output_, Vector<float>(cell_state),
                         cell_output_partial_sum, cell_output);
  ApplyUnary<24>(activations::Sigmoid, cell_output);

  // Computes the hidden state,
  //
  // $h_t = o_t \cdot tanh(c_t)$
  ApplyBinary<16>(HiddenStateFunction, Vector<float>(cell_output),
                  Vector<float>(cell_state), next_hidden);

  return tensorflow::Status::OK();
}

template <typename MatrixElementType>
double LstmCellFunction<MatrixElementType>::FlopsPerRun(bool is_initial) const {
  double sum = 0;
  for (const auto &matrix :
       {input_to_cell_input_state_output_, cell_state_to_cell_output_}) {
    sum += (2 * matrix.matrix().num_rows() * matrix.matrix().num_columns());
  }
  if (!is_initial) {
    for (const auto &matrix : {last_hidden_to_cell_input_state_output_,
                               last_cell_state_to_cell_input_}) {
      sum += (2 * matrix.matrix().num_rows() * matrix.matrix().num_columns());
    }
  }

  // Element-wise activation calculations.
  sum += (26 +  // i_t sigmoid
          26 +  // c_t tanh (23) plus 3 more
          26 +  // o_t sigmoid
          24    // h_t tanh and multiplication
          ) *
         hidden_size_;

  return sum;
}

// Instantiate the class for floats and TruncatedFloat16's.
template class LstmCellFunction<float>;
template class LstmCellFunction<TruncatedFloat16>;

}  // namespace runtime
}  // namespace dragnn
}  // namespace syntaxnet
