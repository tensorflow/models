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

// Computes `[y_1, y_2, ...] = M * [v_1, v_2, ...] + [b_1, b_2, ...]`, where
//
//    M is a `m x n` dense matrix.
//    v_i are `n`-dimensional dense vectors.
//    b_i and y_i are `m`-dimensional dense vectors.
//
// Unfortunately even larger (e.g. 128x128) matrix sizes are not sufficient to
// hide the latency of a function call. So the entire implementation needs to
// live in this header file. Please make sure to use all of the optimization
// flags mentioned in the BUILD file in any client libraries.

#ifndef DRAGNN_RUNTIME_MATH_SGEMVV_H_
#define DRAGNN_RUNTIME_MATH_SGEMVV_H_

#if defined(__SSE2__)
#include <xmmintrin.h>
#endif

#include "dragnn/runtime/math/avx_vector_array.h"
#include "dragnn/runtime/math/types.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"


#define DRAGNN_SGEMVV_ATTRIBUTE_ALWAYS_INLINE __attribute__((always_inline))
#ifdef __clang__
#define DRAGNN_SGEMVV_GCC_UNROLL
#else
#define DRAGNN_SGEMVV_GCC_UNROLL __attribute__((optimize("unroll-loops")))
#endif

namespace syntaxnet {
namespace dragnn {
namespace runtime {

// Represents `v, b` from one operation `y = M * v + b`.
template <int num_ops>
struct SgemvInputBatch {
  const float *input[num_ops];
  const float *initial[num_ops];
};

template <int num_ops>
struct SgemvOutputBatch {
  float *output[num_ops];
};

// Matrix argument for the SGEMV/SGEMVV operation. Based on row-batched
// column-major matrices, but pulls the batch size into a template argument
// so code can be compiled more efficiently.
template <int sse_batch_size, typename ElementType = float>
class SgemvMatrix final {
 public:
  // Convenience type alias.
  using MatrixType =
      BlockedMatrix<ElementType, BlockedMatrixFormat::kRowBlockedColumnMajor>;

  // Creates an empty SgemvMatrix.
  SgemvMatrix() = default;

  // Initializes the new matrix. Returns an InvalidArgumentError if the block
  // size of `matrix` is not equal to `sse_batch_size.
  ::tensorflow::Status Initialize(const MatrixType &matrix);

  // Computes the matrix-vector product with a set of other inputs. See
  // top-level comment for the general algorithm.
  template <int num_ops, int lookahead_1 = 8, int lookahead_2 = 8>
  void DRAGNN_SGEMVV_ATTRIBUTE_ALWAYS_INLINE DRAGNN_SGEMVV_GCC_UNROLL
  MatrixMultiVectorProduct(const SgemvInputBatch<num_ops> &inputs,
                           SgemvOutputBatch<num_ops> *outputs) const {
    MatrixMultiVectorProductImpl<num_ops, /*mask_input_output=*/false,
                                 /*read_initial=*/true, lookahead_1,
                                 lookahead_2>(inputs, -1, outputs);
  }

  // Computes the matrix-vector product with a set of other inputs. See
  // top-level comment for the general algorithm. This variant allows another
  // parameter, `output_vector_elements`, to write to outputs which are a
  // multiple of kAvxWidth (8 floats, or 32 bytes) but not necessarily
  // sse_batch_size. It is slightly slower, but probably more than noise.
  //
  // |lookahead_1| and |lookahead_2| parameters control prefetching, and should
  // usually be tuned via a script. They issue prefetch instructions that are
  // `lookahead_1 * sse_batch_size` values ahead of the current matrix entry
  // being read, if `lookahead_1 != 0` (and `(lookahead_1 + lookahead_2) *
  // sse_batch_size` values, if lookahead_2 != 0). To reiterate, all prefetching
  // can be disabled by setting |lookahead_1| to 0, or the second prefetch can
  // be disabled by setting |lookahead_2| to 0.
  template <int num_ops, int lookahead_1 = 8, int lookahead_2 = 8>
  void DRAGNN_SGEMVV_ATTRIBUTE_ALWAYS_INLINE DRAGNN_SGEMVV_GCC_UNROLL
  MaskedMatrixMultiVectorProduct(const SgemvInputBatch<num_ops> &inputs,
                                 int output_vector_elements,
                                 SgemvOutputBatch<num_ops> *outputs) const {
    MatrixMultiVectorProductImpl<num_ops, /*mask_input_output=*/true,
                                 /*read_initial=*/true, lookahead_1,
                                 lookahead_2>(inputs, output_vector_elements,
                                              outputs);
  }

  // Like the above, but assumes existing values are zero instead of reading
  // them.
  template <int num_ops>
  void DRAGNN_SGEMVV_ATTRIBUTE_ALWAYS_INLINE DRAGNN_SGEMVV_GCC_UNROLL
  MaskedMatrixMultiVectorProductNoInitial(
      const SgemvInputBatch<num_ops> &inputs, int output_vector_elements,
      SgemvOutputBatch<num_ops> *outputs) const {
    MatrixMultiVectorProductImpl<num_ops, /*mask_input_output=*/true,
                                 /*read_initial=*/false>(
        inputs, output_vector_elements, outputs);
  }

  // Read-only accessor.
  const MatrixType &matrix() const { return matrix_; }

 private:
  template <int num_ops, bool mask_input_output, bool read_initial,
            int lookahead_1 = 8, int lookahead_2 = 8>
  DRAGNN_SGEMVV_ATTRIBUTE_ALWAYS_INLINE DRAGNN_SGEMVV_GCC_UNROLL void
  MatrixMultiVectorProductImpl(const SgemvInputBatch<num_ops> &inputs,
                               int output_vector_elements,
                               SgemvOutputBatch<num_ops> *outputs) const;

  MatrixType matrix_;
};

// Implementation details.
template <int sse_batch_size, typename ElementType>
template <int num_ops, bool mask_input_output, bool read_initial,
          int lookahead_1, int lookahead_2>
inline void DRAGNN_SGEMVV_ATTRIBUTE_ALWAYS_INLINE DRAGNN_SGEMVV_GCC_UNROLL
SgemvMatrix<sse_batch_size, ElementType>::MatrixMultiVectorProductImpl(
    const SgemvInputBatch<num_ops> &inputs, int output_vector_elements,
    SgemvOutputBatch<num_ops> *outputs) const {
  static_assert(sse_batch_size % kAvxWidth == 0,
                "sse_batch_size must be a multiple of kAvxWidth (8).");
  if (mask_input_output) {
    DCHECK_EQ(output_vector_elements % kAvxWidth, 0)
        << "output_vector_elements must be padded to alignment";
  }

  const ElementType *curr_matrix_ptr = matrix_.vector(0).data();

  // Loop over blocks of output rows. Each block of output rows will get a
  // partial sum of the [matrix-vector] dot product, where the range of that
  // partial sum is designated by start_col and end_col.
  for (int row_start = 0; row_start < matrix_.num_rows();
       row_start += sse_batch_size) {
    const int load_store_max_idx =
        (output_vector_elements - row_start) / kAvxWidth;
    AvxFloatVecArray<sse_batch_size / kAvxWidth> accumulators[num_ops];

    // Read inputs.
    for (int op = 0; op < num_ops; ++op) {
      if (read_initial) {
        if (mask_input_output) {
          accumulators[op].Load(&inputs.initial[op][row_start],
                                load_store_max_idx);
        } else {
          accumulators[op].Load(&inputs.initial[op][row_start]);
        }
      } else {
        accumulators[op].LoadConstVector(0.0f);
      }
    }

    // Compute matrix-vector product.
    for (int col = 0; col < matrix_.num_columns(); ++col) {
      if (lookahead_1 != 0) {
#if defined(__SSE2__)
        _mm_prefetch(curr_matrix_ptr + lookahead_1 * sse_batch_size,
                     _MM_HINT_T0);
        if (lookahead_2 != 0) {
          _mm_prefetch(
              curr_matrix_ptr + (lookahead_1 + lookahead_2) * sse_batch_size,
              _MM_HINT_T0);
        }
#endif
      }

      // These are the coefficients from each vector at column `col` (just
      // broadcast over the whole AVX array).
      AvxFloatVec weights[num_ops];
      for (int op = 0; op < num_ops; ++op) {
        weights[op].LoadConstVector(inputs.input[op][col]);
      }

      // Loop over each AVX vector and add the current sub-product.
      AvxFloatVecArray<sse_batch_size / kAvxWidth> matrix_block;
      matrix_block.Load(curr_matrix_ptr);
      curr_matrix_ptr += sse_batch_size;
      for (int row_offset = 0; row_offset < sse_batch_size / kAvxWidth;
           row_offset++) {
        for (int op = 0; op < num_ops; ++op) {
          accumulators[op].vectors[row_offset].AddProductOf(
              weights[op], matrix_block.vectors[row_offset]);
        }
      }
    }

    // Save the results.
    for (int op = 0; op < num_ops; ++op) {
      if (mask_input_output) {
        accumulators[op].Store(&outputs->output[op][row_start],
                               load_store_max_idx);
      } else {
        accumulators[op].Store(&outputs->output[op][row_start]);
      }
    }
  }
}

template <int sse_batch_size, typename ElementType>
::tensorflow::Status SgemvMatrix<sse_batch_size, ElementType>::Initialize(
    const SgemvMatrix<sse_batch_size, ElementType>::MatrixType &matrix) {
  if (matrix.block_size() != sse_batch_size) {
    return ::tensorflow::errors::InvalidArgument(
        "Blocked matrix block_size (", matrix.block_size(),
        ") must be equal to sse_batch_size (", sse_batch_size, ")");
  }
  matrix_ = matrix;
  return ::tensorflow::Status::OK();
}

}  // namespace runtime
}  // namespace dragnn
}  // namespace syntaxnet

#undef DRAGNN_SGEMVV_ATTRIBUTE_ALWAYS_INLINE
#undef DRAGNN_SGEMVV_GCC_UNROLL

#endif  // DRAGNN_RUNTIME_MATH_SGEMVV_H_
