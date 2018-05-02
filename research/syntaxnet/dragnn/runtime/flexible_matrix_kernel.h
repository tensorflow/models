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

#ifndef DRAGNN_RUNTIME_FLEXIBLE_MATRIX_KERNEL_H_
#define DRAGNN_RUNTIME_FLEXIBLE_MATRIX_KERNEL_H_

#include "dragnn/runtime/alignment.h"
#include "dragnn/runtime/math/arithmetic.h"
#include "dragnn/runtime/math/sgemvv.h"
#include "dragnn/runtime/math/types.h"
#include "dragnn/runtime/variable_store.h"

#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"

#define DRAGNN_FMK_ATTRIBUTE_ALWAYS_INLINE __attribute__((always_inline)) inline

namespace syntaxnet {
namespace dragnn {
namespace runtime {

// Matrix-vector multiplication helper, which will switch the type of the
// underlying matrix based on estimates of how well it will perform. For
// example, a 32x32 matrix-multiplication might get a specialized SGEMV routine,
// while a 2x128 matrix-multiplication might use a naive (non-SSE/AVX)
// algorithm.
//
// Call Initialize() before using, then call one of the MatrixVectorProduct()
// routines.
class FlexibleMatrixKernel {
 public:
  // Suffix appended to variable lookups issued by the kernel.
  static constexpr char kSuffix[] = "/FlexibleMatrixKernel";

  FlexibleMatrixKernel() = default;

  // Initializes the underlying matrices for this kernel; call this method
  // before using this class. Arguments: |debug_name| is the name of the matrix
  // being accessed, which usually should specify the component name and other
  // relevant aspects; |weights_name| is the name of the variable in the
  // TensorFlow graph to access; |output_dimension| is the real output
  // dimension, which is comparable to the number of rows in the matrix but does
  // not include padding; |variable_store| is the store which is queried for
  // variables.
  tensorflow::Status Initialize(const string &debug_name,
                                const string &weights_name,
                                int output_dimension,
                                VariableStore *variable_store);

  // Number of columns for the matrix. This may be padded, if a blocked format
  // is chosen.
  int NumPaddedRows() const;

  // Number of columns for the matrix. This should not be padded.
  int NumColumns() const;

  // Whether a layer's logical output dimension matches the shape of this class'
  // underlying matrix.
  bool MatchesOutputDimension(int output_dimension) const;

  // Computes the matrix-vector product of a single vector, with an initial
  // value. This runs different code based on what kind of blocked matrix was
  // chosen. There are generally no restrictions, i.e. it is fairly common to
  // have initial == output.
  DRAGNN_FMK_ATTRIBUTE_ALWAYS_INLINE
  void MatrixVectorProduct(Vector<float> input, Vector<float> initial,
                           MutableVector<float> output) const;

  // Computes the matrix-vector product of two vectors at once. This is the
  // entrypoint for SGEMVV, and is more efficient.

  DRAGNN_FMK_ATTRIBUTE_ALWAYS_INLINE
  void MatrixVectorVectorProduct(Vector<float> input0, Vector<float> input1,
                                 Vector<float> initial0, Vector<float> initial1,
                                 MutableVector<float> output0,
                                 MutableVector<float> output1) const;

  // Convenience function, calculating `output += M * input`.
  void AddMatrixVectorProduct(Vector<float> input,
                              MutableVector<float> output) const {
    MatrixVectorProduct(input, Vector<float>(output), output);
  }

  // Same as above, without initial bias.
  DRAGNN_FMK_ATTRIBUTE_ALWAYS_INLINE
  void MatrixVectorProduct(Vector<float> input,
                           MutableVector<float> output) const;

 private:


  enum class WeightsType { kNormal, kBlocked32, kBlocked48 };


  // Returns the human-readable name of a WeightsType.
  static string TypeName(WeightsType value);

  WeightsType weights_type_;

  // Actual matrix data. Which matrix is active is determined by
  // |weights_type_|.
  Matrix<float> weights_;
  SgemvMatrix<32> fast_weights_32_;
  SgemvMatrix<48> fast_weights_48_;

  // Output dimension padded to alignment.
  int padded_output_dimension_;
};

// Implementation details below.

DRAGNN_FMK_ATTRIBUTE_ALWAYS_INLINE
void FlexibleMatrixKernel::MatrixVectorProduct(
    Vector<float> input, Vector<float> initial,
    MutableVector<float> output) const {
  SgemvOutputBatch<1> outputs = {{output.data()}};
  SgemvInputBatch<1> inputs = {{input.data()}, {initial.data()}};
  switch (weights_type_) {
    case WeightsType::kNormal:
      MultiplyMatrixAndVectorWithBias(weights_, initial, input, output);
      return;
    case WeightsType::kBlocked32:
      fast_weights_32_.MaskedMatrixMultiVectorProduct(
          inputs, padded_output_dimension_, &outputs);
      return;
    case WeightsType::kBlocked48:
      fast_weights_48_.MaskedMatrixMultiVectorProduct(
          inputs, padded_output_dimension_, &outputs);
      return;
  }
}

DRAGNN_FMK_ATTRIBUTE_ALWAYS_INLINE
void FlexibleMatrixKernel::MatrixVectorVectorProduct(
    Vector<float> input0, Vector<float> input1, Vector<float> initial0,
    Vector<float> initial1, MutableVector<float> output0,
    MutableVector<float> output1) const {
  SgemvOutputBatch<2> outputs = {{output0.data(), output1.data()}};
  SgemvInputBatch<2> inputs = {{input0.data(), input1.data()},
                               {initial0.data(), initial1.data()}};
  switch (weights_type_) {
    case WeightsType::kNormal:
      MultiplyMatrixAndVectorWithBias(weights_, initial0, input0, output0);
      MultiplyMatrixAndVectorWithBias(weights_, initial1, input1, output1);
      return;
    case WeightsType::kBlocked32:
      fast_weights_32_.MaskedMatrixMultiVectorProduct(
          inputs, padded_output_dimension_, &outputs);
      return;
    case WeightsType::kBlocked48:
      fast_weights_48_.MaskedMatrixMultiVectorProduct(
          inputs, padded_output_dimension_, &outputs);
      return;
  }
}

DRAGNN_FMK_ATTRIBUTE_ALWAYS_INLINE
void FlexibleMatrixKernel::MatrixVectorProduct(
    Vector<float> input, MutableVector<float> output) const {
  SgemvOutputBatch<1> outputs = {{output.data()}};
  SgemvInputBatch<1> inputs = {{input.data()}, {nullptr}};
  switch (weights_type_) {
    case WeightsType::kNormal:
      MultiplyMatrixAndVector(weights_, input, output);
      return;
    case WeightsType::kBlocked32:
      fast_weights_32_.MaskedMatrixMultiVectorProductNoInitial(
          inputs, padded_output_dimension_, &outputs);
      return;
    case WeightsType::kBlocked48:
      fast_weights_48_.MaskedMatrixMultiVectorProductNoInitial(
          inputs, padded_output_dimension_, &outputs);
      return;
  }
}

}  // namespace runtime
}  // namespace dragnn
}  // namespace syntaxnet

#undef DRAGNN_FMK_ATTRIBUTE_ALWAYS_INLINE

#endif  // DRAGNN_RUNTIME_FLEXIBLE_MATRIX_KERNEL_H_
