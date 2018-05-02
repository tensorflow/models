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

// Shape inference functions for SyntaxNet ops.

#ifndef SYNTAXNET_OPS_SHAPE_HELPERS_H_
#define SYNTAXNET_OPS_SHAPE_HELPERS_H_

#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"

namespace syntaxnet {

// Returns OK if the |input_index|'th input is a tensor of the |rank| with
// unknown dimensions.
inline tensorflow::Status TensorInputShape(
    int input_index, int rank,
    tensorflow::shape_inference::InferenceContext *context) {
  tensorflow::shape_inference::ShapeHandle unused;
  return context->WithRank(context->input(input_index), rank, &unused);
}

// Returns OK if the |input_index|'th input is a scalar.
inline tensorflow::Status ScalarInputShape(
    int input_index, tensorflow::shape_inference::InferenceContext *context) {
  return TensorInputShape(input_index, 0, context);
}

// Returns OK if the |input_index|'th input is a vector of unknown dimension.
inline tensorflow::Status VectorInputShape(
    int input_index, tensorflow::shape_inference::InferenceContext *context) {
  return TensorInputShape(input_index, 1, context);
}

// Returns OK if the |input_index|'th input is a matrix of unknown dimensions.
inline tensorflow::Status MatrixInputShape(
    int input_index, tensorflow::shape_inference::InferenceContext *context) {
  return TensorInputShape(input_index, 2, context);
}

// Sets the |output_index|'th output to a scalar.
inline void ScalarOutputShape(
    int output_index, tensorflow::shape_inference::InferenceContext *context) {
  context->set_output(output_index, context->Scalar());
}

// Sets the |output_index|'th output to a vector of unknown dimension.
inline void VectorOutputShape(
    int output_index, tensorflow::shape_inference::InferenceContext *context) {
  context->set_output(output_index, context->UnknownShapeOfRank(1));
}

// Sets the |output_index|'th output to a matrix of unknown dimensions.
inline void MatrixOutputShape(
    int output_index, tensorflow::shape_inference::InferenceContext *context) {
  context->set_output(output_index, context->UnknownShapeOfRank(2));
}

}  // namespace syntaxnet

#endif  // SYNTAXNET_OPS_SHAPE_HELPERS_H_
