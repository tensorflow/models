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

// Shape inference functions for DRAGNN ops.

#ifndef DRAGNN_CORE_OPS_SHAPE_HELPERS_H_
#define DRAGNN_CORE_OPS_SHAPE_HELPERS_H_

#include "syntaxnet/ops/shape_helpers.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"

namespace syntaxnet {
namespace dragnn {

// Returns OK if the 0'th input of the |context| is compatible with the shape of
// a ComputeSession handle.
inline tensorflow::Status ComputeSessionHandleInputShape(
    tensorflow::shape_inference::InferenceContext *context) {
  tensorflow::shape_inference::ShapeHandle unused;
  return context->Merge(context->input(0), context->Vector(2), &unused);
}

// Sets the 0'th output of the |context| to have the shape of a ComputeSession
// handle.  Always returns OK.
inline tensorflow::Status ComputeSessionHandleOutputShape(
    tensorflow::shape_inference::InferenceContext *context) {
  context->set_output(0, context->Vector(2));
  return tensorflow::Status::OK();
}

// For convenience, combines ComputeSessionHandle{Input,Output}Shape().
inline tensorflow::Status ComputeSessionHandleInputAndOutputShape(
    tensorflow::shape_inference::InferenceContext *context) {
  TF_RETURN_IF_ERROR(ComputeSessionHandleInputShape(context));
  return ComputeSessionHandleOutputShape(context);
}

}  // namespace dragnn
}  // namespace syntaxnet

#endif  // DRAGNN_CORE_OPS_SHAPE_HELPERS_H_
