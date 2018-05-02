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

// Definitions of activation functions for neural netowrks.

#ifndef DRAGNN_RUNTIME_ACTIVATION_FUNCTIONS_H_
#define DRAGNN_RUNTIME_ACTIVATION_FUNCTIONS_H_

#include "dragnn/runtime/math/arithmetic.h"
#include "dragnn/runtime/math/types.h"

namespace syntaxnet {
namespace dragnn {
namespace runtime {

// Possible types of activation functions.
//
// TODO(googleuser): If many activation functions are added, or if functions start
// using configuration parameters (e.g., leakiness of a leaky ReLU), then switch
// to a registered class.
enum class ActivationFunction {
  kIdentity,  // pass-through, useful for classification logits
  kRelu,      // ReLU; i.e., max(0,x)
};

// Applies the |activation_function| to the |values|.
template <class T>
void ApplyActivationFunction(ActivationFunction activation_function,
                             MutableVector<T> values);

// Implementation details below.

template <class T>
void ApplyActivationFunction(ActivationFunction activation_function,
                             MutableVector<T> values) {
  switch (activation_function) {
    case ActivationFunction::kIdentity:
      break;

    case ActivationFunction::kRelu:
      MaxElements(T(), values);
      break;
  }
}

}  // namespace runtime
}  // namespace dragnn
}  // namespace syntaxnet

#endif  // DRAGNN_RUNTIME_ACTIVATION_FUNCTIONS_H_
