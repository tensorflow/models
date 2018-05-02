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

#ifndef DRAGNN_RUNTIME_MATH_ARITHMETIC_SSE_H_
#define DRAGNN_RUNTIME_MATH_ARITHMETIC_SSE_H_
#if defined(__SSE4_2__)

#include <stddef.h>

#include "dragnn/runtime/math/arithmetic_common.h"
#include "dragnn/runtime/math/types.h"
#include "tensorflow/core/platform/logging.h"

namespace syntaxnet {
namespace dragnn {
namespace runtime {

// TODO(googleuser): Leaving this empty means that the definitions
// from arithmetic_common.h carry through.  Provide template specializations
// that use architecture-specific intrinsics.

}  // namespace runtime
}  // namespace dragnn
}  // namespace syntaxnet

#endif  // defined(__SSE4_2__)
#endif  // DRAGNN_RUNTIME_MATH_ARITHMETIC_SSE_H_
