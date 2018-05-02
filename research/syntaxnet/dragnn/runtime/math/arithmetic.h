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

// Top-level organizational header for arithmetic operations.  Users should
// include this instead of directly including the sub-headers below.  See
// arithmetic_common.h for function declarations and comments.
//
// NB: If you wish to use an architecture-specific implementation, make sure to
// add the relevant copts to the cc_library whose .cc file includes this header.

#ifndef DRAGNN_RUNTIME_MATH_ARITHMETIC_H_
#define DRAGNN_RUNTIME_MATH_ARITHMETIC_H_

// Select an architecture-specific implementation, if possible, or fall back to
// the trivial generic implementations.  The order of the clauses is important:
// in cases where architectures may overlap the newer version should be checked
// first (e.g., AVX before SSE).
#if defined(__AVX2__)
#include "dragnn/runtime/math/arithmetic_avx.h"
#elif defined(__SSE4_2__)
#include "dragnn/runtime/math/arithmetic_sse.h"
#elif defined(__ARM_NEON) || defined(__ARM_NEON__)
#include "dragnn/runtime/math/arithmetic_neon.h"
#else  // no architecture-specific implementation
#include "dragnn/runtime/math/arithmetic_common.h"
#endif

#endif  // DRAGNN_RUNTIME_MATH_ARITHMETIC_H_
