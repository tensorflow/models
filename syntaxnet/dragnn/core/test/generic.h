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

#ifndef NLP_SAFT_OPENSOURCE_DRAGNN_CORE_TEST_GENERIC_H_
#define NLP_SAFT_OPENSOURCE_DRAGNN_CORE_TEST_GENERIC_H_

#include <utility>

#include <gmock/gmock.h>

#include "syntaxnet/base.h"
#include "tensorflow/core/platform/protobuf.h"
#include "tensorflow/core/platform/test.h"

namespace syntaxnet {
namespace test {

MATCHER_P(EqualsProto, a, "Protos are not equivalent:") {
  return a.DebugString() == arg.DebugString();
}

// Returns the prefix for where the test data is stored.
string GetTestDataPrefix();

}  // namespace test
}  // namespace syntaxnet

#endif  // NLP_SAFT_OPENSOURCE_DRAGNN_CORE_TEST_GENERIC_H_
