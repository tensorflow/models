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

#ifndef DRAGNN_CORE_TEST_GENERIC_H_
#define DRAGNN_CORE_TEST_GENERIC_H_

#include <utility>

#include <gmock/gmock.h>

#include "syntaxnet/base.h"
#include "tensorflow/core/platform/protobuf.h"
#include "tensorflow/core/platform/test.h"

namespace syntaxnet {
namespace test {

MATCHER_P(EqualsProto, a,
          "Protos " + string(negation ? "aren't" : "are") + " equivalent:") {
  return a.DebugString() == arg.DebugString();
}

// Matches an error status whose message matches |substr|.
MATCHER_P(IsErrorWithSubstr, substr,
          string(negation ? "isn't" : "is") +
          " an error Status whose message matches the substring '" +
          ::testing::PrintToString(substr) + "'") {
  return !arg.ok() && arg.error_message().find(substr) != string::npos;
}

// Matches an error status whose code and message match |code| and |substr|.
MATCHER_P2(IsErrorWithCodeAndSubstr, code, substr,
           string(negation ? "isn't" : "is") +
           " an error Status whose code is " + ::testing::PrintToString(code) +
           " and whose message matches the substring '" +
           ::testing::PrintToString(substr) + "'") {
  return !arg.ok() && arg.code() == code &&
         arg.error_message().find(substr) != string::npos;
}

// Returns the prefix for where the test data is stored.
string GetTestDataPrefix();

}  // namespace test
}  // namespace syntaxnet

#endif  // DRAGNN_CORE_TEST_GENERIC_H_
