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

#include "dragnn/runtime/activation_functions.h"

#include "dragnn/runtime/math/types.h"
#include "dragnn/runtime/test/helpers.h"
#include "tensorflow/core/platform/test.h"

namespace syntaxnet {
namespace dragnn {
namespace runtime {
namespace {

// Tests that kIdentity is a pass-through.
TEST(ActivationFunctionsTest, ApplyIdentity) {
  UniqueVector<float> values({1.25f, -1.5f, 0.0f, 0.0625f, -0.03125});

  ApplyActivationFunction(ActivationFunction::kIdentity, *values);

  EXPECT_EQ((*values)[0], 1.25);
  EXPECT_EQ((*values)[1], -1.5);
  EXPECT_EQ((*values)[2], 0.0);
  EXPECT_EQ((*values)[3], 0.0625);
  EXPECT_EQ((*values)[4], -0.03125);
}

// Tests that kRelu clips to zero.
TEST(ActivationFunctionsTest, ApplyRelu) {
  UniqueVector<float> values({1.25f, -1.5f, 0.0f, 0.0625f, -0.03125});

  ApplyActivationFunction(ActivationFunction::kRelu, *values);

  EXPECT_EQ((*values)[0], 1.25);
  EXPECT_EQ((*values)[1], 0.0);  // clipped
  EXPECT_EQ((*values)[2], 0.0);  // boundary
  EXPECT_EQ((*values)[3], 0.0625);
  EXPECT_EQ((*values)[4], 0.0);  // clipped
}

}  // namespace
}  // namespace runtime
}  // namespace dragnn
}  // namespace syntaxnet
