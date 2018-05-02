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

#include "dragnn/runtime/math/float16_types.h"

#include "tensorflow/core/platform/test.h"

namespace syntaxnet {
namespace dragnn {
namespace runtime {
namespace {

// C++11 doesn't support binary literals like 0b01001, so add a helper. :(
uint16 ParseBinaryString(const string &bits) {
  CHECK_EQ(bits.size(), 16) << "ParseBinaryString expects full 16-bit values";
  uint16 value = 0;
  for (const char bit : bits) {
    CHECK(bit == '0' || bit == '1') << "String must be 0's and 1's.";
    value = (value << 1) + (bit == '0' ? 0 : 1);
  }
  return value;
}

TEST(Float16TypesTest, IeeeFloat16Accuracy) {
#if defined(__F16C__)
  bool some_not_exact = false;
  for (int i = -100; i < 100; ++i) {
    float value = i / 10.0f;
    IeeeFloat16 half = IeeeFloat16::DebugFromFloat(value);
    float unpacked = half.DebugToFloat();
    EXPECT_NEAR(value, unpacked, 0.01);
    some_not_exact = some_not_exact || (value != unpacked);
  }
  EXPECT_TRUE(some_not_exact);
#else
  LOG(INFO) << "Test binary wasn't compiled with F16C support, so skipping "
            << "this test.";
#endif
}

TEST(Float16TypesTest, TruncatedAccuracy) {
  bool some_not_exact = false;
  for (int i = -100; i < 100; ++i) {
    float value = i / 10.0f;
    TruncatedFloat16 half = TruncatedFloat16::DebugFromFloat(value);
    float unpacked = half.DebugToFloat();
    EXPECT_NEAR(value, unpacked, 0.06);
    some_not_exact = some_not_exact || (value != unpacked);
  }
  EXPECT_TRUE(some_not_exact);
}

TEST(Float16TypesTest, TruncatedKnownBinaryRepresentation) {
  uint16 neg_1 = ParseBinaryString("1011111110000000");
  uint16 one = ParseBinaryString("0011111110000000");
  EXPECT_EQ((TruncatedFloat16{neg_1}).DebugToFloat(), -1.0f);
  EXPECT_EQ((TruncatedFloat16{one}).DebugToFloat(), 1.0f);
}

TEST(Float16TypesTest, IeeeFloat16KnownBinaryRepresentation) {
#if defined(__F16C__)
  uint16 neg_1 = ParseBinaryString("1011110000000000");
  uint16 one = ParseBinaryString("0011110000000000");
  EXPECT_EQ((IeeeFloat16{neg_1}).DebugToFloat(), -1.0f);
  EXPECT_EQ((IeeeFloat16{one}).DebugToFloat(), 1.0f);
#else
  LOG(INFO) << "Test binary wasn't compiled with F16C support, so skipping "
            << "this test.";
#endif
}

}  // namespace
}  // namespace runtime
}  // namespace dragnn
}  // namespace syntaxnet
