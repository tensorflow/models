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

#include "dragnn/runtime/math/avx_vector_array.h"

#include <cmath>

#include "dragnn/runtime/test/helpers.h"
#include "tensorflow/core/platform/test.h"

namespace syntaxnet {
namespace dragnn {
namespace runtime {
namespace {

TEST(AvxVectorTest, LoadAndStore) {
  UniqueVector<float> input(kAvxWidth);
  UniqueVector<float> output(kAvxWidth);
  InitRandomVector(*input);
  InitRandomVector(*output);

  AvxFloatVec vec;
  vec.Load(input->data());
  vec.Store(output->data());

  for (int i = 0; i < kAvxWidth; ++i) {
    EXPECT_EQ((*input)[i], (*output)[i]);
  }
}

// Test flooring with assignment, just to make the compiler not erase aliases.
TEST(AvxVectorTest, AssignmentAndFloor) {
  UniqueVector<float> input(kAvxWidth);
  UniqueVector<float> output(kAvxWidth);
  UniqueVector<float> floored(kAvxWidth);
  InitRandomVector(*input);
  InitRandomVector(*output);

  AvxFloatVec vec;
  vec.Load(input->data());
  AvxFloatVec vec2 = vec;
  vec.Floor();
  vec.Store(floored->data());
  vec2.Store(output->data());

  for (int i = 0; i < kAvxWidth; ++i) {
    EXPECT_EQ((*input)[i], (*output)[i]);
    EXPECT_EQ(floor((*input)[i]), (*floored)[i]);
  }
}

TEST(AvxVectorTest, ClampTest) {
  bool modified = false;  // check that some value was clamped.
  AvxVectorFuzzTest(
      [](AvxFloatVec *vec) { vec->Clamp(-0.314f, 0.314f); },
      [&modified](float input_value, float output_value) {
        modified = modified || input_value < -0.314 || input_value > 0.314;
        EXPECT_EQ(fmax(-0.314f, fmin(0.314f, input_value)), output_value);
      });
  EXPECT_TRUE(modified) << "No values fell outside test range for ClampTest().";
}

TEST(AvxVectorTest, LoadConstAndStore) {
  UniqueVector<float> output(kAvxWidth);
  InitRandomVector(*output);

  AvxFloatVec vec;
  vec.LoadConstVector(3.14f);
  vec.Store(output->data());

  for (int i = 0; i < kAvxWidth; ++i) {
    EXPECT_EQ((*output)[i], 3.14f);
  }
}

TEST(AvxVectorTest, AddTest) {
  AvxVectorFuzzTest(  //
      [](AvxFloatVec *vec) { (*vec) += *vec; },
      [](float input_value, float output_value) {
        EXPECT_EQ(input_value * 2, output_value);
      });
}

TEST(AvxVectorTest, SubtractTest) {
  AvxVectorFuzzTest(
      [](AvxFloatVec *vec) {
        AvxFloatVec one;
        one.LoadConstVector(1.0f);
        (*vec) -= one;
      },
      [](float input_value, float output_value) {
        EXPECT_EQ(input_value - 1.0f, output_value);
      });
}

TEST(AvxVectorTest, DivideTest) {
  AvxVectorFuzzTest(
      [](AvxFloatVec *vec) {
        AvxFloatVec result;
        result.LoadConstVector(1.0f);
        result /= *vec;
        *vec = result;
      },
      [](float input_value, float output_value) {
        EXPECT_EQ(1.0f / input_value, output_value);
      });
}

// This is a really basic test; half of the purpose is to ensure that the float
// API is still OK (i.e. compiles) for odd-sized arrays. If you try to add a
// call to array.Load(TruncatedFloat16 *source), it should produce a compiler
// error.
TEST(AvxFloatVecArrayTest, SingletonArrayLoadsAndStores) {
  AvxFloatVecArray<1> array;

  UniqueVector<float> input(kAvxWidth);
  UniqueVector<float> output(kAvxWidth);
  InitRandomVector(*input);
  InitRandomVector(*output);

  array.Load(input->data());
  array.Store(output->data());

  for (int i = 0; i < kAvxWidth; ++i) {
    EXPECT_EQ((*input)[i], (*output)[i]);
  }
}

TEST(AvxFloatVecArrayTest, LoadTruncatedFloat16) {
  AvxFloatVecArray<2> array;
  UniqueVector<TruncatedFloat16> values(2 * kAvxWidth);
  UniqueVector<float> decompressed(2 * kAvxWidth);

  for (int i = 0; i < 2 * kAvxWidth; ++i) {
    int permuted = FastUnpackPermutation(i);
    (*values)[i] = TruncatedFloat16::DebugFromFloat(permuted / 10.0);
  }

  // Ensure that state persisted from other tests won't cause this test to
  // erroneously pass.
  array.LoadConstVector(-1.0f);

  array.Load(values->data());
  array.Store(decompressed->data());
  for (int i = 0; i < 2 * kAvxWidth; ++i) {
    ASSERT_NEAR((*decompressed)[i], i / 10.0, 0.01);
  }
}

TEST(AvxFloatVecArrayTest, LoadIeeeFloat16) {
#if defined(__F16C__)
  AvxFloatVecArray<2> array;
  UniqueVector<IeeeFloat16> values(2 * kAvxWidth);
  UniqueVector<float> decompressed(2 * kAvxWidth);
  for (int i = 0; i < 2 * kAvxWidth; ++i) {
    (*values)[i] = IeeeFloat16::DebugFromFloat(i / 10.0);
  }

  // Ensure that state persisted from other tests won't cause this test to
  // erroneously pass.
  array.LoadConstVector(-1.0f);

  array.Load(values->data());
  array.Store(decompressed->data());
  for (int i = 0; i < 2 * kAvxWidth; ++i) {
    ASSERT_NEAR((*decompressed)[i], i / 10.0, 0.01);
  }
#else
  LOG(INFO) << "Test binary wasn't compiled with F16C support, so skipping "
            << "this test.";
#endif
}

TEST(AvxFloatVecArrayTest, PermutationFunctionIsEqualToTable) {
  std::vector<int> permutation = {0, 1, 2, 3, 8,  9,  10, 11,
                                  4, 5, 6, 7, 12, 13, 14, 15};

  for (int i = 0; i < kAvxWidthHalfPrecision; ++i) {
    EXPECT_EQ(FastUnpackPermutation(i), permutation[i]);
  }
}

}  // namespace
}  // namespace runtime
}  // namespace dragnn
}  // namespace syntaxnet
