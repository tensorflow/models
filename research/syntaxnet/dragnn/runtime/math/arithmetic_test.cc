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

#include "dragnn/runtime/math/arithmetic.h"

#include <stddef.h>
#include <vector>

#include "dragnn/runtime/math/types.h"
#include "dragnn/runtime/test/helpers.h"
#include "tensorflow/core/platform/test.h"

namespace syntaxnet {
namespace dragnn {
namespace runtime {
namespace {

// Tests that ScaleElements() doesn't crash on empty vectors.
TEST(ScaleElementsTest, Empty) {
  Vector<float> input;
  MutableVector<float> output;

  ScaleElements(1.5f, input, output);
}

// Tests that ScaleElements() copies scaled values from one vector to another.
TEST(ScaleElementsTest, Populated) {
  UniqueVector<float> input({-2.0f, -3.0f, 5.0f});
  UniqueVector<float> output({7.0f, 11.0f, 13.0f});  // gets overwritten

  ScaleElements(1.5f, Vector<float>(*input), *output);

  EXPECT_EQ((*output)[0], 1.5 * -2.0);
  EXPECT_EQ((*output)[1], 1.5 * -3.0);
  EXPECT_EQ((*output)[2], 1.5 * 5.0);
}

// Tests that AddScaledElements() doesn't crash on empty vectors.
TEST(AddScaledElementsTest, Empty) {
  Vector<float> input;
  MutableVector<float> output;

  AddScaledElements(1.5f, input, output);
}

// Tests that AddScaledElements() adds scaled values from one vector to another.
TEST(AddScaledElementsTest, Populated) {
  UniqueVector<float> input({-2.0f, -3.0f, 5.0f});
  UniqueVector<float> output({7.0f, 11.0f, 13.0f});  // gets added to

  AddScaledElements(1.5f, Vector<float>(*input), *output);

  EXPECT_EQ((*output)[0], 1.5 * -2.0 + 7.0);
  EXPECT_EQ((*output)[1], 1.5 * -3.0 + 11.0);
  EXPECT_EQ((*output)[2], 1.5 * 5.0 + 13.0);
}

// Tests that MaxElements() doesn't crash on empty vectors.
TEST(MaxElementsTest, Empty) {
  MutableVector<float> values;

  MaxElements(1.5f, values);
}

// Tests that MaxElements() performs an in-place element-wise maximum.
TEST(MaxElementsTest, Populated) {
  UniqueVector<float> values({-1.0f, 2.0f, 0.25f, -0.5f, 0.375f});

  MaxElements(0.125f, *values);

  EXPECT_EQ((*values)[0], 0.125);
  EXPECT_EQ((*values)[1], 2.0);
  EXPECT_EQ((*values)[2], 0.25);
  EXPECT_EQ((*values)[3], 0.125);
  EXPECT_EQ((*values)[4], 0.375);
}

// Tests that MultiplyMatrixAndVector() doesn't crash on empty inputs.
TEST(MultiplyMatrixAndVectorTest, Empty) {
  Matrix<float> matrix;
  Vector<float> input;
  MutableVector<float> output;

  MultiplyMatrixAndVector(matrix, input, output);
}

// Tests that MultiplyMatrixAndVector() computes a matrix-vector product.
TEST(MultiplyMatrixAndVectorTest, Populated) {
  UniqueMatrix<float> matrix({{2.0f, 3.0f},  //
                              {5.0f, 7.0f},  //
                              {11.0f, 13.0f}});
  UniqueVector<float> input({-0.5f, 2.0f});
  UniqueVector<float> output({9.8f, 7.6f, 5.4f});  // gets overwritten

  MultiplyMatrixAndVector(Matrix<float>(*matrix), Vector<float>(*input),
                          *output);

  EXPECT_EQ((*output)[0], 2.0 * -0.5 + 3.0 * 2.0);
  EXPECT_EQ((*output)[1], 5.0 * -0.5 + 7.0 * 2.0);
  EXPECT_EQ((*output)[2], 11.0 * -0.5 + 13.0 * 2.0);
}

// Tests that MultiplyMatrixAndVectorWithBias() doesn't crash on empty inputs.
TEST(MultiplyMatrixAndVectorWithBiasTest, Empty) {
  Matrix<float> matrix;
  Vector<float> bias;
  Vector<float> input;
  MutableVector<float> output;

  MultiplyMatrixAndVectorWithBias(matrix, bias, input, output);
}

// Tests that MultiplyMatrixAndVectorWithBias() computes a matrix-vector product
// with an additive bias.
TEST(MultiplyMatrixAndVectorWithBiasTest, Populated) {
  UniqueMatrix<float> matrix({{2.0f, 3.0f},  //
                              {5.0f, 7.0f},  //
                              {11.0f, 13.0f}});
  UniqueVector<float> bias({100.5f, 200.25f, 300.75f});
  UniqueVector<float> input({-0.5f, 2.0f});
  UniqueVector<float> output({9.8f, 7.6f, 5.4f});  // gets overwritten

  MultiplyMatrixAndVectorWithBias(Matrix<float>(*matrix), Vector<float>(*bias),
                                  Vector<float>(*input), *output);

  EXPECT_EQ((*output)[0], 100.5 + 2.0 * -0.5 + 3.0 * 2.0);
  EXPECT_EQ((*output)[1], 200.25 + 5.0 * -0.5 + 7.0 * 2.0);
  EXPECT_EQ((*output)[2], 300.75 + 11.0 * -0.5 + 13.0 * 2.0);
}

// A dummy type for the specializations below.  Specializing on this unique
// dummy type ensures we don't conflict with any existing specialization.
struct Foo {
  float value;
};

}  // namespace

// Dummy specializations for use in the subsequent tests.
template <>
void ScaleElements(Foo scale, Vector<Foo> input, MutableVector<Foo> output) {
  for (Foo &foo : output) foo.value = 777.0;
}

namespace {

// Tests that the template specialization overrides the generic implementation.
TEST(ScaleElementsTest, OverriddenByTemplateSpecialization) {
  // These values are uninitialized, but it doesn't matter because the
  // specialization never looks at them.
  UniqueVector<Foo> input(3);
  UniqueVector<Foo> output(3);

  ScaleElements(Foo(), Vector<Foo>(*input), *output);

  EXPECT_EQ((*output)[0].value, 777.0);
  EXPECT_EQ((*output)[1].value, 777.0);
  EXPECT_EQ((*output)[2].value, 777.0);
}

}  // namespace
}  // namespace runtime
}  // namespace dragnn
}  // namespace syntaxnet
