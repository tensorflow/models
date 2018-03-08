/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "light_curve_util/cc/view_generator.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "light_curve_util/cc/test_util.h"

using std::vector;
using testing::Pointwise;

namespace astronet {
namespace {

TEST(ViewGenerator, CreationError) {
  vector<double> time = {1, 2, 3};
  vector<double> flux = {2, 3};
  std::string error;

  std::unique_ptr<ViewGenerator> generator =
      ViewGenerator::Create(time, flux, 1, 0.5, &error);
  EXPECT_EQ(nullptr, generator);
  EXPECT_FALSE(error.empty());
}

TEST(ViewGenerator, GenerateViews) {
  vector<double> time = range(0, 2, 0.1);
  vector<double> flux = range(0, 20, 1);
  std::string error;

  // Create the ViewGenerator.
  std::unique_ptr<ViewGenerator> generator =
      ViewGenerator::Create(time, flux, 2.0, 0.15, &error);
  EXPECT_NE(nullptr, generator);
  EXPECT_TRUE(error.empty());

  vector<double> result;

  // Error: t_max <= t_min. We do not test all failure cases here since they
  // are tested in light_curve_util_test.cc.
  EXPECT_FALSE(generator->GenerateView(10, 1, -1, -1, false, &result, &error));
  EXPECT_FALSE(error.empty());
  error.clear();

  // Global view, unnormalized.
  EXPECT_TRUE(generator->GenerateView(10, 0.2, -1, 1, false, &result, &error));
  EXPECT_TRUE(error.empty());
  vector<double> expected = {12.5, 14.5, 16.5, 18.5, 0.5,
                             2.5,  4.5,  6.5,  8.5,  10.5};
  EXPECT_THAT(result, Pointwise(DoubleNear(), expected));

  // Global view, normalized.
  EXPECT_TRUE(generator->GenerateView(10, 0.2, -1, 1, true, &result, &error));
  EXPECT_TRUE(error.empty());
  expected = {3.0 / 9,  5.0 / 9,  7.0 / 9,  9.0 / 9,  -9.0 / 9,
              -7.0 / 9, -5.0 / 9, -3.0 / 9, -1.0 / 9, 1.0 / 9};
  EXPECT_THAT(result, Pointwise(DoubleNear(), expected));

  // Local view, unnormalized.
  EXPECT_TRUE(
      generator->GenerateView(5, 0.2, -0.5, 0.5, false, &result, &error));
  EXPECT_TRUE(error.empty());
  expected = {17.5, 9.5, 1.5, 3.5, 5.5};
  EXPECT_THAT(result, Pointwise(DoubleNear(), expected));

  // Local view, normalized.
  EXPECT_TRUE(
      generator->GenerateView(5, 0.2, -0.5, 0.5, true, &result, &error));
  EXPECT_TRUE(error.empty());
  expected = {3, 1, -1, -0.5, 0};
  EXPECT_THAT(result, Pointwise(DoubleNear(), expected));
}

}  // namespace
}  // namespace astronet
