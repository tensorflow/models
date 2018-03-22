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

#include "light_curve_util/cc/normalize.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "light_curve_util/cc/test_util.h"

using std::vector;
using testing::Pointwise;

namespace astronet {
namespace {

TEST(NormalizeMedianAndMinimum, Error) {
  vector<double> x = {-1, -1, -1, -1, -1, -1};
  vector<double> result;
  std::string error;

  EXPECT_FALSE(NormalizeMedianAndMinimum(x, &result, &error));
  EXPECT_EQ(error, "Minimum and median have the same value: -1");
}

TEST(NormalizeMedianAndMinimum, TooFewElements) {
  vector<double> x = {1};
  vector<double> result;
  std::string error;

  EXPECT_FALSE(NormalizeMedianAndMinimum(x, &result, &error));
  EXPECT_EQ(error, "x.size() must be greater than 1. Got: 1");
}

TEST(NormalizeMedianAndMinimum, NonNegative) {
  vector<double> x = {0, 1, 2, 3, 4, 5, 6, 7, 8};  // Median 4, Min 0.
  vector<double> result;
  std::string error;

  EXPECT_TRUE(NormalizeMedianAndMinimum(x, &result, &error));
  EXPECT_TRUE(error.empty());

  vector<double> expected = {-1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1};
  EXPECT_THAT(result, Pointwise(DoubleNear(), expected));
}

TEST(NormalizeMedianAndMinimum, NonPositive) {
  vector<double> x = {0, -1, -2, -3, -4, -5, -6, -7, -8};  // Median -4, Min -8.
  vector<double> result;
  std::string error;

  EXPECT_TRUE(NormalizeMedianAndMinimum(x, &result, &error));
  EXPECT_TRUE(error.empty());

  vector<double> expected = {1, 0.75, 0.5, 0.25, 0, -0.25, -0.5, -0.75, -1};
  EXPECT_THAT(result, Pointwise(DoubleNear(), expected));
}

TEST(NormalizeMedianAndMinimum, PositiveNegative) {
  vector<double> x = {-4, -3, -2, -1, 0, 1, 2, 3, 4};  // Median 0, Min -4.
  vector<double> result;
  std::string error;

  EXPECT_TRUE(NormalizeMedianAndMinimum(x, &result, &error));
  EXPECT_TRUE(error.empty());

  vector<double> expected = {-1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1};
  EXPECT_THAT(result, Pointwise(DoubleNear(), expected));
}

TEST(NormalizeMedianAndMinimum, InPlace) {
  vector<double> x = {-4, -3, -2, -1, 0, 1, 2, 3, 4};  // Median 0, Min -4.
  std::string error;

  EXPECT_TRUE(NormalizeMedianAndMinimum(x, &x, &error));
  EXPECT_TRUE(error.empty());

  vector<double> expected = {-1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1};
  EXPECT_THAT(x, Pointwise(DoubleNear(), expected));
}

}  // namespace
}  // namespace astronet
