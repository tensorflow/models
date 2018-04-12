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

#include "light_curve_util/cc/median_filter.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "light_curve_util/cc/test_util.h"

using std::vector;
using testing::Pointwise;

namespace astronet {
namespace {

TEST(MedianFilter, Errors) {
  vector<double> x;
  vector<double> y;
  vector<double> result;
  std::string error;

  // x size less than 2.
  x = {1};
  y = {2};
  EXPECT_FALSE(MedianFilter(x, y, 2, 1, 0, 2, &result, &error));
  EXPECT_EQ(error, "x.size() must be greater than 1. Got: 1");

  // x and y not the same size.
  x = {1, 2};
  y = {4, 5, 6};
  EXPECT_FALSE(MedianFilter(x, y, 2, 1, 0, 2, &result, &error));
  EXPECT_EQ(error, "x.size() (got: 2) must equal y.size() (got: 3)");

  // x out of order.
  x = {2, 0, 1};
  EXPECT_FALSE(MedianFilter(x, y, 2, 1, 0, 2, &result, &error));
  EXPECT_EQ(error,
            "The first element of x (got: 2) must be less than the last element"
            " (got: 1). Either x is not sorted or all elements are equal.");

  // x all equal.
  x = {1, 1, 1};
  EXPECT_FALSE(MedianFilter(x, y, 2, 1, 0, 2, &result, &error));
  EXPECT_EQ(error,
            "The first element of x (got: 1) must be less than the last element"
            " (got: 1). Either x is not sorted or all elements are equal.");

  // x_min not less than x_max
  x = {1, 2, 3};
  EXPECT_FALSE(MedianFilter(x, y, 2, 1, -1, -1, &result, &error));
  EXPECT_EQ(error, "x_min (got: -1) must be less than x_max (got: -1)");

  // x_min greater than the last element of x.
  x = {1, 2, 3};
  EXPECT_FALSE(MedianFilter(x, y, 2, 0.25, 3.5, 4, &result, &error));
  EXPECT_EQ(error,
            "x_min (got: 3.5) must be less than or equal to the largest value "
            "of x (got: 3)");

  // bin_width nonpositive.
  x = {1, 2, 3};
  EXPECT_FALSE(MedianFilter(x, y, 2, 0, 1, 3, &result, &error));
  EXPECT_EQ(error, "bin_width must be positive. Got: 0");

  // bin_width greater than or equal to x_max - x_min.
  x = {1, 2, 3};
  EXPECT_FALSE(MedianFilter(x, y, 2, 1, 1.5, 2.5, &result, &error));
  EXPECT_EQ(error,
            "bin_width (got: 1) must be less than x_max - x_min (got: 1)");

  // num_bins less than 2.
  x = {1, 2, 3};
  EXPECT_FALSE(MedianFilter(x, y, 1, 1, 0, 2, &result, &error));
  EXPECT_EQ(error, "num_bins must be greater than 1. Got: 1");
}

TEST(MedianFilter, BucketBoundaries) {
  vector<double> x = {-6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6};
  vector<double> y = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13};
  vector<double> result;
  std::string error;

  EXPECT_TRUE(MedianFilter(x, y, 5, 2, -5, 5, &result, &error));
  EXPECT_TRUE(error.empty());

  vector<double> expected = {2.5, 4.5, 6.5, 8.5, 10.5};
  EXPECT_THAT(result, Pointwise(DoubleNear(), expected));
}

TEST(MedianFilter, MultiSizeBins) {
  // Construct bins with size 0, 1, 2, 3, 4, 5, 10, respectively.
  vector<double> x = {1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5,
                      5, 5, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6};
  vector<double> y = {0, -1, 1, 4, 5, 6, 2, 2, 4, 4, 1, 1, 1,
                      1, -1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  vector<double> result;
  std::string error;

  EXPECT_TRUE(MedianFilter(x, y, 7, 1, 0, 7, &result, &error));
  EXPECT_TRUE(error.empty());

  // expected[0] = 3 is the median of y.
  vector<double> expected = {3, 0, 0, 5, 3, 1, 5.5};
  EXPECT_THAT(result, Pointwise(DoubleNear(), expected));
}

TEST(MedianFilter, EmptyBins) {
  vector<double> x = {-1, 0, 1};
  vector<double> y = {2, 3, 1};
  vector<double> result;
  std::string error;

  EXPECT_TRUE(MedianFilter(x, y, 5, 1, -5, 5, &result, &error));
  EXPECT_TRUE(error.empty());

  // The center bin is the only nonempty bin.
  vector<double> expected = {2, 2, 3, 2, 2};
  EXPECT_THAT(result, Pointwise(DoubleNear(), expected));
}

TEST(MedianFilter, WideBins) {
  vector<double> x = {-6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6};
  vector<double> y = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13};
  vector<double> result;
  std::string error;

  EXPECT_TRUE(MedianFilter(x, y, 7, 5, -10, 10, &result, &error));
  EXPECT_TRUE(error.empty());

  vector<double> expected = {1, 2.5, 4, 7, 9, 11.5, 12.5};
  EXPECT_THAT(result, Pointwise(DoubleNear(), expected));
}

TEST(MedianFilter, NarrowBins) {
  vector<double> x = {-6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6};
  vector<double> y = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13};
  vector<double> result;
  std::string error;

  EXPECT_TRUE(MedianFilter(x, y, 9, 0.5, -2.25, 2.25, &result, &error));
  EXPECT_TRUE(error.empty());

  // Bins 1, 3, 5, 7 are empty.
  vector<double> expected = {5, 7, 6, 7, 7, 7, 8, 7, 9};
  EXPECT_THAT(result, Pointwise(DoubleNear(), expected));
}

}  // namespace
}  // namespace astronet
