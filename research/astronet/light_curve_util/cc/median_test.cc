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

#include "light_curve_util/cc/median.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"

using testing::ElementsAreArray;

namespace astronet {
namespace {

TEST(InPlaceMedian, SingleFloat) {
  std::vector<double> v = {1.0};
  EXPECT_FLOAT_EQ(1.0, InPlaceMedian(v.begin(), v.end()));
  EXPECT_THAT(v, ElementsAreArray({1.0}));
}

TEST(InPlaceMedian, TwoInts) {
  std::vector<int> v = {3, 2};
  // Note that integer division is used, so the median is (2 + 3) / 2 = 2.
  EXPECT_EQ(2, InPlaceMedian(v.begin(), v.end()));
  EXPECT_THAT(v, ElementsAreArray({2, 3}));
}

TEST(InPlaceMedian, OddElements) {
  std::vector<double> v = {1.0, 0.0, 2.0};
  EXPECT_FLOAT_EQ(1.0, InPlaceMedian(v.begin(), v.end()));
  EXPECT_THAT(v, ElementsAreArray({0.0, 1.0, 2.0}));
}

TEST(InPlaceMedian, EvenElements) {
  std::vector<double> v = {1.0, 0.0, 4.0, 3.0};
  EXPECT_FLOAT_EQ(2.0, InPlaceMedian(v.begin(), v.end()));
  EXPECT_FLOAT_EQ(3.0, v[2]);
  EXPECT_FLOAT_EQ(4.0, v[3]);
}

TEST(InPlaceMedian, SubRanges) {
  std::vector<double> v = {1.0, 4.0, 0.0, 3.0, -1.0, 6.0, 9.0, -10.0};

  // [0, 1)
  EXPECT_FLOAT_EQ(1.0, InPlaceMedian(v.begin(), v.begin() + 1));
  EXPECT_FLOAT_EQ(1.0, v[0]);

  // [1, 4)
  EXPECT_FLOAT_EQ(3.0, InPlaceMedian(v.begin() + 1, v.begin() + 4));
  EXPECT_FLOAT_EQ(0.0, v[1]);
  EXPECT_FLOAT_EQ(3.0, v[2]);
  EXPECT_FLOAT_EQ(4.0, v[3]);

  // [4, 8)
  EXPECT_FLOAT_EQ(2.5, InPlaceMedian(v.begin() + 4, v.end()));
  EXPECT_FLOAT_EQ(6.0, v[6]);
  EXPECT_FLOAT_EQ(9.0, v[7]);
}

TEST(Median, SingleFloat) {
  std::vector<double> v = {-5.0};
  EXPECT_FLOAT_EQ(-5.0, Median(v.begin(), v.end()));
  EXPECT_THAT(v, ElementsAreArray({-5.0}));
}

TEST(Median, TwoInts) {
  std::vector<int> v = {3, 2};
  // Note that integer division is used, so the median is (2 + 3) / 2 = 2.
  EXPECT_EQ(2, Median(v.begin(), v.end()));
  EXPECT_THAT(v, ElementsAreArray({3, 2}));  // Unmodified.
}

TEST(Median, SubRanges) {
  std::vector<double> v = {1.0, 4.0, 0.0, 3.0, -1.0, 6.0, 9.0, -10.0};

  // [0, 1)
  EXPECT_FLOAT_EQ(1.0, Median(v.begin(), v.begin() + 1));
  EXPECT_THAT(v, ElementsAreArray({1.0, 4.0, 0.0, 3.0, -1.0, 6.0, 9.0, -10.0}));

  // [1, 4)
  EXPECT_FLOAT_EQ(3.0, Median(v.begin() + 1, v.begin() + 4));
  EXPECT_THAT(v, ElementsAreArray({1.0, 4.0, 0.0, 3.0, -1.0, 6.0, 9.0, -10.0}));

  // [4, 8)
  EXPECT_FLOAT_EQ(2.5, Median(v.begin() + 4, v.end()));
  EXPECT_THAT(v, ElementsAreArray({1.0, 4.0, 0.0, 3.0, -1.0, 6.0, 9.0, -10.0}));
}

}  // namespace
}  // namespace astronet
