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

#include "light_curve_util/cc/phase_fold.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "light_curve_util/cc/test_util.h"

using std::vector;
using testing::Pointwise;

namespace astronet {
namespace {

TEST(PhaseFoldTime, Empty) {
  vector<double> time = {};
  vector<double> result;
  PhaseFoldTime(time, 1, 0.45, &result);
  EXPECT_TRUE(result.empty());
}

TEST(PhaseFoldTime, Simple) {
  vector<double> time = range(0, 2, 0.1);
  vector<double> result;
  PhaseFoldTime(time, 1, 0.45, &result);
  vector<double> expected = {
      -0.45, -0.35, -0.25, -0.15, -0.05, 0.05, 0.15, 0.25, 0.35, 0.45,
      -0.45, -0.35, -0.25, -0.15, -0.05, 0.05, 0.15, 0.25, 0.35, 0.45,
  };
  EXPECT_THAT(result, Pointwise(DoubleNear(), expected));
}

TEST(PhaseFoldTime, LargeT0) {
  vector<double> time = range(0, 2, 0.1);
  vector<double> result;
  PhaseFoldTime(time, 1, 1.25, &result);
  vector<double> expected = {
      -0.25, -0.15, -0.05, 0.05, 0.15, 0.25, 0.35, 0.45, -0.45, -0.35,
      -0.25, -0.15, -0.05, 0.05, 0.15, 0.25, 0.35, 0.45, -0.45, -0.35,
  };
  EXPECT_THAT(result, Pointwise(DoubleNear(), expected));
}

TEST(PhaseFoldTime, NegativeT0) {
  vector<double> time = range(0, 2, 0.1);
  vector<double> result;
  PhaseFoldTime(time, 1, -1.65, &result);
  vector<double> expected = {
      -0.35, -0.25, -0.15, -0.05, 0.05, 0.15, 0.25, 0.35, 0.45, -0.45,
      -0.35, -0.25, -0.15, -0.05, 0.05, 0.15, 0.25, 0.35, 0.45, -0.45,
  };
  EXPECT_THAT(result, Pointwise(DoubleNear(), expected));
}

TEST(PhaseFoldTime, NegativeTime) {
  vector<double> time = range(-3, -1, 0.1);
  vector<double> result;
  PhaseFoldTime(time, 1, 0.55, &result);
  vector<double> expected = {
      0.45, -0.45, -0.35, -0.25, -0.15, -0.05, 0.05, 0.15, 0.25, 0.35,
      0.45, -0.45, -0.35, -0.25, -0.15, -0.05, 0.05, 0.15, 0.25, 0.35,
  };
  EXPECT_THAT(result, Pointwise(DoubleNear(), expected));
}

TEST(PhaseFoldTime, InPlace) {
  vector<double> time = range(0, 2, 0.1);
  PhaseFoldTime(time, 0.5, 1.15, &time);
  vector<double> expected = {
      -0.15, -0.05, 0.05, 0.15, -0.25, -0.15, -0.05, 0.05, 0.15, -0.25,
      -0.15, -0.05, 0.05, 0.15, -0.25, -0.15, -0.05, 0.05, 0.15, -0.25,
  };
  EXPECT_THAT(time, Pointwise(DoubleNear(), time));
}

TEST(PhaseFoldAndSortLightCurve, Error) {
  vector<double> time = {1.0, 2.0, 3.0};
  vector<double> flux = {7.5, 8.6};
  vector<double> folded_time;
  vector<double> folded_flux;
  std::string error;

  EXPECT_FALSE(PhaseFoldAndSortLightCurve(time, flux, 1.0, 0.5, &folded_time,
                                          &folded_flux, &error));
  EXPECT_EQ(error, "time.size() (got: 3) must equal flux.size() (got: 2)");
}

TEST(PhaseFoldAndSortLightCurve, Empty) {
  vector<double> time = {};
  vector<double> flux = {};
  vector<double> folded_time;
  vector<double> folded_flux;
  std::string error;

  EXPECT_TRUE(PhaseFoldAndSortLightCurve(time, flux, 1.0, 0.5, &folded_time,
                                         &folded_flux, &error));
  EXPECT_TRUE(error.empty());
  EXPECT_TRUE(folded_time.empty());
  EXPECT_TRUE(folded_flux.empty());
}

TEST(PhaseFoldAndSortLightCurve, FoldAndSort) {
  vector<double> time = range(0, 2, 0.1);
  vector<double> flux = range(0, 20, 1);
  vector<double> folded_time;
  vector<double> folded_flux;
  std::string error;

  EXPECT_TRUE(PhaseFoldAndSortLightCurve(time, flux, 2.0, 0.15, &folded_time,
                                         &folded_flux, &error));
  EXPECT_TRUE(error.empty());

  vector<double> expected_time = {
      -0.95, -0.85, -0.75, -0.65, -0.55, -0.45, -0.35, -0.25, -0.15, -0.05,
      0.05,  0.15,  0.25,  0.35,  0.45,  0.55,  0.65,  0.75,  0.85,  0.95};
  EXPECT_THAT(folded_time, Pointwise(DoubleNear(), expected_time));

  vector<double> expected_flux = {12, 13, 14, 15, 16, 17, 18, 19, 0,  1,
                                  2,  3,  4,  5,  6,  7,  8,  9,  10, 11};
  EXPECT_THAT(folded_flux, Pointwise(DoubleNear(), expected_flux));
}

}  // namespace
}  // namespace astronet
