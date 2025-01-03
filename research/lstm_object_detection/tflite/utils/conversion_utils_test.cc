/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include "utils/conversion_utils.h"

#include <vector>

#include <glog/logging.h>
#include <gmock/gmock.h>
#include "gtest/gtest.h"

using testing::ContainerEq;

namespace lstm_object_detection {
namespace tflite {
namespace {

TEST(ConversionUtilsTests, HasPaddingNonPositiveDimensions) {
  EXPECT_DEATH(HasPadding(/* width= */ 0, /* height= */ 4,
                          /* bytes_per_pixel= */ 4, /* bytes_per_row= */ 12),
               "");
  EXPECT_DEATH(HasPadding(/* width= */ 3, /* height= */ 0,
                          /* bytes_per_pixel= */ 4, /* bytes_per_row= */ 12),
               "");
}

TEST(ConversionUtilsTests, HasPaddingIllegalDepth) {
  for (int bytes_per_pixel : {-1, 0, 2, 5, 6}) {
    EXPECT_DEATH(HasPadding(/* width= */ 3, /* height= */ 4, bytes_per_pixel,
                            /* bytes_per_row= */ 12),
                 "");
  }
}

TEST(ConversionUtilsTests, HasPaddingWithRGBAImage) {
  const int kWidth = 3;
  const int kHeight = 4;
  const int kBytesPerPixel = 4;
  EXPECT_DEATH(
      HasPadding(kWidth, kHeight, kBytesPerPixel, /* bytes_per_row= */ 11), "");
  EXPECT_TRUE(
      HasPadding(kWidth, kHeight, kBytesPerPixel, /* bytes_per_row= */ 12));
  EXPECT_TRUE(
      HasPadding(kWidth, kHeight, kBytesPerPixel, /* bytes_per_row= */ 13));
}

TEST(ConversionUtilsTests, HasPaddingWithRGBImage) {
  const int kWidth = 3;
  const int kHeight = 4;
  const int kBytesPerPixel = 3;
  EXPECT_DEATH(
      HasPadding(kWidth, kHeight, kBytesPerPixel, /* bytes_per_row= */ 8), "");
  EXPECT_FALSE(
      HasPadding(kWidth, kHeight, kBytesPerPixel, /* bytes_per_row= */ 9));
  EXPECT_TRUE(
      HasPadding(kWidth, kHeight, kBytesPerPixel, /* bytes_per_row= */ 10));
}

TEST(ConversionUtilsTests, HasPaddingWithGrayscaleImage) {
  const int kWidth = 3;
  const int kHeight = 4;
  const int kBytesPerPixel = 1;
  EXPECT_DEATH(
      HasPadding(kWidth, kHeight, kBytesPerPixel,
                          /* bytes_per_row= */ 2), "");
  EXPECT_FALSE(
      HasPadding(kWidth, kHeight, kBytesPerPixel,
                          /* bytes_per_row= */ 3));
  EXPECT_TRUE(
      HasPadding(kWidth, kHeight, kBytesPerPixel,
                         /* bytes_per_row= */ 4));
}

TEST(ConversionUtilsTests, RemovePaddingWithRGBAImage) {
  constexpr int kWidth = 4;
  constexpr int kHeight = 2;
  constexpr int kBytesPerPixel = 4;
  constexpr int kStride = kBytesPerPixel * kWidth * sizeof(uint8_t);
  const std::vector<uint8_t> kImageData{
      1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16,
      21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36};
  ASSERT_EQ(kHeight * kStride, kImageData.size());

  std::vector<uint8_t> actual =
      RemovePadding(&kImageData[0], kWidth, kHeight, kBytesPerPixel, kStride);

  const std::vector<uint8_t> kExpected = {
      1,  2,  3,  5,  6,  7,  9,  10, 11, 13, 14, 15,
      21, 22, 23, 25, 26, 27, 29, 30, 31, 33, 34, 35,
  };
  EXPECT_EQ(3 * kWidth * kHeight, actual.size());
  EXPECT_THAT(actual, ContainerEq(kExpected));
}

TEST(ConversionUtilsTests, RemovePaddingWithRGBImage) {
  constexpr int kWidth = 4;
  constexpr int kHeight = 2;
  constexpr int kBytesPerPixel = 3;
  constexpr int kBytesPerRow = kBytesPerPixel * kWidth * sizeof(uint8_t);
  const std::vector<uint8_t> kImageData{1,  2,  3,  5,  6,  7,  9,  10,
                                        11, 13, 14, 15, 21, 22, 23, 25,
                                        26, 27, 29, 30, 31, 33, 34, 35};
  ASSERT_EQ(kHeight * kBytesPerRow, kImageData.size());

  std::vector<uint8_t> actual = RemovePadding(&kImageData[0], kWidth, kHeight,
                                              kBytesPerPixel, kBytesPerRow);

  EXPECT_EQ(3 * kWidth * kHeight, actual.size());
  EXPECT_THAT(actual, ContainerEq(kImageData));
}

TEST(ConversionUtilsTests, RemovePaddingWithGrayscaleImage) {
  constexpr int kWidth = 8;
  constexpr int kHeight = 2;
  constexpr int kBytesPerPixel = 1;
  constexpr int kBytesPerRow = kBytesPerPixel * kWidth * sizeof(uint8_t);
  const std::vector<uint8_t> kImageData{
      1, 2, 3, 4, 5, 6, 7, 8, 21, 22, 23, 24, 25, 26, 27, 28,
  };
  ASSERT_EQ(kHeight * kBytesPerRow, kImageData.size());

  std::vector<uint8_t> actual = RemovePadding(&kImageData[0], kWidth, kHeight,
                                              kBytesPerPixel, kBytesPerRow);

  EXPECT_EQ(kWidth * kHeight, actual.size());
  EXPECT_THAT(actual, ContainerEq(kImageData));
}

TEST(ConversionUtilsTests, RemovePaddingWithPadding) {
  constexpr int kWidth = 8;
  constexpr int kHeight = 2;
  constexpr int kBytesPerPixel = 1;
  // Pad each row with two bytes.
  constexpr int kBytesPerRow = kBytesPerPixel * (kWidth + 2) * sizeof(uint8_t);
  const std::vector<uint8_t> kImageData{1,  2,  3,  4,  5,  6,  7,  8,  21, 22,
                                        23, 24, 25, 26, 27, 28, 29, 30, 31, 32};
  ASSERT_EQ(kHeight * kBytesPerRow, kImageData.size());

  std::vector<uint8_t> actual = RemovePadding(&kImageData[0], kWidth, kHeight,
                                              kBytesPerPixel, kBytesPerRow);

  const std::vector<uint8_t> kExpected = {
      1, 2, 3, 4, 5, 6, 7, 8, 23, 24, 25, 26, 27, 28, 29, 30,
  };
  EXPECT_EQ(kWidth * kHeight, actual.size());
  EXPECT_THAT(actual, ContainerEq(kExpected));
}

}  // namespace
}  // namespace tflite
}  // namespace lstm_object_detection
