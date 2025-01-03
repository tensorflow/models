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

#include <glog/logging.h>

namespace lstm_object_detection {
namespace tflite {

bool HasPadding(int width, int height, int bytes_per_pixel, int bytes_per_row) {
  CHECK_LT(0, width);
  CHECK_LT(0, height);
  CHECK(bytes_per_pixel == 1 || bytes_per_pixel == 3 || bytes_per_pixel == 4);
  CHECK_LE(width * bytes_per_pixel, bytes_per_row);

  if (bytes_per_pixel == 4) {
    return true;
  }
  return (width * bytes_per_pixel < bytes_per_row);
}

std::vector<uint8_t> RemovePadding(const uint8_t* image_data, int width,
                                   int height, int bytes_per_pixel,
                                   int bytes_per_row) {
  CHECK_LT(0, width);
  CHECK_LT(0, height);
  CHECK(bytes_per_pixel == 1 || bytes_per_pixel == 3 || bytes_per_pixel == 4);
  CHECK_LE(width * bytes_per_pixel, bytes_per_row);

  const int unpadded_bytes_per_pixel = (bytes_per_pixel == 1 ? 1 : 3);
  const int pixel_padding = (bytes_per_pixel == 4 ? 1 : 0);
  std::vector<uint8_t> unpadded_image_data(width * height *
                                           unpadded_bytes_per_pixel);

  const uint8_t* row_ptr = image_data;
  int index = 0;
  for (int y = 0; y < height; ++y) {
    const uint8_t* ptr = row_ptr;
    for (int x = 0; x < width; ++x) {
      for (int d = 0; d < unpadded_bytes_per_pixel; ++d) {
        unpadded_image_data[index++] = *ptr++;
      }
      ptr += pixel_padding;
    }
    row_ptr += bytes_per_row;
  }

  return unpadded_image_data;
}

}  // namespace tflite
}  // namespace lstm_object_detection
