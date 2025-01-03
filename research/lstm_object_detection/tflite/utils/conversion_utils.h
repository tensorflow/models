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

// Lightweight utilities related to conversion of input images.

#ifndef TENSORFLOW_MODELS_LSTM_OBJECT_DETECTION_TFLITE_UTILS_CONVERSION_UTILS_H_
#define TENSORFLOW_MODELS_LSTM_OBJECT_DETECTION_TFLITE_UTILS_CONVERSION_UTILS_H_

#include <vector>

#include <cstdint>

namespace lstm_object_detection {
namespace tflite {

// Finds out whether a call to 'RemovePadding()' is needed to process the given
// pixel data constellation in order to make it suitable for model input layer.
// All integers must be positive, 'bytes_per_row' must be sufficiently large,
// and for 'bytes_per_pixel' only values 1, 3, 4 may be passed and implies a
// grayscale, RGB, or RGBA image. Returns true iff excessive bytes exist in the
// associated pixel data.
bool HasPadding(int width, int height, int bytes_per_pixel, int bytes_per_row);

// Removes padding at the pixel and row level of pixel data which is stored in
// the usual row major order ("interleaved"). Produces pixel data which is
// suitable for model input layer. If 'HasPadding()' is false then this
// function will return an identical copy of 'image'. For restrictions on the
// integer parameters see comment on 'HasPadding()'.
std::vector<uint8_t> RemovePadding(const uint8_t* image, int width, int height,
                                   int bytes_per_pixel, int bytes_per_row);

}  // namespace tflite
}  // namespace lstm_object_detection

#endif  // TENSORFLOW_MODELS_LSTM_OBJECT_DETECTION_TFLITE_UTILS_CONVERSION_UTILS_H_
