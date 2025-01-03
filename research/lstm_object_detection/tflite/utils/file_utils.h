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

#ifndef TENSORFLOW_MODELS_LSTM_OBJECT_DETECTION_TFLITE_UTILS_FILE_UTILS_H_
#define TENSORFLOW_MODELS_LSTM_OBJECT_DETECTION_TFLITE_UTILS_FILE_UTILS_H_

#include <string>

#include "absl/strings/string_view.h"
#include "protos/labelmap.pb.h"

namespace lstm_object_detection {
namespace tflite {

std::string ReadFileToString(absl::string_view filename);

// Load labelmap from a binary proto file or bytes string.
// labelmap_bytes takes precedence over labelmap_file.
bool LoadLabelMapFromFileOrBytes(const std::string& labelmap_file,
                                 const std::string& labelmap_bytes,
                                 protos::StringIntLabelMapProto* labelmap);

}  // namespace tflite
}  // namespace lstm_object_detection

#endif  // TENSORFLOW_MODELS_LSTM_OBJECT_DETECTION_TFLITE_UTILS_FILE_UTILS_H_
