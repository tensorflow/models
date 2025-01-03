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

#include "utils/file_utils.h"

#include <fstream>

#include <glog/logging.h>

namespace lstm_object_detection {
namespace tflite {

std::string ReadFileToString(absl::string_view filename) {
  std::ifstream file(filename.data(), std::ios::binary | std::ios::ate);
  CHECK(file.is_open());
  int filesize = file.tellg();
  std::string result;
  result.resize(filesize);
  CHECK_EQ(result.size(), filesize);
  file.seekg(0);
  CHECK(file.read(&(result)[0], filesize));
  file.close();
  return result;
}

bool LoadLabelMapFromFileOrBytes(const std::string& labelmap_file,
                                 const std::string& labelmap_bytes,
                                 protos::StringIntLabelMapProto* labelmap) {
  if (!labelmap_bytes.empty()) {
    CHECK(labelmap->ParseFromString(labelmap_bytes));
  } else {
    if (labelmap_file.empty()) {
      LOG(ERROR) << "labelmap file empty.";
      return false;
    }
    const std::string proto_bytes = ReadFileToString(labelmap_file);
    CHECK(labelmap->ParseFromString(proto_bytes));
  }
  return true;
}

}  // namespace tflite
}  // namespace lstm_object_detection
