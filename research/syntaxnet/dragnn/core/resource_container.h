// Copyright 2017 Google Inc. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// =============================================================================

#ifndef DRAGNN_CORE_RESOURCE_CONTAINER_H_
#define DRAGNN_CORE_RESOURCE_CONTAINER_H_

#include <memory>

#include "syntaxnet/base.h"
#include "tensorflow/core/framework/resource_mgr.h"

namespace syntaxnet {
namespace dragnn {

using tensorflow::strings::StrCat;

// Wrapper to store a data type T in the ResourceMgr. There should be one per
// Session->Run() call that may happen concurrently.
template <class T>
class ResourceContainer : public tensorflow::ResourceBase {
 public:
  explicit ResourceContainer(std::unique_ptr<T> data)
      : data_(std::move(data)) {}

  ~ResourceContainer() override {}

  T *get() { return data_.get(); }
  std::unique_ptr<T> release() { return std::move(data_); }

  string DebugString() override { return "ResourceContainer"; }

 private:
  std::unique_ptr<T> data_;
};

}  // namespace dragnn
}  // namespace syntaxnet

#endif  // DRAGNN_CORE_RESOURCE_CONTAINER_H_
