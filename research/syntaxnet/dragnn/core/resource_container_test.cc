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

// Tests the methods of ResourceContainer.
//
// NOTE(danielandor): For all tests: ResourceContainer is derived from
// RefCounted, which requires the use of Unref to reduce the ref count
// to zero and automatically delete the pointer.
#include "dragnn/core/resource_container.h"

#include <gmock/gmock.h>
#include "tensorflow/core/platform/test.h"

namespace syntaxnet {
namespace dragnn {

class MockDatatype {};

TEST(ResourceContainerTest, Get) {
  std::unique_ptr<MockDatatype> data(new MockDatatype());
  MockDatatype *data_ptr = data.get();
  auto *container = new ResourceContainer<MockDatatype>(std::move(data));
  EXPECT_EQ(data_ptr, container->get());
  container->Unref();
}

TEST(ResourceContainerTest, Release) {
  std::unique_ptr<MockDatatype> data(new MockDatatype());
  MockDatatype *data_ptr = data.get();
  auto *container = new ResourceContainer<MockDatatype>(std::move(data));
  std::unique_ptr<MockDatatype> data_again = container->release();
  container->Unref();
  EXPECT_EQ(data_ptr, data_again.get());
}

TEST(ResourceContainerTest, NullptrOnGetAfterRelease) {
  std::unique_ptr<MockDatatype> data(new MockDatatype());
  auto *container = new ResourceContainer<MockDatatype>(std::move(data));
  container->release();
  EXPECT_EQ(nullptr, container->get());
  container->Unref();
}

TEST(ResourceContainerTest, DebugString) {
  std::unique_ptr<MockDatatype> data(new MockDatatype());
  auto *container = new ResourceContainer<MockDatatype>(std::move(data));
  EXPECT_EQ("ResourceContainer", container->DebugString());
  container->Unref();
}

}  // namespace dragnn
}  // namespace syntaxnet
