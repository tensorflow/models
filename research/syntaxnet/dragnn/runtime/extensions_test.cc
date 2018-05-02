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

#include "dragnn/runtime/extensions.h"

#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include "tensorflow/core/platform/test.h"

namespace syntaxnet {
namespace dragnn {
namespace runtime {
namespace {

using ::testing::ElementsAre;

// Dummy struct for tests.
struct Foo {
  Foo() = default;
  explicit Foo(float real, int num) : real(real) {
    for (int i = 0; i < num; ++i) ints.push_back(i);
  }

  float real = 0.0;
  std::vector<int> ints;
};

// Returns a shared extension handle from the |manager|.
template<class T>
SharedExtensionHandle<T> GetShared(ExtensionManager *manager) {
  SharedExtensionHandle<T> handle;
  manager->GetShared(&handle);
  return handle;
}

// Returns a local extension handle from the |manager|.
template<class T>
LocalExtensionHandle<T> AddLocal(ExtensionManager *manager) {
  LocalExtensionHandle<T> handle;
  manager->AddLocal(&handle);
  return handle;
}

// Tests that GetShared() reuses existing extensions.
TEST(ExtensionManagerTest, GetShared) {
  ExtensionManager manager;
  const auto foo_handle1 = GetShared<Foo>(&manager);
  const auto int_handle = GetShared<int>(&manager);
  const auto foo_handle2 = GetShared<Foo>(&manager);

  Extensions extensions;
  extensions.Reset(&manager);
  Foo &foo1 = extensions.Get(foo_handle1);
  Foo &foo2 = extensions.Get(foo_handle2);

  EXPECT_EQ(&foo1, &foo2);
  EXPECT_EQ(foo1.real, 0.0);
  EXPECT_TRUE(foo1.ints.empty());
  EXPECT_EQ(extensions.Get(int_handle), 0);  // T() zero-initializes POD
}

// Tests that AddLocal() always adds a new extension.
TEST(ExtensionManagerTest, AddLocal) {
  ExtensionManager manager;
  const auto foo_handle1 = AddLocal<Foo>(&manager);
  const auto int_handle = AddLocal<int>(&manager);
  const auto foo_handle2 = AddLocal<Foo>(&manager);

  Extensions extensions;
  extensions.Reset(&manager);
  Foo &foo1 = extensions.Get(foo_handle1);
  Foo &foo2 = extensions.Get(foo_handle2);

  EXPECT_NE(&foo1, &foo2);
  EXPECT_EQ(foo1.real, 0.0);
  EXPECT_EQ(foo2.real, 0.0);
  EXPECT_TRUE(foo1.ints.empty());
  EXPECT_TRUE(foo2.ints.empty());
  EXPECT_EQ(extensions.Get(int_handle), 0);  // T() zero-initializes POD
}

// Tests that Get() always returns the same object.
TEST(ExtensionManagerTest, GetReturnsSameObject) {
  ExtensionManager manager;
  const auto foo_shared = GetShared<Foo>(&manager);
  const auto int_shared = GetShared<int>(&manager);
  const auto foo_local = AddLocal<Foo>(&manager);
  const auto int_local = AddLocal<int>(&manager);

  Extensions extensions;
  extensions.Reset(&manager);
  Foo &foo_shared1 = extensions.Get(foo_shared);
  int &int_shared1 = extensions.Get(int_shared);
  Foo &foo_local1 = extensions.Get(foo_local);
  int &int_local1 = extensions.Get(int_local);

  Foo &foo_shared2 = extensions.Get(foo_shared);
  int &int_shared2 = extensions.Get(int_shared);
  Foo &foo_local2 = extensions.Get(foo_local);
  int &int_local2 = extensions.Get(int_local);

  EXPECT_EQ(&foo_shared1, &foo_shared2);
  EXPECT_EQ(&int_shared1, &int_shared2);
  EXPECT_EQ(&foo_local1, &foo_local2);
  EXPECT_EQ(&int_local1, &int_local2);
}

// Tests that local extensions can use non-default constructors.
TEST(ExtensionManagerTest, LocalAllowsNonDefaultConstructor) {
  ExtensionManager manager;
  const auto foo_handle = AddLocal<Foo>(&manager);
  const auto int_handle = AddLocal<int>(&manager);

  Extensions extensions;
  extensions.Reset(&manager);

  // Use non-default constructors to get initialized values.
  Foo &foo1 = extensions.Get(foo_handle, 0.5, 5);
  EXPECT_EQ(foo1.real, 0.5);
  EXPECT_THAT(foo1.ints, ElementsAre(0, 1, 2, 3, 4));
  EXPECT_EQ(extensions.Get(int_handle, -123), -123);

  // However, once created, the non-default constructor args are ignored.
  Foo &foo2 = extensions.Get(foo_handle, 1.23, 1000);
  EXPECT_EQ(foo2.real, 0.5);
  EXPECT_THAT(foo2.ints, ElementsAre(0, 1, 2, 3, 4));
  EXPECT_EQ(extensions.Get(int_handle, -456), -123);
}

// Tests that calling Reset() with the same manager is a NOP.
TEST(ExtensionManagerTest, ResetWithSameManager) {
  ExtensionManager manager;
  const auto foo_shared = GetShared<Foo>(&manager);
  const auto int_shared = GetShared<int>(&manager);
  const auto foo_local = AddLocal<Foo>(&manager);
  const auto int_local = AddLocal<int>(&manager);

  Extensions extensions;
  extensions.Reset(&manager);
  Foo &foo_shared1 = extensions.Get(foo_shared);
  int &int_shared1 = extensions.Get(int_shared);
  Foo &foo_local1 = extensions.Get(foo_local);
  int &int_local1 = extensions.Get(int_local);

  extensions.Reset(&manager);
  Foo &foo_shared2 = extensions.Get(foo_shared);
  int &int_shared2 = extensions.Get(int_shared);
  Foo &foo_local2 = extensions.Get(foo_local);
  int &int_local2 = extensions.Get(int_local);

  EXPECT_EQ(&foo_shared1, &foo_shared2);
  EXPECT_EQ(&int_shared1, &int_shared2);
  EXPECT_EQ(&foo_local1, &foo_local2);
  EXPECT_EQ(&int_local1, &int_local2);
}

// Tests that Reset() can be used to switch managers.
TEST(ExtensionManagerTest, ResetWithDifferentManager) {
  ExtensionManager manager1;
  const auto foo_shared = GetShared<Foo>(&manager1);
  const auto foo_local = AddLocal<Foo>(&manager1);

  ExtensionManager manager2;
  const auto int_shared = GetShared<int>(&manager2);
  const auto int_local = AddLocal<int>(&manager2);

  Extensions extensions;
  extensions.Reset(&manager1);
  EXPECT_EQ(extensions.Get(foo_shared).real, 0.0);
  EXPECT_EQ(extensions.Get(foo_local, 0.75, 3).real, 0.75);

  extensions.Reset(&manager2);
  EXPECT_EQ(extensions.Get(int_shared), 0);
  EXPECT_EQ(extensions.Get(int_local, 5), 5);
}

// Tests that Extensions supports move construction.
TEST(ExtensionManagerTest, MoveConstruction) {
  ExtensionManager manager;
  const auto foo_shared = GetShared<Foo>(&manager);
  const auto int_shared = GetShared<int>(&manager);
  const auto foo_local = AddLocal<Foo>(&manager);
  const auto int_local = AddLocal<int>(&manager);

  // Add a couple more spurious extensions that are never set, to exercise
  // movement of non-present extensions.
  GetShared<float>(&manager);
  AddLocal<float>(&manager);

  Extensions extensions1;
  extensions1.Reset(&manager);
  Foo &foo_shared1 = extensions1.Get(foo_shared);
  int &int_shared1 = extensions1.Get(int_shared);
  Foo &foo_local1 = extensions1.Get(foo_local);
  int &int_local1 = extensions1.Get(int_local);

  Extensions extensions2 = std::move(extensions1);
  Foo &foo_shared2 = extensions2.Get(foo_shared);
  int &int_shared2 = extensions2.Get(int_shared);
  Foo &foo_local2 = extensions2.Get(foo_local);
  int &int_local2 = extensions2.Get(int_local);

  EXPECT_EQ(&foo_shared1, &foo_shared2);
  EXPECT_EQ(&int_shared1, &int_shared2);
  EXPECT_EQ(&foo_local1, &foo_local2);
  EXPECT_EQ(&int_local1, &int_local2);
}

// Tests that Extensions supports move assignment.
TEST(ExtensionManagerTest, MoveAssignment) {
  ExtensionManager manager1;
  const auto foo_shared = GetShared<Foo>(&manager1);
  const auto foo_local = AddLocal<Foo>(&manager1);

  ExtensionManager manager2;
  const auto int_shared = GetShared<int>(&manager2);
  const auto int_local = AddLocal<int>(&manager2);

  // Add a couple more spurious extensions that are never set, to exercise
  // movement of non-present extensions.
  GetShared<float>(&manager1);
  GetShared<float>(&manager2);
  AddLocal<float>(&manager1);
  AddLocal<float>(&manager2);

  // Fill two sets of extensions.
  Extensions extensions1;
  extensions1.Reset(&manager1);
  extensions1.Get(foo_shared).real = 1.0;
  extensions1.Get(foo_local).real = 1.0;

  Extensions extensions2;
  extensions2.Reset(&manager2);
  extensions2.Get(int_shared) = 2;
  extensions2.Get(int_local) = 2;

  // Use a third set of extensions to perform a swap.
  Extensions extensions3;
  extensions3 = std::move(extensions1);
  extensions1 = std::move(extensions2);
  extensions2 = std::move(extensions3);

  EXPECT_EQ(extensions1.Get(int_shared), 2);
  EXPECT_EQ(extensions1.Get(int_local), 2);
  EXPECT_EQ(extensions2.Get(foo_shared).real, 1.0);
  EXPECT_EQ(extensions2.Get(foo_local).real, 1.0);
}

}  // namespace
}  // namespace runtime
}  // namespace dragnn
}  // namespace syntaxnet
