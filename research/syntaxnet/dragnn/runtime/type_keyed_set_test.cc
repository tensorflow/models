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

#include "dragnn/runtime/type_keyed_set.h"

#include <utility>

#include "tensorflow/core/platform/test.h"

namespace syntaxnet {
namespace dragnn {
namespace runtime {
namespace {

// Dummy struct for tests.
struct Foo {
  float value = -1.5;
};

// Type aliases to exercise usage of aliases as type keys.
using OtherInt = int;
using OtherFoo = Foo;

// Tests that TypeKeyedSet::Get() returns the same object once created.
TEST(TypeKeyedSetTest, Get) {
  TypeKeyedSet set;

  // Get a couple types, and check for default-constructed values.
  int &int_object = set.Get<int>();
  ASSERT_NE(&int_object, nullptr);
  EXPECT_EQ(int_object, 0);  // due to T()
  int_object = 2718;
  Foo &foo_object = set.Get<Foo>();
  ASSERT_NE(&foo_object, nullptr);
  EXPECT_EQ(foo_object.value, -1.5);  // due to T()
  foo_object.value = 3141.5;

  // Get the same types again, this time using type aliases, and check for
  // address and value equality.
  OtherInt &other_int_object = set.Get<OtherInt>();
  EXPECT_EQ(&other_int_object, &int_object);
  EXPECT_EQ(other_int_object, 2718);
  OtherFoo &other_foo_object = set.Get<OtherFoo>();
  EXPECT_EQ(&other_foo_object, &foo_object);
  EXPECT_EQ(other_foo_object.value, 3141.5);
}

// Tests that TypeKeyedSet::Clear() removes existing values.
TEST(TypeKeyedSetTest, Clear) {
  // Create a set with some values.
  TypeKeyedSet set;
  int &int_object = set.Get<int>();
  int_object = 2718;
  Foo &foo_object = set.Get<Foo>();
  foo_object.value = 3141.5;

  // Clear the set and check that the values are now defaulted.
  set.Clear();
  EXPECT_EQ(set.Get<int>(), 0);
  EXPECT_EQ(set.Get<Foo>().value, -1.5);
}

// Tests that TypeKeyedSet supports move construction.
TEST(TypeKeyedSetTest, MoveConstruction) {
  TypeKeyedSet set1;

  // Insert a couple of values.
  int &int_object = set1.Get<int>();
  int_object = 2718;
  Foo &foo_object = set1.Get<Foo>();
  foo_object.value = 3141.5;

  // Move-construct another set, and check address and value equality.
  TypeKeyedSet set2(std::move(set1));
  OtherInt &other_int_object = set2.Get<OtherInt>();
  EXPECT_EQ(&other_int_object, &int_object);
  EXPECT_EQ(other_int_object, 2718);
  OtherFoo &other_foo_object = set2.Get<OtherFoo>();
  EXPECT_EQ(&other_foo_object, &foo_object);
  EXPECT_EQ(other_foo_object.value, 3141.5);
}

// Tests that TypeKeyedSet supports move assignment.
TEST(TypeKeyedSetTest, MoveAssignment) {
  // Create one set with some values.
  TypeKeyedSet set1;
  int &int_object = set1.Get<int>();
  int_object = 2718;
  Foo &foo_object = set1.Get<Foo>();
  foo_object.value = 3141.5;

  // Create another set with different values, to be overwritten.
  TypeKeyedSet set2;
  set2.Get<int>() = 123;
  set2.Get<Foo>().value = 76.5;

  // Move-assign to another set, and check address and value equality.
  set2 = std::move(set1);
  OtherInt &other_int_object = set2.Get<OtherInt>();
  EXPECT_EQ(&other_int_object, &int_object);
  EXPECT_EQ(other_int_object, 2718);
  OtherFoo &other_foo_object = set2.Get<OtherFoo>();
  EXPECT_EQ(&other_foo_object, &foo_object);
  EXPECT_EQ(other_foo_object.value, 3141.5);
}

}  // namespace
}  // namespace runtime
}  // namespace dragnn
}  // namespace syntaxnet
