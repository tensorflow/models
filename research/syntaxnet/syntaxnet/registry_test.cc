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

#include "syntaxnet/registry.h"

#include <memory>

#include "dragnn/core/test/generic.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"

namespace syntaxnet {

class ThingDoer : public RegisterableClass<ThingDoer> {};

DECLARE_SYNTAXNET_CLASS_REGISTRY("Thing doer", ThingDoer);
REGISTER_SYNTAXNET_CLASS_REGISTRY("Thing doer", ThingDoer);

class Foo : public ThingDoer {};
class Bar : public ThingDoer {};
class Bar2 : public ThingDoer {};

REGISTER_SYNTAXNET_CLASS_COMPONENT(ThingDoer, "foo", Foo);
REGISTER_SYNTAXNET_CLASS_COMPONENT(ThingDoer, "bar", Bar);

#if DRAGNN_REGISTRY_TEST_WITH_DUPLICATE
REGISTER_SYNTAXNET_CLASS_COMPONENT(ThingDoer, "bar", Bar2);  // bad

constexpr char kDuplicateError[] =
    "Multiple classes named 'bar' have been registered as Thing doer";

#endif

namespace {

#if !DRAGNN_REGISTRY_TEST_WITH_DUPLICATE

// Tests that CreateOrError() is successful for a properly registered component.
TEST(RegistryTest, CreateOrErrorSuccess) {
  std::unique_ptr<ThingDoer> object;
  TF_ASSERT_OK(ThingDoer::CreateOrError("foo", &object));
  ASSERT_NE(object, nullptr);
}

#else

// Tests that CreateOrError() fails if the registry is misconfigured.
TEST(RegistryTest, CreateOrErrorFailure) {
  std::unique_ptr<ThingDoer> object;
  EXPECT_THAT(ThingDoer::CreateOrError("bar", &object),
              test::IsErrorWithSubstr(kDuplicateError));
  ASSERT_EQ(object, nullptr);

  // Any call to Create has the same error.
  EXPECT_THAT(ThingDoer::CreateOrError("foo", &object),
              test::IsErrorWithSubstr(kDuplicateError));
}

// Tests that Create() dies if the registry is misconfigured.
TEST(RegistryTest, CreateFailure) {
  EXPECT_DEATH(ThingDoer::Create("bar"), kDuplicateError);
}

#endif

// Tests that CreateOrError() returns error if the component is unknown.
TEST(RegistryTest, CreateOrErrorUnknown) {
  std::unique_ptr<ThingDoer> object;
  EXPECT_FALSE(ThingDoer::CreateOrError("unknown", &object).ok());
}

// Tests that Validate() returns OK only when the registry is fine.
TEST(RegistryTest, Validate) {
#if DRAGNN_REGISTRY_TEST_WITH_DUPLICATE
  EXPECT_THAT(RegistryMetadata::Validate(),
              test::IsErrorWithSubstr(kDuplicateError));
#else
  TF_EXPECT_OK(RegistryMetadata::Validate());
#endif
}

}  // namespace
}  // namespace syntaxnet
