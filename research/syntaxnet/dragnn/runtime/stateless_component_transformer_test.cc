// Copyright 2018 Google Inc. All Rights Reserved.
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

#include "dragnn/core/test/generic.h"
#include "dragnn/protos/spec.pb.h"
#include "dragnn/runtime/component_transformation.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"

namespace syntaxnet {
namespace dragnn {
namespace runtime {
namespace {

// Arbitrary supported component type.
constexpr char kSupportedComponentType[] = "SyntaxNetHeadSelectionComponent";

// Returns a ComponentSpec that is supported by the transformer.
ComponentSpec MakeSupportedSpec() {
  ComponentSpec component_spec;
  component_spec.mutable_component_builder()->set_registered_name(
      kSupportedComponentType);
  return component_spec;
}

// Tests that a compatible spec is modified to use StatelessComponent.
TEST(StatelessComponentTransformerTest, Compatible) {
  ComponentSpec component_spec = MakeSupportedSpec();

  ComponentSpec expected_spec = component_spec;
  expected_spec.mutable_backend()->set_registered_name("StatelessComponent");

  TF_ASSERT_OK(ComponentTransformer::ApplyAll(&component_spec));
  EXPECT_THAT(component_spec, test::EqualsProto(expected_spec));
}

// Tests that other component specs are not modified.
TEST(StatelessComponentTransformerTest, Incompatible) {
  ComponentSpec component_spec = MakeSupportedSpec();
  component_spec.mutable_component_builder()->set_registered_name("other");

  const ComponentSpec expected_spec = component_spec;

  TF_ASSERT_OK(ComponentTransformer::ApplyAll(&component_spec));
  EXPECT_THAT(component_spec, test::EqualsProto(expected_spec));
}

}  // namespace
}  // namespace runtime
}  // namespace dragnn
}  // namespace syntaxnet
