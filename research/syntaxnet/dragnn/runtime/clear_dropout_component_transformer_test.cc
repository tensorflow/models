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

#include "dragnn/core/test/generic.h"
#include "dragnn/protos/spec.pb.h"
#include "dragnn/runtime/component_transformation.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"

namespace syntaxnet {
namespace dragnn {
namespace runtime {
namespace {

// Tests that a spec with no dropout features is unmodified.
TEST(ClearDropoutComponentTransformerTest, DoesNotModifyIfNoDropout) {
  ComponentSpec component_spec;
  component_spec.mutable_component_builder()->set_registered_name("foo");
  component_spec.add_fixed_feature()->set_name("words");

  const ComponentSpec expected_spec = component_spec;

  TF_ASSERT_OK(ComponentTransformer::ApplyAll(&component_spec));
  EXPECT_THAT(component_spec, test::EqualsProto(expected_spec));
}

// Tests that a spec with dropout features is modified.
TEST(ClearDropoutComponentTransformerTest, ClearsDropout) {
  ComponentSpec component_spec;
  component_spec.mutable_component_builder()->set_registered_name("foo");
  FixedFeatureChannel *channel = component_spec.add_fixed_feature();
  channel->set_name("words");
  channel->set_dropout_id(100);
  channel->add_dropout_keep_probability(1.0);
  channel->add_dropout_keep_probability(0.5);
  channel->add_dropout_keep_probability(0.1);

  ComponentSpec expected_spec = component_spec;
  expected_spec.clear_fixed_feature();
  expected_spec.add_fixed_feature()->set_name("words");

  TF_ASSERT_OK(ComponentTransformer::ApplyAll(&component_spec));
  EXPECT_THAT(component_spec, test::EqualsProto(expected_spec));
}

}  // namespace
}  // namespace runtime
}  // namespace dragnn
}  // namespace syntaxnet
