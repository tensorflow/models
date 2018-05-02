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

#include <memory>
#include <string>
#include <vector>

#include "dragnn/core/input_batch_cache.h"
#include "dragnn/core/test/generic.h"
#include "dragnn/protos/spec.pb.h"
#include "dragnn/runtime/sequence_linker.h"
#include "syntaxnet/base.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"

namespace syntaxnet {
namespace dragnn {
namespace runtime {
namespace {

// Returns a ComponentSpec that the linker will support.
ComponentSpec MakeSupportedSpec() {
  ComponentSpec component_spec;
  component_spec.set_name("test_component");
  component_spec.mutable_transition_system()->set_registered_name("shift-only");
  LinkedFeatureChannel *channel = component_spec.add_linked_feature();
  channel->set_fml("bias");
  channel->set_source_component("test_component");
  channel->set_source_translator("history");
  return component_spec;
}

// Tests that the linker supports appropriate specs.
TEST(RecurrentSequenceLinkerTest, Supported) {
  string name;
  ComponentSpec component_spec = MakeSupportedSpec();
  LinkedFeatureChannel &channel = *component_spec.mutable_linked_feature(0);

  TF_ASSERT_OK(SequenceLinker::Select(channel, component_spec, &name));
  EXPECT_EQ(name, "LeftToRightRecurrentSequenceLinker");

  (*component_spec.mutable_transition_system()
        ->mutable_parameters())["left_to_right"] = "false";
  TF_ASSERT_OK(SequenceLinker::Select(channel, component_spec, &name));
  EXPECT_EQ(name, "RightToLeftRecurrentSequenceLinker");

  channel.set_fml("bias(0)");
  TF_ASSERT_OK(SequenceLinker::Select(channel, component_spec, &name));
  EXPECT_EQ(name, "RightToLeftRecurrentSequenceLinker");

  (*component_spec.mutable_transition_system()
        ->mutable_parameters())["left_to_right"] = "true";
  TF_ASSERT_OK(SequenceLinker::Select(channel, component_spec, &name));
  EXPECT_EQ(name, "LeftToRightRecurrentSequenceLinker");
}

// Tests that the linker requires the right transition system.
TEST(RecurrentSequenceLinkerTest, WrongTransitionSystem) {
  string name;
  ComponentSpec component_spec = MakeSupportedSpec();
  const LinkedFeatureChannel &channel = component_spec.linked_feature(0);

  component_spec.mutable_transition_system()->set_registered_name("bad");
  EXPECT_THAT(SequenceLinker::Select(channel, component_spec, &name),
              test::IsErrorWithSubstr("No SequenceLinker supports channel"));
}

// Tests that the linker requires the right FML.
TEST(RecurrentSequenceLinkerTest, WrongFml) {
  string name;
  ComponentSpec component_spec = MakeSupportedSpec();
  LinkedFeatureChannel &channel = *component_spec.mutable_linked_feature(0);

  channel.set_fml("bad");
  EXPECT_THAT(SequenceLinker::Select(channel, component_spec, &name),
              test::IsErrorWithSubstr("No SequenceLinker supports channel"));
}

// Tests that the linker requires a recurrent link.
TEST(RecurrentSequenceLinkerTest, WrongSource) {
  string name;
  ComponentSpec component_spec = MakeSupportedSpec();
  LinkedFeatureChannel &channel = *component_spec.mutable_linked_feature(0);

  channel.set_source_component("bad");
  EXPECT_THAT(SequenceLinker::Select(channel, component_spec, &name),
              test::IsErrorWithSubstr("No SequenceLinker supports channel"));
}

// Tests that the linker requires the right translator.
TEST(RecurrentSequenceLinkerTest, WrongTranslator) {
  string name;
  ComponentSpec component_spec = MakeSupportedSpec();
  LinkedFeatureChannel &channel = *component_spec.mutable_linked_feature(0);

  channel.set_source_translator("bad");
  EXPECT_THAT(SequenceLinker::Select(channel, component_spec, &name),
              test::IsErrorWithSubstr("No SequenceLinker supports channel"));
}

// Tests that the linker can be initialized and used to extract links.
TEST(RecurrentSequenceLinkerTest, InitializeAndGetLinks) {
  const ComponentSpec component_spec = MakeSupportedSpec();
  const LinkedFeatureChannel &channel = component_spec.linked_feature(0);

  std::unique_ptr<SequenceLinker> linker;
  TF_ASSERT_OK(SequenceLinker::New("LeftToRightRecurrentSequenceLinker",
                                   channel, component_spec, &linker));

  InputBatchCache input;
  std::vector<int32> links = {123, 456, 789};  // gets overwritten
  TF_ASSERT_OK(linker->GetLinks(10, &input, &links));

  const std::vector<int32> expected_links = {-1, 0, 1, 2, 3, 4, 5, 6, 7, 8};
  EXPECT_EQ(links, expected_links);
}

// Tests that the links are reversed for right-to-left components.
TEST(RecurrentSequenceLinkerTest, InitializeAndGetLinksRightToLeft) {
  ComponentSpec component_spec = MakeSupportedSpec();
  const LinkedFeatureChannel &channel = component_spec.linked_feature(0);

  std::unique_ptr<SequenceLinker> linker;
  TF_ASSERT_OK(SequenceLinker::New("RightToLeftRecurrentSequenceLinker",
                                   channel, component_spec, &linker));

  InputBatchCache input;
  std::vector<int32> links = {123, 456, 789};  // gets overwritten
  TF_ASSERT_OK(linker->GetLinks(10, &input, &links));

  const std::vector<int32> expected_links = {8, 7, 6, 5, 4, 3, 2, 1, 0, -1};
  EXPECT_EQ(links, expected_links);
}

}  // namespace
}  // namespace runtime
}  // namespace dragnn
}  // namespace syntaxnet
