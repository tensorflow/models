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

#include "dragnn/runtime/myelin/myelin_spec_utils.h"

#include <set>
#include <string>

#include "dragnn/core/test/generic.h"
#include "dragnn/protos/spec.pb.h"
#include "syntaxnet/base.h"
#include "sling/file/file.h"
#include "sling/myelin/compute.h"
#include "sling/myelin/flow.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/platform/test.h"

namespace syntaxnet {
namespace dragnn {
namespace runtime {
namespace {

TEST(MyelinSpecUtilsTest, AddAndLookupMyelinFlowResource) {
  ComponentSpec component_spec;
  TF_ASSERT_OK(AddMyelinFlowResource("/dev/null", &component_spec));

  const Resource *resource = nullptr;
  TF_ASSERT_OK(LookupMyelinFlowResource(component_spec, &resource));

  ASSERT_NE(resource, nullptr);
  EXPECT_EQ(resource->name(), kMyelinFlowResourceName);
  ASSERT_EQ(resource->part_size(), 1);
  EXPECT_EQ(resource->part(0).file_pattern(), "/dev/null");
  EXPECT_EQ(resource->part(0).file_format(), kMyelinFlowResourceFileFormat);
  EXPECT_EQ(resource->part(0).record_format(), kMyelinFlowResourceRecordFormat);
}

TEST(MyelinSpecUtilsTest, LookupMyelinFlowResourceMissing) {
  ComponentSpec component_spec;
  const Resource *resource = nullptr;
  EXPECT_THAT(LookupMyelinFlowResource(component_spec, &resource),
              test::IsErrorWithSubstr("has no Myelin Flow resource"));

  component_spec.add_resource()->set_name("foo");
  EXPECT_THAT(LookupMyelinFlowResource(component_spec, &resource),
              test::IsErrorWithSubstr("has no Myelin Flow resource"));

  component_spec.add_resource()->set_name("bar");
  EXPECT_THAT(LookupMyelinFlowResource(component_spec, &resource),
              test::IsErrorWithSubstr("has no Myelin Flow resource"));
}

TEST(MyelinSpecUtilsTest, LookupMyelinFlowResourceWrongName) {
  ComponentSpec component_spec;
  TF_ASSERT_OK(AddMyelinFlowResource("/dev/null", &component_spec));
  component_spec.mutable_resource(0)->set_name("bad");

  const Resource *resource = nullptr;
  EXPECT_THAT(LookupMyelinFlowResource(component_spec, &resource),
              test::IsErrorWithSubstr("has no Myelin Flow resource"));
}

TEST(MyelinSpecUtilsTest, LookupMyelinFlowResourceWrongFileFormat) {
  ComponentSpec component_spec;
  TF_ASSERT_OK(AddMyelinFlowResource("/dev/null", &component_spec));
  component_spec.mutable_resource(0)->mutable_part(0)->set_file_format("bad");

  const Resource *resource = nullptr;
  EXPECT_THAT(LookupMyelinFlowResource(component_spec, &resource),
              test::IsErrorWithSubstr("wrong file format"));
}

TEST(MyelinSpecUtilsTest, LookupMyelinFlowResourceWrongRecordFormat) {
  ComponentSpec component_spec;
  TF_ASSERT_OK(AddMyelinFlowResource("/dev/null", &component_spec));
  component_spec.mutable_resource(0)->mutable_part(0)->set_record_format("bad");

  const Resource *resource = nullptr;
  EXPECT_THAT(LookupMyelinFlowResource(component_spec, &resource),
              test::IsErrorWithSubstr("wrong record format"));
}

TEST(MyelinSpecUtilsTest, LookupMyelinFlowResourceWrongNumberOfParts) {
  ComponentSpec component_spec;
  TF_ASSERT_OK(AddMyelinFlowResource("/dev/null", &component_spec));
  component_spec.mutable_resource(0)->add_part();

  const Resource *resource = nullptr;
  EXPECT_THAT(LookupMyelinFlowResource(component_spec, &resource),
              test::IsErrorWithSubstr("expected 1 part"));
}

TEST(MyelinSpecUtilsTest, LookupMyelinFlowResourceDuplicate) {
  ComponentSpec component_spec;
  TF_ASSERT_OK(AddMyelinFlowResource("/dev/null", &component_spec));
  component_spec.add_resource()->set_name(kMyelinFlowResourceName);

  const Resource *resource = nullptr;
  EXPECT_THAT(
      LookupMyelinFlowResource(component_spec, &resource),
      test::IsErrorWithSubstr("contains duplicate Myelin Flow resource"));
}

TEST(MyelinSpecUtilsTest, AddMyelinFlowResourceDuplicate) {
  ComponentSpec component_spec;
  TF_ASSERT_OK(AddMyelinFlowResource("/dev/null", &component_spec));

  EXPECT_THAT(
      AddMyelinFlowResource("another/flow", &component_spec),
      test::IsErrorWithSubstr("already contains a Myelin Flow resource"));
}

TEST(MyelinSpecUtilsTest, LoadMyelinFlowInvalidPath) {
  sling::myelin::Flow flow;
  EXPECT_THAT(LoadMyelinFlow("invalid/path", &flow),
              test::IsErrorWithSubstr("Failed to load Myelin Flow"));
}

TEST(MyelinSpecUtilsTest, LoadMyelinFlowValidFile) {
  // Build and write a Flow file with some variables that are annotated with
  // input and output aliases.
  sling::myelin::Flow original_flow;
  original_flow
      .AddVariable("input", sling::myelin::DT_FLOAT, sling::myelin::Shape())
      ->aliases = {"INPUT/a"};
  original_flow
      .AddVariable("output", sling::myelin::DT_FLOAT, sling::myelin::Shape())
      ->aliases = {"OUTPUT/b"};
  original_flow
      .AddVariable("both", sling::myelin::DT_FLOAT, sling::myelin::Shape())
      ->aliases = {"INPUT/c", "OUTPUT/d"};
  original_flow.AddVariable("neither", sling::myelin::DT_FLOAT,
                            sling::myelin::Shape());

  const string flow_path =
      tensorflow::io::JoinPath(tensorflow::testing::TmpDir(), "foo.flow");
  sling::File::Init();
  original_flow.Save(flow_path);

  // Load the Flow file into a fresh Flow and check that inputs and outputs are
  // marked as such.
  sling::myelin::Flow flow;
  TF_ASSERT_OK(LoadMyelinFlow(flow_path, &flow));

  ASSERT_NE(flow.Var("input"), nullptr);
  EXPECT_TRUE(flow.Var("input")->in);
  EXPECT_FALSE(flow.Var("input")->out);

  ASSERT_NE(flow.Var("output"), nullptr);
  EXPECT_FALSE(flow.Var("output")->in);
  EXPECT_TRUE(flow.Var("output")->out);

  ASSERT_NE(flow.Var("both"), nullptr);
  EXPECT_TRUE(flow.Var("both")->in);
  EXPECT_TRUE(flow.Var("both")->out);

  ASSERT_NE(flow.Var("neither"), nullptr);
  EXPECT_FALSE(flow.Var("neither")->in);
  EXPECT_FALSE(flow.Var("neither")->out);
}

TEST(MyelinSpecUtilsTest, RegisterMyelinLibraries) {
  sling::myelin::Library library;
  RegisterMyelinLibraries(&library);

  // The |library| should contain something.
  EXPECT_GT(library.transformers().size() + library.typers().size(), 0);
}

TEST(MyelinSpecUtilsTest, GetRecurrentLayerNamesEmpty) {
  sling::myelin::Flow flow;

  const std::set<string> expected_names;
  EXPECT_EQ(GetRecurrentLayerNames(flow), expected_names);
}

TEST(MyelinSpecUtilsTest, GetRecurrentLayerNamesVariablesWithNoAliases) {
  sling::myelin::Flow flow;
  flow.AddVariable("x", sling::myelin::DT_FLOAT, {});
  flow.AddVariable("y", sling::myelin::DT_INT32, {});

  const std::set<string> expected_names;
  EXPECT_EQ(GetRecurrentLayerNames(flow), expected_names);
}

TEST(MyelinSpecUtilsTest, GetRecurrentLayerNamesVariablesWithAliases) {
  sling::myelin::Flow flow;
  flow.AddVariable("x", sling::myelin::DT_FLOAT, {})->aliases = {"foo", "bar"};
  flow.AddVariable("y", sling::myelin::DT_INT32, {})->aliases = {
      "INPUT/y",                            //
      "INPUT/fixed_channel_0_index_0_ids",  //
      "INPUT/linked_channel_0_activations"};
  flow.AddVariable("z", sling::myelin::DT_INT32, {})->aliases = {"OUTPUT/z"};

  const std::set<string> expected_names = {"y"};
  EXPECT_EQ(GetRecurrentLayerNames(flow), expected_names);
}

TEST(MyelinSpecUtilsTest, GetRecurrentLayerNamesVariablesWithMultipleAliases) {
  sling::myelin::Flow flow;
  flow.AddVariable("x", sling::myelin::DT_FLOAT, {})->aliases = {"foo", "bar"};
  flow.AddVariable("y", sling::myelin::DT_INT32, {})->aliases = {
      "INPUT/recurrent_1",                  //
      "INPUT/recurrent_2",                  //
      "INPUT/fixed_channel_0_index_0_ids",  //
      "INPUT/linked_channel_0_activations"};
  flow.AddVariable("z", sling::myelin::DT_INT32, {})->aliases = {
      "OUTPUT/output_1",  //
      "OUTPUT/output_2"};

  const std::set<string> expected_names = {"recurrent_1", "recurrent_2"};
  EXPECT_EQ(GetRecurrentLayerNames(flow), expected_names);
}

TEST(MyelinSpecUtilsTest, GetOutputLayerNamesEmpty) {
  sling::myelin::Flow flow;

  const std::set<string> expected_names;
  EXPECT_EQ(GetOutputLayerNames(flow), expected_names);
}

TEST(MyelinSpecUtilsTest, GetOutputLayerNamesVariablesWithNoAliases) {
  sling::myelin::Flow flow;
  flow.AddVariable("x", sling::myelin::DT_FLOAT, {});
  flow.AddVariable("y", sling::myelin::DT_INT32, {});

  const std::set<string> expected_names;
  EXPECT_EQ(GetOutputLayerNames(flow), expected_names);
}

TEST(MyelinSpecUtilsTest, GetOutputLayerNamesVariablesWithAliases) {
  sling::myelin::Flow flow;
  flow.AddVariable("x", sling::myelin::DT_FLOAT, {})->aliases = {"foo", "bar"};
  flow.AddVariable("y", sling::myelin::DT_INT32, {})->aliases = {
      "INPUT/y",                            //
      "INPUT/fixed_channel_0_index_0_ids",  //
      "INPUT/linked_channel_0_activations"};
  flow.AddVariable("z", sling::myelin::DT_INT32, {})->aliases = {"OUTPUT/z"};

  const std::set<string> expected_names = {"z"};
  EXPECT_EQ(GetOutputLayerNames(flow), expected_names);
}

TEST(MyelinSpecUtilsTest, GetOutputLayerNamesVariablesWithMultipleAliases) {
  sling::myelin::Flow flow;
  flow.AddVariable("x", sling::myelin::DT_FLOAT, {})->aliases = {"foo", "bar"};
  flow.AddVariable("y", sling::myelin::DT_INT32, {})->aliases = {
      "INPUT/recurrent_1",                  //
      "INPUT/recurrent_2",                  //
      "INPUT/fixed_channel_0_index_0_ids",  //
      "INPUT/linked_channel_0_activations"};
  flow.AddVariable("z", sling::myelin::DT_INT32, {})->aliases = {
      "OUTPUT/output_1",  //
      "OUTPUT/output_2"};

  const std::set<string> expected_names = {"output_1", "output_2"};
  EXPECT_EQ(GetOutputLayerNames(flow), expected_names);
}

TEST(MyelinSpecUtilsTest, MakeMyelinInputFixedFeatureIdName) {
  EXPECT_EQ(MakeMyelinInputFixedFeatureIdName(0, 1),
            "INPUT/fixed_channel_0_index_1_ids");
  EXPECT_EQ(MakeMyelinInputFixedFeatureIdName(1, 0),
            "INPUT/fixed_channel_1_index_0_ids");
}

TEST(MyelinSpecUtilsTest, MakeMyelinInputLinkedActivationVectorName) {
  EXPECT_EQ(MakeMyelinInputLinkedActivationVectorName(0),
            "INPUT/linked_channel_0_activations");
  EXPECT_EQ(MakeMyelinInputLinkedActivationVectorName(1),
            "INPUT/linked_channel_1_activations");
}

TEST(MyelinSpecUtilsTest, MakeMyelinInputLinkedOutOfBoundsIndicatorName) {
  EXPECT_EQ(MakeMyelinInputLinkedOutOfBoundsIndicatorName(0),
            "INPUT/linked_channel_0_out_of_bounds");
  EXPECT_EQ(MakeMyelinInputLinkedOutOfBoundsIndicatorName(1),
            "INPUT/linked_channel_1_out_of_bounds");
}

TEST(MyelinSpecUtilsTest, MakeMyelinInputRecurrentLayerName) {
  EXPECT_EQ(MakeMyelinInputRecurrentLayerName("foo"), "INPUT/foo");
  EXPECT_EQ(MakeMyelinInputRecurrentLayerName("bar_baz"), "INPUT/bar_baz");
}

TEST(MyelinSpecUtilsTest, MakeMyelinOutputLayerName) {
  EXPECT_EQ(MakeMyelinOutputLayerName("foo"), "OUTPUT/foo");
  EXPECT_EQ(MakeMyelinOutputLayerName("bar_baz"), "OUTPUT/bar_baz");
}

}  // namespace
}  // namespace runtime
}  // namespace dragnn
}  // namespace syntaxnet
