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

#include "dragnn/runtime/xla/xla_spec_utils.h"

#include <set>
#include <string>

#include "dragnn/core/test/generic.h"
#include "dragnn/protos/spec.pb.h"
#include "syntaxnet/base.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/platform/test.h"

namespace syntaxnet {
namespace dragnn {
namespace runtime {
namespace {

TEST(XlaSpecUtilsTest, ModelNameForComponent) {
  ComponentSpec component_spec;
  component_spec.MutableExtension(CompilationSpec::component_spec_extension)
      ->set_model_name("ModelName");

  EXPECT_EQ(ModelNameForComponent(component_spec), "ModelName");
}

TEST(XlaSpecUtilsTest, ModelNameForComponent_Empty) {
  ComponentSpec component_spec;
  EXPECT_EQ(ModelNameForComponent(component_spec), "");

  component_spec.MutableExtension(CompilationSpec::component_spec_extension);
  EXPECT_EQ(ModelNameForComponent(component_spec), "");
}

TEST(XlaSpecUtilsTest, GetCellSubgraphSpecForComponent) {
  ComponentSpec component_spec;

  CellSubgraphSpec expected_cell_subgraph_spec;
  auto *input = expected_cell_subgraph_spec.add_input();
  input->set_name("fixed_channel_0_index_0_ids");
  input->set_tensor("cell/id:0");
  input->set_type(CellSubgraphSpec::Input::TYPE_FEATURE);
  auto *output = expected_cell_subgraph_spec.add_output();
  output->set_name("logits");
  output->set_tensor("cell/lookup:0");
  *(component_spec.MutableExtension(CompilationSpec::component_spec_extension)
        ->mutable_cell_subgraph_spec()) = expected_cell_subgraph_spec;

  CellSubgraphSpec actual_cell_subgraph_spec;
  TF_ASSERT_OK(GetCellSubgraphSpecForComponent(component_spec,
                                               &actual_cell_subgraph_spec));
  EXPECT_THAT(actual_cell_subgraph_spec,
              test::EqualsProto(expected_cell_subgraph_spec));
}

TEST(XlaSpecUtilsTest, GetCellSubgraphSpecForComponent_Missing) {
  ComponentSpec component_spec;
  CellSubgraphSpec cell_subgraph_spec;

  EXPECT_THAT(
      GetCellSubgraphSpecForComponent(component_spec, &cell_subgraph_spec),
      test::IsErrorWithSubstr("does not have a CellSubgraphSpec"));

  component_spec.MutableExtension(CompilationSpec::component_spec_extension);
  EXPECT_THAT(
      GetCellSubgraphSpecForComponent(component_spec, &cell_subgraph_spec),
      test::IsErrorWithSubstr("does not have a CellSubgraphSpec"));
}

TEST(XlaSpecUtilsTest, AddAndLookupFrozenGraphDefResource) {
  ComponentSpec component_spec;
  TF_ASSERT_OK(AddFrozenGraphDefResource("/dev/null", &component_spec));

  const Resource *resource = nullptr;
  TF_ASSERT_OK(LookupFrozenGraphDefResource(component_spec, &resource));

  ASSERT_NE(resource, nullptr);
  EXPECT_EQ(resource->name(), kFrozenGraphDefResourceName);
  ASSERT_EQ(resource->part_size(), 1);
  EXPECT_EQ(resource->part(0).file_pattern(), "/dev/null");
  EXPECT_EQ(resource->part(0).file_format(), kFrozenGraphDefResourceFileFormat);
  EXPECT_EQ(resource->part(0).record_format(),
            kFrozenGraphDefResourceRecordFormat);
}

TEST(XlaSpecUtilsTest, LookupFrozenGraphDefResourceMissing) {
  ComponentSpec component_spec;
  const Resource *resource = nullptr;
  EXPECT_THAT(LookupFrozenGraphDefResource(component_spec, &resource),
              test::IsErrorWithSubstr("has no frozen TF GraphDef resource"));

  component_spec.add_resource()->set_name("foo");
  EXPECT_THAT(LookupFrozenGraphDefResource(component_spec, &resource),
              test::IsErrorWithSubstr("has no frozen TF GraphDef resource"));

  component_spec.add_resource()->set_name("bar");
  EXPECT_THAT(LookupFrozenGraphDefResource(component_spec, &resource),
              test::IsErrorWithSubstr("has no frozen TF GraphDef resource"));
}

TEST(XlaSpecUtilsTest, LookupFrozenGraphDefResourceWrongName) {
  ComponentSpec component_spec;
  TF_ASSERT_OK(AddFrozenGraphDefResource("/dev/null", &component_spec));
  component_spec.mutable_resource(0)->set_name("bad");

  const Resource *resource = nullptr;
  EXPECT_THAT(LookupFrozenGraphDefResource(component_spec, &resource),
              test::IsErrorWithSubstr("has no frozen TF GraphDef resource"));
}

TEST(XlaSpecUtilsTest, LookupFrozenGraphDefResourceWrongFileFormat) {
  ComponentSpec component_spec;
  TF_ASSERT_OK(AddFrozenGraphDefResource("/dev/null", &component_spec));
  component_spec.mutable_resource(0)->mutable_part(0)->set_file_format("bad");

  const Resource *resource = nullptr;
  EXPECT_THAT(LookupFrozenGraphDefResource(component_spec, &resource),
              test::IsErrorWithSubstr("wrong file format"));
}

TEST(XlaSpecUtilsTest, LookupFrozenGraphDefResourceWrongRecordFormat) {
  ComponentSpec component_spec;
  TF_ASSERT_OK(AddFrozenGraphDefResource("/dev/null", &component_spec));
  component_spec.mutable_resource(0)->mutable_part(0)->set_record_format("bad");

  const Resource *resource = nullptr;
  EXPECT_THAT(LookupFrozenGraphDefResource(component_spec, &resource),
              test::IsErrorWithSubstr("wrong record format"));
}

TEST(XlaSpecUtilsTest, LookupFrozenGraphDefResourceWrongNumberOfParts) {
  ComponentSpec component_spec;
  TF_ASSERT_OK(AddFrozenGraphDefResource("/dev/null", &component_spec));
  component_spec.mutable_resource(0)->add_part();

  const Resource *resource = nullptr;
  EXPECT_THAT(LookupFrozenGraphDefResource(component_spec, &resource),
              test::IsErrorWithSubstr("expected 1 part"));
}

TEST(XlaSpecUtilsTest, LookupFrozenGraphDefResourceDuplicate) {
  ComponentSpec component_spec;
  TF_ASSERT_OK(AddFrozenGraphDefResource("/dev/null", &component_spec));
  component_spec.add_resource()->set_name(kFrozenGraphDefResourceName);

  const Resource *resource = nullptr;
  EXPECT_THAT(LookupFrozenGraphDefResource(component_spec, &resource),
              test::IsErrorWithSubstr(
                  "contains duplicate frozen TF GraphDef resource"));
}

TEST(XlaSpecUtilsTest, AddFrozenGraphDefResourceDuplicate) {
  ComponentSpec component_spec;
  TF_ASSERT_OK(AddFrozenGraphDefResource("/dev/null", &component_spec));

  EXPECT_THAT(AddFrozenGraphDefResource("another/graph", &component_spec),
              test::IsErrorWithSubstr(
                  "already contains a frozen TF GraphDef resource"));
}

TEST(XlaSpecUtilsTest, MakeXlaInputFixedFeatureIdName) {
  EXPECT_EQ(MakeXlaInputFixedFeatureIdName(0, 1),
            "INPUT__fixed_channel_0_index_1_ids");
  EXPECT_EQ(MakeXlaInputFixedFeatureIdName(1, 0),
            "INPUT__fixed_channel_1_index_0_ids");
}

TEST(XlaSpecUtilsTest, MakeXlaInputLinkedActivationVectorName) {
  EXPECT_EQ(MakeXlaInputLinkedActivationVectorName(0),
            "INPUT__linked_channel_0_activations");
  EXPECT_EQ(MakeXlaInputLinkedActivationVectorName(1),
            "INPUT__linked_channel_1_activations");
}

TEST(XlaSpecUtilsTest, MakeXlaInputLinkedOutOfBoundsIndicatorName) {
  EXPECT_EQ(MakeXlaInputLinkedOutOfBoundsIndicatorName(0),
            "INPUT__linked_channel_0_out_of_bounds");
  EXPECT_EQ(MakeXlaInputLinkedOutOfBoundsIndicatorName(1),
            "INPUT__linked_channel_1_out_of_bounds");
}

TEST(XlaSpecUtilsTest, MakeXlaInputRecurrentLayerName) {
  EXPECT_EQ(MakeXlaInputRecurrentLayerName("foo"), "INPUT__foo");
  EXPECT_EQ(MakeXlaInputRecurrentLayerName("bar_baz"), "INPUT__bar_baz");
}

TEST(XlaSpecUtilsTest, MakeXlaInputLayerName) {
  EXPECT_EQ(MakeXlaInputLayerName("foo"), "INPUT__foo");
  EXPECT_EQ(MakeXlaInputLayerName("bar_baz"), "INPUT__bar_baz");
}

TEST(XlaSpecUtilsTest, MakeXlaOutputLayerName) {
  EXPECT_EQ(MakeXlaOutputLayerName("foo"), "OUTPUT__foo");
  EXPECT_EQ(MakeXlaOutputLayerName("bar_baz"), "OUTPUT__bar_baz");
}

}  // namespace
}  // namespace runtime
}  // namespace dragnn
}  // namespace syntaxnet
