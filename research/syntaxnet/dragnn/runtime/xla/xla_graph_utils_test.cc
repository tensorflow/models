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

#include "dragnn/runtime/xla/xla_graph_utils.h"

#include <set>
#include <string>

#include "dragnn/core/test/generic.h"
#include "dragnn/protos/export.pb.h"
#include "syntaxnet/base.h"
#include "tensorflow/compiler/tf2xla/tf2xla.pb.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/tensor_shape.pb.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/test.h"

namespace syntaxnet {
namespace dragnn {
namespace runtime {
namespace {

class XlaGraphUtilsTest : public ::testing::Test {
 protected:
  tensorflow::TensorProto CreateStringTensor(const string &s) {
    tensorflow::Tensor tensor(tensorflow::DT_STRING,
                              tensorflow::TensorShape({1}));
    tensor.vec<string>()(0) = s;
    tensorflow::TensorProto proto;
    tensor.AsProtoField(&proto);
    return proto;
  }

  void AddSimpleSpec(const string &output_name, CellSubgraphSpec *spec) {
    auto *input1 = spec->add_input();
    input1->set_name("id");
    input1->set_tensor("cell/id:0");
    input1->set_type(CellSubgraphSpec::Input::TYPE_FEATURE);

    auto *input2 = spec->add_input();
    input2->set_name("val");
    input2->set_tensor("cell/val:0");
    input2->set_type(CellSubgraphSpec::Input::TYPE_RECURRENT);

    auto *output1 = spec->add_output();
    output1->set_name(output_name);
    output1->set_tensor("cell/also_val:0");
  }

  void AddSimpleConfig(const string &output_name,
                       tensorflow::tf2xla::Config *xla_config) {
    auto *feed1 = xla_config->add_feed();
    feed1->mutable_id()->set_node_name("cell/id");
    feed1->mutable_shape()->add_dim()->set_size(1);
    feed1->set_name("INPUT__id");

    auto *feed2 = xla_config->add_feed();
    feed2->mutable_id()->set_node_name("cell/val");
    auto *feed2_shape = feed2->mutable_shape();
    feed2_shape->add_dim()->set_size(16);
    feed2_shape->add_dim()->set_size(1);
    feed2->set_name("INPUT__val");

    auto *fetch1 = xla_config->add_fetch();
    fetch1->mutable_id()->set_node_name("cell/also_val");
    fetch1->set_name(output_name);
  }

  tensorflow::Status AddCellSubgraphSpecNode(const string &serialized_spec,
                                             tensorflow::GraphDef *graph) {
    return tensorflow::NodeDefBuilder(kFrozenCellSubgraphSpecNodeName, "Const")
        .Attr("dtype", tensorflow::DT_STRING)
        .Attr("value", CreateStringTensor(serialized_spec))
        .Attr("shape", tensorflow::TensorShape({1}))
        .Finalize(graph->add_node());
  }

  tensorflow::Status AddCellSubgraphSpecNode(const CellSubgraphSpec &spec,
                                             tensorflow::GraphDef *graph) {
    string serialized_spec;
    if (!spec.SerializeToString(&serialized_spec)) {
      return tensorflow::errors::InvalidArgument("Invalid CellSubgraphSpec: ",
                                                 spec.DebugString());
    }
    return AddCellSubgraphSpecNode(serialized_spec, graph);
  }

  tensorflow::Status AddSimpleGraph(tensorflow::GraphDef *graph) {
    TF_RETURN_IF_ERROR(tensorflow::NodeDefBuilder("cell/id", "Placeholder")
                           .Attr("dtype", tensorflow::DT_INT32)
                           .Attr("shape", tensorflow::TensorShape({1}))
                           .Finalize(graph->add_node()));
    TF_RETURN_IF_ERROR(tensorflow::NodeDefBuilder("cell/val", "Placeholder")
                           .Attr("dtype", tensorflow::DT_FLOAT)
                           .Attr("shape", tensorflow::TensorShape({16, 1}))
                           .Finalize(graph->add_node()));

    TF_RETURN_IF_ERROR(tensorflow::NodeDefBuilder("cell/also_val", "Identity")
                           .Input("val", 0, tensorflow::DT_FLOAT)
                           .Attr("dtype", tensorflow::DT_FLOAT)
                           .Finalize(graph->add_node()));

    return tensorflow::Status::OK();
  }
};

TEST_F(XlaGraphUtilsTest, LoadFrozenGraphDefInvalidPath) {
  tensorflow::GraphDef graph;
  EXPECT_EQ(tensorflow::error::NOT_FOUND,
            LoadFrozenGraphDef("invalid/path", &graph).code());
}

TEST_F(XlaGraphUtilsTest, LoadFrozenGraphDefInvalidProto) {
  const string path =
      tensorflow::io::JoinPath(tensorflow::testing::TmpDir(), "bad-graph");
  TF_ASSERT_OK(WriteStringToFile(tensorflow::Env::Default(), path, "junk"));

  // The file is found but there is still an error.
  tensorflow::GraphDef graph;
  tensorflow::Status status = LoadFrozenGraphDef(path, &graph);
  EXPECT_FALSE(status.ok());
  EXPECT_NE(tensorflow::error::NOT_FOUND, status.code());
}

TEST_F(XlaGraphUtilsTest, LoadFrozenGraphDefValidFile) {
  tensorflow::GraphDef graph;
  TF_ASSERT_OK(AddSimpleGraph(&graph));
  const string path =
      tensorflow::io::JoinPath(tensorflow::testing::TmpDir(), "graph-frozen");
  TF_ASSERT_OK(SaveFrozenGraphDef(path, graph));

  tensorflow::GraphDef loaded_graph;
  TF_ASSERT_OK(LoadFrozenGraphDef(path, &loaded_graph));
  EXPECT_THAT(loaded_graph, test::EqualsProto(graph));
}

TEST_F(XlaGraphUtilsTest, ParseTensorName_Valid) {
  string name;
  uint32 index;
  TF_ASSERT_OK(ParseTensorName("value:0", &name, &index));
  EXPECT_EQ("value", name);
  EXPECT_EQ(0, index);

  TF_ASSERT_OK(ParseTensorName("some/value:3", &name, &index));
  EXPECT_EQ("some/value", name);
  EXPECT_EQ(3, index);

  TF_ASSERT_OK(ParseTensorName("value", &name, &index));
  EXPECT_EQ("value", name);
  EXPECT_EQ(0, index);
}

TEST_F(XlaGraphUtilsTest, ParseTensorName_Invalid) {
  string name;
  uint32 index = -1;
  EXPECT_THAT(
      ParseTensorName("value:zero", &name, &index),
      test::IsErrorWithCodeAndSubstr(tensorflow::error::INVALID_ARGUMENT,
                                     "Malformed tensor name"));
  EXPECT_EQ("", name);
  EXPECT_EQ(-1, index);

  EXPECT_THAT(
      ParseTensorName("^value", &name, &index),
      test::IsErrorWithCodeAndSubstr(tensorflow::error::INVALID_ARGUMENT,
                                     "Cannot parse name of control input"));
  EXPECT_EQ("", name);
  EXPECT_EQ(-1, index);

  EXPECT_THAT(
      ParseTensorName("^value:0", &name, &index),
      test::IsErrorWithCodeAndSubstr(tensorflow::error::INVALID_ARGUMENT,
                                     "Cannot parse name of control input"));
  EXPECT_EQ("", name);
  EXPECT_EQ(-1, index);
}

TEST_F(XlaGraphUtilsTest, GetSpecAndMakeXlaConfig_NoSpecNodeFails) {
  tensorflow::GraphDef graph;
  TF_ASSERT_OK(AddSimpleGraph(&graph));

  CellSubgraphSpec cell_subgraph_spec;
  tensorflow::tf2xla::Config xla_config;
  EXPECT_THAT(
      GetSpecAndMakeXlaConfig(graph, &cell_subgraph_spec, &xla_config),
      test::IsErrorWithCodeAndSubstr(tensorflow::error::NOT_FOUND,
                                     "Cannot find node CellSubgraphSpec"));
}

TEST_F(XlaGraphUtilsTest, GetSpecAndMakeXlaConfig_SpecNodeMissingValueFails) {
  tensorflow::GraphDef graph;
  TF_ASSERT_OK(
      tensorflow::NodeDefBuilder(kFrozenCellSubgraphSpecNodeName, "Const")
          .Attr("dtype", tensorflow::DT_STRING)
          .Attr("shape", tensorflow::TensorShape({1}))
          .Finalize(graph.add_node()));
  TF_ASSERT_OK(AddSimpleGraph(&graph));

  CellSubgraphSpec cell_subgraph_spec;
  tensorflow::tf2xla::Config xla_config;
  EXPECT_THAT(
      GetSpecAndMakeXlaConfig(graph, &cell_subgraph_spec, &xla_config),
      test::IsErrorWithCodeAndSubstr(tensorflow::error::NOT_FOUND,
                                     "Cannot find CellSubgraphSpec value"));
}

TEST_F(XlaGraphUtilsTest, GetSpecAndMakeXlaConfig_UnparseableSpecFails) {
  tensorflow::GraphDef graph;
  TF_ASSERT_OK(AddCellSubgraphSpecNode("junk", &graph));
  TF_ASSERT_OK(AddSimpleGraph(&graph));

  CellSubgraphSpec cell_subgraph_spec;
  tensorflow::tf2xla::Config xla_config;
  EXPECT_THAT(
      GetSpecAndMakeXlaConfig(graph, &cell_subgraph_spec, &xla_config),
      test::IsErrorWithCodeAndSubstr(tensorflow::error::INVALID_ARGUMENT,
                                     "Failed to parse CellSubgraphSpec"));
}

TEST_F(XlaGraphUtilsTest, GetSpecAndMakeXlaConfig_MissingGraphInputNodeFails) {
  CellSubgraphSpec spec_in_graph;
  auto *input = spec_in_graph.add_input();
  input->set_name("id");
  input->set_tensor("cell/id:0");
  input->set_type(CellSubgraphSpec::Input::TYPE_FEATURE);

  tensorflow::GraphDef graph;
  TF_ASSERT_OK(AddCellSubgraphSpecNode(spec_in_graph, &graph));

  CellSubgraphSpec cell_subgraph_spec;
  tensorflow::tf2xla::Config xla_config;
  EXPECT_THAT(GetSpecAndMakeXlaConfig(graph, &cell_subgraph_spec, &xla_config),
              test::IsErrorWithCodeAndSubstr(tensorflow::error::NOT_FOUND,
                                             "Cannot find node cell/id"));
}

TEST_F(XlaGraphUtilsTest, GetSpecAndMakeXlaConfig_InvalidTensorNameFails) {
  CellSubgraphSpec spec_in_graph;
  auto *input = spec_in_graph.add_input();
  input->set_name("id");
  input->set_tensor("cell/id:zero");
  input->set_type(CellSubgraphSpec::Input::TYPE_FEATURE);

  tensorflow::GraphDef graph;
  TF_ASSERT_OK(AddCellSubgraphSpecNode(spec_in_graph, &graph));

  CellSubgraphSpec cell_subgraph_spec;
  tensorflow::tf2xla::Config xla_config;
  EXPECT_THAT(
      GetSpecAndMakeXlaConfig(graph, &cell_subgraph_spec, &xla_config),
      test::IsErrorWithCodeAndSubstr(tensorflow::error::INVALID_ARGUMENT,
                                     "Malformed tensor name cell/id:zero"));
}

TEST_F(XlaGraphUtilsTest, GetSpecAndMakeXlaConfig_NonPlaceholderInputFails) {
  CellSubgraphSpec spec_in_graph;
  auto *input = spec_in_graph.add_input();
  input->set_name("id");
  input->set_tensor("cell/id:0");
  input->set_type(CellSubgraphSpec::Input::TYPE_FEATURE);

  tensorflow::GraphDef graph;
  TF_ASSERT_OK(AddCellSubgraphSpecNode(spec_in_graph, &graph));
  TF_ASSERT_OK(tensorflow::NodeDefBuilder("cell/id", "Const")
                   .Attr("dtype", tensorflow::DT_INT32)
                   .Attr("shape", tensorflow::TensorShape({1}))
                   .Finalize(graph.add_node()));

  CellSubgraphSpec cell_subgraph_spec;
  tensorflow::tf2xla::Config xla_config;
  EXPECT_THAT(GetSpecAndMakeXlaConfig(graph, &cell_subgraph_spec, &xla_config),
              test::IsErrorWithCodeAndSubstr(
                  tensorflow::error::INVALID_ARGUMENT,
                  "Input node 'cell/id' is not a Placeholder"));
}

TEST_F(XlaGraphUtilsTest, GetSpecAndMakeXlaConfig_Valid) {
  CellSubgraphSpec spec_in_graph;
  AddSimpleSpec("val", &spec_in_graph);

  tensorflow::GraphDef graph;
  TF_ASSERT_OK(AddCellSubgraphSpecNode(spec_in_graph, &graph));
  TF_ASSERT_OK(AddSimpleGraph(&graph));

  CellSubgraphSpec cell_subgraph_spec;
  tensorflow::tf2xla::Config xla_config;
  TF_ASSERT_OK(
      GetSpecAndMakeXlaConfig(graph, &cell_subgraph_spec, &xla_config));

  EXPECT_THAT(cell_subgraph_spec, test::EqualsProto(spec_in_graph));

  tensorflow::tf2xla::Config expected_xla_config;
  AddSimpleConfig("OUTPUT__val", &expected_xla_config);
  EXPECT_THAT(xla_config, test::EqualsProto(expected_xla_config));
}

TEST_F(XlaGraphUtilsTest, GetSpecAndMakeXlaConfig_WithAlias) {
  CellSubgraphSpec spec_in_graph;
  AddSimpleSpec("val", &spec_in_graph);

  // Adding this alias doesn't change the output Config.
  auto *extra_output = spec_in_graph.add_output();
  extra_output->set_name("val_two");
  extra_output->set_tensor("cell/also_val:0");

  tensorflow::GraphDef graph;
  TF_ASSERT_OK(AddCellSubgraphSpecNode(spec_in_graph, &graph));
  TF_ASSERT_OK(AddSimpleGraph(&graph));

  CellSubgraphSpec cell_subgraph_spec;
  tensorflow::tf2xla::Config xla_config;
  TF_ASSERT_OK(
      GetSpecAndMakeXlaConfig(graph, &cell_subgraph_spec, &xla_config));

  EXPECT_THAT(cell_subgraph_spec, test::EqualsProto(spec_in_graph));

  tensorflow::tf2xla::Config expected_xla_config;
  AddSimpleConfig("OUTPUT__val", &expected_xla_config);
  EXPECT_THAT(xla_config, test::EqualsProto(expected_xla_config));
}

TEST_F(XlaGraphUtilsTest, GetSpecAndMakeXlaConfig_OutputWithAliasTakesFirst) {
  CellSubgraphSpec spec_in_graph;
  AddSimpleSpec("val_two", &spec_in_graph);

  // This is the same as GetSpecAndMakeXlaConfig_WithAlias except that the
  // output and its alias names are switched. The Config below will contain
  // the first one specified.
  auto *extra_output = spec_in_graph.add_output();
  extra_output->set_name("val");
  extra_output->set_tensor("cell/also_val:0");

  tensorflow::GraphDef graph;
  TF_ASSERT_OK(AddCellSubgraphSpecNode(spec_in_graph, &graph));
  TF_ASSERT_OK(AddSimpleGraph(&graph));

  CellSubgraphSpec cell_subgraph_spec;
  tensorflow::tf2xla::Config xla_config;
  TF_ASSERT_OK(
      GetSpecAndMakeXlaConfig(graph, &cell_subgraph_spec, &xla_config));

  EXPECT_THAT(cell_subgraph_spec, test::EqualsProto(spec_in_graph));

  tensorflow::tf2xla::Config expected_xla_config;
  AddSimpleConfig("OUTPUT__val_two", &expected_xla_config);
  EXPECT_THAT(xla_config, test::EqualsProto(expected_xla_config));
}

}  // namespace
}  // namespace runtime
}  // namespace dragnn
}  // namespace syntaxnet
