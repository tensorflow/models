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

#include "dragnn/runtime/trained_model.h"

#include <stddef.h>
#include <string>

#include "dragnn/core/test/generic.h"
#include "syntaxnet/base.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/platform/test.h"

namespace syntaxnet {
namespace dragnn {
namespace runtime {
namespace {

// Relative path to a saved model.
constexpr char kSavedModelDir[] = "dragnn/runtime/testdata/rnn_tagger";

// A valid tensor name in the test model and its dimensions.
constexpr char kTensorName[] = "tagger/weights_0/ExponentialMovingAverage";
constexpr size_t kTensorRows = 160;
constexpr size_t kTensorColumns = 64;

// Returns a valid saved model directory.
string GetSavedModelDir() {
  return tensorflow::io::JoinPath(test::GetTestDataPrefix(), kSavedModelDir);
}

// Tests that TrainedModel can initialize itself from a valid saved model,
// retrieve tensors and nodes, and close itself.  This is done in one test to
// avoid multiple (expensive) saved model loads.
TEST(TrainedModelTest, ResetQueryAndClose) {
  TrainedModel trained_model;
  TF_ASSERT_OK(trained_model.Reset(GetSavedModelDir()));

  // Look up a valid tensor.
  tensorflow::Tensor tensor;
  TF_ASSERT_OK(trained_model.EvaluateTensor(kTensorName, &tensor));
  ASSERT_EQ(tensor.dims(), 2);
  EXPECT_EQ(tensor.dim_size(0), kTensorRows);
  EXPECT_EQ(tensor.dim_size(1), kTensorColumns);

  // Look up an invalid tensor.
  EXPECT_FALSE(trained_model.EvaluateTensor("invalid", &tensor).ok());

  // Still have the old tensor contents.
  ASSERT_EQ(tensor.dims(), 2);
  EXPECT_EQ(tensor.dim_size(0), kTensorRows);
  EXPECT_EQ(tensor.dim_size(1), kTensorColumns);

  // Look up a valid node.  Note that the tensor name doubles as a node name.
  const tensorflow::NodeDef *node = nullptr;
  TF_ASSERT_OK(trained_model.LookupNode(kTensorName, &node));
  ASSERT_NE(node, nullptr);
  EXPECT_EQ(node->name(), kTensorName);

  // Look up an invalid node.
  ASSERT_THAT(trained_model.LookupNode("invalid", &node),
              test::IsErrorWithSubstr("Unknown node"));

  // Still have the old node.
  ASSERT_NE(node, nullptr);
  EXPECT_EQ(node->name(), kTensorName);

  // Get the current Graph.
  const tensorflow::GraphDef *graph_def = nullptr;
  TF_ASSERT_OK(trained_model.GraphDef(&graph_def));
  EXPECT_GT(graph_def->node_size(), 0);

  // First Close() is OK, second fails because already closed.
  TF_EXPECT_OK(trained_model.Close());
  EXPECT_THAT(trained_model.Close(),
              test::IsErrorWithSubstr("TF Session is not active"));
}

// Tests that TrainedModel::Reset() fails on an invalid path.
TEST(TrainedModelTest, InvalidPath) {
  TrainedModel trained_model;
  EXPECT_FALSE(trained_model.Reset("invalid/path").ok());
}

// Tests that TrainedModel::Close() fails if there is no model.
TEST(TrainedModelTest, CloseFailsBeforeReset) {
  TrainedModel trained_model;
  EXPECT_THAT(trained_model.Close(),
              test::IsErrorWithSubstr("TF Session is not active"));
}

// Tests that TrainedModel::GraphDef() fails if there is no active session.
TEST(TrainedModelTest, GraphDefFailsBeforeReset) {
  const tensorflow::GraphDef *graph_def = nullptr;
  TrainedModel trained_model;
  EXPECT_THAT(trained_model.GraphDef(&graph_def),
              test::IsErrorWithSubstr("TF Session is not active"));
}

// Tests that TrainedModel::EvaluateTensor() fails if there is no model.
TEST(TrainedModelTest, EvaluateTensorFailsBeforeReset) {
  TrainedModel trained_model;
  tensorflow::Tensor tensor;
  EXPECT_THAT(trained_model.EvaluateTensor("whatever", &tensor),
              test::IsErrorWithSubstr("TF Session is not active"));
}

// Tests that TrainedModel::LookupNode() fails if there is no model.
TEST(TrainedModelTest, LookupNodeFailsBeforeReset) {
  TrainedModel trained_model;
  const tensorflow::NodeDef *node = nullptr;
  EXPECT_THAT(trained_model.LookupNode("whatever", &node),
              test::IsErrorWithSubstr("TF Session is not active"));
}

}  // namespace
}  // namespace runtime
}  // namespace dragnn
}  // namespace syntaxnet
