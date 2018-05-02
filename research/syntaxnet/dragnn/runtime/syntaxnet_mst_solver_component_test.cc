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

#include <memory>
#include <string>
#include <vector>

#include "dragnn/core/compute_session.h"
#include "dragnn/core/input_batch_cache.h"
#include "dragnn/core/test/generic.h"
#include "dragnn/io/sentence_input_batch.h"
#include "dragnn/io/syntaxnet_sentence.h"
#include "dragnn/protos/spec.pb.h"
#include "dragnn/runtime/component.h"
#include "dragnn/runtime/math/types.h"
#include "dragnn/runtime/test/network_test_base.h"
#include "dragnn/runtime/variable_store.h"
#include "syntaxnet/sentence.pb.h"
#include <gmock/gmock.h>
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"

namespace syntaxnet {
namespace dragnn {
namespace runtime {
namespace {

using ::testing::Return;

constexpr char kPreviousComponentName[] = "previous_component";
constexpr char kAdjacencyLayerName[] = "adjacency_layer";

// Returns a ComponentSpec that works with the head selection component.
ComponentSpec MakeGoodSpec() {
  ComponentSpec component_spec;
  component_spec.mutable_component_builder()->set_registered_name(
      "SyntaxNetMstSolverComponent");
  component_spec.mutable_backend()->set_registered_name("SyntaxNetComponent");
  component_spec.mutable_transition_system()->set_registered_name("heads");
  component_spec.mutable_network_unit()->set_registered_name(
      "some.path.to.MstSolverNetwork");
  LinkedFeatureChannel *link = component_spec.add_linked_feature();
  link->set_source_component(kPreviousComponentName);
  link->set_source_layer(kAdjacencyLayerName);
  return component_spec;
}

// Returns a sentence containing |num_tokens| tokens.  All heads are set to
// self-loops, which are normally invalid, to ensure that the head selector
// touches all tokens.
Sentence MakeSentence(int num_tokens) {
  Sentence sentence;
  for (int i = 0; i < num_tokens; ++i) {
    Token *token = sentence.add_token();
    token->set_start(0);  // never used; set because required field
    token->set_end(0);  // never used; set because required field
    token->set_word("foo");  // never used; set because required field
    token->set_head(i);
  }
  return sentence;
}

class SyntaxNetMstSolverComponentTest : public NetworkTestBase {
 protected:
  // Initializes a parser head selection component from the |component_spec|,
  // feeds it the |adjacency| matrix, and applies the resulting heads to the
  // |sentence|.  Returs non-OK on error.
  tensorflow::Status Run(const ComponentSpec &component_spec,
                         const std::vector<std::vector<float>> &adjacency,
                         Sentence *sentence) {
    AddComponent(kPreviousComponentName);
    AddPairwiseLayer(kAdjacencyLayerName, 1);

    std::unique_ptr<Component> component;
    TF_RETURN_IF_ERROR(Component::CreateOrError(
        "SyntaxNetMstSolverComponent", &component));

    TF_RETURN_IF_ERROR(component->Initialize(component_spec, &variable_store_,
                                             &network_state_manager_,
                                             &extension_manager_));

    network_states_.Reset(&network_state_manager_);
    const int num_steps = adjacency.size();
    StartComponent(num_steps);

    MutableMatrix<float> adjacency_layer =
        GetPairwiseLayer(kPreviousComponentName, kAdjacencyLayerName);
    for (size_t target = 0; target < num_steps; ++target) {
      for (size_t source = 0; source < num_steps; ++source) {
        adjacency_layer.row(target)[source] = adjacency[target][source];
      }
    }

    string data;
    CHECK(sentence->SerializeToString(&data));
    InputBatchCache input(data);
    EXPECT_CALL(compute_session_, GetInputBatchCache())
        .WillRepeatedly(Return(&input));

    session_state_.extensions.Reset(&extension_manager_);
    TF_RETURN_IF_ERROR(
        component->Evaluate(&session_state_, &compute_session_, nullptr));

    CHECK(sentence->ParseFromString(input.SerializedData()[0]));
    return tensorflow::Status::OK();
  }
};

// Tests the head selector on a single-token input.
TEST_F(SyntaxNetMstSolverComponentTest, ParseOneToken) {
  const std::vector<std::vector<float>> adjacency = {{0.0}};

  Sentence sentence = MakeSentence(1);
  TF_ASSERT_OK(Run(MakeGoodSpec(), adjacency, &sentence));

  EXPECT_FALSE(sentence.token(0).has_head());
}

// Tests the head selector on a two-token input.
TEST_F(SyntaxNetMstSolverComponentTest, ParseTwoTokens) {
  const std::vector<std::vector<float>> adjacency = {{0.0, 1.0},  //
                                                     {0.9, 1.0}};

  Sentence sentence = MakeSentence(2);
  TF_ASSERT_OK(Run(MakeGoodSpec(), adjacency, &sentence));

  EXPECT_EQ(sentence.token(0).head(), 1);
  EXPECT_EQ(sentence.token(1).head(), -1);
}

// Tests the head selector on a three-token input.
TEST_F(SyntaxNetMstSolverComponentTest, ParseThreeTokens) {
  // This adjacency matrix forms a left-headed chain.
  const std::vector<std::vector<float>> adjacency = {{1.0, 0.0, 0.0},  //
                                                     {1.0, 0.0, 0.0},  //
                                                     {0.0, 1.0, 0.0}};

  Sentence sentence = MakeSentence(3);
  TF_ASSERT_OK(Run(MakeGoodSpec(), adjacency, &sentence));

  EXPECT_FALSE(sentence.token(0).has_head());
  EXPECT_EQ(sentence.token(1).head(), 0);
  EXPECT_EQ(sentence.token(2).head(), 1);
}

// Tests the head selector on a four-token input.
TEST_F(SyntaxNetMstSolverComponentTest, ParseFourTokens) {
  // This adjacency matrix forms a right-headed chain.
  const std::vector<std::vector<float>> adjacency = {{0.0, 1.0, 0.0, 0.0},  //
                                                     {0.0, 0.0, 1.0, 0.0},  //
                                                     {0.0, 0.0, 0.0, 1.0},  //
                                                     {0.0, 0.0, 0.0, 1.0}};

  Sentence sentence = MakeSentence(4);
  TF_ASSERT_OK(Run(MakeGoodSpec(), adjacency, &sentence));

  EXPECT_EQ(sentence.token(0).head(), 1);
  EXPECT_EQ(sentence.token(1).head(), 2);
  EXPECT_EQ(sentence.token(2).head(), 3);
  EXPECT_FALSE(sentence.token(3).has_head());
}

// Tests that the component supports the good spec.
TEST_F(SyntaxNetMstSolverComponentTest, Supported) {
  const ComponentSpec component_spec = MakeGoodSpec();

  string name;
  TF_ASSERT_OK(Component::Select(component_spec, &name));
  EXPECT_EQ(name, "SyntaxNetMstSolverComponent");
}

// Tests that the component requires the proper backend.
TEST_F(SyntaxNetMstSolverComponentTest, WrongComponentBuilder) {
  ComponentSpec component_spec = MakeGoodSpec();
  component_spec.mutable_component_builder()->set_registered_name("bad");

  string name;
  EXPECT_THAT(
      Component::Select(component_spec, &name),
      test::IsErrorWithSubstr("Could not find a best spec for component"));
}

// Tests that the component requires the proper backend.
TEST_F(SyntaxNetMstSolverComponentTest, WrongBackend) {
  ComponentSpec component_spec = MakeGoodSpec();
  component_spec.mutable_backend()->set_registered_name("bad");

  string name;
  EXPECT_THAT(
      Component::Select(component_spec, &name),
      test::IsErrorWithSubstr("Could not find a best spec for component"));
}

// Tests that Evaluate() fails if the batch is null.
TEST_F(SyntaxNetMstSolverComponentTest, NullBatch) {
  std::unique_ptr<Component> component;
  TF_ASSERT_OK(
      Component::CreateOrError("SyntaxNetMstSolverComponent", &component));

  EXPECT_CALL(compute_session_, GetInputBatchCache())
      .WillRepeatedly(Return(nullptr));

  EXPECT_THAT(component->Evaluate(&session_state_, &compute_session_, nullptr),
              test::IsErrorWithSubstr("Null input batch"));
}

// Tests that Evaluate() fails if the batch is the wrong size.
TEST_F(SyntaxNetMstSolverComponentTest, WrongBatchSize) {
  std::unique_ptr<Component> component;
  TF_ASSERT_OK(
      Component::CreateOrError("SyntaxNetMstSolverComponent", &component));

  InputBatchCache input({MakeSentence(1).SerializeAsString(),
                         MakeSentence(2).SerializeAsString(),
                         MakeSentence(3).SerializeAsString(),
                         MakeSentence(4).SerializeAsString()});
  EXPECT_CALL(compute_session_, GetInputBatchCache())
      .WillRepeatedly(Return(&input));

  EXPECT_THAT(component->Evaluate(&session_state_, &compute_session_, nullptr),
              test::IsErrorWithSubstr("Non-singleton batch: got 4 elements"));
}

// Tests that Evaluate() fails if the adjacency matrix and sentence disagree on
// the number of tokens.
TEST_F(SyntaxNetMstSolverComponentTest, WrongNumTokens) {
  const std::vector<std::vector<float>> adjacency = {{1.0, 0.0, 0.0, 0.0},  //
                                                     {0.0, 1.0, 0.0, 0.0},  //
                                                     {0.0, 0.0, 1.0, 0.0},  //
                                                     {0.0, 0.0, 0.0, 1.0}};

  // 4-token adjacency matrix with 3-token sentence.
  Sentence sentence = MakeSentence(3);
  EXPECT_THAT(Run(MakeGoodSpec(), adjacency, &sentence),
              test::IsErrorWithSubstr(
                  "Sentence size mismatch: expected 4 tokens but got 3"));
}

}  // namespace
}  // namespace runtime
}  // namespace dragnn
}  // namespace syntaxnet
