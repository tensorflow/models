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

#include "dragnn/runtime/network_unit_base.h"

#include <stddef.h>
#include <memory>
#include <string>

#include "dragnn/core/test/generic.h"
#include "dragnn/protos/spec.pb.h"
#include "dragnn/runtime/extensions.h"
#include "dragnn/runtime/fixed_embeddings.h"
#include "dragnn/runtime/linked_embeddings.h"
#include "dragnn/runtime/math/types.h"
#include "dragnn/runtime/network_states.h"
#include "dragnn/runtime/session_state.h"
#include "dragnn/runtime/test/network_test_base.h"
#include "dragnn/runtime/variable_store.h"
#include "syntaxnet/base.h"
#include <gmock/gmock.h>
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"

namespace syntaxnet {
namespace dragnn {
namespace runtime {
namespace {

using ::testing::_;
using ::testing::Invoke;
using ::testing::Return;

// Dimensions of the layers in the network.
static constexpr size_t kPreviousDim = 77;
static constexpr size_t kRecurrentDim = 123;

// Contents of the layers in the network.
static constexpr float kPreviousValue = -2.75;
static constexpr float kRecurrentValue = 6.25;

// Number of steps taken in each component.
static constexpr size_t kNumSteps = 10;

// A trivial network unit that exposes the concatenated inputs.  Note that
// NetworkUnitBase does not override the interface methods, so we need a
// concrete subclass for testing.
class FooNetwork : public NetworkUnitBase {
 public:
  void RequestConcatenation() { request_concatenation_ = true; }
  void ProvideConcatenatedInput() { provide_concatenated_input_ = true; }
  Vector<float> concatenated_input() const { return concatenated_input_; }

  // Implements NetworkUnit.
  tensorflow::Status Initialize(const ComponentSpec &component_spec,
                                VariableStore *variable_store,
                                NetworkStateManager *network_state_manager,
                                ExtensionManager *extension_manager) override {
    TF_RETURN_IF_ERROR(network_state_manager->AddLayer(
        "recurrent_layer", kRecurrentDim, &recurrent_handle_));
    return InitializeBase(request_concatenation_, component_spec,
                          variable_store, network_state_manager,
                          extension_manager);
  }
  string GetLogitsName() const override { return ""; }
  tensorflow::Status Evaluate(size_t unused_step_index,
                              SessionState *session_state,
                              ComputeSession *compute_session) const override {
    return EvaluateBase(
        session_state, compute_session,
        provide_concatenated_input_ ? &concatenated_input_ : nullptr);
  }

 private:
  bool request_concatenation_ = false;
  bool provide_concatenated_input_ = false;
  LayerHandle<float> recurrent_handle_;
  mutable Vector<float> concatenated_input_;  // Evaluate() sets this
};

class NetworkUnitBaseTest : public NetworkTestBase {
 protected:
  // Initializes the |network_unit_| based on the |component_spec_text| and
  // evaluates it.  On error, returns non-OK.
  tensorflow::Status Run(const string &component_spec_text) {
    ComponentSpec component_spec;
    CHECK(TextFormat::ParseFromString(component_spec_text, &component_spec));
    component_spec.set_name(kTestComponentName);

    AddComponent("previous_component");
    AddLayer("previous_layer", kPreviousDim);
    AddComponent(kTestComponentName);

    TF_RETURN_IF_ERROR(
        network_unit_.Initialize(component_spec, &variable_store_,
                                 &network_state_manager_, &extension_manager_));

    // Create and populate the network states.
    network_states_.Reset(&network_state_manager_);
    StartComponent(kNumSteps);
    StartComponent(kNumSteps);
    FillLayer("previous_component", "previous_layer", kPreviousValue);
    FillLayer(kTestComponentName, "recurrent_layer", kRecurrentValue);
    session_state_.extensions.Reset(&extension_manager_);

    // Neither FooNetwork nor NetworkUnitBase look at the step index, so use an
    // arbitrary value.
    return network_unit_.Evaluate(0, &session_state_, &compute_session_);
  }

  FooNetwork network_unit_;
  std::vector<std::vector<float>> concatenated_inputs_;
};

// Tests that NetworkUnitBase produces an empty vector when concatenating and
// there are no input embeddings.
TEST_F(NetworkUnitBaseTest, ConcatenateNoInputs) {
  network_unit_.RequestConcatenation();
  network_unit_.ProvideConcatenatedInput();
  TF_ASSERT_OK(Run(""));

  EXPECT_EQ(network_unit_.fixed_embedding_manager().num_channels(), 0);
  EXPECT_EQ(network_unit_.linked_embedding_manager().num_channels(), 0);
  EXPECT_EQ(network_unit_.num_actions(), 0);
  EXPECT_EQ(network_unit_.concatenated_input_dim(), 0);

  EXPECT_TRUE(network_unit_.concatenated_input().empty());
}

// Tests that NetworkUnitBase produces a copy of the single input embedding when
// concatenating a single fixed channel.
TEST_F(NetworkUnitBaseTest, ConcatenateOneFixedChannel) {
  const float kEmbedding = 1.5;
  const float kFeature = 0.5;
  const size_t kDim = 13;
  const string kSpec = R"(num_actions: 42
                          fixed_feature {
                            vocabulary_size: 11
                            embedding_dim: 13
                            size: 1
                          })";
  AddFixedEmbeddingMatrix(0, 11, kDim, kEmbedding);

  EXPECT_CALL(compute_session_, GetInputFeatures(_, _, _, _, _))
      .WillOnce(Invoke(ExtractFeatures(0, {{1, kFeature}})));
  const float kValue = kEmbedding * kFeature;

  network_unit_.RequestConcatenation();
  network_unit_.ProvideConcatenatedInput();
  TF_ASSERT_OK(Run(kSpec));

  EXPECT_EQ(network_unit_.fixed_embedding_manager().num_channels(), 1);
  EXPECT_EQ(network_unit_.linked_embedding_manager().num_channels(), 0);
  EXPECT_EQ(network_unit_.num_actions(), 42);
  EXPECT_EQ(network_unit_.concatenated_input_dim(), kDim);

  ExpectVector(network_unit_.concatenated_input(),
               network_unit_.concatenated_input_dim(), kValue);
}

// Tests that NetworkUnitBase does not concatenate if concatenation is requested
// and the concatenated input vector is not provided.
TEST_F(NetworkUnitBaseTest, ConcatenatedInputVectorNotProvided) {
  const float kEmbedding = 1.5;
  const float kFeature = 0.5;
  const size_t kDim = 13;
  const string kSpec = R"(num_actions: 37
                          fixed_feature {
                            vocabulary_size: 11
                            embedding_dim: 13
                            size: 1
                          })";
  AddFixedEmbeddingMatrix(0, 11, kDim, kEmbedding);

  EXPECT_CALL(compute_session_, GetInputFeatures(_, _, _, _, _))
      .WillOnce(Invoke(ExtractFeatures(0, {{1, kFeature}})));

  network_unit_.RequestConcatenation();
  TF_ASSERT_OK(Run(kSpec));

  // Embedding managers and other config is set up properly.
  EXPECT_EQ(network_unit_.fixed_embedding_manager().num_channels(), 1);
  EXPECT_EQ(network_unit_.linked_embedding_manager().num_channels(), 0);
  EXPECT_EQ(network_unit_.num_actions(), 37);
  EXPECT_EQ(network_unit_.concatenated_input_dim(), kDim);

  // But the concatenation was not performed.
  EXPECT_TRUE(network_unit_.concatenated_input().empty());
}

// As above, but with the converse condition: does not request concatenation,
// but does provide the concatenated input vector.
TEST_F(NetworkUnitBaseTest, ConcatenationNotRequested) {
  const float kEmbedding = 1.5;
  const float kFeature = 0.5;
  const size_t kDim = 13;
  const string kSpec = R"(num_actions: 31
                          fixed_feature {
                            vocabulary_size: 11
                            embedding_dim: 13
                            size: 1
                          })";
  AddFixedEmbeddingMatrix(0, 11, kDim, kEmbedding);

  EXPECT_CALL(compute_session_, GetInputFeatures(_, _, _, _, _))
      .WillOnce(Invoke(ExtractFeatures(0, {{1, kFeature}})));

  network_unit_.ProvideConcatenatedInput();
  TF_ASSERT_OK(Run(kSpec));

  // Embedding managers and other config is set up properly.
  EXPECT_EQ(network_unit_.fixed_embedding_manager().num_channels(), 1);
  EXPECT_EQ(network_unit_.linked_embedding_manager().num_channels(), 0);
  EXPECT_EQ(network_unit_.num_actions(), 31);
  EXPECT_EQ(network_unit_.concatenated_input_dim(), kDim);

  // But the concatenation was not performed.
  EXPECT_TRUE(network_unit_.concatenated_input().empty());
}

// Tests that NetworkUnitBase produces a copy of the single input embedding when
// concatenating a single linked channel.
TEST_F(NetworkUnitBaseTest, ConcatenateOneLinkedChannel) {
  const string kSpec = R"(num_actions: 37
                         linked_feature {
                           embedding_dim: -1
                           source_component: 'previous_component'
                           source_layer: 'previous_layer'
                           size: 1
                         })";

  EXPECT_CALL(compute_session_, GetTranslatedLinkFeatures(_, _))
      .WillOnce(Invoke(ExtractLinks(0, {"step_idx: 5"})));
  EXPECT_CALL(compute_session_, SourceComponentBeamSize(_, _))
      .WillRepeatedly(Return(1));

  network_unit_.RequestConcatenation();
  network_unit_.ProvideConcatenatedInput();
  TF_ASSERT_OK(Run(kSpec));

  EXPECT_EQ(network_unit_.fixed_embedding_manager().num_channels(), 0);
  EXPECT_EQ(network_unit_.linked_embedding_manager().num_channels(), 1);
  EXPECT_EQ(network_unit_.num_actions(), 37);
  EXPECT_EQ(network_unit_.concatenated_input_dim(), kPreviousDim);

  ExpectVector(network_unit_.concatenated_input(),
               network_unit_.concatenated_input_dim(), kPreviousValue);
}

// Tests that NetworkUnitBase concatenates a fixed and linked channel in that
// order.
TEST_F(NetworkUnitBaseTest, ConcatenateOneChannelOfEachType) {
  const float kEmbedding = 1.25;
  const float kFeature = 0.75;
  const size_t kFixedDim = 13;
  const string kSpec = R"(num_actions: 77
                          fixed_feature {
                            vocabulary_size: 11
                            embedding_dim: 13
                            size: 1
                          }
                          linked_feature {
                            embedding_dim: -1
                            source_component: 'previous_component'
                            source_layer: 'previous_layer'
                            size: 1
                          })";
  AddFixedEmbeddingMatrix(0, 11, kFixedDim, kEmbedding);

  EXPECT_CALL(compute_session_, GetInputFeatures(_, _, _, _, _))
      .WillOnce(Invoke(ExtractFeatures(0, {{1, kFeature}})));
  const float kFixedValue = kEmbedding * kFeature;

  EXPECT_CALL(compute_session_, GetTranslatedLinkFeatures(_, _))
      .WillOnce(Invoke(ExtractLinks(0, {"step_idx: 5"})));
  EXPECT_CALL(compute_session_, SourceComponentBeamSize(_, _))
      .WillRepeatedly(Return(1));

  network_unit_.RequestConcatenation();
  network_unit_.ProvideConcatenatedInput();
  TF_ASSERT_OK(Run(kSpec));

  EXPECT_EQ(network_unit_.fixed_embedding_manager().num_channels(), 1);
  EXPECT_EQ(network_unit_.linked_embedding_manager().num_channels(), 1);
  EXPECT_EQ(network_unit_.num_actions(), 77);
  EXPECT_EQ(network_unit_.concatenated_input_dim(), kFixedDim + kPreviousDim);

  // Check that each sub-segment is equal to one of the input embeddings.
  const Vector<float> input = network_unit_.concatenated_input();
  EXPECT_EQ(input.size(), network_unit_.concatenated_input_dim());
  size_t index = 0;
  size_t end = kFixedDim;
  for (; index < end; ++index) EXPECT_EQ(input[index], kFixedValue);
  end += kPreviousDim;
  for (; index < end; ++index) EXPECT_EQ(input[index], kPreviousValue);
}

// Tests that NetworkUnitBase produces a properly-ordered concatenation of
// multiple fixed and linked channels, including a recurrent channel.
TEST_F(NetworkUnitBaseTest, ConcatenateMultipleChannelsOfEachType) {
  const float kEmbedding0 = 1.25;
  const float kEmbedding1 = -0.125;
  const float kFeature0 = 0.75;
  const float kFeature1 = -2.5;
  const size_t kFixedDim0 = 13;
  const size_t kFixedDim1 = 19;
  const string kSpec = R"(num_actions: 99
                          fixed_feature {
                            vocabulary_size: 11
                            embedding_dim: 13
                            size: 1
                          }
                          fixed_feature {
                            vocabulary_size: 17
                            embedding_dim: 19
                            size: 1
                          }
                          linked_feature {
                            embedding_dim: -1
                            source_component: 'previous_component'
                            source_layer: 'previous_layer'
                            size: 1
                          }
                          linked_feature {
                            embedding_dim: -1
                            source_component: 'test_component'
                            source_layer: 'recurrent_layer'
                            size: 1
                          })";
  AddFixedEmbeddingMatrix(0, 11, kFixedDim0, kEmbedding0);
  AddFixedEmbeddingMatrix(1, 17, kFixedDim1, kEmbedding1);

  EXPECT_CALL(compute_session_, GetInputFeatures(_, _, _, _, _))
      .WillOnce(Invoke(ExtractFeatures(0, {{1, kFeature0}})))
      .WillOnce(Invoke(ExtractFeatures(1, {{1, kFeature1}})));
  const float kFixedValue0 = kEmbedding0 * kFeature0;
  const float kFixedValue1 = kEmbedding1 * kFeature1;

  EXPECT_CALL(compute_session_, GetTranslatedLinkFeatures(_, _))
      .WillOnce(Invoke(ExtractLinks(0, {"step_idx: 5"})))
      .WillOnce(Invoke(ExtractLinks(1, {"step_idx: 6"})));
  EXPECT_CALL(compute_session_, SourceComponentBeamSize(_, _))
      .WillRepeatedly(Return(1));

  network_unit_.RequestConcatenation();
  network_unit_.ProvideConcatenatedInput();
  TF_ASSERT_OK(Run(kSpec));

  EXPECT_EQ(network_unit_.fixed_embedding_manager().num_channels(), 2);
  EXPECT_EQ(network_unit_.linked_embedding_manager().num_channels(), 2);
  EXPECT_EQ(network_unit_.num_actions(), 99);
  EXPECT_EQ(network_unit_.concatenated_input_dim(),
            kFixedDim0 + kFixedDim1 + kPreviousDim + kRecurrentDim);

  // Check that each sub-segment is equal to one of the input embeddings.  For
  // compatibility with the Python codebase, fixed channels must appear before
  // linked channels, and among each type order follows the ComponentSpec.
  const Vector<float> input = network_unit_.concatenated_input();
  EXPECT_EQ(input.size(), network_unit_.concatenated_input_dim());
  size_t index = 0;
  size_t end = kFixedDim0;
  for (; index < end; ++index) EXPECT_EQ(input[index], kFixedValue0);
  end += kFixedDim1;
  for (; index < end; ++index) EXPECT_EQ(input[index], kFixedValue1);
  end += kPreviousDim;
  for (; index < end; ++index) EXPECT_EQ(input[index], kPreviousValue);
  end += kRecurrentDim;
  for (; index < end; ++index) EXPECT_EQ(input[index], kRecurrentValue);
}

// Tests that NetworkUnitBase refuses to concatenate if there are non-embedded
// fixed embeddings.
TEST_F(NetworkUnitBaseTest, CannotConcatenateNonEmbeddedFixedFeatures) {
  const string kBadSpec = R"(fixed_feature {
                               embedding_dim: -1
                               size: 1
                             })";

  network_unit_.RequestConcatenation();
  network_unit_.ProvideConcatenatedInput();
  EXPECT_THAT(Run(kBadSpec),
              test::IsErrorWithSubstr(
                  "Non-embedded fixed features cannot be concatenated"));
}

}  // namespace
}  // namespace runtime
}  // namespace dragnn
}  // namespace syntaxnet
