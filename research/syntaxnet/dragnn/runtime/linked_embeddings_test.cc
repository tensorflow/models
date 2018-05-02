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

#include "dragnn/runtime/linked_embeddings.h"

#include <string>

#include "dragnn/core/test/generic.h"
#include "dragnn/protos/spec.pb.h"
#include "dragnn/runtime/network_states.h"
#include "dragnn/runtime/test/network_test_base.h"
#include "syntaxnet/base.h"
#include <gmock/gmock.h>
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/test.h"

namespace syntaxnet {
namespace dragnn {
namespace runtime {
namespace {

using ::testing::_;
using ::testing::Invoke;
using ::testing::Return;

// Dimensions of the layers in the network (see ResetManager() below).
const size_t kPrevious1LayerDim = 16;
const size_t kPrevious2LayerDim = 32;
const size_t kRecurrentLayerDim = 48;

// Dimensions of the transformed links in the network.
const size_t kPrevious2EmbeddingDim = 24;
const size_t kRecurrentEmbeddingDim = 40;

// Number of transition steps to take in each component in the network.
const size_t kNumSteps = 10;

// A working one-channel ComponentSpec.
const char kSingleSpec[] = R"(linked_feature {
                                embedding_dim: -1
                                source_component: 'source_component_1'
                                source_layer: 'previous_1'
                                size: 1
                              })";

// A working multi-channel ComponentSpec.
const char kMultiSpec[] = R"(linked_feature {
                               embedding_dim: -1
                               source_component: 'source_component_1'
                               source_layer: 'previous_1'
                               size: 1
                             }
                             linked_feature {
                               embedding_dim: 24
                               source_component: 'source_component_2'
                               source_layer: 'previous_2'
                               size: 1
                             }
                             linked_feature {
                               embedding_dim: 40
                               source_component: 'test_component'
                               source_layer: 'recurrent'
                               size: 1
                             })";

class LinkedEmbeddingManagerTest : public NetworkTestBase {
 protected:
  // Creates a LinkedEmbeddingManager and returns the result of Reset()-ing it
  // using the |component_spec_text|.
  tensorflow::Status ResetManager(const string &component_spec_text) {
    ComponentSpec component_spec;
    CHECK(TextFormat::ParseFromString(component_spec_text, &component_spec));
    component_spec.set_name(kTestComponentName);

    AddComponent("source_component_0");
    AddComponent("source_component_1");
    AddLayer("previous_1", kPrevious1LayerDim);
    AddComponent("source_component_2");
    AddLayer("previous_2", kPrevious2LayerDim);
    AddComponent(kTestComponentName);
    AddLayer("recurrent", kRecurrentLayerDim);

    return manager_.Reset(component_spec, &variable_store_,
                          &network_state_manager_);
  }

  LinkedEmbeddingManager manager_;
};

// Tests that LinkedEmbeddingManager is empty by default.
TEST_F(LinkedEmbeddingManagerTest, EmptyByDefault) {
  EXPECT_EQ(manager_.num_channels(), 0);
  EXPECT_EQ(manager_.num_embeddings(), 0);
}

// Tests that LinkedEmbeddingManager is empty when reset to an empty spec.
TEST_F(LinkedEmbeddingManagerTest, EmptySpec) {
  TF_EXPECT_OK(ResetManager(""));

  EXPECT_EQ(manager_.component_name(), kTestComponentName);
  EXPECT_EQ(manager_.num_channels(), 0);
  EXPECT_EQ(manager_.num_embeddings(), 0);
}

// Tests that LinkedEmbeddingManager works with a single channel.
TEST_F(LinkedEmbeddingManagerTest, OneChannel) {
  TF_EXPECT_OK(ResetManager(kSingleSpec));

  EXPECT_EQ(manager_.component_name(), kTestComponentName);
  EXPECT_EQ(manager_.num_channels(), 1);
  EXPECT_EQ(manager_.embedding_dim(0), kPrevious1LayerDim);
  EXPECT_EQ(manager_.num_embeddings(), 1);
}

// Tests that LinkedEmbeddingManager works with multiple channels.
TEST_F(LinkedEmbeddingManagerTest, MultipleChannels) {
  AddLinkedWeightMatrix(1, kPrevious2LayerDim, kPrevious2EmbeddingDim, 0.0);
  AddLinkedWeightMatrix(2, kRecurrentLayerDim, kRecurrentEmbeddingDim, 0.0);
  AddLinkedOutOfBoundsVector(1, kPrevious2EmbeddingDim, 0.0);
  AddLinkedOutOfBoundsVector(2, kRecurrentEmbeddingDim, 0.0);

  TF_EXPECT_OK(ResetManager(kMultiSpec));

  EXPECT_EQ(manager_.component_name(), kTestComponentName);
  EXPECT_EQ(manager_.num_channels(), 3);
  EXPECT_EQ(manager_.embedding_dim(0), kPrevious1LayerDim);
  EXPECT_EQ(manager_.embedding_dim(1), kPrevious2EmbeddingDim);
  EXPECT_EQ(manager_.embedding_dim(2), kRecurrentEmbeddingDim);
  EXPECT_EQ(manager_.num_embeddings(), 3);
}

// Tests that LinkedEmbeddingManager fails when the channel size is 0.
TEST_F(LinkedEmbeddingManagerTest, InvalidChannelSize) {
  const string kBadSpec = R"(linked_feature {
                               embedding_dim: -1
                               source_component: 'source_component_1'
                               source_layer: 'previous_1'
                               size: 0  # bad
                             })";
  EXPECT_THAT(ResetManager(kBadSpec),
              test::IsErrorWithSubstr("Invalid channel size"));
}

// Tests that LinkedEmbeddingManager fails when the channel size is > 1.
TEST_F(LinkedEmbeddingManagerTest, UnsupportedChannelSize) {
  const string kBadSpec = R"(linked_feature {
                               embedding_dim: -1
                               source_component: 'source_component_1'
                               source_layer: 'previous_1'
                               size: 2  # bad
                             })";
  EXPECT_THAT(ResetManager(kBadSpec),
              test::IsErrorWithSubstr(
                  "Multi-instance linked features are not supported"));
}

// Tests that LinkedEmbeddingManager fails when the source component is unknown.
TEST_F(LinkedEmbeddingManagerTest, UnknownComponent) {
  const string kBadSpec = R"(linked_feature {
                               embedding_dim: -1
                               source_component: 'missing_component'  # bad
                               source_layer: 'previous_1'
                               size: 1
                             })";
  EXPECT_THAT(ResetManager(kBadSpec),
              test::IsErrorWithSubstr("Unknown component"));
}

// Tests that LinkedEmbeddingManager fails when the source layer is unknown.
TEST_F(LinkedEmbeddingManagerTest, UnknownLayer) {
  const string kBadSpec = R"(linked_feature {
                               embedding_dim: -1
                               source_component: 'source_component_0'
                               source_layer: 'missing_layer'  # bad
                               size: 1
                             })";
  EXPECT_THAT(ResetManager(kBadSpec),
              test::IsErrorWithSubstr("Unknown layer"));
}

// Tests that LinkedEmbeddingManager fails for a missing weight matrix.
TEST_F(LinkedEmbeddingManagerTest, MissingWeightMatrix) {
  // Only the weight matrix for channel 2 is missing.
  AddLinkedWeightMatrix(1, kPrevious2LayerDim, kPrevious2EmbeddingDim, 0.0);
  AddLinkedOutOfBoundsVector(1, kPrevious2EmbeddingDim, 0.0);
  AddLinkedOutOfBoundsVector(2, kRecurrentEmbeddingDim, 0.0);

  EXPECT_THAT(ResetManager(kMultiSpec),
              test::IsErrorWithSubstr("Unknown variable"));
}

// Tests that LinkedEmbeddingManager fails for a missing out-of-bounds vector.
TEST_F(LinkedEmbeddingManagerTest, MissingOutOfBoundsVector) {
  // Only the out-of-bounds vector for channel 1 is missing.
  AddLinkedWeightMatrix(1, kPrevious2LayerDim, kPrevious2EmbeddingDim, 0.0);
  AddLinkedWeightMatrix(2, kRecurrentLayerDim, kRecurrentEmbeddingDim, 0.0);
  AddLinkedOutOfBoundsVector(2, kRecurrentEmbeddingDim, 0.0);

  EXPECT_THAT(ResetManager(kMultiSpec),
              test::IsErrorWithSubstr("Unknown variable"));
}

// Tests that LinkedEmbeddingManager fails for a weight matrix with the wrong
// number of rows.
TEST_F(LinkedEmbeddingManagerTest, WeightMatrixRowMismatch) {
  AddLinkedWeightMatrix(1, kPrevious2LayerDim + 1, kPrevious2EmbeddingDim, 0.0);
  AddLinkedWeightMatrix(2, kRecurrentLayerDim, kRecurrentEmbeddingDim, 0.0);
  AddLinkedOutOfBoundsVector(1, kPrevious2EmbeddingDim, 0.0);
  AddLinkedOutOfBoundsVector(2, kRecurrentEmbeddingDim, 0.0);

  EXPECT_THAT(ResetManager(kMultiSpec),
              test::IsErrorWithSubstr(
                  "Weight matrix does not match source layer in link 1"));
}

// Tests that LinkedEmbeddingManager fails for a weight matrix with the wrong
// number of columns.
TEST_F(LinkedEmbeddingManagerTest, WeightMatrixColumnMismatch) {
  AddLinkedWeightMatrix(1, kPrevious2LayerDim, kPrevious2EmbeddingDim, 0.0);
  AddLinkedWeightMatrix(2, kRecurrentLayerDim, kRecurrentEmbeddingDim - 1, 0.0);
  AddLinkedOutOfBoundsVector(1, kPrevious2EmbeddingDim, 0.0);
  AddLinkedOutOfBoundsVector(2, kRecurrentEmbeddingDim, 0.0);

  EXPECT_THAT(ResetManager(kMultiSpec),
              test::IsErrorWithSubstr(
                  "Weight matrix shape should be output dimension plus "
                  "padding"));
}

// Tests that LinkedEmbeddingManager fails for a weight matrix with the wrong
// number of rows.
TEST_F(LinkedEmbeddingManagerTest, OutOfBoundsVectorSizeMismatch) {
  AddLinkedWeightMatrix(1, kPrevious2LayerDim, kPrevious2EmbeddingDim, 0.0);
  AddLinkedWeightMatrix(2, kRecurrentLayerDim, kRecurrentEmbeddingDim, 0.0);
  AddLinkedOutOfBoundsVector(1, kPrevious2EmbeddingDim + 1, 0.0);
  AddLinkedOutOfBoundsVector(2, kRecurrentEmbeddingDim, 0.0);

  EXPECT_THAT(
      ResetManager(kMultiSpec),
      test::IsErrorWithSubstr(
          "Out-of-bounds vector does not match embedding_dim in link 1"));
}

// Values to fill each layer with.
const float kLayerValues[] = {1.0, 2.0, 3.0};

class LinkedEmbeddingsTest : public LinkedEmbeddingManagerTest {
 protected:
  // Resets the |linked_embeddings_| using the |manager_|, |network_states_|,
  // and |compute_session_|, and returns the resulting status.
  tensorflow::Status ResetLinkedEmbeddings() {
    network_states_.Reset(&network_state_manager_);

    // Fill components with steps.
    StartComponent(kNumSteps);  // source_component_0
    StartComponent(kNumSteps);  // source_component_1
    StartComponent(kNumSteps);  // source_component_2
    StartComponent(kNumSteps);  // current component

    // Fill layers with values.
    FillLayer("source_component_1", "previous_1", kLayerValues[0]);
    FillLayer("source_component_2", "previous_2", kLayerValues[1]);
    FillLayer(kTestComponentName, "recurrent", kLayerValues[2]);

    return linked_embeddings_.Reset(&manager_, network_states_,
                                    &compute_session_);
  }

  LinkedEmbeddings linked_embeddings_;
};

// Tests that LinkedEmbeddings is empty by default.
TEST_F(LinkedEmbeddingsTest, EmptyByDefault) {
  EXPECT_EQ(linked_embeddings_.num_embeddings(), 0);
}

// Tests that LinkedEmbeddings is empty when reset by an empty manager.
TEST_F(LinkedEmbeddingsTest, EmptyManager) {
  TF_ASSERT_OK(ResetManager(""));

  TF_EXPECT_OK(ResetLinkedEmbeddings());
  EXPECT_EQ(linked_embeddings_.num_embeddings(), 0);
}

// Tests that LinkedEmbeddings fails when no linked features are extracted.
TEST_F(LinkedEmbeddingsTest, OneChannelNoFeatures) {
  TF_ASSERT_OK(ResetManager(kSingleSpec));

  EXPECT_CALL(compute_session_, GetTranslatedLinkFeatures(_, _))
      .WillOnce(Invoke(ExtractLinks(0, {})));
  EXPECT_CALL(compute_session_, SourceComponentBeamSize(_, _))
      .WillRepeatedly(Return(1));

  EXPECT_THAT(ResetLinkedEmbeddings(),
              test::IsErrorWithSubstr("Got 0 linked features; expected 1"));
}

// Tests that LinkedEmbeddings works when exactly one linked feature is
// extracted.
TEST_F(LinkedEmbeddingsTest, OneChannelOneFeature) {
  TF_ASSERT_OK(ResetManager(kSingleSpec));

  EXPECT_CALL(compute_session_, GetTranslatedLinkFeatures(_, _))
      .WillOnce(Invoke(ExtractLinks(0, {"step_idx: 5"})));
  EXPECT_CALL(compute_session_, SourceComponentBeamSize(_, _))
      .WillRepeatedly(Return(1));

  TF_ASSERT_OK(ResetLinkedEmbeddings());
  ASSERT_EQ(linked_embeddings_.num_embeddings(), 1);
  ExpectVector(linked_embeddings_.embedding(0), kPrevious1LayerDim, 1.0);
  EXPECT_FALSE(linked_embeddings_.is_out_of_bounds(0));
}

// Tests that LinkedEmbeddings fails when more than one linked feature is
// extracted.
TEST_F(LinkedEmbeddingsTest, OneChannelManyFeatures) {
  TF_ASSERT_OK(ResetManager(kSingleSpec));

  EXPECT_CALL(compute_session_, GetTranslatedLinkFeatures(_, _))
      .WillOnce(Invoke(
          ExtractLinks(0, {"step_idx: 5", "step_idx: 6", "step_idx: 7"})));
  EXPECT_CALL(compute_session_, SourceComponentBeamSize(_, _))
      .WillRepeatedly(Return(1));

  EXPECT_THAT(ResetLinkedEmbeddings(),
              test::IsErrorWithSubstr("Got 3 linked features; expected 1"));
}

// Tests that LinkedEmbeddings fails if the linked feature has a batch index.
TEST_F(LinkedEmbeddingsTest, BatchesUnsupported) {
  TF_ASSERT_OK(ResetManager(kSingleSpec));

  EXPECT_CALL(compute_session_, GetTranslatedLinkFeatures(_, _))
      .WillOnce(Invoke(ExtractLinks(0, {"step_idx: 5 batch_idx: 1"})));
  EXPECT_CALL(compute_session_, SourceComponentBeamSize(_, _))
      .WillRepeatedly(Return(1));

  EXPECT_THAT(ResetLinkedEmbeddings(),
              test::IsErrorWithSubstr("Batches are not supported"));
}

// Tests that LinkedEmbeddings fails if the linked feature has a beam index.
TEST_F(LinkedEmbeddingsTest, BeamsUnsupported) {
  TF_ASSERT_OK(ResetManager(kSingleSpec));

  EXPECT_CALL(compute_session_, GetTranslatedLinkFeatures(_, _))
      .WillOnce(Invoke(ExtractLinks(0, {"step_idx: 5 beam_idx: 1"})));
  EXPECT_CALL(compute_session_, SourceComponentBeamSize(_, _))
      .WillRepeatedly(Return(1));

  EXPECT_THAT(ResetLinkedEmbeddings(),
              test::IsErrorWithSubstr("Beams are not supported"));
}

// Tests that LinkedEmbeddings fails if the source component of the link has
// beam size > 1.
TEST_F(LinkedEmbeddingsTest, OneChannelWithSourceBeam) {
  TF_ASSERT_OK(ResetManager(kSingleSpec));

  EXPECT_CALL(compute_session_, GetTranslatedLinkFeatures(_, _))
      .WillOnce(Invoke(ExtractLinks(0, {"step_idx: 5"})));
  EXPECT_CALL(compute_session_, SourceComponentBeamSize(_, _))
      .WillOnce(Return(2));

  EXPECT_THAT(ResetLinkedEmbeddings(),
              test::IsErrorWithSubstr("Source beams are not supported"));
}

// Tests that LinkedEmbeddings produces zeros when the extracted linked feature
// has no step index.
TEST_F(LinkedEmbeddingsTest, OneChannelNoStep) {
  TF_ASSERT_OK(ResetManager(kSingleSpec));

  EXPECT_CALL(compute_session_, GetTranslatedLinkFeatures(_, _))
      .WillOnce(Invoke(ExtractLinks(0, {""})));
  EXPECT_CALL(compute_session_, SourceComponentBeamSize(_, _))
      .WillRepeatedly(Return(1));

  TF_ASSERT_OK(ResetLinkedEmbeddings());
  ASSERT_EQ(linked_embeddings_.num_embeddings(), 1);
  ExpectVector(linked_embeddings_.embedding(0), kPrevious1LayerDim, 0.0);
  EXPECT_TRUE(linked_embeddings_.is_out_of_bounds(0));
}

// Tests that LinkedEmbeddings produces zeros when the extracted linked feature
// has step index -1.
TEST_F(LinkedEmbeddingsTest, OneChannelNegativeOneStep) {
  TF_ASSERT_OK(ResetManager(kSingleSpec));

  EXPECT_CALL(compute_session_, GetTranslatedLinkFeatures(_, _))
      .WillOnce(Invoke(ExtractLinks(0, {"step_idx: -1"})));
  EXPECT_CALL(compute_session_, SourceComponentBeamSize(_, _))
      .WillRepeatedly(Return(1));

  TF_ASSERT_OK(ResetLinkedEmbeddings());
  ASSERT_EQ(linked_embeddings_.num_embeddings(), 1);
  ExpectVector(linked_embeddings_.embedding(0), kPrevious1LayerDim, 0.0);
  EXPECT_TRUE(linked_embeddings_.is_out_of_bounds(0));
}

// Tests that LinkedEmbeddings produces zeros when the extracted linked feature
// has a large negative step index.
TEST_F(LinkedEmbeddingsTest, OneChannelLargeNegativeStep) {
  TF_ASSERT_OK(ResetManager(kSingleSpec));

  EXPECT_CALL(compute_session_, GetTranslatedLinkFeatures(_, _))
      .WillOnce(Invoke(ExtractLinks(0, {"step_idx: -100"})));
  EXPECT_CALL(compute_session_, SourceComponentBeamSize(_, _))
      .WillRepeatedly(Return(1));

  TF_ASSERT_OK(ResetLinkedEmbeddings());
  ASSERT_EQ(linked_embeddings_.num_embeddings(), 1);
  ExpectVector(linked_embeddings_.embedding(0), kPrevious1LayerDim, 0.0);
  EXPECT_TRUE(linked_embeddings_.is_out_of_bounds(0));
}

// Tests that LinkedEmbeddings works with multiple linked channels.
TEST_F(LinkedEmbeddingsTest, ManyChannels) {
  AddLinkedWeightMatrix(1, kPrevious2LayerDim, kPrevious2EmbeddingDim, 0.5);
  AddLinkedWeightMatrix(2, kRecurrentLayerDim, kRecurrentEmbeddingDim, 1.5);
  AddLinkedOutOfBoundsVector(1, kPrevious2EmbeddingDim, 5.5);
  AddLinkedOutOfBoundsVector(2, kRecurrentEmbeddingDim, 7.75);

  const size_t kEmbeddingDims[] = {kPrevious1LayerDim,      //
                                   kPrevious2EmbeddingDim,  //
                                   kRecurrentEmbeddingDim};
  const float kExpected[] = {kLayerValues[0],                              //
                             kLayerValues[1] * kPrevious2LayerDim * 0.5f,  //
                             kLayerValues[2] * kRecurrentLayerDim * 1.5f};

  TF_ASSERT_OK(ResetManager(kMultiSpec));

  EXPECT_CALL(compute_session_, GetTranslatedLinkFeatures(_, _))
      .WillOnce(Invoke(ExtractLinks(0, {"step_idx: 5"})))
      .WillOnce(Invoke(ExtractLinks(1, {"step_idx: 6"})))
      .WillOnce(Invoke(ExtractLinks(2, {"step_idx: 7"})));
  EXPECT_CALL(compute_session_, SourceComponentBeamSize(_, _))
      .Times(3)
      .WillRepeatedly(Return(1));

  TF_ASSERT_OK(ResetLinkedEmbeddings());
  ASSERT_EQ(linked_embeddings_.num_embeddings(), 3);
  for (int channel_id = 0; channel_id < linked_embeddings_.num_embeddings();
       ++channel_id) {
    ExpectVector(linked_embeddings_.embedding(channel_id),
                 kEmbeddingDims[channel_id], kExpected[channel_id]);
    EXPECT_FALSE(linked_embeddings_.is_out_of_bounds(channel_id));
  }
}

// Tests that LinkedEmbeddings produces the relevant out-of-bounds embeddings
// when multiple linked channels have invalid step indices.
TEST_F(LinkedEmbeddingsTest, ManyChannelsOutOfBounds) {
  AddLinkedWeightMatrix(1, kPrevious2LayerDim, kPrevious2EmbeddingDim, 0.5);
  AddLinkedWeightMatrix(2, kRecurrentLayerDim, kRecurrentEmbeddingDim, 1.5);
  AddLinkedOutOfBoundsVector(1, kPrevious2EmbeddingDim, 5.5);
  AddLinkedOutOfBoundsVector(2, kRecurrentEmbeddingDim, 7.75);

  const size_t kEmbeddingDims[] = {kPrevious1LayerDim,      //
                                   kPrevious2EmbeddingDim,  //
                                   kRecurrentEmbeddingDim};
  const float kExpected[] = {0.0f, 5.5f, 7.75f};

  TF_ASSERT_OK(ResetManager(kMultiSpec));

  EXPECT_CALL(compute_session_, GetTranslatedLinkFeatures(_, _))
      .WillOnce(Invoke(ExtractLinks(0, {"step_idx: -1"})))
      .WillOnce(Invoke(ExtractLinks(1, {"step_idx: -10"})))
      .WillOnce(Invoke(ExtractLinks(2, {"step_idx: -999"})));
  EXPECT_CALL(compute_session_, SourceComponentBeamSize(_, _))
      .Times(3)
      .WillRepeatedly(Return(1));

  TF_ASSERT_OK(ResetLinkedEmbeddings());
  ASSERT_EQ(linked_embeddings_.num_embeddings(), 3);
  for (int channel_id = 0; channel_id < linked_embeddings_.num_embeddings();
       ++channel_id) {
    ExpectVector(linked_embeddings_.embedding(channel_id),
                 kEmbeddingDims[channel_id], kExpected[channel_id]);
    EXPECT_TRUE(linked_embeddings_.is_out_of_bounds(channel_id));
  }
}

}  // namespace
}  // namespace runtime
}  // namespace dragnn
}  // namespace syntaxnet
