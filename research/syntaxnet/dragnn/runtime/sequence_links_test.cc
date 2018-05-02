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

#include "dragnn/runtime/sequence_links.h"

#include <stddef.h>
#include <string>
#include <vector>

#include "dragnn/core/input_batch_cache.h"
#include "dragnn/core/test/generic.h"
#include "dragnn/protos/spec.pb.h"
#include "dragnn/runtime/linked_embeddings.h"
#include "dragnn/runtime/network_states.h"
#include "dragnn/runtime/sequence_linker.h"
#include "dragnn/runtime/test/network_test_base.h"
#include "syntaxnet/base.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/test.h"

namespace syntaxnet {
namespace dragnn {
namespace runtime {
namespace {

// Dimensions of the layers in the network (see ResetManager() below).
const size_t kPrevious1LayerDim = 16;
const size_t kPrevious2LayerDim = 32;
const size_t kRecurrentLayerDim = 48;

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
                               embedding_dim: -1
                               source_component: 'source_component_2'
                               source_layer: 'previous_2'
                               size: 1
                             }
                             linked_feature {
                               embedding_dim: -1
                               source_component: 'test_component'
                               source_layer: 'recurrent'
                               size: 1
                             })";

// A recurrent-only ComponentSpec.
const char kRecurrentSpec[] = R"(linked_feature {
                                   embedding_dim: -1
                                   source_component: 'test_component'
                                   source_layer: 'recurrent'
                                   size: 1
                                 })";

// Fails to initialize.
class FailToInitialize : public SequenceLinker {
 public:
  // Implements SequenceLinker.
  bool Supports(const LinkedFeatureChannel &,
                const ComponentSpec &component_spec) const override {
    LOG(FATAL) << "Should never be called.";
  }
  tensorflow::Status Initialize(const LinkedFeatureChannel &,
                                const ComponentSpec &) override {
    return tensorflow::errors::Internal("No initialization for you!");
  }
  tensorflow::Status GetLinks(size_t, InputBatchCache *,
                              std::vector<int32> *) const override {
    LOG(FATAL) << "Should never be called.";
  }
};

DRAGNN_RUNTIME_REGISTER_SEQUENCE_LINKER(FailToInitialize);

// Initializes OK, then fails to extract links.
class FailToGetLinks : public FailToInitialize {
 public:
  // Implements SequenceLinker.
  tensorflow::Status Initialize(const LinkedFeatureChannel &,
                                const ComponentSpec &) override {
    return tensorflow::Status::OK();
  }
  tensorflow::Status GetLinks(size_t, InputBatchCache *,
                              std::vector<int32> *) const override {
    return tensorflow::errors::Internal("No links for you!");
  }
};

DRAGNN_RUNTIME_REGISTER_SEQUENCE_LINKER(FailToGetLinks);

// Initializes OK and links to the previous step.
class LinkToPrevious : public FailToGetLinks {
 public:
  // Implements SequenceLinker.
  tensorflow::Status GetLinks(size_t source_num_steps, InputBatchCache *,
                              std::vector<int32> *links) const override {
    links->resize(source_num_steps);
    for (int i = 0; i < links->size(); ++i) (*links)[i] = i - 1;
    return tensorflow::Status::OK();
  }
};

DRAGNN_RUNTIME_REGISTER_SEQUENCE_LINKER(LinkToPrevious);

// Initializes OK but produces the wrong number of links.
class WrongNumberOfLinks : public FailToGetLinks {
 public:
  // Implements SequenceLinker.
  tensorflow::Status GetLinks(size_t, InputBatchCache *,
                              std::vector<int32> *links) const override {
    links->resize(kNumSteps + 1);
    return tensorflow::Status::OK();
  }
};

DRAGNN_RUNTIME_REGISTER_SEQUENCE_LINKER(WrongNumberOfLinks);

class SequenceLinkManagerTest : public NetworkTestBase {
 protected:
  // Sets up previous components and layers.
  void AddComponentsAndLayers() {
    AddComponent("source_component_0");
    AddComponent("source_component_1");
    AddLayer("previous_1", kPrevious1LayerDim);
    AddComponent("source_component_2");
    AddLayer("previous_2", kPrevious2LayerDim);
    AddComponent(kTestComponentName);
    AddLayer("recurrent", kRecurrentLayerDim);
  }

  // Creates a SequenceLinkManager and returns the result of Reset()-ing it
  // using the |component_spec_text|.
  tensorflow::Status ResetManager(
      const string &component_spec_text,
      const std::vector<string> &sequence_linker_types) {
    ComponentSpec component_spec;
    CHECK(TextFormat::ParseFromString(component_spec_text, &component_spec));
    component_spec.set_name(kTestComponentName);

    AddComponentsAndLayers();

    TF_RETURN_IF_ERROR(linked_embedding_manager_.Reset(
        component_spec, &variable_store_, &network_state_manager_));

    return manager_.Reset(&linked_embedding_manager_, component_spec,
                          sequence_linker_types);
  }

  LinkedEmbeddingManager linked_embedding_manager_;
  SequenceLinkManager manager_;
};

// Tests that SequenceLinkManager is empty by default.
TEST_F(SequenceLinkManagerTest, EmptyByDefault) {
  EXPECT_EQ(manager_.num_channels(), 0);
}

// Tests that SequenceLinkManager is empty when reset to an empty spec.
TEST_F(SequenceLinkManagerTest, EmptySpec) {
  TF_EXPECT_OK(ResetManager("", {}));

  EXPECT_EQ(manager_.num_channels(), 0);
}

// Tests that SequenceLinkManager works with a single channel.
TEST_F(SequenceLinkManagerTest, OneChannel) {
  TF_EXPECT_OK(ResetManager(kSingleSpec, {"LinkToPrevious"}));

  EXPECT_EQ(manager_.num_channels(), 1);
}

// Tests that SequenceLinkManager works with multiple channels.
TEST_F(SequenceLinkManagerTest, MultipleChannels) {
  TF_EXPECT_OK(ResetManager(
      kMultiSpec, {"LinkToPrevious", "LinkToPrevious", "LinkToPrevious"}));

  EXPECT_EQ(manager_.num_channels(), 3);
}

// Tests that SequenceLinkManager fails if the LinkedEmbeddingManager and
// ComponentSpec are mismatched.
TEST_F(SequenceLinkManagerTest, MismatchedLinkedManagerAndComponentSpec) {
  ComponentSpec component_spec;
  CHECK(TextFormat::ParseFromString(kMultiSpec, &component_spec));
  component_spec.set_name(kTestComponentName);

  AddComponentsAndLayers();

  TF_ASSERT_OK(linked_embedding_manager_.Reset(component_spec, &variable_store_,
                                               &network_state_manager_));

  // Remove one linked feature, resulting in a mismatch.
  component_spec.mutable_linked_feature()->RemoveLast();

  EXPECT_THAT(
      manager_.Reset(&linked_embedding_manager_, component_spec,
                     {"LinkToPrevious", "LinkToPrevious", "LinkToPrevious"}),
      test::IsErrorWithSubstr("Channel mismatch between LinkedEmbeddingManager "
                              "(3) and ComponentSpec (2)"));
}

// Tests that SequenceLinkManager fails if the LinkedEmbeddingManager and
// SequenceLinkers are mismatched.
TEST_F(SequenceLinkManagerTest, MismatchedLinkedManagerAndSequenceLinkers) {
  EXPECT_THAT(
      ResetManager(kMultiSpec, {"LinkToPrevious", "LinkToPrevious"}),
      test::IsErrorWithSubstr("Channel mismatch between LinkedEmbeddingManager "
                              "(3) and SequenceLinkers (2)"));
}

// Tests that SequenceLinkManager fails when the link is transformed.
TEST_F(SequenceLinkManagerTest, UnsupportedTransformedLink) {
  const string kBadSpec = R"(linked_feature {
                               embedding_dim: 16  # bad
                               source_component: 'source_component_1'
                               source_layer: 'previous_1'
                               size: 1
                             })";
  AddLinkedWeightMatrix(0, kPrevious1LayerDim, 16, 0.0);
  AddLinkedOutOfBoundsVector(0, 16, 0.0);

  EXPECT_THAT(
      ResetManager(kBadSpec, {"LinkToPrevious"}),
      test::IsErrorWithSubstr("Transformed linked features are not supported"));
}

// Tests that SequenceLinkManager fails if one of the SequenceLinkers fails to
// initialize.
TEST_F(SequenceLinkManagerTest, FailToInitializeSequenceLinker) {
  EXPECT_THAT(ResetManager(kMultiSpec, {"LinkToPrevious", "FailToInitialize",
                                        "LinkToPrevious"}),
              test::IsErrorWithSubstr("No initialization for you!"));
}

// Tests that SequenceLinkManager is OK even if the SequenceLinkers would fail
// in GetLinks().
TEST_F(SequenceLinkManagerTest, ManagerDoesntCareAboutGetLinks) {
  TF_EXPECT_OK(ResetManager(
      kMultiSpec, {"FailToGetLinks", "FailToGetLinks", "FailToGetLinks"}));
}

// Values to fill each layer with.
const float kPrevious1LayerValue = 1.0;
const float kPrevious2LayerValue = 2.0;
const float kRecurrentLayerValue = 3.0;

class SequenceLinksTest : public SequenceLinkManagerTest {
 protected:
  // Resets the |sequence_links_| using the |manager_|, |network_states_|, and
  // |input_batch_cache_|, and returns the resulting status.  Passes |add_steps|
  // to Reset() and advances the current component by |num_steps|.
  tensorflow::Status ResetLinks(bool add_steps = false,
                                size_t num_steps = kNumSteps) {
    network_states_.Reset(&network_state_manager_);

    // Fill components with steps.
    StartComponent(kNumSteps);  // source_component_0
    StartComponent(kNumSteps);  // source_component_1
    StartComponent(kNumSteps);  // source_component_2
    StartComponent(num_steps);  // current component

    // Fill layers with values.
    FillLayer("source_component_1", "previous_1", kPrevious1LayerValue);
    FillLayer("source_component_2", "previous_2", kPrevious2LayerValue);
    FillLayer(kTestComponentName, "recurrent", kRecurrentLayerValue);

    return sequence_links_.Reset(add_steps, &manager_, &network_states_,
                                 &input_batch_cache_);
  }

  InputBatchCache input_batch_cache_;
  SequenceLinks sequence_links_;
};

// Tests that SequenceLinks is empty by default.
TEST_F(SequenceLinksTest, EmptyByDefault) {
  EXPECT_EQ(sequence_links_.num_channels(), 0);
  EXPECT_EQ(sequence_links_.num_steps(), 0);
}

// Tests that SequenceLinks is empty when reset by an empty manager.
TEST_F(SequenceLinksTest, EmptyManager) {
  TF_ASSERT_OK(ResetManager("", {}));

  TF_EXPECT_OK(ResetLinks());
  EXPECT_EQ(sequence_links_.num_channels(), 0);
  EXPECT_EQ(sequence_links_.num_steps(), 0);
}

// Tests that SequenceLinks fails when one of the non-recurrent SequenceLinkers
// fails.
TEST_F(SequenceLinksTest, FailToGetNonRecurrentLinks) {
  TF_ASSERT_OK(ResetManager(
      kMultiSpec, {"LinkToPrevious", "FailToGetLinks", "LinkToPrevious"}));

  EXPECT_THAT(ResetLinks(), test::IsErrorWithSubstr("No links for you!"));
}

// Tests that SequenceLinks fails when one of the recurrent SequenceLinkers
// fails.
TEST_F(SequenceLinksTest, FailToGetRecurrentLinks) {
  TF_ASSERT_OK(ResetManager(
      kMultiSpec, {"LinkToPrevious", "LinkToPrevious", "FailToGetLinks"}));

  EXPECT_THAT(ResetLinks(), test::IsErrorWithSubstr("No links for you!"));
}

// Tests that SequenceLinks fails when the non-recurrent SequenceLinkers produce
// different numbers of links.
TEST_F(SequenceLinksTest, MismatchedNumbersOfNonRecurrentLinks) {
  TF_ASSERT_OK(ResetManager(
      kMultiSpec, {"LinkToPrevious", "WrongNumberOfLinks", "LinkToPrevious"}));

  EXPECT_THAT(ResetLinks(),
              test::IsErrorWithSubstr("Inconsistent link sequence lengths at "
                                      "channel ID 1: got 11 but expected 10"));
}

// Tests that SequenceLinks fails when the recurrent SequenceLinkers produce
// different numbers of links.
TEST_F(SequenceLinksTest, MismatchedNumbersOfRecurrentLinks) {
  TF_ASSERT_OK(ResetManager(
      kMultiSpec, {"LinkToPrevious", "LinkToPrevious", "WrongNumberOfLinks"}));

  EXPECT_THAT(ResetLinks(),
              test::IsErrorWithSubstr("Inconsistent link sequence lengths at "
                                      "channel ID 2: got 11 but expected 10"));
}

// Tests that SequenceLinks works as expected on one channel.
TEST_F(SequenceLinksTest, SingleChannel) {
  TF_ASSERT_OK(ResetManager(kSingleSpec, {"LinkToPrevious"}));

  TF_ASSERT_OK(ResetLinks());
  ASSERT_EQ(sequence_links_.num_channels(), 1);
  ASSERT_EQ(sequence_links_.num_steps(), kNumSteps);

  const Matrix<float> previous1(GetLayer("source_component_1", "previous_1"));
  Vector<float> embedding;
  bool is_out_of_bounds = false;

  // LinkToPrevious links the 0'th index to -1, which is out of bounds.
  sequence_links_.Get(0, 0, &embedding, &is_out_of_bounds);
  EXPECT_TRUE(is_out_of_bounds);
  ExpectVector(embedding, kPrevious1LayerDim, 0.0);

  // The remaining links point to the previous item.
  for (int i = 1; i < kNumSteps; ++i) {
    sequence_links_.Get(0, i, &embedding, &is_out_of_bounds);
    EXPECT_FALSE(is_out_of_bounds);
    ExpectVector(embedding, kPrevious1LayerDim, kPrevious1LayerValue);
    EXPECT_EQ(embedding.data(), previous1.row(i - 1).data());
  }
}

// Tests that SequenceLinks works as expected on multiple channels.
TEST_F(SequenceLinksTest, ManyChannels) {
  TF_ASSERT_OK(ResetManager(
      kMultiSpec, {"LinkToPrevious", "LinkToPrevious", "LinkToPrevious"}));

  TF_ASSERT_OK(ResetLinks());
  ASSERT_EQ(sequence_links_.num_channels(), 3);
  ASSERT_EQ(sequence_links_.num_steps(), kNumSteps);

  const Matrix<float> previous1(GetLayer("source_component_1", "previous_1"));
  const Matrix<float> previous2(GetLayer("source_component_2", "previous_2"));
  const Matrix<float> recurrent(GetLayer(kTestComponentName, "recurrent"));
  Vector<float> embedding;
  bool is_out_of_bounds = false;

  // LinkToPrevious links the 0'th index to -1, which is out of bounds.
  sequence_links_.Get(0, 0, &embedding, &is_out_of_bounds);
  EXPECT_TRUE(is_out_of_bounds);
  ExpectVector(embedding, kPrevious1LayerDim, 0.0);

  sequence_links_.Get(1, 0, &embedding, &is_out_of_bounds);
  EXPECT_TRUE(is_out_of_bounds);
  ExpectVector(embedding, kPrevious2LayerDim, 0.0);

  sequence_links_.Get(2, 0, &embedding, &is_out_of_bounds);
  EXPECT_TRUE(is_out_of_bounds);
  ExpectVector(embedding, kRecurrentLayerDim, 0.0);

  // The remaining links point to the previous item.
  for (int i = 1; i < kNumSteps; ++i) {
    sequence_links_.Get(0, i, &embedding, &is_out_of_bounds);
    EXPECT_FALSE(is_out_of_bounds);
    ExpectVector(embedding, kPrevious1LayerDim, kPrevious1LayerValue);
    EXPECT_EQ(embedding.data(), previous1.row(i - 1).data());

    sequence_links_.Get(1, i, &embedding, &is_out_of_bounds);
    EXPECT_FALSE(is_out_of_bounds);
    ExpectVector(embedding, kPrevious2LayerDim, kPrevious2LayerValue);
    EXPECT_EQ(embedding.data(), previous2.row(i - 1).data());

    sequence_links_.Get(2, i, &embedding, &is_out_of_bounds);
    EXPECT_FALSE(is_out_of_bounds);
    ExpectVector(embedding, kRecurrentLayerDim, kRecurrentLayerValue);
    EXPECT_EQ(embedding.data(), recurrent.row(i - 1).data());
  }
}

// Tests that SequenceLinks is emptied when resetting to an empty manager after
// being reset to a non-empty manager.
TEST_F(SequenceLinksTest, ResetToEmptyAfterNonEmpty) {
  TF_ASSERT_OK(ResetManager(kSingleSpec, {"LinkToPrevious"}));

  TF_ASSERT_OK(ResetLinks());
  ASSERT_EQ(sequence_links_.num_channels(), 1);
  ASSERT_EQ(sequence_links_.num_steps(), kNumSteps);

  SequenceLinkManager manager;
  TF_ASSERT_OK(sequence_links_.Reset(/*add_steps=*/false, &manager,
                                     &network_states_, &input_batch_cache_));
  ASSERT_EQ(sequence_links_.num_channels(), 0);
  ASSERT_EQ(sequence_links_.num_steps(), 0);
}

// Tests that SequenceLinks fails when adding steps to a component with no
// non-recurrent links.
TEST_F(SequenceLinksTest, AddStepsWithNoNonRecurrentLinks) {
  TF_ASSERT_OK(ResetManager(kRecurrentSpec, {"LinkToPrevious"}));

  EXPECT_THAT(
      ResetLinks(/*add_steps=*/true),
      test::IsErrorWithSubstr("Cannot infer the number of steps to add because "
                              "there are no non-recurrent links"));
}

// Tests that SequenceLinks produces no links when processing a component with
// only recurrent links, and when the NetworkStates has no steps.
TEST_F(SequenceLinksTest, RecurrentLinksWithNoSteps) {
  TF_ASSERT_OK(ResetManager(kRecurrentSpec, {"LinkToPrevious"}));

  TF_ASSERT_OK(ResetLinks(/*add_steps=*/false, /*num_steps=*/0));
  ASSERT_EQ(sequence_links_.num_channels(), 1);
  ASSERT_EQ(sequence_links_.num_steps(), 0);
}

// Tests that SequenceLinks properly infers the number of steps and adds them
// when processing a component with both non-recurrent and recurrent links.
TEST_F(SequenceLinksTest, AddStepsWithNonRecurrentAndRecurrentLinks) {
  TF_ASSERT_OK(ResetManager(
      kMultiSpec, {"LinkToPrevious", "LinkToPrevious", "LinkToPrevious"}));

  TF_ASSERT_OK(ResetLinks(/*add_steps=*/true, /*num_steps=*/0));
  ASSERT_EQ(sequence_links_.num_channels(), 3);
  ASSERT_EQ(sequence_links_.num_steps(), kNumSteps);
}

}  // namespace
}  // namespace runtime
}  // namespace dragnn
}  // namespace syntaxnet
