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

#include "dragnn/runtime/head_selection_component_base.h"

#include <stddef.h>
#include <string>
#include <vector>

#include "dragnn/core/compute_session.h"
#include "dragnn/core/input_batch_cache.h"
#include "dragnn/core/test/generic.h"
#include "dragnn/protos/spec.pb.h"
#include "dragnn/protos/trace.pb.h"
#include "dragnn/runtime/network_states.h"
#include "dragnn/runtime/session_state.h"
#include "dragnn/runtime/test/network_test_base.h"
#include "dragnn/runtime/variable_store.h"
#include "syntaxnet/base.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"

namespace syntaxnet {
namespace dragnn {
namespace runtime {
namespace {

constexpr size_t kNumSteps = 12;
constexpr size_t kRootIndex = 7;  // the root and head of all other tokens

constexpr char kTestBuilder[] = "TestBuilder";
constexpr char kTestBackend[] = "TestBackend";
constexpr char kPreviousComponentName[] = "previous_component";
constexpr char kAdjacencyLayerName[] = "adjacency_layer";
constexpr char kBadDimLayerName[] = "bad_layer";

// A subclass for tests.
class BasicHeadSelectionComponent : public HeadSelectionComponentBase {
 public:
  BasicHeadSelectionComponent()
      : HeadSelectionComponentBase(kTestBuilder, kTestBackend) {}

  // Implements Component.  These methods are never called, but must be defined
  // so the class is not abstract.
  tensorflow::Status Evaluate(
      SessionState *session_state, ComputeSession *compute_session,
      ComponentTrace *component_trace) const override {
    return tensorflow::Status::OK();
  }

  // Publicizes the base class's method.
  using HeadSelectionComponentBase::ComputeHeads;
};

// Returns a ComponentSpec that works with the head selection component.
ComponentSpec MakeGoodSpec() {
  ComponentSpec component_spec;
  component_spec.mutable_component_builder()->set_registered_name(kTestBuilder);
  component_spec.mutable_backend()->set_registered_name(kTestBackend);
  component_spec.mutable_transition_system()->set_registered_name("heads");
  component_spec.mutable_network_unit()->set_registered_name("IdentityNetwork");
  LinkedFeatureChannel *link = component_spec.add_linked_feature();
  link->set_source_component(kPreviousComponentName);
  link->set_source_layer(kAdjacencyLayerName);
  return component_spec;
}

class HeadSelectionComponentBaseTest : public NetworkTestBase {
 protected:
  // Initializes a head selection component from the |component_spec| and sets
  // |heads| to the extracted head indices.  Returs non-OK on error.
  tensorflow::Status Run(const ComponentSpec &component_spec,
                         std::vector<int> *heads) {
    AddComponent(kPreviousComponentName);
    AddPairwiseLayer(kAdjacencyLayerName, 1);
    AddPairwiseLayer(kBadDimLayerName, 2);

    BasicHeadSelectionComponent component;
    TF_RETURN_IF_ERROR(component.Initialize(component_spec, &variable_store_,
                                            &network_state_manager_,
                                            &extension_manager_));

    network_states_.Reset(&network_state_manager_);
    StartComponent(kNumSteps);

    // Fill the |kRootIndex|'th column of the adjacency matrix with higher
    // scores, so all tokens select it as head.  The |kRootIndex|'th token
    // itself is a self-loop, so it becomes a root.
    MutableMatrix<float> adjacency =
        GetPairwiseLayer(kPreviousComponentName, kAdjacencyLayerName);
    for (size_t target = 0; target < kNumSteps; ++target) {
      for (size_t source = 0; source < kNumSteps; ++source) {
        adjacency.row(target)[source] = source == kRootIndex ? 1.0 : 0.0;
      }
    }

    session_state_.extensions.Reset(&extension_manager_);
    *heads = component.ComputeHeads(&session_state_);

    return tensorflow::Status::OK();
  }
};

// Tests that the expected heads are produced for a good spec.
TEST_F(HeadSelectionComponentBaseTest, RunsGoodSpec) {
  std::vector<int> heads;
  TF_ASSERT_OK(Run(MakeGoodSpec(), &heads));

  std::vector<int> expected_heads(kNumSteps, kRootIndex);
  expected_heads[kRootIndex] = -1;
  EXPECT_EQ(heads, expected_heads);
}

// Tests that a layer with the wrong dimension is rejected
TEST_F(HeadSelectionComponentBaseTest, WrongDimension) {
  ComponentSpec component_spec = MakeGoodSpec();
  component_spec.mutable_linked_feature(0)->set_source_layer(kBadDimLayerName);

  std::vector<int> heads;
  EXPECT_THAT(Run(component_spec, &heads),
              test::IsErrorWithSubstr(
                  "Adjacency matrix has dimension 2 but expected 1"));
}

// Tests that the component is always dis-preferred.
TEST_F(HeadSelectionComponentBaseTest, NotPreferred) {
  BasicHeadSelectionComponent component;
  EXPECT_FALSE(component.PreferredTo(component));
}

// Tests that the good spec is supported.
TEST_F(HeadSelectionComponentBaseTest, SupportsGoodSpec) {
  ComponentSpec component_spec = MakeGoodSpec();

  BasicHeadSelectionComponent component;
  EXPECT_TRUE(component.Supports(component_spec, kTestBuilder));
}

// Tests that various bad specs are rejected.
TEST_F(HeadSelectionComponentBaseTest, RejectsBadSpecs) {
  ComponentSpec component_spec = MakeGoodSpec();
  BasicHeadSelectionComponent component;
  EXPECT_FALSE(component.Supports(component_spec, "bad"));

  component_spec = MakeGoodSpec();
  component_spec.mutable_backend()->set_registered_name("bad");
  EXPECT_FALSE(component.Supports(component_spec, kTestBuilder));

  component_spec = MakeGoodSpec();
  component_spec.mutable_transition_system()->set_registered_name("bad");
  EXPECT_FALSE(component.Supports(component_spec, kTestBuilder));

  component_spec = MakeGoodSpec();
  component_spec.mutable_network_unit()->set_registered_name("bad");
  EXPECT_FALSE(component.Supports(component_spec, kTestBuilder));

  component_spec = MakeGoodSpec();
  component_spec.add_fixed_feature();
  EXPECT_FALSE(component.Supports(component_spec, kTestBuilder));

  component_spec = MakeGoodSpec();
  component_spec.add_linked_feature();
  EXPECT_FALSE(component.Supports(component_spec, kTestBuilder));

  component_spec = MakeGoodSpec();
  component_spec.clear_linked_feature();
  EXPECT_FALSE(component.Supports(component_spec, kTestBuilder));
}

}  // namespace
}  // namespace runtime
}  // namespace dragnn
}  // namespace syntaxnet
