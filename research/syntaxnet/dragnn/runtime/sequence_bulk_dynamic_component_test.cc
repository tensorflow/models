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

#include <stddef.h>
#include <memory>
#include <string>
#include <vector>

#include "dragnn/core/input_batch_cache.h"
#include "dragnn/core/test/generic.h"
#include "dragnn/protos/spec.pb.h"
#include "dragnn/runtime/bulk_network_unit.h"
#include "dragnn/runtime/component.h"
#include "dragnn/runtime/extensions.h"
#include "dragnn/runtime/math/types.h"
#include "dragnn/runtime/network_states.h"
#include "dragnn/runtime/sequence_backend.h"
#include "dragnn/runtime/sequence_extractor.h"
#include "dragnn/runtime/sequence_linker.h"
#include "dragnn/runtime/sequence_predictor.h"
#include "dragnn/runtime/test/network_test_base.h"
#include "dragnn/runtime/variable_store.h"
#include "syntaxnet/base.h"
#include <gmock/gmock.h>
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"

namespace syntaxnet {
namespace dragnn {
namespace runtime {
namespace {

using ::testing::Return;

constexpr size_t kNumSteps = 50;
constexpr size_t kFixedDim = 11;
constexpr size_t kFixedVocabularySize = 123;
constexpr float kFixedValue = 0.5;
constexpr size_t kLinkedDim = 13;
constexpr float kLinkedValue = 1.25;
constexpr char kPreviousComponentName[] = "previous_component";
constexpr char kPreviousLayerName[] = "previous_layer";
constexpr char kLogitsName[] = "logits";
constexpr size_t kLogitsDim = kFixedDim + kLinkedDim;

// Adds one to all inputs.
class BulkAddOne : public BulkNetworkUnit {
 public:
  // Implements BulkNetworkUnit.
  tensorflow::Status Initialize(const ComponentSpec &component_spec,
                                VariableStore *variable_store,
                                NetworkStateManager *network_state_manager,
                                ExtensionManager *extension_manager) override {
    return network_state_manager->AddLayer(kLogitsName, kLogitsDim,
                                           &logits_handle_);
  }
  tensorflow::Status ValidateInputDimension(size_t dimension) const override {
    return tensorflow::Status::OK();
  }
  string GetLogitsName() const override { return kLogitsName; }
  tensorflow::Status Evaluate(Matrix<float> inputs,
                              SessionState *session_state) const override {
    const MutableMatrix<float> logits =
        session_state->network_states.GetLayer(logits_handle_);
    for (size_t row = 0; row < inputs.num_rows(); ++row) {
      for (size_t column = 0; column < inputs.num_columns(); ++column) {
        logits.row(row)[column] = inputs.row(row)[column] + 1.0;
      }
    }
    return tensorflow::Status::OK();
  }

 private:
  // Output logits.
  LayerHandle<float> logits_handle_;
};

DRAGNN_RUNTIME_REGISTER_BULK_NETWORK_UNIT(BulkAddOne);

// A component that also prefers other but is triggered on the presence of a
// resource.  This can be used to cause a component selection conflict.
class ImTheWorst : public Component {
 public:
  // Implements Component.
  tensorflow::Status Initialize(const ComponentSpec &component_spec,
                                VariableStore *variable_store,
                                NetworkStateManager *network_state_manager,
                                ExtensionManager *extension_manager) override {
    return tensorflow::Status::OK();
  }
  tensorflow::Status Evaluate(SessionState *session_state,
                              ComputeSession *compute_session,
                              ComponentTrace *component_trace) const override {
    return tensorflow::Status::OK();
  }
  bool Supports(const ComponentSpec &component_spec,
                const string &normalized_builder_name) const override {
    return component_spec.resource_size() > 0;
  }
  bool PreferredTo(const Component &other) const override { return false; }
};

DRAGNN_RUNTIME_REGISTER_COMPONENT(ImTheWorst);

// Extractor that produces a sequence of zeros.
class ExtractZeros : public SequenceExtractor {
 public:
  // Implements SequenceExtractor.
  bool Supports(const FixedFeatureChannel &,
                const ComponentSpec &) const override {
    return true;
  }
  tensorflow::Status Initialize(const FixedFeatureChannel &,
                                const ComponentSpec &) override {
    return tensorflow::Status::OK();
  }
  tensorflow::Status GetIds(InputBatchCache *,
                            std::vector<int32> *ids) const override {
    ids->assign(kNumSteps, 0);
    return tensorflow::Status::OK();
  }
};

DRAGNN_RUNTIME_REGISTER_SEQUENCE_EXTRACTOR(ExtractZeros);

// Linker that produces a sequence of zeros.
class LinkZeros : public SequenceLinker {
 public:
  // Implements SequenceLinker.
  bool Supports(const LinkedFeatureChannel &,
                const ComponentSpec &) const override {
    return true;
  }
  tensorflow::Status Initialize(const LinkedFeatureChannel &,
                                const ComponentSpec &) override {
    return tensorflow::Status::OK();
  }
  tensorflow::Status GetLinks(size_t, InputBatchCache *,
                              std::vector<int32> *links) const override {
    links->assign(kNumSteps, 0);
    return tensorflow::Status::OK();
  }
};

DRAGNN_RUNTIME_REGISTER_SEQUENCE_LINKER(LinkZeros);

// Predictor that captures the logits.
class CaptureLogits : public SequencePredictor {
 public:
  // Implements SequencePredictor.
  bool Supports(const ComponentSpec &) const override { return true; }
  tensorflow::Status Initialize(const ComponentSpec &) override {
    return tensorflow::Status::OK();
  }
  tensorflow::Status Predict(Matrix<float> logits,
                             InputBatchCache *) const override {
    logits_ = logits;
    return tensorflow::Status::OK();
  }

  // Returns the captured logits.
  static Matrix<float> GetCapturedLogits() { return logits_; }

 private:
  // Logits from the most recent call to Predict().
  static Matrix<float> logits_;
};

Matrix<float> CaptureLogits::logits_;

DRAGNN_RUNTIME_REGISTER_SEQUENCE_PREDICTOR(CaptureLogits);

class SequenceBulkDynamicComponentTest : public NetworkTestBase {
 protected:
  SequenceBulkDynamicComponentTest() {
    EXPECT_CALL(compute_session_, GetInputBatchCache())
        .WillRepeatedly(Return(&input_));
    EXPECT_CALL(compute_session_, GetReadiedComponent(kTestComponentName))
        .WillRepeatedly(Return(&backend_));
  }

  // Returns a spec that the network supports.
  ComponentSpec GetSupportedSpec() {
    ComponentSpec component_spec;
    component_spec.set_name(kTestComponentName);
    component_spec.set_num_actions(kLogitsDim);

    component_spec.mutable_network_unit()->set_registered_name("AddOne");
    component_spec.mutable_backend()->set_registered_name("SequenceBackend");
    component_spec.mutable_component_builder()->set_registered_name(
        "SequenceBulkDynamicComponent");

    auto &component_parameters =
        *component_spec.mutable_component_builder()->mutable_parameters();
    component_parameters["sequence_extractors"] = "ExtractZeros";
    component_parameters["sequence_linkers"] = "LinkZeros";
    component_parameters["sequence_predictor"] = "CaptureLogits";

    FixedFeatureChannel *fixed_feature = component_spec.add_fixed_feature();
    fixed_feature->set_size(1);
    fixed_feature->set_embedding_dim(kFixedDim);
    fixed_feature->set_vocabulary_size(kFixedVocabularySize);

    LinkedFeatureChannel *linked_feature = component_spec.add_linked_feature();
    linked_feature->set_size(1);
    linked_feature->set_embedding_dim(-1);
    linked_feature->set_source_component(kPreviousComponentName);
    linked_feature->set_source_layer(kPreviousLayerName);

    return component_spec;
  }

  // Creates a network unit, initializes it based on the |component_spec_text|,
  // and evaluates it.  On error, returns non-OK.
  tensorflow::Status Run(const ComponentSpec &component_spec) {
    AddComponent(kPreviousComponentName);
    AddLayer(kPreviousLayerName, kLinkedDim);
    AddComponent(kTestComponentName);
    AddFixedEmbeddingMatrix(0, kFixedVocabularySize, kFixedDim, kFixedValue);

    std::unique_ptr<Component> component;
    TF_RETURN_IF_ERROR(
        Component::CreateOrError("SequenceBulkDynamicComponent", &component));
    TF_RETURN_IF_ERROR(component->Initialize(component_spec, &variable_store_,
                                             &network_state_manager_,
                                             &extension_manager_));

    // Allocates network states for a few steps.
    network_states_.Reset(&network_state_manager_);
    StartComponent(kNumSteps);
    FillLayer(kPreviousComponentName, kPreviousLayerName, kLinkedValue);
    StartComponent(0);
    session_state_.extensions.Reset(&extension_manager_);

    return component->Evaluate(&session_state_, &compute_session_, nullptr);
  }

  // Input batch injected into Evaluate() by default.
  InputBatchCache input_;

  // Backend injected into Evaluate().
  SequenceBackend backend_;
};

// Tests that the supported spec is supported.
TEST_F(SequenceBulkDynamicComponentTest, Supported) {
  const ComponentSpec component_spec = GetSupportedSpec();

  string component_type;
  TF_ASSERT_OK(Component::Select(component_spec, &component_type));
  EXPECT_EQ(component_type, "SequenceBulkDynamicComponent");

  TF_ASSERT_OK(Run(component_spec));

  const Matrix<float> logits = CaptureLogits::GetCapturedLogits();
  ASSERT_EQ(logits.num_rows(), kNumSteps);
  ASSERT_EQ(logits.num_columns(), kFixedDim + kLinkedDim);

  for (size_t row = 0; row < kNumSteps; ++row) {
    size_t column = 0;
    for (; column < kFixedDim; ++column) {
      EXPECT_EQ(logits.row(row)[column], kFixedValue + 1.0);
    }
    for (; column < kFixedDim + kLinkedDim; ++column) {
      EXPECT_EQ(logits.row(row)[column], kLinkedValue + 1.0);
    }
  }
}

// Tests that links cannot be recurrent.
TEST_F(SequenceBulkDynamicComponentTest, ForbidRecurrences) {
  ComponentSpec component_spec = GetSupportedSpec();
  component_spec.mutable_linked_feature(0)->set_source_component(
      kTestComponentName);

  string component_type;
  EXPECT_THAT(
      Component::Select(component_spec, &component_type),
      test::IsErrorWithSubstr("Could not find a best spec for component"));
}

// Tests that the component prefers others.
TEST_F(SequenceBulkDynamicComponentTest, PrefersOthers) {
  ComponentSpec component_spec = GetSupportedSpec();
  component_spec.add_resource();

  // Adding a resource triggers the ImTheWorst component, which also prefers
  // itself and leads to a selection conflict.
  string component_type;
  EXPECT_THAT(
      Component::Select(component_spec, &component_type),
      test::IsErrorWithSubstr("both think they should be dis-preferred"));
}

}  // namespace
}  // namespace runtime
}  // namespace dragnn
}  // namespace syntaxnet
