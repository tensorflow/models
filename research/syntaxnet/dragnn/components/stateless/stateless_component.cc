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

#include "dragnn/core/component_registry.h"
#include "dragnn/core/interfaces/component.h"
#include "dragnn/core/interfaces/transition_state.h"
#include "dragnn/protos/data.pb.h"
#include "syntaxnet/base.h"

namespace syntaxnet {
namespace dragnn {
namespace {

// A component that does not create its own transition states; instead, it
// simply forwards the states of the previous component.  Requires that some
// previous component has converted the input batch.  Does not support all
// methods.  Intended for "compute-only" bulk components that only use linked
// features, which use only a small subset of DRAGNN functionality.
class StatelessComponent : public Component {
 public:
  void InitializeComponent(const ComponentSpec &spec) override {
    name_ = spec.name();
  }

  // Stores the |parent_states| for forwarding to downstream components.
  void InitializeData(
      const std::vector<std::vector<const TransitionState *>> &parent_states,
      int max_beam_size, InputBatchCache *input_data) override {
    batch_size_ = input_data->Size();
    beam_size_ = max_beam_size;
    parent_states_ = parent_states;

    // The beam should be wide enough for the previous component.
    for (const auto &beam : parent_states) {
      CHECK_LE(beam.size(), beam_size_);
    }
  }

  // Forwards the states of the previous component.
  std::vector<std::vector<const TransitionState *>> GetBeam() override {
    return parent_states_;
  }

  // Forwards the |current_index| to the previous component.
  int GetSourceBeamIndex(int current_index, int batch) const override {
    return current_index;
  }

  string Name() const override { return name_; }
  int BeamSize() const override { return beam_size_; }
  int BatchSize() const override { return batch_size_; }
  int StepsTaken(int batch_index) const override { return 0; }
  bool IsReady() const override { return true; }
  bool IsTerminal() const override { return true; }
  void FinalizeData() override {}
  void ResetComponent() override {}
  void InitializeTracing() override {}
  void DisableTracing() override {}
  std::vector<std::vector<ComponentTrace>> GetTraceProtos() const override {
    return {};
  }

  // Unsupported methods.
  int GetBeamIndexAtStep(int step, int current_index,
                         int batch) const override {
    LOG(FATAL) << "[" << name_ << "] Method not supported";
    return 0;
  }
  std::function<int(int, int, int)> GetStepLookupFunction(
      const string &method) override {
    LOG(FATAL) << "[" << name_ << "] Method not supported";
    return nullptr;
  }
  bool AdvanceFromPrediction(const float *transition_matrix, int num_items,
                             int num_actions) override {
    LOG(FATAL) << "[" << name_ << "] AdvanceFromPrediction not supported";
  }
  void AdvanceFromOracle() override {
    LOG(FATAL) << "[" << name_ << "] AdvanceFromOracle not supported";
  }
  std::vector<std::vector<int>> GetOracleLabels() const override {
    LOG(FATAL) << "[" << name_ << "] Method not supported";
  }
  int GetFixedFeatures(std::function<int32 *(int)> allocate_indices,
                       std::function<int64 *(int)> allocate_ids,
                       std::function<float *(int)> allocate_weights,
                       int channel_id) const override {
    LOG(FATAL) << "[" << name_ << "] Method not supported";
  }
  int BulkGetFixedFeatures(const BulkFeatureExtractor &extractor) override {
    LOG(FATAL) << "[" << name_ << "] Method not supported";
  }
  void BulkEmbedFixedFeatures(
      int batch_size_padding, int num_steps_padding, int output_array_size,
      const vector<const float *> &per_channel_embeddings,
      float *embedding_output) override {
    LOG(FATAL) << "[" << name_ << "] Method not supported";
  }

  std::vector<LinkFeatures> GetRawLinkFeatures(int channel_id) const override {
    LOG(FATAL) << "[" << name_ << "] Method not supported";
  }
  void AddTranslatedLinkFeaturesToTrace(
      const std::vector<LinkFeatures> &features, int channel_id) override {
    LOG(FATAL) << "[" << name_ << "] Method not supported";
  }

 private:
  string name_;  // component name
  int batch_size_ = 1;  // number of sentences in current batch
  int beam_size_ = 1;  // maximum beam size

  // Parent states passed to InitializeData(), and passed along in GetBeam().
  std::vector<std::vector<const TransitionState *>> parent_states_;
};

REGISTER_DRAGNN_COMPONENT(StatelessComponent);

}  // namespace
}  // namespace dragnn
}  // namespace syntaxnet
