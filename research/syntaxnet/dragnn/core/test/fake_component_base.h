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

#ifndef DRAGNN_CORE_TEST_FAKE_COMPONENT_BASE_H_
#define DRAGNN_CORE_TEST_FAKE_COMPONENT_BASE_H_

#include "dragnn/core/interfaces/component.h"
#include "dragnn/protos/data.pb.h"

namespace syntaxnet {
namespace dragnn {

// Define a test component to validate registered construction.
class FakeComponentBase : public Component {
 public:
  FakeComponentBase() {}
  void InitializeComponent(const ComponentSpec &spec) override {
    name_ = spec.name();
  }
  void InitializeData(
      const std::vector<std::vector<const TransitionState *>> &states,
      int max_beam_size, InputBatchCache *input_data) override {}
  void InitializeTracing() override {}
  void DisableTracing() override {}
  bool IsReady() const override { return true; }
  string Name() const override { return name_; }
  int BeamSize() const override { return 1; }
  int BatchSize() const override { return 1; }
  int StepsTaken(int batch_index) const override { return 0; }
  int GetBeamIndexAtStep(int step, int current_index,
                         int batch) const override {
    return 0;
  }
  int GetSourceBeamIndex(int current_index, int batch) const override {
    return 0;
  }
  bool AdvanceFromPrediction(const float *score_matrix, int num_items,
                             int num_actions) override {
    return true;
  }
  void AdvanceFromOracle() override {}
  bool IsTerminal() const override { return true; }
  std::function<int(int, int, int)> GetStepLookupFunction(
      const string &method) override {
    return nullptr;
  }
  std::vector<std::vector<const TransitionState *>> GetBeam() override {
    std::vector<std::vector<const TransitionState *>> states;
    return states;
  }
  int GetFixedFeatures(std::function<int32 *(int)> allocate_indices,
                       std::function<int64 *(int)> allocate_ids,
                       std::function<float *(int)> allocate_weights,
                       int channel_id) const override {
    return 0;
  }
  void BulkEmbedFixedFeatures(
      int batch_size_padding, int num_steps_padding, int embedding_size,
      const vector<const float *> &per_channel_embeddings,
      float *embedding_output) override {}
  void BulkEmbedDenseFixedFeatures(
      const vector<const float *> &per_channel_embeddings,
      float *embedding_output, int embedding_output_size,
      int *offset_array_output, int offset_array_size) override {}
  int BulkDenseFeatureSize() const override { return 0; }
  int BulkGetFixedFeatures(const BulkFeatureExtractor &extractor) override {
    return 0;
  }
  std::vector<LinkFeatures> GetRawLinkFeatures(int channel_id) const override {
    std::vector<LinkFeatures> ret;
    return ret;
  }
  std::vector<std::vector<std::vector<Label>>> GetOracleLabels()
      const override {
    std::vector<std::vector<std::vector<Label>>> ret;
    return ret;
  }
  void FinalizeData() override {}
  void ResetComponent() override {}

  std::vector<std::vector<ComponentTrace>> GetTraceProtos() const override {
    std::vector<std::vector<ComponentTrace>> ret;
    return ret;
  }
  void AddTranslatedLinkFeaturesToTrace(
      const std::vector<LinkFeatures> &features, int channel_id) override {}

  string name_;
};

}  // namespace dragnn
}  // namespace syntaxnet

#endif  // DRAGNN_CORE_TEST_FAKE_COMPONENT_BASE_H_
