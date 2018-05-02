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

#ifndef DRAGNN_RUNTIME_SEQUENCE_BACKEND_H_
#define DRAGNN_RUNTIME_SEQUENCE_BACKEND_H_

#include <functional>
#include <vector>

#include "dragnn/core/input_batch_cache.h"
#include "dragnn/core/interfaces/component.h"
#include "dragnn/core/interfaces/transition_state.h"
#include "dragnn/core/util/label.h"
#include "dragnn/protos/data.pb.h"
#include "dragnn/protos/spec.pb.h"
#include "dragnn/protos/trace.pb.h"
#include "syntaxnet/base.h"

namespace syntaxnet {
namespace dragnn {
namespace runtime {

// Runtime-only component backend for sequence-based models.  This is not used
// at training time, and provides trivial implementations of most methods.  This
// is intended to be used with alternative feature extraction approaches, such
// as SequenceExtractor.
class SequenceBackend : public dragnn::Component {
 public:
  // Sets the size of the sequence in the current input.
  void SetSequenceSize(int size) { sequence_size_ = size; }

  // Implements dragnn::Component.
  std::function<int(int, int, int)> GetStepLookupFunction(
      const string &method) override;
  void InitializeComponent(const ComponentSpec &spec) override;
  void InitializeData(
      const std::vector<std::vector<const TransitionState *>> &parent_states,
      int max_beam_size, InputBatchCache *input_data) override;
  std::vector<std::vector<const TransitionState *>> GetBeam() override;
  int GetSourceBeamIndex(int current_index, int batch) const override;
  int GetBeamIndexAtStep(int step, int current_index, int batch) const override;
  std::vector<std::vector<ComponentTrace>> GetTraceProtos() const override;
  string Name() const override;
  int BeamSize() const override;
  int BatchSize() const override;
  bool IsReady() const override;
  bool IsTerminal() const override;
  void FinalizeData() override;
  void ResetComponent() override;
  void InitializeTracing() override;
  void DisableTracing() override;

  // Not implemented, crashes when called.
  int StepsTaken(int batch_index) const override;

  // Not implemented, crashes when called.
  bool AdvanceFromPrediction(const float *transition_matrix, int num_items,
                             int num_actions) override;

  // Not implemented, crashes when called.
  void AdvanceFromOracle() override;

  // Not implemented, crashes when called.
  std::vector<std::vector<std::vector<Label>>> GetOracleLabels() const override;

  // Not implemented, crashes when called.
  int GetFixedFeatures(std::function<int32 *(int)> allocate_indices,
                       std::function<int64 *(int)> allocate_ids,
                       std::function<float *(int)> allocate_weights,
                       int channel_id) const override;

  // Not implemented, crashes when called.
  int BulkGetFixedFeatures(const BulkFeatureExtractor &extractor) override;

  // Not implemented, crashes when called.
  void BulkEmbedFixedFeatures(
      int batch_size_padding, int num_steps_padding, int output_array_size,
      const vector<const float *> &per_channel_embeddings,
      float *embedding_output) override;

  // Not implemented, crashes when called.
  void BulkEmbedDenseFixedFeatures(
      const vector<const float *> &per_channel_embeddings,
      float *embedding_output, int embedding_output_size,
      int *offset_array_output, int offset_array_size) override;

  // Not implemented, crashes when called.
  int BulkDenseFeatureSize() const override;

  // Not implemented, crashes when called.
  std::vector<LinkFeatures> GetRawLinkFeatures(int channel_id) const override;

  // Not implemented, crashes when called.
  void AddTranslatedLinkFeaturesToTrace(
      const std::vector<LinkFeatures> &features, int channel_id) override;

 private:
  // Name of the component that this backend supports.
  string name_;

  // Size of the current input sequence.
  int sequence_size_ = 0;

  // Parent states passed to InitializeData(), and passed along in GetBeam().
  std::vector<std::vector<const TransitionState *>> parent_states_;
};

}  // namespace runtime
}  // namespace dragnn
}  // namespace syntaxnet

#endif  // DRAGNN_RUNTIME_SEQUENCE_BACKEND_H_
