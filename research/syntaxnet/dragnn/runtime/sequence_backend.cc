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

#include "dragnn/runtime/sequence_backend.h"

#include "dragnn/core/component_registry.h"
#include "tensorflow/core/platform/logging.h"

namespace syntaxnet {
namespace dragnn {
namespace runtime {

std::function<int(int, int, int)> SequenceBackend::GetStepLookupFunction(
    const string &method) {
  if (method == "reverse-char" || method == "reverse-token") {
    // Reverses the |index| in the sequence.  We are agnostic to whether the
    // input is a sequence of tokens or chars.
    return [this](int unused_batch_index, int unused_beam_index, int index) {
      index = sequence_size_ - index - 1;
      return index >= 0 && index < sequence_size_ ? index : -1;
    };
  }

  LOG(FATAL) << "[" << name_ << "] Unknown step lookup function: " << method;
}

void SequenceBackend::InitializeComponent(const ComponentSpec &spec) {
  name_ = spec.name();
}

void SequenceBackend::InitializeData(
    const std::vector<std::vector<const TransitionState *>> &parent_states,
    int max_beam_size, InputBatchCache *input_data) {
  // Store the |parent_states| for forwarding to downstream components.
  parent_states_ = parent_states;
}

std::vector<std::vector<const TransitionState *>> SequenceBackend::GetBeam() {
  // Forward the states of the previous component.
  return parent_states_;
}

int SequenceBackend::GetSourceBeamIndex(int current_index, int batch) const {
  // Forward the |current_index| to the previous component.
  return current_index;
}

int SequenceBackend::GetBeamIndexAtStep(int step, int current_index,
                                        int batch) const {
  // Always return 0 since there is only one beam.
  return 0;
}

std::vector<std::vector<ComponentTrace>> SequenceBackend::GetTraceProtos()
    const {
  // Return a single trace, since the beam and batch sizes are fixed at 1.
  return {{ComponentTrace()}};
}

string SequenceBackend::Name() const { return name_; }

int SequenceBackend::BeamSize() const { return 1; }

int SequenceBackend::BatchSize() const { return 1; }

bool SequenceBackend::IsReady() const { return true; }

bool SequenceBackend::IsTerminal() const { return true; }

void SequenceBackend::FinalizeData() {}

void SequenceBackend::ResetComponent() {}

void SequenceBackend::InitializeTracing() {}

void SequenceBackend::DisableTracing() {}

int SequenceBackend::StepsTaken(int batch_index) const {
  LOG(FATAL) << "[" << name_ << "] Not supported";
}

bool SequenceBackend::AdvanceFromPrediction(const float *transition_matrix,
                                            int num_items, int num_actions) {
  LOG(FATAL) << "[" << name_ << "] Not supported";
}

void SequenceBackend::AdvanceFromOracle() {
  LOG(FATAL) << "[" << name_ << "] Not supported";
}

std::vector<std::vector<std::vector<Label>>> SequenceBackend::GetOracleLabels()
    const {
  LOG(FATAL) << "[" << name_ << "] Not supported";
}

int SequenceBackend::GetFixedFeatures(
    std::function<int32 *(int)> allocate_indices,
    std::function<int64 *(int)> allocate_ids,
    std::function<float *(int)> allocate_weights, int channel_id) const {
  LOG(FATAL) << "[" << name_ << "] Not supported";
}

int SequenceBackend::BulkGetFixedFeatures(
    const BulkFeatureExtractor &extractor) {
  LOG(FATAL) << "[" << name_ << "] Not supported";
}

void SequenceBackend::BulkEmbedFixedFeatures(
    int batch_size_padding, int num_steps_padding, int output_array_size,
    const vector<const float *> &per_channel_embeddings,
    float *embedding_output) {
  LOG(FATAL) << "[" << name_ << "] Not supported";
}

void SequenceBackend::BulkEmbedDenseFixedFeatures(
    const vector<const float *> &per_channel_embeddings,
    float *embedding_output, int embedding_output_size,
    int *offset_array_output, int offset_array_size) {
  LOG(FATAL) << "[" << name_ << "] Not supported";
}

int SequenceBackend::BulkDenseFeatureSize() const {
  LOG(FATAL) << "[" << name_ << "] Not supported";
}

std::vector<LinkFeatures> SequenceBackend::GetRawLinkFeatures(
    int channel_id) const {
  LOG(FATAL) << "[" << name_ << "] Not supported";
}

void SequenceBackend::AddTranslatedLinkFeaturesToTrace(
    const std::vector<LinkFeatures> &features, int channel_id) {
  LOG(FATAL) << "[" << name_ << "] Not supported";
}

REGISTER_DRAGNN_COMPONENT(SequenceBackend);

}  // namespace runtime
}  // namespace dragnn
}  // namespace syntaxnet
