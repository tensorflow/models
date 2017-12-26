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

#ifndef DRAGNN_CORE_INTERFACES_COMPONENT_H_
#define DRAGNN_CORE_INTERFACES_COMPONENT_H_

#include <vector>

#include "dragnn/components/util/bulk_feature_extractor.h"
#include "dragnn/core/input_batch_cache.h"
#include "dragnn/core/interfaces/transition_state.h"
#include "dragnn/protos/spec.pb.h"
#include "dragnn/protos/trace.pb.h"
#include "syntaxnet/registry.h"

namespace syntaxnet {
namespace dragnn {

class Component : public RegisterableClass<Component> {
 public:
  virtual ~Component() {}

  // Initializes this component from the spec.
  virtual void InitializeComponent(const ComponentSpec &spec) = 0;

  // Provides the previous beam to the component.
  virtual void InitializeData(
      const std::vector<std::vector<const TransitionState *>> &states,
      int max_beam_size, InputBatchCache *input_data) = 0;

  // Returns true if the component has had InitializeData called on it since
  // the last time it was reset.
  virtual bool IsReady() const = 0;

  // Initializes the component for tracing execution, resetting any existing
  // traces. This will typically have the side effect of slowing down all
  // subsequent Component calculations and storing a trace in memory that can be
  // returned by GetTraceProtos().
  virtual void InitializeTracing() = 0;

  // Disables tracing, freeing any associated traces and avoiding triggering
  // additional computation in the future.
  virtual void DisableTracing() = 0;

  // Returns the string name of this component.
  virtual string Name() const = 0;

  // Returns the current batch size of the component's underlying data.
  virtual int BatchSize() const = 0;

  // Returns the maximum beam size of this component.
  virtual int BeamSize() const = 0;

  // Returns the number of steps taken by this component so far.
  virtual int StepsTaken(int batch_index) const = 0;

  // Return the beam index of the item which is currently at index
  // 'index', when the beam was at step 'step', for batch element 'batch'.
  virtual int GetBeamIndexAtStep(int step, int current_index,
                                 int batch) const = 0;

  // Return the source index of the item which is currently at index 'index'
  // for batch element 'batch'. This index is into the final beam of the
  // Component that this Component was initialized from.
  virtual int GetSourceBeamIndex(int current_index, int batch) const = 0;

  // Request a translation function based on the given method string.
  // The translation function will be called with arguments (beam, batch, value)
  // and should return the step index corresponding to the given value, for the
  // data in the given beam and batch.
  virtual std::function<int(int, int, int)> GetStepLookupFunction(
      const string &method) = 0;

  // Advances this component from the given transition matrix, which is
  // |num_items| x |num_actions|.
  virtual bool AdvanceFromPrediction(const float *score_matrix, int num_items,
                                     int num_actions) = 0;

  // Advances this component from the state oracles. There is no return from
  // this, since it should always succeed.
  virtual void AdvanceFromOracle() = 0;

  // Returns true if all states within this component are terminal.
  virtual bool IsTerminal() const = 0;

  // Returns the current batch of beams for this component.
  virtual std::vector<std::vector<const TransitionState *>> GetBeam() = 0;

  // Extracts and populates the vector of FixedFeatures for the specified
  // channel. Each functor allocates storage space for the indices, the IDs, and
  // the weights (respectively).
  virtual int GetFixedFeatures(
      std::function<int32 *(int num_elements)> allocate_indices,
      std::function<int64 *(int num_elements)> allocate_ids,
      std::function<float *(int num_elements)> allocate_weights,
      int channel_id) const = 0;

  // Extracts and populates all FixedFeatures for all channels, advancing this
  // component via the oracle until it is terminal. This call uses a
  // BulkFeatureExtractor object to contain the functors and other information.
  virtual int BulkGetFixedFeatures(const BulkFeatureExtractor &extractor) = 0;

  // Directly computes the embedding matrix for all channels, advancing the
  // component via the oracle until it is terminal. This call takes a vector
  // of EmbeddingMatrix structs, one per channel, in channel order.
  virtual void BulkEmbedFixedFeatures(
      int batch_size_padding, int num_steps_padding, int output_array_size,
      const vector<const float *> &per_channel_embeddings,
      float *embedding_output) = 0;

  // Extracts and returns the vector of LinkFeatures for the specified
  // channel. Note: these are NOT translated.
  virtual std::vector<LinkFeatures> GetRawLinkFeatures(
      int channel_id) const = 0;

  // Returns a vector of oracle labels for each element in the beam and
  // batch.
  virtual std::vector<std::vector<int>> GetOracleLabels() const = 0;

  // Annotate the underlying data object with the results of this Component's
  // calculation.
  virtual void FinalizeData() = 0;

  // Reset this component.
  virtual void ResetComponent() = 0;

  // Get a vector of all traces managed by this component.
  virtual std::vector<std::vector<ComponentTrace>> GetTraceProtos() const = 0;

  // Add the translated link features (done outside the component) to the traces
  // managed by this component.
  virtual void AddTranslatedLinkFeaturesToTrace(
      const std::vector<LinkFeatures> &features, int channel_id) = 0;
};

}  // namespace dragnn
}  // namespace syntaxnet

#endif  // DRAGNN_CORE_INTERFACES_COMPONENT_H_
