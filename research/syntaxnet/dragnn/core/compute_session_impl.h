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

#ifndef DRAGNN_CORE_COMPUTE_SESSION_IMPL_H_
#define DRAGNN_CORE_COMPUTE_SESSION_IMPL_H_

#include <memory>

#include "dragnn/components/util/bulk_feature_extractor.h"
#include "dragnn/core/compute_session.h"
#include "dragnn/core/index_translator.h"
#include "dragnn/core/input_batch_cache.h"
#include "dragnn/protos/data.pb.h"
#include "dragnn/protos/spec.pb.h"
#include "dragnn/protos/trace.pb.h"

namespace syntaxnet {
namespace dragnn {

class ComputeSessionImpl : public ComputeSession {
 public:
  // Creates a ComputeSessionImpl with the provided component builder function.
  ComputeSessionImpl(
      int id,
      std::function<std::unique_ptr<Component>(const string &component_name,
                                               const string &backend_type)>
          component_builder);

  void Init(const MasterSpec &master_spec,
            const GridPoint &hyperparams) override;

  void InitializeComponentData(const string &component_name,
                               int max_beam_size) override;

  int BatchSize(const string &component_name) const override;

  int BeamSize(const string &component_name) const override;

  const ComponentSpec &Spec(const string &component_name) const override;

  int SourceComponentBeamSize(const string &component_name,
                              int channel_id) override;

  void AdvanceFromOracle(const string &component_name) override;

  bool AdvanceFromPrediction(const string &component_name,
                             const float *score_matrix, int num_items,
                             int num_actions) override;

  int GetInputFeatures(const string &component_name,
                       std::function<int32 *(int)> allocate_indices,
                       std::function<int64 *(int)> allocate_ids,
                       std::function<float *(int)> allocate_weights,
                       int channel_id) const override;

  int BulkGetInputFeatures(const string &component_name,
                           const BulkFeatureExtractor &extractor) override;

  void BulkEmbedFixedFeatures(
      const string &component_name, int batch_size_padding,
      int num_steps_padding, int output_array_size,
      const vector<const float *> &per_channel_embeddings,
      float *embedding_output) override;

  std::vector<LinkFeatures> GetTranslatedLinkFeatures(
      const string &component_name, int channel_id) override;

  std::vector<std::vector<int>> EmitOracleLabels(
      const string &component_name) override;

  bool IsTerminal(const string &component_name) override;

  void FinalizeData(const string &component_name) override;

  std::vector<string> GetSerializedPredictions() override;

  std::vector<MasterTrace> GetTraceProtos() override;

  void SetInputData(const std::vector<string> &data) override;

  void SetInputBatchCache(std::unique_ptr<InputBatchCache> batch) override;

  void ResetSession() override;

  void SetTracing(bool tracing_on) override;

  int Id() const override;

  string GetDescription(const string &component_name) const override;

  const std::vector<const IndexTranslator *> Translators(
      const string &component_name) const override;

  // Get a given component. CHECK-fail if the component's IsReady method
  // returns false.
  Component *GetReadiedComponent(const string &component_name) const override;

 private:
  // Get a given component. Fails if the component is not found.
  Component *GetComponent(const string &component_name) const;

  // Get the index translators for the given component.
  const std::vector<IndexTranslator *> &GetTranslators(
      const string &component_name) const;

  // Create an index translator.
  std::unique_ptr<IndexTranslator> CreateTranslator(
      const LinkedFeatureChannel &channel, Component *start_component);

  // Perform initialization on the given Component.
  void InitComponent(Component *component);

  // Holds all of the components owned by this ComputeSession, associated with
  // their names in the MasterSpec.
  std::map<string, std::unique_ptr<Component>> components_;

  // Holds a vector of translators for each component, indexed by the name
  // of the component they belong to.
  std::map<string, std::vector<IndexTranslator *>> translators_;

  // Holds ownership of all the IndexTranslators for this compute session.
  std::vector<std::unique_ptr<IndexTranslator>> owned_translators_;

  // The predecessor component for every component.
  // If a component is not in this map, it has no predecessor component and
  // will have its beam initialized without any data from other components.
  std::map<Component *, Component *> predecessors_;

  // Holds the current input data for this ComputeSession.
  std::unique_ptr<InputBatchCache> input_data_;

  // Function that, given a string, will return a Component.
  std::function<std::unique_ptr<Component>(const string &component_name,
                                           const string &backend_type)>
      component_builder_;

  // The master spec for this compute session.
  MasterSpec spec_;

  // The hyperparameters for this compute session.
  GridPoint grid_point_;

  // Unique identifier, assigned at construction.
  int id_;

  // Whether or not to perform tracing.
  bool do_tracing_ = false;
};

}  // namespace dragnn
}  // namespace syntaxnet

#endif  // DRAGNN_CORE_COMPUTE_SESSION_IMPL_H_
