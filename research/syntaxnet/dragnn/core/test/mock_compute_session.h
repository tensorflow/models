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

#ifndef DRAGNN_CORE_TEST_MOCK_COMPUTE_SESSION_H_
#define DRAGNN_CORE_TEST_MOCK_COMPUTE_SESSION_H_

#include <memory>

#include "dragnn/components/util/bulk_feature_extractor.h"
#include "dragnn/core/compute_session.h"
#include "dragnn/core/input_batch_cache.h"
#include "dragnn/protos/data.pb.h"
#include "dragnn/protos/spec.pb.h"
#include "syntaxnet/base.h"
#include <gmock/gmock.h>
#include "tensorflow/core/platform/test.h"

namespace syntaxnet {
namespace dragnn {

class MockComputeSession : public ComputeSession {
 public:
  MOCK_METHOD2(Init, void(const MasterSpec &master_spec,
                          const GridPoint &hyperparams));
  MOCK_METHOD2(InitializeComponentData,
               void(const string &component_name, int max_beam_size));
  MOCK_CONST_METHOD1(BatchSize, int(const string &component_name));
  MOCK_CONST_METHOD1(BeamSize, int(const string &component_name));
  MOCK_CONST_METHOD1(Spec, const ComponentSpec &(const string &component_name));
  MOCK_METHOD2(SourceComponentBeamSize,
               int(const string &component_name, int channel_id));
  MOCK_METHOD1(AdvanceFromOracle, void(const string &component_name));
  MOCK_METHOD4(AdvanceFromPrediction,
               bool(const string &component_name, const float *score_matrix,
                    int num_items, int num_actions));
  MOCK_CONST_METHOD5(GetInputFeatures,
                     int(const string &component_name,
                         std::function<int32 *(int)> allocate_indices,
                         std::function<int64 *(int)> allocate_ids,
                         std::function<float *(int)> allocate_weights,
                         int channel_id));
  MOCK_METHOD2(BulkGetInputFeatures,
               int(const string &component_name,
                   const BulkFeatureExtractor &extractor));
  MOCK_METHOD6(BulkEmbedFixedFeatures,
               void(const string &component_name, int batch_size_padding,
                    int num_steps_padding, int output_array_size,
                    const vector<const float *> &per_channel_embedding,
                    float *embedding_output));
  MOCK_METHOD2(GetTranslatedLinkFeatures,
               std::vector<LinkFeatures>(const string &component_name,
                                         int channel_id));
  MOCK_METHOD1(EmitOracleLabels,
               std::vector<std::vector<int>>(const string &component_name));
  MOCK_METHOD1(IsTerminal, bool(const string &component_name));
  MOCK_METHOD1(FinalizeData, void(const string &component_name));
  MOCK_METHOD0(GetSerializedPredictions, std::vector<string>());
  MOCK_METHOD0(GetTraceProtos, std::vector<MasterTrace>());
  MOCK_METHOD1(SetInputData, void(const std::vector<string> &data));
  MOCK_METHOD0(ResetSession, void());
  MOCK_METHOD1(SetTracing, void(bool tracing_on));
  MOCK_CONST_METHOD0(Id, int());
  MOCK_CONST_METHOD1(GetDescription, string(const string &component_name));
  MOCK_CONST_METHOD1(Translators, const std::vector<const IndexTranslator *>(
                                      const string &component_name));
  MOCK_CONST_METHOD1(GetReadiedComponent, Component *(const string &name));

  // TODO(googleuser): Upgrade gMock to a version that supports mocking methods
  // with move-only types, then remove this workaround.
  MOCK_METHOD1(DoSetInputBatchCache, void(InputBatchCache *batch));
  void SetInputBatchCache(std::unique_ptr<InputBatchCache> batch) override {
    DoSetInputBatchCache(batch.get());
  }
};

}  // namespace dragnn
}  // namespace syntaxnet

#endif  // DRAGNN_CORE_TEST_MOCK_COMPUTE_SESSION_H_
