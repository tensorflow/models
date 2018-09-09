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

#ifndef DRAGNN_CORE_TEST_MOCK_COMPONENT_H_
#define DRAGNN_CORE_TEST_MOCK_COMPONENT_H_

#include <gmock/gmock.h>

#include "dragnn/components/util/bulk_feature_extractor.h"
#include "dragnn/core/index_translator.h"
#include "dragnn/core/interfaces/component.h"
#include "dragnn/core/interfaces/transition_state.h"
#include "dragnn/core/util/label.h"
#include "dragnn/protos/data.pb.h"
#include "dragnn/protos/spec.pb.h"
#include "syntaxnet/base.h"
#include "tensorflow/core/platform/test.h"

namespace syntaxnet {
namespace dragnn {

class MockComponent : public Component {
 public:
  MOCK_METHOD1(InitializeComponent, void(const ComponentSpec &spec));
  MOCK_METHOD3(
      InitializeData,
      void(const std::vector<std::vector<const TransitionState *>> &states,
           int max_beam_size, InputBatchCache *input_data));
  MOCK_CONST_METHOD0(IsReady, bool());
  MOCK_METHOD0(InitializeTracing, void());
  MOCK_METHOD0(DisableTracing, void());
  MOCK_CONST_METHOD0(Name, string());
  MOCK_CONST_METHOD0(BatchSize, int());
  MOCK_CONST_METHOD0(BeamSize, int());
  MOCK_CONST_METHOD1(StepsTaken, int(int batch_index));
  MOCK_CONST_METHOD3(GetBeamIndexAtStep,
                     int(int step, int current_index, int batch));
  MOCK_CONST_METHOD2(GetSourceBeamIndex, int(int current_index, int batch));
  MOCK_METHOD3(AdvanceFromPrediction, bool(const float *transition_matrix,
                                           int num_items, int num_actions));
  MOCK_METHOD0(AdvanceFromOracle, void());
  MOCK_CONST_METHOD0(IsTerminal, bool());
  MOCK_METHOD0(GetBeam, std::vector<std::vector<const TransitionState *>>());
  MOCK_CONST_METHOD4(GetFixedFeatures,
                     int(std::function<int32 *(int)> allocate_indices,
                         std::function<int64 *(int)> allocate_ids,
                         std::function<float *(int)> allocate_weights,
                         int channel_id));
  MOCK_METHOD1(BulkGetFixedFeatures,
               int(const BulkFeatureExtractor &extractor));
  MOCK_METHOD5(BulkEmbedFixedFeatures,
               void(int batch_size_padding, int num_steps_padding,
                    int output_array_size,
                    const vector<const float *> &per_channel_embeddings,
                    float *embedding_output));
  MOCK_METHOD5(BulkEmbedDenseFixedFeatures,
               void(const vector<const float *> &per_channel_embeddings,
                    float *embedding_output, int embedding_output_size,
                    int32 *offset_array_output, int offset_array_size));
  MOCK_CONST_METHOD0(BulkDenseFeatureSize, int());
  MOCK_CONST_METHOD1(GetRawLinkFeatures,
                     std::vector<LinkFeatures>(int channel_id));
  MOCK_CONST_METHOD0(GetOracleLabels,
                     std::vector<std::vector<std::vector<Label>>>());
  MOCK_METHOD0(ResetComponent, void());
  MOCK_METHOD1(GetStepLookupFunction,
               std::function<int(int, int, int)>(const string &method));
  MOCK_METHOD0(FinalizeData, void());
  MOCK_CONST_METHOD0(GetTraceProtos,
                     std::vector<std::vector<ComponentTrace>>());
  MOCK_METHOD2(AddTranslatedLinkFeaturesToTrace,
               void(const std::vector<LinkFeatures> &features, int channel_id));
};

}  // namespace dragnn
}  // namespace syntaxnet

#endif  // DRAGNN_CORE_TEST_MOCK_COMPONENT_H_
