#ifndef NLP_SAFT_OPENSOURCE_DRAGNN_CORE_TEST_MOCK_COMPUTE_SESSION_H_
#define NLP_SAFT_OPENSOURCE_DRAGNN_CORE_TEST_MOCK_COMPUTE_SESSION_H_

#include <gmock/gmock.h>

#include "dragnn/components/util/bulk_feature_extractor.h"
#include "dragnn/core/compute_session.h"
#include "dragnn/protos/data.pb.h"
#include "dragnn/protos/spec.pb.h"
#include "syntaxnet/base.h"
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
  MOCK_METHOD3(AdvanceFromPrediction,
               void(const string &component_name, const float score_matrix[],
                    int score_matrix_length));
  MOCK_CONST_METHOD5(GetInputFeatures,
                     int(const string &component_name,
                         std::function<int32 *(int)> allocate_indices,
                         std::function<int64 *(int)> allocate_ids,
                         std::function<float *(int)> allocate_weights,
                         int channel_id));
  MOCK_METHOD2(BulkGetInputFeatures,
               int(const string &component_name,
                   const BulkFeatureExtractor &extractor));
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
};

}  // namespace dragnn
}  // namespace syntaxnet

#endif  // NLP_SAFT_OPENSOURCE_DRAGNN_CORE_TEST_MOCK_COMPUTE_SESSION_H_
