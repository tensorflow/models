#ifndef NLP_SAFT_OPENSOURCE_DRAGNN_CORE_COMPUTE_SESSION_H_
#define NLP_SAFT_OPENSOURCE_DRAGNN_CORE_COMPUTE_SESSION_H_

#include <string>

#include "dragnn/components/util/bulk_feature_extractor.h"
#include "dragnn/core/index_translator.h"
#include "dragnn/core/interfaces/component.h"
#include "dragnn/protos/spec.pb.h"
#include "dragnn/protos/trace.pb.h"

namespace syntaxnet {
namespace dragnn {

// This defines the interface for a ComputeSession object. We only ever expect
// ComputeSessionImpl to implement the ComputeSession - this is only used
// to provide a mocking seam.

class ComputeSession {
 public:
  virtual ~ComputeSession() {}

  // Initialize this ComputeSession to compute the graph defined in the given
  // MasterSpec with the hyperparameters passed in the GridPoint. This should
  // only be called once, when the ComputeSession is created.
  virtual void Init(const MasterSpec &master_spec,
                    const GridPoint &hyperparams) = 0;

  // Initialize a component with data and a given maximum beam
  // size. Note that attempting to initialize a component that depends on
  // another component that has not yet finished will cause a CHECK failure.
  virtual void InitializeComponentData(const string &component_name,
                                       int max_beam_size) = 0;

  // Return the batch size for the given component.
  virtual int BatchSize(const string &component_name) const = 0;

  // Return the beam size for the given component.
  virtual int BeamSize(const string &component_name) const = 0;

  // Returns the spec used to create this ComputeSession.
  virtual const ComponentSpec &Spec(const string &component_name) const = 0;

  // For a given component and linked feature channel, get the beam size of the
  // component that is the source of the linked features.
  virtual int SourceComponentBeamSize(const string &component_name,
                                      int channel_id) = 0;

  // Advance the given component using the component's oracle.
  virtual void AdvanceFromOracle(const string &component_name) = 0;

  // Advance the given component using the given score matrix.
  virtual void AdvanceFromPrediction(const string &component_name,
                                     const float score_matrix[],
                                     int score_matrix_length) = 0;

  // Get the input features for the given component and channel. This passes
  // through to the relevant Component's GetFixedFeatures() call.
  virtual int GetInputFeatures(
      const string &component_name,
      std::function<int32 *(int num_items)> allocate_indices,
      std::function<int64 *(int num_items)> allocate_ids,
      std::function<float *(int num_items)> allocate_weights,
      int channel_id) const = 0;

  // Get the input features for the given component and channel, advancing via
  // the oracle until the state is final. This passes through to the relevant
  // Component's BulkGetFixedFeatures() call.
  virtual int BulkGetInputFeatures(const string &component_name,
                                   const BulkFeatureExtractor &extractor) = 0;

  // Get the input features for the given component and channel. This function
  // can return empty LinkFeatures protos, which represent unused padding slots
  // in the output weight tensor.
  virtual std::vector<LinkFeatures> GetTranslatedLinkFeatures(
      const string &component_name, int channel_id) = 0;

  // Get the oracle labels for the given component.
  virtual std::vector<std::vector<int>> EmitOracleLabels(
      const string &component_name) = 0;

  // Returns true if the given component is terminal.
  virtual bool IsTerminal(const string &component_name) = 0;

  // Force the given component to write out its predictions to the backing data.
  virtual void FinalizeData(const string &component_name) = 0;

  // Return the finalized predictions from this compute session.
  virtual std::vector<string> GetSerializedPredictions() = 0;

  // Returns the trace protos. This will CHECK fail or be empty if the
  // SetTracing() has not been called to initialize the underlying Component
  // traces.
  virtual std::vector<MasterTrace> GetTraceProtos() = 0;

  // Provides the ComputeSession with a batch of data to compute.
  virtual void SetInputData(const std::vector<string> &data) = 0;

  // Resets all components owned by this ComputeSession.
  virtual void ResetSession() = 0;

  // Set the tracing for this ComputeSession.
  virtual void SetTracing(bool tracing_on) = 0;

  // Returns a unique identifier for this ComputeSession.
  virtual int Id() const = 0;

  // Returns a string describing the given component.
  virtual string GetDescription(const string &component_name) const = 0;

  // Get all the translators for the given component. Should only be used to
  // validate correct construction of translators in tests.
  virtual const std::vector<const IndexTranslator *> Translators(
      const string &component_name) const = 0;
};

}  // namespace dragnn
}  // namespace syntaxnet

#endif  // NLP_SAFT_OPENSOURCE_DRAGNN_CORE_COMPUTE_SESSION_H_
