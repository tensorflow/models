#ifndef NLP_SAFT_OPENSOURCE_DRAGNN_COMPONENTS_SYNTAXNET_SYNTAXNET_COMPONENT_H_
#define NLP_SAFT_OPENSOURCE_DRAGNN_COMPONENTS_SYNTAXNET_SYNTAXNET_COMPONENT_H_

#include <vector>

#include "dragnn/components/syntaxnet/syntaxnet_link_feature_extractor.h"
#include "dragnn/components/syntaxnet/syntaxnet_transition_state.h"
#include "dragnn/components/util/bulk_feature_extractor.h"
#include "dragnn/core/beam.h"
#include "dragnn/core/input_batch_cache.h"
#include "dragnn/core/interfaces/component.h"
#include "dragnn/core/interfaces/transition_state.h"
#include "dragnn/protos/data.pb.h"
#include "dragnn/protos/spec.pb.h"
#include "dragnn/protos/trace.pb.h"
#include "syntaxnet/base.h"
#include "syntaxnet/parser_transitions.h"
#include "syntaxnet/registry.h"
#include "syntaxnet/task_context.h"

namespace syntaxnet {
namespace dragnn {

class SyntaxNetComponent : public Component {
 public:
  // Create a SyntaxNet-backed DRAGNN component.
  SyntaxNetComponent();

  // Initializes this component from the spec.
  void InitializeComponent(const ComponentSpec &spec) override;

  // Provides the previous beam to the component.
  void InitializeData(
      const std::vector<std::vector<const TransitionState *>> &states,
      int max_beam_size, InputBatchCache *input_data) override;

  // Returns true if the component has had InitializeData called on it since
  // the last time it was reset.
  bool IsReady() const override;

  // Returns the string name of this component.
  string Name() const override;

  // Returns the number of steps taken by the given batch in this component.
  int StepsTaken(int batch_index) const override;

  // Returns the current batch size of the component's underlying data.
  int BatchSize() const override;

  // Returns the maximum beam size of this component.
  int BeamSize() const override;

  // Return the beam index of the item which is currently at index
  // 'index', when the beam was at step 'step', for batch element 'batch'.
  int GetBeamIndexAtStep(int step, int current_index, int batch) const override;

  // Return the source index of the item which is currently at index 'index'
  // for batch element 'batch'. This index is into the final beam of the
  // Component that this Component was initialized from.
  int GetSourceBeamIndex(int current_index, int batch) const override;

  // Request a translation function based on the given method string.
  // The translation function will be called with arguments (batch, beam, value)
  // and should return the step index corresponding to the given value, for the
  // data in the given beam and batch.
  std::function<int(int, int, int)> GetStepLookupFunction(
      const string &method) override;

  // Advances this component from the given transition matrix.
  void AdvanceFromPrediction(const float transition_matrix[],
                             int transition_matrix_length) override;

  // Advances this component from the state oracles.
  void AdvanceFromOracle() override;

  // Returns true if all states within this component are terminal.
  bool IsTerminal() const override;

  // Returns the current batch of beams for this component.
  std::vector<std::vector<const TransitionState *>> GetBeam() override;

  // Extracts and populates the vector of FixedFeatures for the specified
  // channel.
  int GetFixedFeatures(std::function<int32 *(int)> allocate_indices,
                       std::function<int64 *(int)> allocate_ids,
                       std::function<float *(int)> allocate_weights,
                       int channel_id) const override;

  // Extracts and populates all FixedFeatures for all channels, advancing this
  // component via the oracle until it is terminal.
  int BulkGetFixedFeatures(const BulkFeatureExtractor &extractor) override;

  // Extracts and returns the vector of LinkFeatures for the specified
  // channel. Note: these are NOT translated.
  std::vector<LinkFeatures> GetRawLinkFeatures(int channel_id) const override;

  // Returns a vector of oracle labels for each element in the beam and
  // batch.
  std::vector<std::vector<int>> GetOracleLabels() const override;

  // Annotate the underlying data object with the results of this Component's
  // calculation.
  void FinalizeData() override;

  // Reset this component.
  void ResetComponent() override;

  // Initializes the component for tracing execution. This will typically have
  // the side effect of slowing down all subsequent Component calculations
  // and storing a trace in memory that can be returned by GetTraceProtos().
  void InitializeTracing() override;

  // Disables tracing, freeing any additional memory and avoiding triggering
  // additional computation in the future.
  void DisableTracing() override;

  std::vector<std::vector<ComponentTrace>> GetTraceProtos() const override;

  void AddTranslatedLinkFeaturesToTrace(
      const std::vector<LinkFeatures> &features, int channel_id) override;

 private:
  friend class SyntaxNetComponentTest;
  friend class SyntaxNetTransitionStateTest;

  // Permission function for this component.
  bool IsAllowed(SyntaxNetTransitionState *state, int action) const;

  // Returns true if this state is final
  bool IsFinal(SyntaxNetTransitionState *state) const;

  // Oracle function for this component.
  int GetOracleLabel(SyntaxNetTransitionState *state) const;

  // State advance function for this component.
  void Advance(SyntaxNetTransitionState *state, int action,
               Beam<SyntaxNetTransitionState> *beam);

  // Creates a new state for the given nlp_saft::SentenceExample.
  std::unique_ptr<SyntaxNetTransitionState> CreateState(
      SyntaxNetSentence *example);

  // Creates a newly initialized Beam.
  std::unique_ptr<Beam<SyntaxNetTransitionState>> CreateBeam(int max_size);

  // Transition system.
  std::unique_ptr<ParserTransitionSystem> transition_system_;

  // Label map for transition system.
  const TermFrequencyMap *label_map_;

  // Extractor for fixed features
  ParserEmbeddingFeatureExtractor feature_extractor_;

  // Extractor for linked features.
  SyntaxNetLinkFeatureExtractor link_feature_extractor_;

  // Internal workspace registry for use in feature extraction.
  WorkspaceRegistry workspace_registry_;

  // Switch for simulating legacy parser behaviour.
  bool rewrite_root_labels_;

  // The ComponentSpec used to initialize this component.
  ComponentSpec spec_;

  // State search beams
  std::vector<std::unique_ptr<Beam<SyntaxNetTransitionState>>> batch_;

  // Current max beam size.
  int max_beam_size_;

  // Underlying input data.
  InputBatchCache *input_data_;

  // Whether or not to trace for each batch and beam element.
  bool do_tracing_ = false;
};

}  // namespace dragnn
}  // namespace syntaxnet

#endif  // NLP_SAFT_OPENSOURCE_DRAGNN_COMPONENTS_SYNTAXNET_SYNTAXNET_COMPONENT_H_
