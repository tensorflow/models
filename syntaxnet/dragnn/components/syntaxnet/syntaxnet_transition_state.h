#ifndef NLP_SAFT_OPENSOURCE_DRAGNN_COMPONENTS_SYNTAXNET_SYNTAXNET_TRANSITION_STATE_H_
#define NLP_SAFT_OPENSOURCE_DRAGNN_COMPONENTS_SYNTAXNET_SYNTAXNET_TRANSITION_STATE_H_

#include <vector>

#include "dragnn/core/interfaces/cloneable_transition_state.h"
#include "dragnn/core/interfaces/transition_state.h"
#include "dragnn/io/syntaxnet_sentence.h"
#include "dragnn/protos/trace.pb.h"
#include "syntaxnet/base.h"
#include "syntaxnet/parser_state.h"

namespace syntaxnet {
namespace dragnn {

class SyntaxNetTransitionState
    : public CloneableTransitionState<SyntaxNetTransitionState> {
 public:
  // Create a SyntaxNetTransitionState to wrap this nlp_saft::ParserState.
  SyntaxNetTransitionState(std::unique_ptr<ParserState> parser_state,
                           SyntaxNetSentence *sentence);

  // Initialize this TransitionState from a previous TransitionState. The
  // ParentBeamIndex is the location of that previous TransitionState in the
  // provided beam.
  void Init(const TransitionState &parent) override;

  // Produces a new state with the same backing data as this state.
  std::unique_ptr<SyntaxNetTransitionState> Clone() const override;

  // Return the beam index of the state passed into the initializer of this
  // TransitionState.
  const int ParentBeamIndex() const override;

  // Get the current beam index for this state.
  const int GetBeamIndex() const override;

  // Set the current beam index for this state.
  void SetBeamIndex(const int index) override;

  // Get the score associated with this transition state.
  const float GetScore() const override;

  // Set the score associated with this transition state.
  void SetScore(const float score) override;

  // Depicts this state as an HTML-language string.
  string HTMLRepresentation() const override;

  // **** END INHERITED INTERFACE ****

  // TODO(googleuser): Make these comments actually mean something.
  // Data accessor.
  int step_for_token(int token) {
    if (token < 0 || token >= step_for_token_.size()) {
      return -1;
    } else {
      return step_for_token_.at(token);
    }
  }

  // Data setter.
  void set_step_for_token(int token, int step) {
    step_for_token_.insert(step_for_token_.begin() + token, step);
  }

  // Data accessor.
  int parent_step_for_token(int token) {
    if (token < 0 || token >= step_for_token_.size()) {
      return -1;
    } else {
      return parent_step_for_token_.at(token);
    }
  }

  // Data setter.
  void set_parent_step_for_token(int token, int parent_step) {
    parent_step_for_token_.insert(parent_step_for_token_.begin() + token,
                                  parent_step);
  }

  // Data accessor.
  int parent_for_token(int token) {
    if (token < 0 || token >= step_for_token_.size()) {
      return -1;
    } else {
      return parent_for_token_.at(token);
    }
  }

  // Data setter.
  void set_parent_for_token(int token, int parent) {
    parent_for_token_.insert(parent_for_token_.begin() + token, parent);
  }

  // Accessor for the underlying nlp_saft::ParserState.
  ParserState *parser_state() { return parser_state_.get(); }

  // Accessor for the underlying sentence object.
  SyntaxNetSentence *sentence() { return sentence_; }

  ComponentTrace *mutable_trace() {
    CHECK(trace_) << "Trace is not initialized";
    return trace_.get();
  }
  void set_trace(std::unique_ptr<ComponentTrace> trace) {
    trace_ = std::move(trace);
  }

 private:
  // Underlying ParserState object that is being wrapped.
  std::unique_ptr<ParserState> parser_state_;

  // Sentence object that is being examined with this state.
  SyntaxNetSentence *sentence_;

  // The current score of this state.
  float score_;

  // The current beam index of this state.
  int current_beam_index_;

  // The parent beam index for this state.
  int parent_beam_index_;

  // Maintains a list of which steps in the history correspond to
  // representations for each of the tokens on the stack.
  std::vector<int> step_for_token_;

  // Maintains a list of which steps in the history correspond to the actions
  // that assigned a parent for tokens when reduced.
  std::vector<int> parent_step_for_token_;

  // Maintain the parent index of a token in the system.
  std::vector<int> parent_for_token_;

  // Trace of the history to produce this state.
  std::unique_ptr<ComponentTrace> trace_;
};

}  // namespace dragnn
}  // namespace syntaxnet

#endif  // NLP_SAFT_OPENSOURCE_DRAGNN_COMPONENTS_SYNTAXNET_SYNTAXNET_TRANSITION_STATE_H_
