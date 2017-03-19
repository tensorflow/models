#ifndef NLP_SAFT_OPENSOURCE_DRAGNN_CORE_INTERFACES_TRANSITION_STATE_H_
#define NLP_SAFT_OPENSOURCE_DRAGNN_CORE_INTERFACES_TRANSITION_STATE_H_

#include <memory>
#include <vector>

#include "syntaxnet/base.h"

namespace syntaxnet {
namespace dragnn {

// TransitionState defines the minimal interface required to pass data between
// Component objects. It is used to initialize one Component from the output of
// another, and every backend should define one. Note that inheriting from
// TransitionState directly is not sufficient to use the Beam class, which
// requires extra functionality given by inheriting from the
// ClonableTransitionState interface. (ClonableTransitionState is a subclass
// of TransitionState, so inheriting from ClonableTransitionState is sufficient
// to allow Components to pass your backing states.)

class TransitionState {
 public:
  virtual ~TransitionState() {}

  // Initialize this TransitionState from a previous TransitionState. The
  // ParentBeamIndex is the location of that previous TransitionState in the
  // provided beam.
  virtual void Init(const TransitionState &parent) = 0;

  // Return the beam index of the state passed into the initializer of this
  // TransitionState.
  virtual const int ParentBeamIndex() const = 0;

  // Get the current beam index for this state.
  virtual const int GetBeamIndex() const = 0;

  // Set the current beam index for this state.
  virtual void SetBeamIndex(const int index) = 0;

  // Get the score associated with this transition state.
  virtual const float GetScore() const = 0;

  // Set the score associated with this transition state.
  virtual void SetScore(const float score) = 0;

  // Depicts this state as an HTML-language string.
  virtual string HTMLRepresentation() const = 0;
};

}  // namespace dragnn
}  // namespace syntaxnet

#endif  // NLP_SAFT_OPENSOURCE_DRAGNN_CORE_INTERFACES_TRANSITION_STATE_H_
