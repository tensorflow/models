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

#ifndef NLP_SAFT_OPENSOURCE_DRAGNN_CORE_INTERFACES_CLONEABLE_TRANSITION_STATE_H_
#define NLP_SAFT_OPENSOURCE_DRAGNN_CORE_INTERFACES_CLONEABLE_TRANSITION_STATE_H_

#include <memory>
#include <vector>

#include "dragnn/core/interfaces/transition_state.h"

namespace syntaxnet {
namespace dragnn {

// This defines a TransitionState object that can be used with the Beam class.
// Any class designed to be used with the Beam must inherit from
// CloneableTransitionState<T>, not TransitionState.

template <class T>
class CloneableTransitionState : public TransitionState {
 public:
  ~CloneableTransitionState<T>() override {}

  // Initialize this TransitionState from a previous TransitionState. The
  // ParentBeamIndex is the location of that previous TransitionState in the
  // provided beam.
  void Init(const TransitionState &parent) override = 0;

  // Return the beam index of the state passed into the initializer of this
  // TransitionState.
  const int ParentBeamIndex() const override = 0;

  // Get the current beam index for this state.
  const int GetBeamIndex() const override = 0;

  // Set the current beam index for this state.
  void SetBeamIndex(const int index) override = 0;

  // Get the score associated with this transition state.
  const float GetScore() const override = 0;

  // Set the score associated with this transition state.
  void SetScore(const float score) override = 0;

  // Depicts this state as an HTML-language string.
  string HTMLRepresentation() const override = 0;

  // Produces a new state with the same backing data as this state.
  virtual std::unique_ptr<T> Clone() const = 0;
};

}  // namespace dragnn
}  // namespace syntaxnet

#endif  // NLP_SAFT_OPENSOURCE_DRAGNN_CORE_INTERFACES_CLONEABLE_TRANSITION_STATE_H_
