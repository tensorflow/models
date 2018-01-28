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

#ifndef DRAGNN_CORE_INTERFACES_TRANSITION_STATE_H_
#define DRAGNN_CORE_INTERFACES_TRANSITION_STATE_H_

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
  virtual int ParentBeamIndex() const = 0;

  // Gets the current beam index for this state.
  virtual int GetBeamIndex() const = 0;

  // Sets the current beam index for this state.
  virtual void SetBeamIndex(int index) = 0;

  // Gets the score associated with this transition state.
  virtual float GetScore() const = 0;

  // Sets the score associated with this transition state.
  virtual void SetScore(float score) = 0;

  // Gets the gold-ness of this state (whether it is on the oracle path)
  virtual bool IsGold() const = 0;

  // Sets the gold-ness of this state.
  virtual void SetGold(bool is_gold) = 0;

  // Depicts this state as an HTML-language string.
  virtual string HTMLRepresentation() const = 0;
};

}  // namespace dragnn
}  // namespace syntaxnet

#endif  // DRAGNN_CORE_INTERFACES_TRANSITION_STATE_H_
