/* Copyright 2016 Google Inc. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "syntaxnet/parser_transitions.h"

#include "syntaxnet/parser_state.h"

namespace syntaxnet {

// Transition system registry.
REGISTER_SYNTAXNET_CLASS_REGISTRY("transition system", ParserTransitionSystem);

void ParserTransitionSystem::PerformAction(ParserAction action,
                                           ParserState *state) const {
  if (state->keep_history()) {
    PerformActionWithoutHistory(action, state);
  } else {
    state->mutable_history()->push_back(action);
    PerformActionWithoutHistory(action, state);
  }
}

}  // namespace syntaxnet
