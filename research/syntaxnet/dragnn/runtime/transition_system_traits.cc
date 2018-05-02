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

#include "dragnn/runtime/transition_system_traits.h"

#include <string>

#include "syntaxnet/base.h"
#include "tensorflow/core/lib/strings/str_util.h"

namespace syntaxnet {
namespace dragnn {
namespace runtime {
namespace {

// Note: The traits are currently simple enough to specify in one file.  We can
// also use a registry-based system if this gets too complex.

// Returns true if the |component_spec| is deterministic.
bool IsDeterministic(const ComponentSpec &component_spec) {
  return component_spec.num_actions() == 1;
}

// Returns true if the |component_spec| is sequential.
bool IsSequential(const ComponentSpec &component_spec) {
  const string &name = component_spec.transition_system().registered_name();
  return name == "char-shift-only" ||  //

         name == "shift-only" ||       //
         name == "tagger" ||           //
         name == "morpher" ||          //
         name == "heads" ||            //
         name == "labels";
}

// Returns true if the |component_spec| specifies a left-to-right transition
// system.  The default when unspecified is true.
bool IsLeftToRight(const ComponentSpec &component_spec) {
  const auto &parameters = component_spec.transition_system().parameters();
  const auto it = parameters.find("left_to_right");
  if (it == parameters.end()) return true;
  return tensorflow::str_util::Lowercase(it->second) != "false";
}

// Returns true if the |transition_system| is character-scale.
bool IsCharacterScale(const ComponentSpec &component_spec) {
  const string &name = component_spec.transition_system().registered_name();
  return                            //

      name == "char-shift-only";
}

// Returns true if the |transition_system| is token-scale.
bool IsTokenScale(const ComponentSpec &component_spec) {
  const string &name = component_spec.transition_system().registered_name();
  return name == "shift-only" ||  //
         name == "tagger" ||      //
         name == "morpher" ||     //
         name == "heads" ||       //
         name == "labels";
}

}  // namespace

TransitionSystemTraits::TransitionSystemTraits(
    const ComponentSpec &component_spec)
    : is_deterministic(IsDeterministic(component_spec)),
      is_sequential(IsSequential(component_spec)),
      is_left_to_right(IsLeftToRight(component_spec)),
      is_character_scale(IsCharacterScale(component_spec)),
      is_token_scale(IsTokenScale(component_spec)) {}

}  // namespace runtime
}  // namespace dragnn
}  // namespace syntaxnet
