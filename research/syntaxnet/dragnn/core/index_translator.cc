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

#include "dragnn/core/index_translator.h"

#include "tensorflow/core/platform/logging.h"

namespace syntaxnet {
namespace dragnn {

using Index = IndexTranslator::Index;

IndexTranslator::IndexTranslator(const std::vector<Component *> &path,
                                 const string &method)
    : path_(path), method_(method) {
  if (method_ == "identity") {
    // Identity lookup: Return the feature index.
    step_lookup_ = [](int batch_index, int beam_index, int feature) {
      return feature;
    };
  } else if (method_ == "history") {
    // History lookup: Return the number of steps taken less the feature.
    step_lookup_ = [this](int batch_index, int beam_index, int feature) {
      if (feature > path_.back()->StepsTaken(batch_index) - 1 || feature < 0) {
        VLOG(2) << "Translation to outside: feature is " << feature
                << " and steps_taken is "
                << path_.back()->StepsTaken(batch_index);
        return -1;
      }
      return ((path_.back()->StepsTaken(batch_index) - 1) - feature);
    };
  } else {
    // Component defined lookup: Get the lookup function from the component.
    // If the lookup function is not defined, this function will CHECK.
    step_lookup_ = path_.back()->GetStepLookupFunction(method_);
  }
}

Index IndexTranslator::Translate(int batch_index, int beam_index,
                                 int feature_value) {
  Index translated_index;
  translated_index.batch_index = batch_index;
  VLOG(2) << "Translation requested (type: " << method_ << ") for batch "
          << batch_index << " beam " << beam_index << " feature "
          << feature_value;

  // For all save the last item in the path, get the source index for the
  // previous component.
  int current_beam_index = beam_index;
  VLOG(2) << "Beam index before walk is " << current_beam_index;
  for (int i = 0; i < path_.size() - 1; ++i) {
    // Backtrack through previous components. For each non-final component,
    // figure out what state in the prior component was used to initialize the
    // state at the current beam index.
    current_beam_index =
        path_.at(i)->GetSourceBeamIndex(current_beam_index, batch_index);
    VLOG(2) << "Beam index updated to " << current_beam_index;
  }
  VLOG(2) << "Beam index after walk is " << current_beam_index;
  translated_index.step_index =
      step_lookup_(batch_index, current_beam_index, feature_value);
  VLOG(2) << "Translated step index is " << translated_index.step_index;
  translated_index.beam_index = path_.back()->GetBeamIndexAtStep(
      translated_index.step_index, current_beam_index, batch_index);
  VLOG(2) << "Translated beam index is " << translated_index.beam_index;
  return translated_index;
}

}  // namespace dragnn
}  // namespace syntaxnet
