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

#ifndef DRAGNN_CORE_TEST_MOCK_TRANSITION_STATE_H_
#define DRAGNN_CORE_TEST_MOCK_TRANSITION_STATE_H_

#include <memory>

#include <gmock/gmock.h>

#include "dragnn/core/interfaces/transition_state.h"
#include "syntaxnet/base.h"
#include "tensorflow/core/platform/test.h"

namespace syntaxnet {
namespace dragnn {

class MockTransitionState : public TransitionState {
 public:
  MOCK_METHOD1(Init, void(const TransitionState &parent));
  MOCK_CONST_METHOD0(Clone, std::unique_ptr<TransitionState>());
  MOCK_CONST_METHOD0(ParentBeamIndex, int());
  MOCK_METHOD1(SetBeamIndex, void(int index));
  MOCK_CONST_METHOD0(GetBeamIndex, int());
  MOCK_CONST_METHOD0(GetScore, float());
  MOCK_METHOD1(SetScore, void(float score));
  MOCK_CONST_METHOD0(IsGold, bool());
  MOCK_METHOD1(SetGold, void(bool is_gold));
  MOCK_CONST_METHOD0(HTMLRepresentation, string());
};

}  // namespace dragnn
}  // namespace syntaxnet

#endif  // DRAGNN_CORE_TEST_MOCK_TRANSITION_STATE_H_
