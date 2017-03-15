#ifndef NLP_SAFT_OPENSOURCE_DRAGNN_CORE_TEST_MOCK_TRANSITION_STATE_H_
#define NLP_SAFT_OPENSOURCE_DRAGNN_CORE_TEST_MOCK_TRANSITION_STATE_H_

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
  MOCK_CONST_METHOD0(ParentBeamIndex, const int());
  MOCK_METHOD1(SetBeamIndex, void(const int index));
  MOCK_CONST_METHOD0(GetBeamIndex, const int());
  MOCK_CONST_METHOD0(GetScore, const float());
  MOCK_METHOD1(SetScore, void(const float score));
  MOCK_CONST_METHOD0(HTMLRepresentation, string());
};

}  // namespace dragnn
}  // namespace syntaxnet

#endif  // NLP_SAFT_OPENSOURCE_DRAGNN_CORE_TEST_MOCK_TRANSITION_STATE_H_
