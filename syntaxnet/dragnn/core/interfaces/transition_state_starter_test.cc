#include "dragnn/core/test/mock_transition_state.h"
#include <gmock/gmock.h>
#include "testing/base/public/googletest.h"
#include "testing/base/public/gunit.h"

// This test suite is intended to validate the contracts that the DRAGNN
// system expects from all transition state subclasses. Developers creating
// new TransitionState subclasses should copy this test and modify it as needed,
// using it to ensure their state conforms to DRAGNN expectations.

namespace syntaxnet {
namespace dragnn {

using testing::Return;

// When this test is instantiated, this function should be changed to
// instantiate a TransitionState subclass of the appropriate type instead
// of Transitionstate->
std::unique_ptr<TransitionState> CreateState() {
  std::unique_ptr<TransitionState> test_state(new TransitionState());
  return test_state;
}

// Validates the consistency of the beam index setter and getter.
TEST(TransitionStateInterfaceTest, CanSetAndGetBeamIndex) {
  // Create and initialize a test state->
  MockTransitionState mock_state;
  auto test_state = CreateState();
  test_state->Init(mock_state);

  constexpr int kOldBeamIndex = 12;
  test_state->SetBeamIndex(kOldBeamIndex);
  EXPECT_EQ(test_state->GetBeamIndex(), kOldBeamIndex);

  constexpr int kNewBeamIndex = 7;
  test_state->SetBeamIndex(kNewBeamIndex);
  EXPECT_EQ(test_state->GetBeamIndex(), kNewBeamIndex);
}

// Validates the consistency of the score setter and getter.
TEST(TransitionStateInterfaceTest, CanSetAndGetScore) {
  // Create and initialize a test state->
  MockTransitionState mock_state;
  auto test_state = CreateState();
  test_state->Init(mock_state);

  constexpr float kOldScore = 12.1;
  test_state->SetScore(kOldScore);
  EXPECT_EQ(test_state->GetScore(), kOldScore);

  constexpr float kNewScore = 7.2;
  test_state->SetScore(kNewScore);
  EXPECT_EQ(test_state->GetScore(), kNewScore);
}

// This test ensures that the initializing state's current index is saved
// as the parent beam index of the state being initialized.
TEST(TransitionStateInterfaceTest, ReportsParentBeamIndex) {
  // Create a mock transition state that wil report a specific current index.
  // This index should become the parent state index for the test state->
  MockTransitionState mock_state;
  constexpr int kParentBeamIndex = 1138;
  EXPECT_CALL(mock_state, GetBeamIndex())
      .WillRepeatedly(Return(kParentBeamIndex));

  auto test_state = CreateState();
  test_state->Init(mock_state);
  EXPECT_EQ(test_state->ParentBeamIndex(), kParentBeamIndex);
}

// This test ensures that the initializing state's current score is saved
// as the current score of the state being initialized.
TEST(TransitionStateInterfaceTest, InitializationCopiesParentScore) {
  // Create a mock transition state that wil report a specific current index.
  // This index should become the parent state index for the test state->
  MockTransitionState mock_state;
  constexpr float kParentScore = 24.12;
  EXPECT_CALL(mock_state, GetScore()).WillRepeatedly(Return(kParentScore));

  auto test_state = CreateState();
  test_state->Init(mock_state);
  EXPECT_EQ(test_state->GetScore(), kParentScore);
}

// This test ensures that calling Clone maintains the state data (parent beam
// index, beam index, score, etc.) of the state that was cloned.
TEST(TransitionStateInterfaceTest, CloningMaintainsState) {
  // Create and initialize the state->
  MockTransitionState mock_state;
  constexpr int kParentBeamIndex = 1138;
  EXPECT_CALL(mock_state, GetBeamIndex())
      .WillRepeatedly(Return(kParentBeamIndex));
  auto test_state = CreateState();
  test_state->Init(mock_state);

  // Validate the internal state of the test state.
  constexpr float kOldScore = 20.0;
  test_state->SetScore(kOldScore);
  EXPECT_EQ(test_state->GetScore(), kOldScore);
  constexpr int kOldBeamIndex = 12;
  test_state->SetBeamIndex(kOldBeamIndex);
  EXPECT_EQ(test_state->GetBeamIndex(), kOldBeamIndex);

  auto clone = test_state->Clone();

  // The clone should have identical state to the old state.
  EXPECT_EQ(clone->ParentBeamIndex(), kParentBeamIndex);
  EXPECT_EQ(clone->GetScore(), kOldScore);
  EXPECT_EQ(clone->GetBeamIndex(), kOldBeamIndex);
}

}  // namespace dragnn
}  // namespace syntaxnet
