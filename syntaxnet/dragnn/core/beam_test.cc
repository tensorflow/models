#include "dragnn/core/beam.h"

#include "dragnn/core/interfaces/cloneable_transition_state.h"
#include "dragnn/core/interfaces/transition_state.h"
#include "dragnn/core/test/mock_transition_state.h"
#include <gmock/gmock.h>
#include "tensorflow/core/platform/test.h"

namespace syntaxnet {
namespace dragnn {

using testing::MockFunction;
using testing::Return;
using testing::Ne;
using testing::_;

namespace {

// *****************************************************************************
// Test-internal class definitions.
// *****************************************************************************

// Create a very basic transition state to test the beam. All it does is keep
// track of its current beam index and score, as well as providing a field
// for the transition function to write in what transition occurred.
// Note that this class does not fulfill the entire TransitionState contract,
// since it is only used in this particular test.
class TestTransitionState
    : public CloneableTransitionState<TestTransitionState> {
 public:
  TestTransitionState() {}

  void Init(const TransitionState &parent) override {}

  std::unique_ptr<TestTransitionState> Clone() const override {
    std::unique_ptr<TestTransitionState> ptr(new TestTransitionState());
    return ptr;
  }

  const int ParentBeamIndex() const override { return parent_beam_index_; }

  // Get the current beam index for this state.
  const int GetBeamIndex() const override { return beam_index_; }

  // Set the current beam index for this state.
  void SetBeamIndex(const int index) override { beam_index_ = index; }

  // Get the score associated with this transition state.
  const float GetScore() const override { return score_; }

  // Set the score associated with this transition state.
  void SetScore(const float score) override { score_ = score; }

  // Depicts this state as an HTML-language string.
  string HTMLRepresentation() const override { return ""; }

  int parent_beam_index_;

  int beam_index_;

  float score_;

  int transition_action_;
};

// This transition function annotates a TestTransitionState with the action that
// was chosen for the transition.
auto transition_function = [](TestTransitionState *state, int action) {
  TestTransitionState *cast_state = dynamic_cast<TestTransitionState *>(state);
  cast_state->transition_action_ = action;
};

// Create oracle and permission functions that do nothing.
auto null_oracle = [](TestTransitionState *) { return 0; };
auto null_permissions = [](TestTransitionState *, int) { return true; };
auto null_finality = [](TestTransitionState *) { return false; };

// Create a unique_ptr with a test transition state in it and set its initial
// score.
std::unique_ptr<TestTransitionState> CreateState(float score) {
  std::unique_ptr<TestTransitionState> state;
  state.reset(new TestTransitionState());
  state->SetScore(score);
  return state;
}

// Convenience accessor for the action field in TestTransitionState.
int GetTransition(const TransitionState *state) {
  return (dynamic_cast<const TestTransitionState *>(state))->transition_action_;
}

// Convenience accessor for the parent_beam_index_ field in TestTransitionState.
void SetParentBeamIndex(TransitionState *state, int index) {
  (dynamic_cast<TestTransitionState *>(state))->parent_beam_index_ = index;
}

}  // namespace

// *****************************************************************************
// Tests begin here.
// *****************************************************************************
TEST(BeamTest, AdvancesFromPredictionWithSingleBeam) {
  // Create a matrix of transitions.
  constexpr int kNumTransitions = 4;
  constexpr int kMatrixSize = kNumTransitions;
  constexpr float matrix[kMatrixSize] = {30.0, 20.0, 40.0, 10.0};
  constexpr int kBestTransition = 2;
  constexpr float kOldScore = 3.0;

  // Create the beam and transition it.
  std::vector<std::unique_ptr<TestTransitionState>> states;
  states.push_back(CreateState(kOldScore));
  constexpr int kBeamSize = 1;
  Beam<TestTransitionState> beam(kBeamSize);
  beam.SetFunctions(null_permissions, null_finality, transition_function,
                    null_oracle);
  beam.Init(std::move(states));
  beam.AdvanceFromPrediction(matrix, kMatrixSize, kNumTransitions);

  // Validate the new beam.
  EXPECT_EQ(beam.beam().size(), kBeamSize);

  // Make sure the state has performed the expected transition.
  EXPECT_EQ(GetTransition(beam.beam().at(0)), kBestTransition);

  // Make sure the state has had its score updated properly.
  EXPECT_EQ(beam.beam().at(0)->GetScore(), kOldScore + matrix[kBestTransition]);

  // Make sure that the beam index field is consistent with the actual beam idx.
  EXPECT_EQ(beam.beam().at(0)->GetBeamIndex(), 0);

  // Make sure that the beam_state accessor actually accesses the beam.
  EXPECT_EQ(beam.beam().at(0), beam.beam_state(0));

  // Validate the beam history field.
  auto history = beam.history();
  EXPECT_EQ(history.at(1).at(0), 0);
}

TEST(BeamTest, AdvancingCreatesNewTransitions) {
  // Create a matrix of transitions.
  constexpr int kMaxBeamSize = 8;
  constexpr int kNumTransitions = 4;
  constexpr int kMatrixSize = kNumTransitions * kMaxBeamSize;
  constexpr float matrix[kMatrixSize] = {
      30.0, 20.0, 40.0, 10.0, 00.0, 00.0, 00.0, 00.0, 00.0, 00.0, 00.0,
      00.0, 00.0, 00.0, 00.0, 00.0, 00.0, 00.0, 00.0, 00.0, 00.0, 00.0,
      00.0, 00.0, 00.0, 00.0, 00.0, 00.0, 00.0, 00.0, 00.0, 00.0};
  constexpr float kOldScore = 4.0;

  // Create the beam and transition it.
  std::vector<std::unique_ptr<TestTransitionState>> states;
  states.push_back(CreateState(kOldScore));

  Beam<TestTransitionState> beam(kMaxBeamSize);
  beam.SetFunctions(null_permissions, null_finality, transition_function,
                    null_oracle);
  beam.Init(std::move(states));
  beam.AdvanceFromPrediction(matrix, kMatrixSize, kNumTransitions);

  // Validate the new beam.
  EXPECT_EQ(beam.beam().size(), 4);

  // Make sure the state has performed the expected transition.
  EXPECT_EQ(GetTransition(beam.beam().at(0)), 2);
  EXPECT_EQ(GetTransition(beam.beam().at(1)), 0);
  EXPECT_EQ(GetTransition(beam.beam().at(2)), 1);
  EXPECT_EQ(GetTransition(beam.beam().at(3)), 3);

  // Make sure the state has had its score updated properly.
  EXPECT_EQ(beam.beam().at(0)->GetScore(), kOldScore + matrix[2]);
  EXPECT_EQ(beam.beam().at(1)->GetScore(), kOldScore + matrix[0]);
  EXPECT_EQ(beam.beam().at(2)->GetScore(), kOldScore + matrix[1]);
  EXPECT_EQ(beam.beam().at(3)->GetScore(), kOldScore + matrix[3]);

  // Make sure that the beam index field is consistent with the actual beam idx.
  for (int i = 0; i < beam.beam().size(); ++i) {
    EXPECT_EQ(beam.beam().at(i)->GetBeamIndex(), i);
  }

  // In this case, we expect the top 4 results to have come from state 0 and
  // the remaining 4 slots to be empty (-1).
  auto history = beam.history();
  EXPECT_EQ(history.at(1).at(0), 0);
  EXPECT_EQ(history.at(1).at(1), 0);
  EXPECT_EQ(history.at(1).at(2), 0);
  EXPECT_EQ(history.at(1).at(3), 0);
  EXPECT_EQ(history.at(1).at(4), -1);
  EXPECT_EQ(history.at(1).at(5), -1);
  EXPECT_EQ(history.at(1).at(6), -1);
  EXPECT_EQ(history.at(1).at(7), -1);
}

TEST(BeamTest, MultipleElementBeamsAdvanceAllElements) {
  // Create a matrix of transitions.
  constexpr int kMaxBeamSize = 8;
  constexpr int kNumTransitions = 4;
  constexpr int kMatrixSize = kNumTransitions * kMaxBeamSize;

  constexpr float matrix[kMatrixSize] = {
      30.0, 20.0, 40.0, 10.0,  // State 0
      31.0, 21.0, 41.0, 11.0, 00.0, 00.0, 00.0, 00.0, 00.0, 00.0,
      00.0, 00.0, 00.0, 00.0, 00.0, 00.0, 00.0, 00.0, 00.0, 00.0,
      00.0, 00.0, 00.0, 00.0, 00.0, 00.0, 00.0, 00.0};

  constexpr float kOldScores[] = {5.0, 7.0};

  // Create the beam and transition it.
  std::vector<std::unique_ptr<TestTransitionState>> states;
  states.push_back(CreateState(kOldScores[0]));
  states.push_back(CreateState(kOldScores[1]));

  Beam<TestTransitionState> beam(kMaxBeamSize);
  beam.SetFunctions(null_permissions, null_finality, transition_function,
                    null_oracle);
  beam.Init(std::move(states));
  beam.AdvanceFromPrediction(matrix, kMatrixSize, kNumTransitions);

  // Validate the new beam.
  EXPECT_EQ(beam.beam().size(), 8);

  // Make sure the state has performed the expected transition.
  // Note that the transition index is not the index into the matrix, but rather
  // the index into the matrix 'row' for that state.
  EXPECT_EQ(GetTransition(beam.beam().at(0)), 2);
  EXPECT_EQ(GetTransition(beam.beam().at(1)), 2);
  EXPECT_EQ(GetTransition(beam.beam().at(2)), 0);
  EXPECT_EQ(GetTransition(beam.beam().at(3)), 0);
  EXPECT_EQ(GetTransition(beam.beam().at(4)), 1);
  EXPECT_EQ(GetTransition(beam.beam().at(5)), 1);
  EXPECT_EQ(GetTransition(beam.beam().at(6)), 3);
  EXPECT_EQ(GetTransition(beam.beam().at(7)), 3);

  // Make sure the state has had its score updated properly.
  EXPECT_EQ(beam.beam().at(0)->GetScore(), kOldScores[1] + matrix[6]);
  EXPECT_EQ(beam.beam().at(1)->GetScore(), kOldScores[0] + matrix[2]);
  EXPECT_EQ(beam.beam().at(2)->GetScore(), kOldScores[1] + matrix[4]);
  EXPECT_EQ(beam.beam().at(3)->GetScore(), kOldScores[0] + matrix[0]);
  EXPECT_EQ(beam.beam().at(4)->GetScore(), kOldScores[1] + matrix[5]);
  EXPECT_EQ(beam.beam().at(5)->GetScore(), kOldScores[0] + matrix[1]);
  EXPECT_EQ(beam.beam().at(6)->GetScore(), kOldScores[1] + matrix[7]);
  EXPECT_EQ(beam.beam().at(7)->GetScore(), kOldScores[0] + matrix[3]);

  // Make sure that the beam index field is consistent with the actual beam idx.
  for (int i = 0; i < beam.beam().size(); ++i) {
    EXPECT_EQ(beam.beam().at(i)->GetBeamIndex(), i);
  }

  // Validate the history at this step.
  auto history = beam.history();
  EXPECT_EQ(history.at(1).at(0), 1);
  EXPECT_EQ(history.at(1).at(1), 0);
  EXPECT_EQ(history.at(1).at(2), 1);
  EXPECT_EQ(history.at(1).at(3), 0);
  EXPECT_EQ(history.at(1).at(4), 1);
  EXPECT_EQ(history.at(1).at(5), 0);
  EXPECT_EQ(history.at(1).at(6), 1);
  EXPECT_EQ(history.at(1).at(7), 0);
}

TEST(BeamTest, AdvancingDropsLowValuePredictions) {
  // Create a matrix of transitions.
  constexpr int kNumTransitions = 4;
  constexpr int kMaxBeamSize = 8;
  constexpr int kMatrixSize = kNumTransitions * kMaxBeamSize;
  constexpr float matrix[kMatrixSize] = {30.0, 20.0, 40.0, 10.0,   // State 0
                                         31.0, 21.0, 41.0, 11.0,   // State 1
                                         32.0, 22.0, 42.0, 12.0,   // State 2
                                         33.0, 23.0, 43.0, 13.0,   // State 3
                                         34.0, 24.0, 44.0, 14.0,   // State 4
                                         35.0, 25.0, 45.0, 15.0,   // State 5
                                         36.0, 26.0, 46.0, 16.0,   // State 6
                                         37.0, 27.0, 47.0, 17.0};  // State 7
  constexpr float kOldScores[] = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8};

  // Create the beam and transition it.
  std::vector<std::unique_ptr<TestTransitionState>> states;
  states.push_back(CreateState(kOldScores[0]));
  states.push_back(CreateState(kOldScores[1]));
  states.push_back(CreateState(kOldScores[2]));
  states.push_back(CreateState(kOldScores[3]));
  states.push_back(CreateState(kOldScores[4]));
  states.push_back(CreateState(kOldScores[5]));
  states.push_back(CreateState(kOldScores[6]));
  states.push_back(CreateState(kOldScores[7]));
  Beam<TestTransitionState> beam(kMaxBeamSize);
  beam.SetFunctions(null_permissions, null_finality, transition_function,
                    null_oracle);
  beam.Init(std::move(states));
  beam.AdvanceFromPrediction(matrix, kMatrixSize, kNumTransitions);

  // Validate the new beam.
  EXPECT_EQ(beam.beam().size(), 8);

  // Make sure the state has performed the expected transition.
  // In this case, every state will perform transition 2.
  EXPECT_EQ(GetTransition(beam.beam().at(0)), 2);
  EXPECT_EQ(GetTransition(beam.beam().at(1)), 2);
  EXPECT_EQ(GetTransition(beam.beam().at(2)), 2);
  EXPECT_EQ(GetTransition(beam.beam().at(3)), 2);
  EXPECT_EQ(GetTransition(beam.beam().at(4)), 2);
  EXPECT_EQ(GetTransition(beam.beam().at(5)), 2);
  EXPECT_EQ(GetTransition(beam.beam().at(6)), 2);
  EXPECT_EQ(GetTransition(beam.beam().at(7)), 2);

  // Make sure the state has had its score updated properly. (Note that row
  // 0 had the smallest transition score, so it ends up on the bottom of the
  // beam, and so forth.) For the matrix index, N*kNumTransitions gets into the
  // correct state row and we add 2 since that was the transition index.
  EXPECT_EQ(beam.beam().at(0)->GetScore(),
            kOldScores[7] + matrix[7 * kNumTransitions + 2]);
  EXPECT_EQ(beam.beam().at(1)->GetScore(),
            kOldScores[6] + matrix[6 * kNumTransitions + 2]);
  EXPECT_EQ(beam.beam().at(2)->GetScore(),
            kOldScores[5] + matrix[5 * kNumTransitions + 2]);
  EXPECT_EQ(beam.beam().at(3)->GetScore(),
            kOldScores[4] + matrix[4 * kNumTransitions + 2]);
  EXPECT_EQ(beam.beam().at(4)->GetScore(),
            kOldScores[3] + matrix[3 * kNumTransitions + 2]);
  EXPECT_EQ(beam.beam().at(5)->GetScore(),
            kOldScores[2] + matrix[2 * kNumTransitions + 2]);
  EXPECT_EQ(beam.beam().at(6)->GetScore(),
            kOldScores[1] + matrix[1 * kNumTransitions + 2]);
  EXPECT_EQ(beam.beam().at(7)->GetScore(),
            kOldScores[0] + matrix[0 * kNumTransitions + 2]);

  // Make sure that the beam index field is consistent with the actual beam idx.
  for (int i = 0; i < beam.beam().size(); ++i) {
    EXPECT_EQ(beam.beam().at(i)->GetBeamIndex(), i);
  }

  auto history = beam.history();
  EXPECT_EQ(history.at(1).at(0), 7);
  EXPECT_EQ(history.at(1).at(1), 6);
  EXPECT_EQ(history.at(1).at(2), 5);
  EXPECT_EQ(history.at(1).at(3), 4);
  EXPECT_EQ(history.at(1).at(4), 3);
  EXPECT_EQ(history.at(1).at(5), 2);
  EXPECT_EQ(history.at(1).at(6), 1);
  EXPECT_EQ(history.at(1).at(7), 0);
}

TEST(BeamTest, AdvancesFromOracleWithSingleBeam) {
  // Create an oracle function for this state.
  constexpr int kOracleLabel = 3;
  auto oracle_function = [](TransitionState *) { return kOracleLabel; };

  // Create the beam and transition it.
  std::vector<std::unique_ptr<TestTransitionState>> states;
  states.push_back(CreateState(0.0));
  constexpr int kBeamSize = 1;
  Beam<TestTransitionState> beam(kBeamSize);
  beam.SetFunctions(null_permissions, null_finality, transition_function,
                    oracle_function);
  beam.Init(std::move(states));
  beam.AdvanceFromOracle();

  // Validate the new beam.
  EXPECT_EQ(beam.beam().size(), kBeamSize);

  // Make sure the state has performed the expected transition.
  EXPECT_EQ(GetTransition(beam.beam().at(0)), kOracleLabel);

  // Make sure the state has had its score held to 0.
  EXPECT_EQ(beam.beam().at(0)->GetScore(), 0.0);

  // Make sure that the beam index field is consistent with the actual beam idx.
  EXPECT_EQ(beam.beam().at(0)->GetBeamIndex(), 0);

  // Validate the beam history field.
  auto history = beam.history();
  EXPECT_EQ(history.at(1).at(0), 0);
}

TEST(BeamTest, AdvancesFromOracleWithMultipleStates) {
  constexpr int kMaxBeamSize = 8;

  // Create a beam with 8 transition states.
  std::vector<std::unique_ptr<TestTransitionState>> states;
  for (int i = 0; i < kMaxBeamSize; ++i) {
    // This is nonzero to test the oracle holding scores to 0.
    states.push_back(CreateState(10.0));
  }

  std::vector<int> expected_actions;

  // Create an oracle function for this state. Use mocks for finer control.
  testing::MockFunction<int(TestTransitionState *)> mock_oracle_function;
  for (int i = 0; i < kMaxBeamSize; ++i) {
    // We expect each state to be queried for its oracle label,
    // and then to be transitioned in place with its oracle label.
    int oracle_label = i % 3;  // 3 is arbitrary.
    EXPECT_CALL(mock_oracle_function, Call(states.at(i).get()))
        .WillOnce(Return(oracle_label));
    expected_actions.push_back(oracle_label);
  }

  Beam<TestTransitionState> beam(kMaxBeamSize);
  beam.SetFunctions(null_permissions, null_finality, transition_function,
                    mock_oracle_function.AsStdFunction());
  beam.Init(std::move(states));
  beam.AdvanceFromOracle();

  // Make sure the state has performed the expected transition, has had its
  // score held to 0, and is self consistent.
  for (int i = 0; i < beam.beam().size(); ++i) {
    EXPECT_EQ(GetTransition(beam.beam().at(i)), expected_actions.at(i));
    EXPECT_EQ(beam.beam().at(i)->GetScore(), 0.0);
    EXPECT_EQ(beam.beam().at(i)->GetBeamIndex(), i);
  }

  auto history = beam.history();
  for (int i = 0; i < beam.beam().size(); ++i) {
    EXPECT_EQ(history.at(1).at(i), i);
  }
}

TEST(BeamTest, ReportsNonFinality) {
  constexpr int kMaxBeamSize = 8;

  // Create a beam with 8 transition states.
  std::vector<std::unique_ptr<TestTransitionState>> states;
  for (int i = 0; i < kMaxBeamSize; ++i) {
    // This is nonzero to test the oracle holding scores to 0.
    states.push_back(CreateState(10.0));
  }

  std::vector<int> expected_actions;

  // Create a finality function for this state. Use mocks for finer control.
  testing::MockFunction<int(TestTransitionState *)> mock_finality_function;

  // Make precisely one call return false, which should cause IsFinal
  // to report false.
  constexpr int incomplete_state = 3;
  EXPECT_CALL(mock_finality_function, Call(states.at(incomplete_state).get()))
      .WillOnce(Return(false));
  EXPECT_CALL(mock_finality_function,
              Call(Ne(states.at(incomplete_state).get())))
      .WillRepeatedly(Return(true));

  Beam<TestTransitionState> beam(kMaxBeamSize);
  beam.SetFunctions(null_permissions, mock_finality_function.AsStdFunction(),
                    transition_function, null_oracle);
  beam.Init(std::move(states));

  EXPECT_FALSE(beam.IsTerminal());
}

TEST(BeamTest, ReportsFinality) {
  constexpr int kMaxBeamSize = 8;

  // Create a beam with 8 transition states.
  std::vector<std::unique_ptr<TestTransitionState>> states;
  for (int i = 0; i < kMaxBeamSize; ++i) {
    // This is nonzero to test the oracle holding scores to 0.
    states.push_back(CreateState(10.0));
  }

  std::vector<int> expected_actions;

  // Create a finality function for this state. Use mocks for finer control.
  testing::MockFunction<int(TransitionState *)> mock_finality_function;

  // All calls will return true, so IsFinal should return true.
  EXPECT_CALL(mock_finality_function, Call(_)).WillRepeatedly(Return(true));

  Beam<TestTransitionState> beam(kMaxBeamSize);
  beam.SetFunctions(null_permissions, mock_finality_function.AsStdFunction(),
                    transition_function, null_oracle);
  beam.Init(std::move(states));

  EXPECT_TRUE(beam.IsTerminal());
}

TEST(BeamTest, IgnoresForbiddenTransitionActions) {
  // Create a matrix of transitions.
  constexpr int kMaxBeamSize = 4;
  constexpr int kNumTransitions = 4;
  constexpr int kMatrixSize = kNumTransitions * kMaxBeamSize;
  constexpr float matrix[kMatrixSize] = {
      10.0, 1000.0, 40.0, 30.0, 00.0, 0000.0, 00.0, 00.0,
      00.0, 0000.0, 00.0, 00.0, 00.0, 0000.0, 00.0, 00.0};
  constexpr float kOldScore = 4.0;

  // Create the beam.
  std::vector<std::unique_ptr<TestTransitionState>> states;
  states.push_back(CreateState(kOldScore));

  // Forbid the second transition (index 1).
  testing::MockFunction<int(TestTransitionState *, int)>
      mock_permission_function;
  EXPECT_CALL(mock_permission_function, Call(states.at(0).get(), 0))
      .WillOnce(Return(true));
  EXPECT_CALL(mock_permission_function, Call(states.at(0).get(), 1))
      .WillOnce(Return(false));
  EXPECT_CALL(mock_permission_function, Call(states.at(0).get(), 2))
      .WillOnce(Return(true));
  EXPECT_CALL(mock_permission_function, Call(states.at(0).get(), 3))
      .WillOnce(Return(true));

  Beam<TestTransitionState> beam(kMaxBeamSize);
  beam.SetFunctions(mock_permission_function.AsStdFunction(), null_finality,
                    transition_function, null_oracle);
  beam.Init(std::move(states));
  beam.AdvanceFromPrediction(matrix, kMatrixSize, kNumTransitions);

  // Validate the new beam.
  EXPECT_EQ(beam.beam().size(), 3);

  // Make sure the state has performed the expected transition.
  EXPECT_EQ(GetTransition(beam.beam().at(0)), 2);
  EXPECT_EQ(GetTransition(beam.beam().at(1)), 3);
  EXPECT_EQ(GetTransition(beam.beam().at(2)), 0);

  // Make sure the state has had its score updated properly.
  EXPECT_EQ(beam.beam().at(0)->GetScore(), kOldScore + matrix[2]);
  EXPECT_EQ(beam.beam().at(1)->GetScore(), kOldScore + matrix[3]);
  EXPECT_EQ(beam.beam().at(2)->GetScore(), kOldScore + matrix[0]);

  // Make sure that the beam index field is consistent with the actual beam idx.
  for (int i = 0; i < beam.beam().size(); ++i) {
    EXPECT_EQ(beam.beam().at(i)->GetBeamIndex(), i);
  }

  // In this case, we expect the top 3 results to have come from state 0 and
  // the remaining 3 slots to be empty (-1).
  auto history = beam.history();
  EXPECT_EQ(history.at(1).at(0), 0);
  EXPECT_EQ(history.at(1).at(1), 0);
  EXPECT_EQ(history.at(1).at(2), 0);
  EXPECT_EQ(history.at(1).at(3), -1);
}

TEST(BeamTest, BadlySizedMatrixDies) {
  // Create a matrix of transitions.
  constexpr int kNumTransitions = 4;
  constexpr int kMatrixSize = 4;  // We have a max beam size of 4; should be 16.
  constexpr float matrix[kMatrixSize] = {30.0, 20.0, 40.0, 10.0};

  // Create the beam and transition it.
  std::vector<std::unique_ptr<TestTransitionState>> states;
  states.push_back(CreateState(0.0));
  states.push_back(CreateState(0.0));
  constexpr int kMaxBeamSize = 8;
  Beam<TestTransitionState> beam(kMaxBeamSize);
  beam.SetFunctions(null_permissions, null_finality, transition_function,
                    null_oracle);
  beam.Init(std::move(states));

  // This matrix should have 8 elements, not 4, so this should die.
  EXPECT_DEATH(beam.AdvanceFromPrediction(matrix, kMatrixSize, kNumTransitions),
               "Transition matrix size does not match max beam size \\* number "
               "of state transitions");
}

TEST(BeamTest, BadlySizedBeamInitializationDies) {
  // Create an initialization beam too large for the max beam size.
  constexpr int kMaxBeamSize = 4;
  std::vector<std::unique_ptr<TestTransitionState>> states;
  for (int i = 0; i < kMaxBeamSize + 1; ++i) {
    states.push_back(CreateState(0.0));
  }

  Beam<TestTransitionState> beam(kMaxBeamSize);
  beam.SetFunctions(null_permissions, null_finality, transition_function,
                    null_oracle);

  // Try to initialize the beam; this should die.
  EXPECT_DEATH(beam.Init(std::move(states)),
               "Attempted to initialize a beam with more states");
}

TEST(BeamTest, ValidBeamIndicesAfterBeamInitialization) {
  // Create a standard beam.
  constexpr int kMaxBeamSize = 4;
  std::vector<std::unique_ptr<TestTransitionState>> states;
  for (int i = 0; i < kMaxBeamSize; ++i) {
    states.push_back(CreateState(0.0));
  }

  Beam<TestTransitionState> beam(kMaxBeamSize);
  beam.SetFunctions(null_permissions, null_finality, transition_function,
                    null_oracle);

  beam.Init(std::move(states));

  // Verify that all beam indices have been initialized.
  for (int i = 0; i < kMaxBeamSize; ++i) {
    EXPECT_EQ(i, beam.beam_state(i)->GetBeamIndex());
  }
}

TEST(BeamTest, FindPreviousIndexTracesHistory) {
  // Create a matrix of transitions.
  constexpr int kNumTransitions = 4;
  constexpr int kMaxBeamSize = 8;
  constexpr int kMatrixSize = kNumTransitions * kMaxBeamSize;
  constexpr float matrix[kMatrixSize] = {
      30.0, 20.0, 40.0, 10.0,  // State 0
      31.0, 21.0, 41.0, 11.0,  // State 1
      00.0, 00.0, 00.0, 00.0, 00.0, 00.0, 00.0, 00.0, 00.0, 00.0, 00.0, 00.0,
      00.0, 00.0, 00.0, 00.0, 00.0, 00.0, 00.0, 00.0, 00.0, 00.0, 00.0, 00.0};
  constexpr float kOldScores[] = {5.0, 7.0};
  constexpr int kParentBeamIndices[] = {1138, 42};

  // Create the beam and transition it.
  std::vector<std::unique_ptr<TestTransitionState>> states;
  states.push_back(CreateState(kOldScores[0]));
  states.push_back(CreateState(kOldScores[1]));

  // Set parent beam indices.
  SetParentBeamIndex(states.at(0).get(), kParentBeamIndices[0]);
  SetParentBeamIndex(states.at(1).get(), kParentBeamIndices[1]);

  Beam<TestTransitionState> beam(kMaxBeamSize);
  beam.SetFunctions(null_permissions, null_finality, transition_function,
                    null_oracle);
  beam.Init(std::move(states));
  beam.AdvanceFromPrediction(matrix, kMatrixSize, kNumTransitions);

  // Validate the new beam.
  EXPECT_EQ(beam.beam().size(), 8);

  // Make sure the state has performed the expected transition.
  // Note that the transition index is not the index into the matrix, but rather
  // the index into the matrix 'row' for that state.
  EXPECT_EQ(GetTransition(beam.beam().at(0)), 2);
  EXPECT_EQ(GetTransition(beam.beam().at(1)), 2);
  EXPECT_EQ(GetTransition(beam.beam().at(2)), 0);
  EXPECT_EQ(GetTransition(beam.beam().at(3)), 0);
  EXPECT_EQ(GetTransition(beam.beam().at(4)), 1);
  EXPECT_EQ(GetTransition(beam.beam().at(5)), 1);
  EXPECT_EQ(GetTransition(beam.beam().at(6)), 3);
  EXPECT_EQ(GetTransition(beam.beam().at(7)), 3);

  // Make sure the state has had its score updated properly.
  EXPECT_EQ(beam.beam().at(0)->GetScore(), kOldScores[1] + matrix[6]);
  EXPECT_EQ(beam.beam().at(1)->GetScore(), kOldScores[0] + matrix[2]);
  EXPECT_EQ(beam.beam().at(2)->GetScore(), kOldScores[1] + matrix[4]);
  EXPECT_EQ(beam.beam().at(3)->GetScore(), kOldScores[0] + matrix[0]);
  EXPECT_EQ(beam.beam().at(4)->GetScore(), kOldScores[1] + matrix[5]);
  EXPECT_EQ(beam.beam().at(5)->GetScore(), kOldScores[0] + matrix[1]);
  EXPECT_EQ(beam.beam().at(6)->GetScore(), kOldScores[1] + matrix[7]);
  EXPECT_EQ(beam.beam().at(7)->GetScore(), kOldScores[0] + matrix[3]);

  // Make sure that the beam index field is consistent with the actual beam idx.
  for (int i = 0; i < beam.beam().size(); ++i) {
    EXPECT_EQ(beam.beam().at(i)->GetBeamIndex(), i);
  }

  // Validate the history at this step.
  auto history = beam.history();
  EXPECT_EQ(history.at(1).at(0), 1);
  EXPECT_EQ(history.at(1).at(1), 0);
  EXPECT_EQ(history.at(1).at(2), 1);
  EXPECT_EQ(history.at(1).at(3), 0);
  EXPECT_EQ(history.at(1).at(4), 1);
  EXPECT_EQ(history.at(1).at(5), 0);
  EXPECT_EQ(history.at(1).at(6), 1);
  EXPECT_EQ(history.at(1).at(7), 0);

  EXPECT_EQ(history.at(0).at(0), kParentBeamIndices[0]);
  EXPECT_EQ(history.at(0).at(1), kParentBeamIndices[1]);
  EXPECT_EQ(history.at(0).at(2), -1);
  EXPECT_EQ(history.at(0).at(3), -1);
  EXPECT_EQ(history.at(0).at(4), -1);
  EXPECT_EQ(history.at(0).at(5), -1);
  EXPECT_EQ(history.at(0).at(6), -1);
  EXPECT_EQ(history.at(0).at(7), -1);

  // Make sure that FindPreviousIndex can read through the history from step 1
  // to step 0.
  constexpr int kDesiredIndex = 0;
  constexpr int kCurrentIndexOne = 4;
  EXPECT_EQ(beam.FindPreviousIndex(kCurrentIndexOne, kDesiredIndex),
            kParentBeamIndices[1]);

  constexpr int kCurrentIndexTwo = 7;
  EXPECT_EQ(beam.FindPreviousIndex(kCurrentIndexTwo, kDesiredIndex),
            kParentBeamIndices[0]);
}

TEST(BeamTest, FindPreviousIndexReturnsInError) {
  // Create the beam. This now has only one history state, 0.
  std::vector<std::unique_ptr<TestTransitionState>> states;
  states.push_back(CreateState(0.0));
  constexpr int kMaxBeamSize = 8;
  Beam<TestTransitionState> beam(kMaxBeamSize);
  beam.SetFunctions(null_permissions, null_finality, transition_function,
                    null_oracle);
  beam.Init(std::move(states));

  // If the requested step is greater than the number of steps taken, expect -1.
  EXPECT_EQ(beam.FindPreviousIndex(0, 1), -1);

  // If the requested step is less than 0, expect -1.
  EXPECT_EQ(beam.FindPreviousIndex(0, -1), -1);

  // If the requested index does not have a state, expect -1.
  EXPECT_EQ(beam.FindPreviousIndex(0, 1), -1);

  // If the requested index is less than 0, expect -1.
  EXPECT_EQ(beam.FindPreviousIndex(0, -1), -1);

  // If the requested index is larger than the maximum beam size -1, expect -1.
  EXPECT_EQ(beam.FindPreviousIndex(0, kMaxBeamSize), -1);
}

TEST(BeamTest, ResetClearsBeamState) {
  // Create the beam
  std::vector<std::unique_ptr<TestTransitionState>> states;
  states.push_back(CreateState(1.0));
  constexpr int kMaxBeamSize = 8;
  Beam<TestTransitionState> beam(kMaxBeamSize);
  beam.SetFunctions(null_permissions, null_finality, transition_function,
                    null_oracle);
  beam.Init(std::move(states));

  // Validate the new beam.
  EXPECT_EQ(beam.beam().size(), 1);

  // Reset the beam.
  beam.Reset();

  // Validate the now-reset beam, which should be empty.
  EXPECT_EQ(beam.beam().size(), 0);
}

TEST(BeamTest, ResetClearsBeamHistory) {
  // Create the beam
  std::vector<std::unique_ptr<TestTransitionState>> states;
  states.push_back(CreateState(1.0));
  constexpr int kMaxBeamSize = 8;
  Beam<TestTransitionState> beam(kMaxBeamSize);
  beam.SetFunctions(null_permissions, null_finality, transition_function,
                    null_oracle);
  beam.Init(std::move(states));

  // Validate the new beam history.
  EXPECT_EQ(beam.history().size(), 1);

  // Reset the beam.
  beam.Reset();

  // Validate the now-reset beam history, which should be empty.
  EXPECT_EQ(beam.history().size(), 0);
}

TEST(BeamTest, SettingMaxSizeResetsBeam) {
  // Create the beam
  std::vector<std::unique_ptr<TestTransitionState>> states;
  states.push_back(CreateState(1.0));
  constexpr int kMaxBeamSize = 8;
  Beam<TestTransitionState> beam(kMaxBeamSize);
  beam.SetFunctions(null_permissions, null_finality, transition_function,
                    null_oracle);
  beam.Init(std::move(states));

  // Validate the new beam history.
  EXPECT_EQ(beam.history().size(), 1);

  // Reset the beam.
  constexpr int kNewMaxBeamSize = 4;
  beam.SetMaxSize(kNewMaxBeamSize);
  EXPECT_EQ(beam.max_size(), kNewMaxBeamSize);

  // Validate the now-reset beam history, which should be empty.
  EXPECT_EQ(beam.history().size(), 0);
}

}  // namespace dragnn
}  // namespace syntaxnet
