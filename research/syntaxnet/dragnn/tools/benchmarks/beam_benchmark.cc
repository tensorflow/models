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

#include "dragnn/core/beam.h"

#include <limits>
#include <random>

#include "dragnn/core/interfaces/cloneable_transition_state.h"
#include "dragnn/core/interfaces/transition_state.h"
#include "dragnn/core/test/mock_transition_state.h"
#include <gmock/gmock.h>
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/test_benchmark.h"

namespace syntaxnet {
namespace dragnn {

using testing::MockFunction;
using testing::Ne;
using testing::Return;
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
  TestTransitionState() : is_gold_(false) {}

  void Init(const TransitionState &parent) override {}

  std::unique_ptr<TestTransitionState> Clone() const override {
    std::unique_ptr<TestTransitionState> ptr(new TestTransitionState());
    return ptr;
  }

  int ParentBeamIndex() const override { return parent_beam_index_; }

  // Gets the current beam index for this state.
  int GetBeamIndex() const override { return beam_index_; }

  // Sets the current beam index for this state.
  void SetBeamIndex(int index) override { beam_index_ = index; }

  // Gets the score associated with this transition state.
  float GetScore() const override { return score_; }

  // Sets the score associated with this transition state.
  void SetScore(float score) override { score_ = score; }

  // Gets the gold-ness of this state (whether it is on the oracle path)
  bool IsGold() const override { return is_gold_; }

  // Sets the gold-ness of this state.
  void SetGold(bool is_gold) override { is_gold_ = is_gold; }

  // Depicts this state as an HTML-language string.
  string HTMLRepresentation() const override { return ""; }

  int parent_beam_index_;

  int beam_index_;

  float score_;

  int transition_action_;

  bool is_gold_;
};

// This transition function annotates a TestTransitionState with the action that
// was chosen for the transition.
auto transition_function = [](TestTransitionState *state, int action) {
  TestTransitionState *cast_state = dynamic_cast<TestTransitionState *>(state);
  cast_state->transition_action_ = action;
};

// Creates oracle and permission functions that do nothing.
auto null_oracle = [](TestTransitionState *) -> const vector<int> {
  return {0};
};
auto null_permissions = [](TestTransitionState *, int) { return true; };
auto null_finality = [](TestTransitionState *) { return false; };

// Creates a unique_ptr with a test transition state in it and set its initial
// score.
std::unique_ptr<TestTransitionState> CreateState(float score) {
  std::unique_ptr<TestTransitionState> state;
  state.reset(new TestTransitionState());
  state->SetScore(score);
  return state;
}

}  // namespace

// *****************************************************************************
// Tests begin here.
// *****************************************************************************
// Helper function for creating random transition matrices of a particular size.
std::vector<float> MakeRandomVector(int size) {
  std::default_random_engine engine;
  std::uniform_real_distribution<float> dist(0., 10.);
  auto gen = std::bind(dist, engine);
  std::vector<float> vec(size);
  std::generate(vec.begin(), vec.end(), gen);
  return vec;
}

// Benchmark Beam::FastAdvanceFromPrediction for a beam size of 1 and
// a variety of transition system sizes.
void BM_FastAdvance(int num_iters, int num_transitions) {
  tensorflow::testing::StopTiming();

  // Create a matrix of transitions.
  constexpr int kMaxBeamSize = 1;
  const int matrix_size = num_transitions * kMaxBeamSize;
  const std::vector<float> matrix = MakeRandomVector(matrix_size);

  // Create the beam and transition it.
  std::vector<std::unique_ptr<TestTransitionState>> states;
  constexpr float kOldScore = 4.0;
  states.push_back(CreateState(kOldScore));

  Beam<TestTransitionState> beam(kMaxBeamSize);
  beam.SetFunctions(null_permissions, null_finality, transition_function,
                    null_oracle);
  beam.Init(std::move(states));
  beam.AdvanceFromPrediction(matrix.data(), matrix_size, num_transitions);
  ASSERT_EQ(beam.beam().size(), kMaxBeamSize);

  tensorflow::testing::StartTiming();
  for (int i = 0; i < num_iters; ++i) {
    beam.FastAdvanceFromPrediction(matrix.data(), num_transitions);
  }
  ASSERT_EQ(beam.beam().size(), kMaxBeamSize);
}
BENCHMARK(BM_FastAdvance)->Range(2, 128);

// Benchmark Beam::BeamAdvanceFromPrediction for a variety of beam
// sizes and transition system sizes.
void BM_BeamAdvance(int num_iters, int num_transitions, int max_beam_size) {
  tensorflow::testing::StopTiming();

  // Create a matrix of transitions.
  const int matrix_size = num_transitions * max_beam_size;
  const std::vector<float> matrix = MakeRandomVector(matrix_size);

  // Create the beam and transition it.
  std::vector<std::unique_ptr<TestTransitionState>> states;
  constexpr float kOldScore = 4.0;
  states.push_back(CreateState(kOldScore));

  Beam<TestTransitionState> beam(max_beam_size);
  beam.SetFunctions(null_permissions, null_finality, transition_function,
                    null_oracle);
  beam.Init(std::move(states));
  while (beam.beam().size() < max_beam_size) {
    beam.AdvanceFromPrediction(matrix.data(), matrix_size, num_transitions);
  }
  ASSERT_EQ(beam.beam().size(), max_beam_size);

  tensorflow::testing::StartTiming();
  for (int i = 0; i < num_iters; ++i) {
    beam.BeamAdvanceFromPrediction(matrix.data(), matrix_size, num_transitions);
  }
  ASSERT_EQ(beam.beam().size(), max_beam_size);
}
BENCHMARK(BM_BeamAdvance)->RangePair(2, 128, 1, 64);

}  // namespace dragnn
}  // namespace syntaxnet
