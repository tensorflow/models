#ifndef NLP_SAFT_OPENSOURCE_DRAGNN_CORE_BEAM_H_
#define NLP_SAFT_OPENSOURCE_DRAGNN_CORE_BEAM_H_

#include <algorithm>
#include <memory>
#include <vector>

#include "dragnn/core/interfaces/cloneable_transition_state.h"
#include "dragnn/core/interfaces/transition_state.h"
#include "tensorflow/core/platform/logging.h"

namespace syntaxnet {
namespace dragnn {

// The Beam class wraps the logic necessary to advance a set of transition
// states for an arbitrary Component. Because the Beam class is generic, it
// doesn't know how to act on the states it is provided - the instantiating
// Component is expected to provide it the three functions it needs to interact
// with that Component's TransitionState subclasses.

template <typename T>
class Beam {
 public:
  // Creates a new Beam which can grow up to max_size elements.
  explicit Beam(int max_size) : max_size_(max_size), num_steps_(0) {
    VLOG(2) << "Creating beam with max size " << max_size_;
    static_assert(
        std::is_base_of<CloneableTransitionState<T>, T>::value,
        "This class must be instantiated to use a CloneableTransitionState");
  }

  // Sets the Beam functions, as follows:
  // bool is_allowed(TransitionState *, int): Return true if transition 'int' is
  //   allowed for transition state 'TransitionState *'.
  // void perform_transition(TransitionState *, int): Performs transition 'int'
  //   on transition state 'TransitionState *'.
  // int oracle_function(TransitionState *): Returns the oracle-specified action
  //   for transition state 'TransitionState *'.
  void SetFunctions(std::function<bool(T *, int)> is_allowed,
                    std::function<bool(T *)> is_final,
                    std::function<void(T *, int)> perform_transition,
                    std::function<int(T *)> oracle_function) {
    is_allowed_ = is_allowed;
    is_final_ = is_final;
    perform_transition_ = perform_transition;
    oracle_function_ = oracle_function;
  }

  // Resets the Beam and initializes it with the given set of states. The Beam
  // takes ownership of these TransitionStates.
  void Init(std::vector<std::unique_ptr<T>> initial_states) {
    VLOG(2) << "Initializing beam. Beam max size is " << max_size_;
    CHECK_LE(initial_states.size(), max_size_)
        << "Attempted to initialize a beam with more states ("
        << initial_states.size() << ") than the max size " << max_size_;
    beam_ = std::move(initial_states);
    std::vector<int> previous_beam_indices(max_size_, -1);
    for (int i = 0; i < beam_.size(); ++i) {
      previous_beam_indices.at(i) = beam_[i]->ParentBeamIndex();
      beam_[i]->SetBeamIndex(i);
    }
    beam_index_history_.emplace_back(previous_beam_indices);
  }

  // Advances the Beam from the given transition matrix.
  void AdvanceFromPrediction(const float transition_matrix[], int matrix_length,
                             int num_actions) {
    // Ensure that the transition matrix is the correct size. All underlying
    // states should have the same transition profile, so using the one at 0
    // should be safe.
    CHECK_EQ(matrix_length, max_size_ * num_actions)
        << "Transition matrix size does not match max beam size * number of "
           "state transitions!";

    if (max_size_ == 1) {
      // In the case where beam size is 1, we can advance by simply finding the
      // highest score and advancing the beam state in place.
      VLOG(2) << "Beam size is 1. Using fast beam path.";
      int best_action = -1;
      float best_score = -INFINITY;
      auto &state = beam_[0];
      for (int action_idx = 0; action_idx < num_actions; ++action_idx) {
        if (is_allowed_(state.get(), action_idx) &&
            transition_matrix[action_idx] > best_score) {
          best_score = transition_matrix[action_idx];
          best_action = action_idx;
        }
      }
      CHECK_GE(best_action, 0) << "Num actions: " << num_actions
                               << " score[0]: " << transition_matrix[0];
      perform_transition_(state.get(), best_action);
      const float new_score = state->GetScore() + best_score;
      state->SetScore(new_score);
      state->SetBeamIndex(0);
    } else {
      // Create the vector of all possible transitions, along with their scores.
      std::vector<Transition> candidates;

      // Iterate through all beams, examining all actions for each beam.
      for (int beam_idx = 0; beam_idx < beam_.size(); ++beam_idx) {
        const auto &state = beam_[beam_idx];
        for (int action_idx = 0; action_idx < num_actions; ++action_idx) {
          // If the action is allowed, calculate the proposed new score and add
          // the candidate action to the vector of all actions at this state.
          if (is_allowed_(state.get(), action_idx)) {
            Transition candidate;

            // The matrix is laid out by beam index, with a linear set of
            // actions for that index - so beam N's actions start at [nr. of
            // actions]*[N].
            const int matrix_idx = action_idx + beam_idx * num_actions;
            CHECK_LT(matrix_idx, matrix_length)
                << "Matrix index out of bounds!";
            const double score_delta = transition_matrix[matrix_idx];
            CHECK(!isnan(score_delta));
            candidate.source_idx = beam_idx;
            candidate.action = action_idx;
            candidate.resulting_score = state->GetScore() + score_delta;
            candidates.emplace_back(candidate);
          }
        }
      }

      // Sort the vector of all possible transitions and scores.
      const auto comparator = [](const Transition &a, const Transition &b) {
        return a.resulting_score > b.resulting_score;
      };
      std::sort(candidates.begin(), candidates.end(), comparator);

      // Apply the top transitions, up to a maximum of 'max_size_'.
      std::vector<std::unique_ptr<T>> new_beam;
      std::vector<int> previous_beam_indices(max_size_, -1);
      const int beam_size =
          std::min(max_size_, static_cast<int>(candidates.size()));
      VLOG(2) << "Previous beam size = " << beam_.size();
      VLOG(2) << "New beam size = " << beam_size;
      VLOG(2) << "Maximum beam size = " << max_size_;
      for (int i = 0; i < beam_size; ++i) {
        // Get the source of the i'th transition.
        const auto &transition = candidates[i];
        VLOG(2) << "Taking transition with score: "
                << transition.resulting_score
                << " and action: " << transition.action;
        VLOG(2) << "transition.source_idx = " << transition.source_idx;
        const auto &source = beam_[transition.source_idx];

        // Put the new transition on the new state beam.
        auto new_state = source->Clone();
        perform_transition_(new_state.get(), transition.action);
        new_state->SetScore(transition.resulting_score);
        new_state->SetBeamIndex(i);
        previous_beam_indices.at(i) = transition.source_idx;
        new_beam.emplace_back(std::move(new_state));
      }

      beam_ = std::move(new_beam);
      beam_index_history_.emplace_back(previous_beam_indices);
    }

    ++num_steps_;
  }

  // Advances the Beam from the state oracles.
  void AdvanceFromOracle() {
    std::vector<int> previous_beam_indices(max_size_, -1);
    for (int i = 0; i < beam_.size(); ++i) {
      previous_beam_indices.at(i) = i;
      if (is_final_(beam_[i].get())) continue;
      const auto oracle_label = oracle_function_(beam_[i].get());
      VLOG(2) << "AdvanceFromOracle beam_index:" << i
              << " oracle_label:" << oracle_label;
      perform_transition_(beam_[i].get(), oracle_label);
      beam_[i]->SetScore(0.0);
      beam_[i]->SetBeamIndex(i);
    }
    if (max_size_ > 1) {
      beam_index_history_.emplace_back(previous_beam_indices);
    }
    num_steps_++;
  }

  // Returns true if all states in the beam are final.
  bool IsTerminal() {
    for (auto &state : beam_) {
      if (!is_final_(state.get())) {
        return false;
      }
    }
    return true;
  }

  // Destroys the states held by this beam and resets its history.
  void Reset() {
    beam_.clear();
    beam_index_history_.clear();
    num_steps_ = 0;
  }

  // Given an index into the current beam, determine the index of the item's
  // parent at beam step "step", which should be less than the total number
  // of steps taken by this beam.
  int FindPreviousIndex(int current_index, int step) const {
    VLOG(2) << "FindPreviousIndex requested for current_index:" << current_index
            << " at step:" << step;
    if (VLOG_IS_ON(2)) {
      int step_index = 0;
      for (const auto &step : beam_index_history_) {
        string row =
            "Step " + std::to_string(step_index) + " element source slot: ";
        for (const auto &index : step) {
          if (index == -1) {
            row += "  X";
          } else {
            row += "  " + std::to_string(index);
          }
        }
        VLOG(2) << row;
        ++step_index;
      }
    }

    // If the max size of the beam is 1, make sure the steps are in sync with
    // the size.
    if (max_size_ > 1) {
      CHECK(num_steps_ == beam_index_history_.size() - 1);
    }

    // Check if the step is too far into the past or future.
    if (step < 0 || step > num_steps_) {
      return -1;
    }

    // Check that the index is within the beam.
    if (current_index < 0 || current_index >= max_size_) {
      return -1;
    }

    // If the max size of the beam is 1, always return 0.
    if (max_size_ == 1) {
      return 0;
    }

    // Check that the start index isn't -1; -1 means that we don't have an
    // actual transition state in that beam slot.
    if (beam_index_history_.back().at(current_index) == -1) {
      return -1;
    }

    int beam_index = current_index;
    for (int i = beam_index_history_.size() - 1; i >= step; --i) {
      beam_index = beam_index_history_.at(i).at(beam_index);
    }
    CHECK_GE(beam_index, 0);
    VLOG(2) << "Index is " << beam_index;
    return beam_index;
  }

  // Returns the current state of the beam.
  std::vector<const TransitionState *> beam() const {
    std::vector<const TransitionState *> state_ptrs;
    for (const auto &beam_state : beam_) {
      state_ptrs.emplace_back(beam_state.get());
    }
    return state_ptrs;
  }

  // Returns the beam at the current state index.
  T *beam_state(int beam_index) { return beam_.at(beam_index).get(); }

  // Returns the raw history vectors for this beam.
  const std::vector<std::vector<int>> &history() {
    if (max_size_ == 1) {
      // If max size is 1, we haven't been keeping track of the beam. Quick
      // create it.
      beam_index_history_.clear();
      beam_index_history_.push_back({beam_[0]->ParentBeamIndex()});
      for (int i = 0; i < num_steps_; ++i) {
        beam_index_history_.push_back({0});
      }
    }
    return beam_index_history_;
  }

  // Sets the max size of the beam.
  void SetMaxSize(int max_size) {
    max_size_ = max_size;
    Reset();
  }

  // Returns the number of steps taken so far.
  const int num_steps() const { return num_steps_; }

  // Returns the max size of this beam.
  const int max_size() const { return max_size_; }

  // Returns the current size of the beam.
  const int size() const { return beam_.size(); }

 private:
  // Associates an action taken on an index into current_state_ with a score.
  struct Transition {
    // The index of the source item.
    int source_idx;

    // The index of the action being taken.
    int action;

    // The score of the full derivation.
    double resulting_score;
  };

  // The maximum beam size.
  int max_size_;

  // The current beam.
  std::vector<std::unique_ptr<T>> beam_;

  // Function to check if a transition is allowed for a given state.
  std::function<bool(T *, int)> is_allowed_;

  // Function to check if a state is final.
  std::function<int(T *)> is_final_;

  // Function to perform a transition on a given state.
  std::function<void(T *, int)> perform_transition_;

  // Function to provide the oracle action for a given state.
  std::function<int(T *)> oracle_function_;

  // The history of the states in this beam. The vector indexes across steps.
  // For every step, there is a vector in the vector. This inner vector denotes
  // the state of the beam at that step, and contains the beam index that
  // was transitioned to create the transition state at that index (so,
  // if at step 2 the transition state at beam index 4 was created by applying
  // a transition to the state in beam index 3 during step 1, the query would
  // be "beam_index_history_.at(2).at(4)" and the value would be 3. Empty beam
  // states will return -1.
  std::vector<std::vector<int>> beam_index_history_;

  // The number of steps taken so far.
  int num_steps_;
};

}  // namespace dragnn
}  // namespace syntaxnet

#endif  // NLP_SAFT_OPENSOURCE_DRAGNN_CORE_BEAM_H_
