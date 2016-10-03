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

#include <algorithm>
#include <deque>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "syntaxnet/base.h"
#include "syntaxnet/parser_state.h"
#include "syntaxnet/parser_transitions.h"
#include "syntaxnet/sentence.pb.h"
#include "syntaxnet/sentence_batch.h"
#include "syntaxnet/shared_store.h"
#include "syntaxnet/sparse.pb.h"
#include "syntaxnet/task_context.h"
#include "syntaxnet/task_spec.pb.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/env.h"

using tensorflow::DEVICE_CPU;
using tensorflow::DT_BOOL;
using tensorflow::DT_FLOAT;
using tensorflow::DT_INT32;
using tensorflow::DT_INT64;
using tensorflow::DT_STRING;
using tensorflow::DataType;
using tensorflow::OpKernel;
using tensorflow::OpKernelConstruction;
using tensorflow::OpKernelContext;
using tensorflow::TTypes;
using tensorflow::Tensor;
using tensorflow::TensorShape;
using tensorflow::errors::FailedPrecondition;
using tensorflow::errors::InvalidArgument;

namespace syntaxnet {

// Wraps ParserState so that the history of transitions (actions
// performed and the beam slot they were performed in) are recorded.
struct ParserStateWithHistory {
 public:
  // New state with an empty history.
  explicit ParserStateWithHistory(const ParserState &s) : state(s.Clone()) {}

  // New state obtained by cloning the given state and applying the given
  // action. The given beam slot and action are appended to the history.
  ParserStateWithHistory(const ParserStateWithHistory &next,
                         const ParserTransitionSystem &transitions, int32 slot,
                         int32 action, float score)
      : state(next.state->Clone()),
        slot_history(next.slot_history),
        action_history(next.action_history),
        score_history(next.score_history) {
    transitions.PerformAction(action, state.get());
    slot_history.push_back(slot);
    action_history.push_back(action);
    score_history.push_back(score);
  }

  std::unique_ptr<ParserState> state;
  std::vector<int32> slot_history;
  std::vector<int32> action_history;
  std::vector<float> score_history;

 private:
  TF_DISALLOW_COPY_AND_ASSIGN(ParserStateWithHistory);
};

struct BatchStateOptions {
  // Maximum number of parser states in a beam.
  int max_beam_size;

  // Number of parallel sentences to decode.
  int batch_size;

  // Argument prefix for context parameters.
  string arg_prefix;

  // Corpus name to read from from context inputs.
  string corpus_name;

  // Whether we allow weights in SparseFeatures protos.
  bool allow_feature_weights;

  // Whether beams should be considered alive until all states are final, or
  // until the gold path falls off.
  bool continue_until_all_final;

  // Whether to skip to a new sentence after each training step.
  bool always_start_new_sentences;

  // Parameter for deciding which tokens to score.
  string scoring_type;
};

// Encapsulates the environment needed to parse with a beam, keeping a
// record of path histories.
class BeamState {
 public:
  // The agenda is keyed by a tuple that is the score followed by an
  // int that is -1 if the path coincides with the gold path and 0
  // otherwise. The lexicographic ordering of the keys therefore
  // ensures that for all paths sharing the same score, the gold path
  // will always be at the bottom. This situation can occur at the
  // onset of training when all weights are zero and therefore all
  // paths have an identically zero score.
  typedef std::pair<double, int> KeyType;
  typedef std::multimap<KeyType, std::unique_ptr<ParserStateWithHistory>>
      AgendaType;
  typedef std::pair<const KeyType, std::unique_ptr<ParserStateWithHistory>>
      AgendaItem;
  typedef Eigen::Tensor<float, 2, Eigen::RowMajor, Eigen::DenseIndex>
      ScoreMatrixType;

  // The beam can be
  //   - ALIVE: parsing is still active, features are being output for at least
  //     some slots in the beam.
  //   - DYING: features should be output for this beam only one more time, then
  //     the beam will be DEAD. This state is reached when the gold path falls
  //     out of the beam and features have to be output one last time.
  //   - DEAD: parsing is not active, features are not being output and the no
  //     actions are taken on the states.
  enum State { ALIVE = 0, DYING = 1, DEAD = 2 };

  explicit BeamState(const BatchStateOptions &options) : options_(options) {}

  void Reset() {
    if (options_.always_start_new_sentences ||
        gold_ == nullptr || transition_system_->IsFinalState(*gold_)) {
      AdvanceSentence();
    }
    slots_.clear();
    if (gold_ == nullptr) {
      state_ = DEAD;  // EOF has been reached.
    } else {
      gold_->set_is_gold(true);
      slots_.emplace(KeyType(0.0, -1), std::unique_ptr<ParserStateWithHistory>(
          new ParserStateWithHistory(*gold_)));
      state_ = ALIVE;
    }
  }

  void UpdateAllFinal() {
    all_final_ = true;
    for (const AgendaItem &item : slots_) {
      if (!transition_system_->IsFinalState(*item.second->state)) {
        all_final_ = false;
        break;
      }
    }
    if (all_final_) {
      state_ = DEAD;
    }
  }

  // This method updates the beam. For all elements of the beam, all
  // allowed transitions are scored and insterted into a new beam. The
  // beam size is capped by discarding the lowest scoring slots at any
  // given time. There is one exception to this process: the gold path
  // is forced to remain in the beam at all times, even if it scores
  // low. This is to ensure that the gold path can be used for
  // training at the moment it would otherwise fall off (and be absent
  // from) the beam.
  void Advance(const ScoreMatrixType &scores) {
    // If the beam was in the state of DYING, it is now DEAD.
    if (state_ == DYING) state_ = DEAD;

    // When to stop advancing depends on the 'continue_until_all_final' arg.
    if (!IsAlive() || gold_ == nullptr) return;

    AdvanceGold();

    const int score_rows = scores.dimension(0);
    const int num_actions = scores.dimension(1);

    // Advance beam.
    AgendaType previous_slots;
    previous_slots.swap(slots_);

    CHECK_EQ(state_, ALIVE);

    int slot = 0;
    for (AgendaItem &item : previous_slots) {
      {
        ParserState *current = item.second->state.get();
        VLOG(2) << "Slot: " << slot;
        VLOG(2) << "Parser state: " << current->ToString();
        VLOG(2) << "Parser state cumulative score: " << item.first.first << " "
                << (item.first.second < 0 ? "golden" : "");
      }
      if (!transition_system_->IsFinalState(*item.second->state)) {
        // Not a final state.
        for (int action = 0; action < num_actions; ++action) {
          // Is action allowed?
          if (!transition_system_->IsAllowedAction(action,
                                                   *item.second->state)) {
            continue;
          }
          CHECK_LT(slot, score_rows);
          MaybeInsertWithNewAction(item, slot, scores(slot, action), action);
          PruneBeam();
        }
      } else {
        // Final state: no need to advance.
        MaybeInsert(&item);
        PruneBeam();
      }
      ++slot;
    }
    UpdateAllFinal();
  }

  void PopulateFeatureOutputs(
      std::vector<std::vector<std::vector<SparseFeatures>>> *features) {
    for (const AgendaItem &item : slots_) {
      VLOG(2) << "State: " << item.second->state->ToString();
      std::vector<std::vector<SparseFeatures>> f =
          features_->ExtractSparseFeatures(*workspace_, *item.second->state);
      for (size_t i = 0; i < f.size(); ++i) (*features)[i].push_back(f[i]);
    }
  }

  int BeamSize() const { return slots_.size(); }

  bool IsAlive() const { return state_ == ALIVE; }

  bool IsDead() const { return state_ == DEAD; }

  bool AllFinal() const { return all_final_; }

  // The current contents of the beam.
  AgendaType slots_;

  // Which batch this refers to.
  int beam_id_ = 0;

  // Sentence batch reader.
  SentenceBatch *sentence_batch_ = nullptr;

  // Label map.
  const TermFrequencyMap *label_map_ = nullptr;

  // Transition system.
  const ParserTransitionSystem *transition_system_ = nullptr;

  // Feature extractor.
  const ParserEmbeddingFeatureExtractor *features_ = nullptr;

  // Feature workspace set.
  WorkspaceSet *workspace_ = nullptr;

  // Internal workspace registry for use in feature extraction.
  WorkspaceRegistry *workspace_registry_ = nullptr;

  // ParserState used to get gold actions.
  std::unique_ptr<ParserState> gold_;

 private:
  // Creates a new ParserState if there's another sentence to be read.
  void AdvanceSentence() {
    gold_.reset();
    if (sentence_batch_->AdvanceSentence(beam_id_)) {
      gold_.reset(new ParserState(sentence_batch_->sentence(beam_id_),
                                  transition_system_->NewTransitionState(true),
                                  label_map_));
      workspace_->Reset(*workspace_registry_);
      features_->Preprocess(workspace_, gold_.get());
    }
  }

  void AdvanceGold() {
    gold_action_ = -1;
    if (!transition_system_->IsFinalState(*gold_)) {
      gold_action_ = transition_system_->GetNextGoldAction(*gold_);
      if (transition_system_->IsAllowedAction(gold_action_, *gold_)) {
        // In cases where the gold annotation is incompatible with the
        // transition system, the action returned as gold might be not allowed.
        transition_system_->PerformAction(gold_action_, gold_.get());
      }
    }
  }

  // Removes the first non-gold beam element if the beam is larger than
  // the maximum beam size. If the gold element was at the bottom of the
  // beam, sets the beam state to DYING, otherwise leaves the state alone.
  void PruneBeam() {
    if (static_cast<int>(slots_.size()) > options_.max_beam_size) {
      auto bottom = slots_.begin();
      if (!options_.continue_until_all_final &&
          bottom->second->state->is_gold()) {
        state_ = DYING;
        ++bottom;
      }
      slots_.erase(bottom);
    }
  }

  // Inserts an item in the beam if
  //   - the item is gold,
  //   - the beam is not full, or
  //   - the item's new score is greater than the lowest score in the beam after
  //     the score has been incremented by given delta_score.
  // Inserted items have slot, delta_score and action appended to their history.
  void MaybeInsertWithNewAction(const AgendaItem &item, const int slot,
                                const double delta_score, const int action) {
    const double score = item.first.first + delta_score;
    const bool is_gold =
        item.second->state->is_gold() && action == gold_action_;
    if (is_gold || static_cast<int>(slots_.size()) < options_.max_beam_size ||
        score > slots_.begin()->first.first) {
      const KeyType key{score, -static_cast<int>(is_gold)};
      slots_.emplace(key, std::unique_ptr<ParserStateWithHistory>(
                              new ParserStateWithHistory(
                                  *item.second, *transition_system_, slot,
                                  action, delta_score)))
          ->second->state->set_is_gold(is_gold);
    }
  }

  // Inserts an item in the beam if
  //   - the item is gold,
  //   - the beam is not full, or
  //   - the item's new score is greater than the lowest score in the beam.
  // The history of inserted items is left untouched.
  void MaybeInsert(AgendaItem *item) {
    const bool is_gold = item->second->state->is_gold();
    const double score = item->first.first;
    if (is_gold || static_cast<int>(slots_.size()) < options_.max_beam_size ||
        score > slots_.begin()->first.first) {
      slots_.emplace(item->first, std::move(item->second));
    }
  }

  // Limits the number of slots on the beam.
  const BatchStateOptions &options_;

  int gold_action_ = -1;
  State state_ = ALIVE;
  bool all_final_ = false;
  TF_DISALLOW_COPY_AND_ASSIGN(BeamState);
};

// Encapsulates the state of a batch of beams. It is an object of this
// type that will persist through repeated Op evaluations as the
// multiple steps are computed in sequence.
class BatchState {
 public:
  explicit BatchState(const BatchStateOptions &options)
      : options_(options), features_(options.arg_prefix) {}

  ~BatchState() { SharedStore::Release(label_map_); }

  void Init(TaskContext *task_context) {
    // Create sentence batch.
    sentence_batch_.reset(
        new SentenceBatch(BatchSize(), options_.corpus_name));
    sentence_batch_->Init(task_context);

    // Create transition system.
    transition_system_.reset(ParserTransitionSystem::Create(task_context->Get(
        tensorflow::strings::StrCat(options_.arg_prefix, "_transition_system"),
        "arc-standard")));
    transition_system_->Setup(task_context);
    transition_system_->Init(task_context);

    // Create label map.
    string label_map_path =
        TaskContext::InputFile(*task_context->GetInput("label-map"));
    label_map_ = SharedStoreUtils::GetWithDefaultName<TermFrequencyMap>(
        label_map_path, 0, 0);

    // Setup features.
    features_.Setup(task_context);
    features_.Init(task_context);
    features_.RequestWorkspaces(&workspace_registry_);

    // Create workspaces.
    workspaces_.resize(BatchSize());

    // Create beams.
    beams_.clear();
    for (int beam_id = 0; beam_id < BatchSize(); ++beam_id) {
      beams_.emplace_back(options_);
      beams_[beam_id].beam_id_ = beam_id;
      beams_[beam_id].sentence_batch_ = sentence_batch_.get();
      beams_[beam_id].transition_system_ = transition_system_.get();
      beams_[beam_id].label_map_ = label_map_;
      beams_[beam_id].features_ = &features_;
      beams_[beam_id].workspace_ = &workspaces_[beam_id];
      beams_[beam_id].workspace_registry_ = &workspace_registry_;
    }
  }

  void ResetBeams() {
    for (BeamState &beam : beams_) {
      beam.Reset();
    }

    // Rewind if no states remain in the batch (we need to rewind the corpus).
    if (sentence_batch_->size() == 0) {
      ++epoch_;
      VLOG(2) << "Starting epoch " << epoch_;
      sentence_batch_->Rewind();
    }
  }

  // Resets the offset vectors required for a single run because we're
  // starting a new matrix of scores.
  void ResetOffsets() {
    beam_offsets_.clear();
    step_offsets_ = {0};
    UpdateOffsets();
  }

  void AdvanceBeam(const int beam_id,
                   const TTypes<float>::ConstMatrix &scores) {
    const int offset = beam_offsets_.back()[beam_id];
    Eigen::array<Eigen::DenseIndex, 2> offsets = {offset, 0};
    Eigen::array<Eigen::DenseIndex, 2> extents = {
        beam_offsets_.back()[beam_id + 1] - offset, NumActions()};
    BeamState::ScoreMatrixType beam_scores = scores.slice(offsets, extents);
    beams_[beam_id].Advance(beam_scores);
  }

  void UpdateOffsets() {
    beam_offsets_.emplace_back(BatchSize() + 1, 0);
    std::vector<int> &offsets = beam_offsets_.back();
    for (int beam_id = 0; beam_id < BatchSize(); ++beam_id) {
      // If the beam is ALIVE or DYING (but not DEAD), we want to
      // output the activations.
      const BeamState &beam = beams_[beam_id];
      const int beam_size = beam.IsDead() ? 0 : beam.BeamSize();
      offsets[beam_id + 1] = offsets[beam_id] + beam_size;
    }
    const int output_size = offsets.back();
    step_offsets_.push_back(step_offsets_.back() + output_size);
  }

  tensorflow::Status PopulateFeatureOutputs(OpKernelContext *context) {
    const int feature_size = FeatureSize();
    std::vector<std::vector<std::vector<SparseFeatures>>> features(
        feature_size);
    for (int beam_id = 0; beam_id < BatchSize(); ++beam_id) {
      if (!beams_[beam_id].IsDead()) {
        beams_[beam_id].PopulateFeatureOutputs(&features);
      }
    }
    CHECK_EQ(features.size(), feature_size);
    Tensor *output;
    const int total_slots = beam_offsets_.back().back();
    for (int i = 0; i < feature_size; ++i) {
      std::vector<std::vector<SparseFeatures>> &f = features[i];
      CHECK_EQ(total_slots, f.size());
      if (total_slots == 0) {
        TF_RETURN_IF_ERROR(
            context->allocate_output(i, TensorShape({0, 0}), &output));
      } else {
        const int size = f[0].size();
        TF_RETURN_IF_ERROR(context->allocate_output(
            i, TensorShape({total_slots, size}), &output));
        for (int j = 0; j < total_slots; ++j) {
          CHECK_EQ(size, f[j].size());
          for (int k = 0; k < size; ++k) {
            if (!options_.allow_feature_weights && f[j][k].weight_size() > 0) {
              return FailedPrecondition(
                  "Feature weights are not allowed when allow_feature_weights "
                  "is set to false.");
            }
            output->matrix<string>()(j, k) = f[j][k].SerializeAsString();
          }
        }
      }
    }
    return tensorflow::Status::OK();
  }

  // Returns the offset (i.e. row number) of a particular beam at a
  // particular step in the final concatenated score matrix.
  int GetOffset(const int step, const int beam_id) const {
    return step_offsets_[step] + beam_offsets_[step][beam_id];
  }

  int FeatureSize() const { return features_.embedding_dims().size(); }

  int NumActions() const {
    return transition_system_->NumActions(label_map_->Size());
  }

  int BatchSize() const { return options_.batch_size; }

  const BeamState &Beam(const int i) const { return beams_[i]; }

  int Epoch() const { return epoch_; }

  const string &ScoringType() const { return options_.scoring_type; }

 private:
  const BatchStateOptions options_;

  // How many times the document source has been rewound.
  int epoch_ = 0;

  // Batch of sentences, and the corresponding parser states.
  std::unique_ptr<SentenceBatch> sentence_batch_;

  // Transition system.
  std::unique_ptr<ParserTransitionSystem> transition_system_;

  // Label map for transition system..
  const TermFrequencyMap *label_map_;

  // Typed feature extractor for embeddings.
  ParserEmbeddingFeatureExtractor features_;

  // Batch: WorkspaceSet objects.
  std::vector<WorkspaceSet> workspaces_;

  // Internal workspace registry for use in feature extraction.
  WorkspaceRegistry workspace_registry_;

  std::deque<BeamState> beams_;
  std::vector<std::vector<int>> beam_offsets_;

  // Keeps track of the slot offset of each step.
  std::vector<int> step_offsets_;
  TF_DISALLOW_COPY_AND_ASSIGN(BatchState);
};

// Creates a BeamState and hooks it up with a parser. This Op needs to
// remain alive for the duration of the parse.
class BeamParseReader : public OpKernel {
 public:
  explicit BeamParseReader(OpKernelConstruction *context) : OpKernel(context) {
    string file_path;
    int feature_size;
    BatchStateOptions options;
    OP_REQUIRES_OK(context, context->GetAttr("task_context", &file_path));
    OP_REQUIRES_OK(context, context->GetAttr("feature_size", &feature_size));
    OP_REQUIRES_OK(context,
                   context->GetAttr("beam_size", &options.max_beam_size));
    OP_REQUIRES_OK(context,
                   context->GetAttr("batch_size", &options.batch_size));
    OP_REQUIRES_OK(context,
                   context->GetAttr("arg_prefix", &options.arg_prefix));
    OP_REQUIRES_OK(context,
                   context->GetAttr("corpus_name", &options.corpus_name));
    OP_REQUIRES_OK(context, context->GetAttr("allow_feature_weights",
                                             &options.allow_feature_weights));
    OP_REQUIRES_OK(context,
                   context->GetAttr("continue_until_all_final",
                                    &options.continue_until_all_final));
    OP_REQUIRES_OK(context,
                   context->GetAttr("always_start_new_sentences",
                                    &options.always_start_new_sentences));

    // Reads task context from file.
    string data;
    OP_REQUIRES_OK(context, ReadFileToString(tensorflow::Env::Default(),
                                             file_path, &data));
    TaskContext task_context;
    OP_REQUIRES(context,
                TextFormat::ParseFromString(data, task_context.mutable_spec()),
                InvalidArgument("Could not parse task context at ", file_path));
    OP_REQUIRES(
        context, options.batch_size > 0,
        InvalidArgument("Batch size ", options.batch_size, " too small."));
    options.scoring_type = task_context.Get(
        tensorflow::strings::StrCat(options.arg_prefix, "_scoring"), "");

    // Create batch state.
    batch_state_.reset(new BatchState(options));
    batch_state_->Init(&task_context);

    // Check number of feature groups matches the task context.
    const int required_size = batch_state_->FeatureSize();
    OP_REQUIRES(
        context, feature_size == required_size,
        InvalidArgument("Task context requires feature_size=", required_size));

    // Set expected signature.
    std::vector<DataType> output_types(feature_size, DT_STRING);
    output_types.push_back(DT_INT64);
    output_types.push_back(DT_INT32);
    OP_REQUIRES_OK(context, context->MatchSignature({}, output_types));
  }

  void Compute(OpKernelContext *context) override {
    mutex_lock lock(mu_);

    // Write features.
    batch_state_->ResetBeams();
    batch_state_->ResetOffsets();
    batch_state_->PopulateFeatureOutputs(context);

    // Forward the beam state vector.
    Tensor *output;
    const int feature_size = batch_state_->FeatureSize();
    OP_REQUIRES_OK(context, context->allocate_output(feature_size,
                                                     TensorShape({}), &output));
    output->scalar<int64>()() = reinterpret_cast<int64>(batch_state_.get());

    // Output number of epochs.
    OP_REQUIRES_OK(context, context->allocate_output(feature_size + 1,
                                                     TensorShape({}), &output));
    output->scalar<int32>()() = batch_state_->Epoch();
  }

 private:
  // mutex to synchronize access to Compute.
  mutex mu_;

  // The object whose handle will be passed among the Ops.
  std::unique_ptr<BatchState> batch_state_;

  TF_DISALLOW_COPY_AND_ASSIGN(BeamParseReader);
};

REGISTER_KERNEL_BUILDER(Name("BeamParseReader").Device(DEVICE_CPU),
                        BeamParseReader);

// Updates the beam based on incoming scores and outputs new feature vectors
// based on the updated beam.
class BeamParser : public OpKernel {
 public:
  explicit BeamParser(OpKernelConstruction *context) : OpKernel(context) {
    int feature_size;
    OP_REQUIRES_OK(context, context->GetAttr("feature_size", &feature_size));

    // Set expected signature.
    std::vector<DataType> output_types(feature_size, DT_STRING);
    output_types.push_back(DT_INT64);
    output_types.push_back(DT_BOOL);
    OP_REQUIRES_OK(context,
                   context->MatchSignature({DT_INT64, DT_FLOAT}, output_types));
  }

  void Compute(OpKernelContext *context) override {
    BatchState *batch_state =
        reinterpret_cast<BatchState *>(context->input(0).scalar<int64>()());

    const TTypes<float>::ConstMatrix scores = context->input(1).matrix<float>();
    VLOG(2) << "Scores: " << scores;
    CHECK_EQ(scores.dimension(1), batch_state->NumActions());

    // In AdvanceBeam we use beam_offsets_[beam_id] to determine the slice of
    // scores that should be used for advancing, but beam_offsets_[beam_id] only
    // exists for beams that have a sentence loaded.
    const int batch_size = batch_state->BatchSize();
    for (int beam_id = 0; beam_id < batch_size; ++beam_id) {
      batch_state->AdvanceBeam(beam_id, scores);
    }
    batch_state->UpdateOffsets();

    // Forward the beam state unmodified.
    Tensor *output;
    const int feature_size = batch_state->FeatureSize();
    OP_REQUIRES_OK(context, context->allocate_output(feature_size,
                                                     TensorShape({}), &output));
    output->scalar<int64>()() = context->input(0).scalar<int64>()();

    // Output the new features of all the slots in all the beams.
    OP_REQUIRES_OK(context, batch_state->PopulateFeatureOutputs(context));

    // Output whether the beams are alive.
    OP_REQUIRES_OK(
        context, context->allocate_output(feature_size + 1,
                                          TensorShape({batch_size}), &output));
    for (int beam_id = 0; beam_id < batch_size; ++beam_id) {
      output->vec<bool>()(beam_id) = batch_state->Beam(beam_id).IsAlive();
    }
  }

 private:
  TF_DISALLOW_COPY_AND_ASSIGN(BeamParser);
};

REGISTER_KERNEL_BUILDER(Name("BeamParser").Device(DEVICE_CPU), BeamParser);

// Extracts the paths for the elements of the current beams and returns
// indices into a scoring matrix that is assumed to have been
// constructed along with the beam search.
class BeamParserOutput : public OpKernel {
 public:
  explicit BeamParserOutput(OpKernelConstruction *context) : OpKernel(context) {
    // Set expected signature.
    OP_REQUIRES_OK(context,
                   context->MatchSignature(
                       {DT_INT64}, {DT_INT32, DT_INT32, DT_INT32, DT_FLOAT}));
  }

  void Compute(OpKernelContext *context) override {
    BatchState *batch_state =
        reinterpret_cast<BatchState *>(context->input(0).scalar<int64>()());

    const int num_actions = batch_state->NumActions();
    const int batch_size = batch_state->BatchSize();

    // Vectors for output.
    //
    // Each step of each batch:path gets its index computed and a
    // unique path id assigned.
    std::vector<int32> indices;
    std::vector<int32> path_ids;

    // Each unique path gets a batch id and a slot (in the beam)
    // id. These are in effect the row and column of the final
    // 'logits' matrix going to CrossEntropy.
    std::vector<int32> beam_ids;
    std::vector<int32> slot_ids;

    // To compute the cross entropy we also need the slot id of the
    // gold path, one per batch.
    std::vector<int32> gold_slot(batch_size, -1);

    // For good measure we also output the path scores as computed by
    // the beam decoder, so it can be compared in tests with the path
    // scores computed via the indices in TF. This has the same length
    // as beam_ids and slot_ids.
    std::vector<float> path_scores;

    // The scores tensor has, conceptually, four dimensions: 1. number
    // of steps, 2. batch size, 3. number of paths on the beam at that
    // step, and 4. the number of actions scored. However this is not
    // a true tensor since the size of the beam at each step may not
    // be equal among all steps and among all batches. Only the batch
    // size and number of actions is fixed.
    int path_id = 0;
    for (int beam_id = 0; beam_id < batch_size; ++beam_id) {
      // This occurs at the end of the corpus, when there aren't enough
      // sentences to fill the batch.
      if (batch_state->Beam(beam_id).gold_ == nullptr) continue;

      // Populate the vectors that will index into the concatenated
      // scores tensor.
      int slot = 0;
      for (const auto &item : batch_state->Beam(beam_id).slots_) {
        beam_ids.push_back(beam_id);
        slot_ids.push_back(slot);
        path_scores.push_back(item.first.first);
        VLOG(2) << "PATH SCORE @ beam_id:" << beam_id << " slot:" << slot
                << " : " << item.first.first << " " << item.first.second;
        VLOG(2) << "SLOT HISTORY: "
                << utils::Join(item.second->slot_history, " ");
        VLOG(2) << "SCORE HISTORY: "
                << utils::Join(item.second->score_history, " ");
        VLOG(2) << "ACTION HISTORY: "
                << utils::Join(item.second->action_history, " ");

        // Record where the gold path ended up.
        if (item.second->state->is_gold()) {
          CHECK_EQ(gold_slot[beam_id], -1);
          gold_slot[beam_id] = slot;
        }

        for (size_t step = 0; step < item.second->slot_history.size(); ++step) {
          const int step_beam_offset = batch_state->GetOffset(step, beam_id);
          const int slot_index = item.second->slot_history[step];
          const int action_index = item.second->action_history[step];
          indices.push_back(num_actions * (step_beam_offset + slot_index) +
                            action_index);
          path_ids.push_back(path_id);
        }
        ++slot;
        ++path_id;
      }

      // One and only path must be the golden one.
      CHECK_GE(gold_slot[beam_id], 0);
    }

    const int num_ix_elements = indices.size();
    Tensor *output;
    OP_REQUIRES_OK(context, context->allocate_output(
                                0, TensorShape({2, num_ix_elements}), &output));
    auto indices_and_path_ids = output->matrix<int32>();
    for (size_t i = 0; i < indices.size(); ++i) {
      indices_and_path_ids(0, i) = indices[i];
      indices_and_path_ids(1, i) = path_ids[i];
    }

    const int num_path_elements = beam_ids.size();
    OP_REQUIRES_OK(context,
                   context->allocate_output(
                       1, TensorShape({2, num_path_elements}), &output));
    auto beam_and_slot_ids = output->matrix<int32>();
    for (size_t i = 0; i < beam_ids.size(); ++i) {
      beam_and_slot_ids(0, i) = beam_ids[i];
      beam_and_slot_ids(1, i) = slot_ids[i];
    }

    OP_REQUIRES_OK(context, context->allocate_output(
                                2, TensorShape({batch_size}), &output));
    std::copy(gold_slot.begin(), gold_slot.end(), output->vec<int32>().data());

    OP_REQUIRES_OK(context, context->allocate_output(
                                3, TensorShape({num_path_elements}), &output));
    std::copy(path_scores.begin(), path_scores.end(),
              output->vec<float>().data());
  }

 private:
  TF_DISALLOW_COPY_AND_ASSIGN(BeamParserOutput);
};

REGISTER_KERNEL_BUILDER(Name("BeamParserOutput").Device(DEVICE_CPU),
                        BeamParserOutput);

// Computes eval metrics for the best path in the input beams.
class BeamEvalOutput : public OpKernel {
 public:
  explicit BeamEvalOutput(OpKernelConstruction *context) : OpKernel(context) {
    // Set expected signature.
    OP_REQUIRES_OK(context,
                   context->MatchSignature({DT_INT64}, {DT_INT32, DT_STRING}));
  }

  void Compute(OpKernelContext *context) override {
    int num_tokens = 0;
    int num_correct = 0;
    int all_final = 0;
    BatchState *batch_state =
        reinterpret_cast<BatchState *>(context->input(0).scalar<int64>()());
    const int batch_size = batch_state->BatchSize();
    vector<Sentence> documents;
    for (int beam_id = 0; beam_id < batch_size; ++beam_id) {
      if (batch_state->Beam(beam_id).gold_ != nullptr &&
          batch_state->Beam(beam_id).AllFinal()) {
        ++all_final;
        const auto &item = *batch_state->Beam(beam_id).slots_.rbegin();
        ComputeTokenAccuracy(*item.second->state, batch_state->ScoringType(),
                             &num_tokens, &num_correct);
        documents.push_back(item.second->state->sentence());
        item.second->state->AddParseToDocument(&documents.back());
      }
    }
    Tensor *output;
    OP_REQUIRES_OK(context,
                   context->allocate_output(0, TensorShape({2}), &output));
    auto eval_metrics = output->vec<int32>();
    eval_metrics(0) = num_tokens;
    eval_metrics(1) = num_correct;

    const int output_size = documents.size();
    OP_REQUIRES_OK(context, context->allocate_output(
                                1, TensorShape({output_size}), &output));
    for (int i = 0; i < output_size; ++i) {
      output->vec<string>()(i) = documents[i].SerializeAsString();
    }
  }

 private:
  // Tallies the # of correct and incorrect tokens for a given ParserState.
  void ComputeTokenAccuracy(const ParserState &state,
                            const string &scoring_type,
                            int *num_tokens, int *num_correct) {
    for (int i = 0; i < state.sentence().token_size(); ++i) {
      const Token &token = state.GetToken(i);
      if (utils::PunctuationUtil::ScoreToken(token.word(), token.tag(),
                                             scoring_type)) {
        ++*num_tokens;
        if (state.IsTokenCorrect(i)) ++*num_correct;
      }
    }
  }

  TF_DISALLOW_COPY_AND_ASSIGN(BeamEvalOutput);
};

REGISTER_KERNEL_BUILDER(Name("BeamEvalOutput").Device(DEVICE_CPU),
                        BeamEvalOutput);

}  // namespace syntaxnet
