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

#ifndef DRAGNN_RUNTIME_SEQUENCE_MODEL_H_
#define DRAGNN_RUNTIME_SEQUENCE_MODEL_H_

#include <stddef.h>
#include <memory>
#include <string>

#include "dragnn/core/compute_session.h"
#include "dragnn/core/input_batch_cache.h"
#include "dragnn/protos/spec.pb.h"
#include "dragnn/runtime/fixed_embeddings.h"
#include "dragnn/runtime/linked_embeddings.h"
#include "dragnn/runtime/network_states.h"
#include "dragnn/runtime/sequence_features.h"
#include "dragnn/runtime/sequence_links.h"
#include "dragnn/runtime/sequence_predictor.h"
#include "dragnn/runtime/session_state.h"
#include "syntaxnet/base.h"
#include "tensorflow/core/lib/core/status.h"

namespace syntaxnet {
namespace dragnn {
namespace runtime {

// A class that configures and helps evaluate a sequence-based model.
//
// This class requires the SequenceBackend component backend and elides most of
// the ComputeSession feature extraction and transition system overhead.
class SequenceModel {
 public:
  // State associated with a single evaluation of the model.
  struct EvaluateState {
    // Number of transition steps in the current sequence.
    size_t num_steps = 0;

    // Current input batch.
    InputBatchCache *input = nullptr;

    // Sequence-based fixed features.
    SequenceFeatures features;

    // Sequence-based linked embeddings.
    SequenceLinks links;
  };

  // Creates an uninitialized model.  Call Initialize() before use.
  SequenceModel() = default;

  // Returns true if the |component_spec| is compatible with a sequence model.
  static bool Supports(const ComponentSpec &component_spec);

  // Initalizes this from the configuration in the |component_spec|.  Wraps the
  // |fixed_embedding_manager| and |linked_embedding_manager| in sequence-based
  // versions, and requests layers from the |network_state_manager|.  All of the
  // managers must outlive this.  If the transition system is non-deterministic,
  // uses the layer named |logits_name| to make predictions later in Predict();
  // otherwise, |logits_name| is ignored and Predict() does nothing.  On error,
  // returns non-OK.
  tensorflow::Status Initialize(
      const ComponentSpec &component_spec, const string &logits_name,
      const FixedEmbeddingManager *fixed_embedding_manager,
      const LinkedEmbeddingManager *linked_embedding_manager,
      NetworkStateManager *network_state_manager);

  // Resets the |evaluate_state| to values derived from the |session_state| and
  // |compute_session|.  Also updates the NetworkStates in the |session_state|
  // and the current component of the |compute_session| with the length of the
  // current sequence.  Call this before producing output layers.  On error,
  // returns non-OK.
  tensorflow::Status Preprocess(SessionState *session_state,
                                ComputeSession *compute_session,
                                EvaluateState *evaluate_state) const;

  // If applicable, makes predictions based on the logits in |network_states|
  // and applies them to the input in the |evaluate_state|.  Call this after
  // producing output layers.  On error, returns non-OK.
  tensorflow::Status Predict(const NetworkStates &network_states,
                             EvaluateState *evaluate_state) const;

  // Accessors.
  bool deterministic() const { return deterministic_; }
  bool left_to_right() const { return left_to_right_; }
  const SequenceLinkManager &sequence_link_manager() const;
  const SequenceFeatureManager &sequence_feature_manager() const;

 private:
  // Name of the component that this model is a part of.
  string component_name_;

  // Whether the underlying transition system is deterministic.
  bool deterministic_ = false;

  // Whether to process sequences from left to right.
  bool left_to_right_ = true;

  // Whether fixed or linked features are present.
  bool have_fixed_features_ = false;
  bool have_linked_features_ = false;

  // Handle to the logits layer.  Only used if |deterministic_| is false.
  LayerHandle<float> logits_handle_;

  // Manager for sequence-based feature extractors.
  SequenceFeatureManager sequence_feature_manager_;

  // Manager for sequence-based linked embeddings.
  SequenceLinkManager sequence_link_manager_;

  // Sequence-based predictor, if |deterministic_| is false.
  std::unique_ptr<SequencePredictor> sequence_predictor_;
};

// Implementation details below.

inline const SequenceLinkManager &SequenceModel::sequence_link_manager() const {
  return sequence_link_manager_;
}

inline const SequenceFeatureManager &SequenceModel::sequence_feature_manager()
    const {
  return sequence_feature_manager_;
}

}  // namespace runtime
}  // namespace dragnn
}  // namespace syntaxnet

#endif  // DRAGNN_RUNTIME_SEQUENCE_MODEL_H_
