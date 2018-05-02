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

#ifndef DRAGNN_RUNTIME_NETWORK_UNIT_BASE_H_
#define DRAGNN_RUNTIME_NETWORK_UNIT_BASE_H_

#include <stddef.h>
#include <utility>
#include <vector>

#include "dragnn/core/compute_session.h"
#include "dragnn/protos/spec.pb.h"
#include "dragnn/runtime/extensions.h"
#include "dragnn/runtime/fixed_embeddings.h"
#include "dragnn/runtime/linked_embeddings.h"
#include "dragnn/runtime/math/types.h"
#include "dragnn/runtime/network_states.h"
#include "dragnn/runtime/network_unit.h"
#include "dragnn/runtime/session_state.h"
#include "dragnn/runtime/variable_store.h"
#include "tensorflow/core/lib/core/status.h"

namespace syntaxnet {
namespace dragnn {
namespace runtime {

// A base class for network units that provides common functionality, analogous
// to NetworkUnitInterface.__init__() in network_units.py.  Specifically, this
// class manages and builds input embeddings and, as an convenience, optionally
// concatenates the input embeddings into a single vector.
//
// Since recurrent layers are both outputs and inputs, they complicate network
// unit initialization.  In particular, the linked embeddings cannot be set up
// until the charateristics of all recurrently-accessible layers are known.  On
// the other hand, some layers cannot be initialized until all inputs, including
// the linked embeddings, are set up.  For example, the IdentityNetwork outputs
// a layer whose dimension is the sum of all input dimensions.
//
// To accommodate recurrent layers, network unit initialization is organized
// into three phases:
//   1. (Subclass) Initialize all recurrently-accessible layers.
//   2. (This class) Initialize embedding managers and other common state.
//   3. (Subclass) Initialize any non-recurrent layers.
//
// Concretely, the subclass's Initialize() should first add recurrent layers,
// then call InitializeBase(), and finally finish initializing.  Evaluation is
// simpler: the subclass's Evaluate() may call EvaluateBase() at any time.
//
// Note: Network unit initialization is similarly interleaved between base and
// subclasses in the Python codebase; see NetworkUnitInterface.get_layer_size()
// and the "init_layers" argument to NetworkUnitInterface.__init__().
class NetworkUnitBase : public NetworkUnit {
 public:
  // Initializes common state as configured in the |component_spec|.  Retrieves
  // pre-trained embedding matrices from the |variable_store|.  Looks up linked
  // embeddings in the |network_state_manager|, which must contain all recurrent
  // layers.  Requests any required extensions from the |extension_manager|.  If
  // |use_concatenated_input| is true, prepares to concatenate input embeddings
  // in EvaluateBase().  On error, returns non-OK.
  tensorflow::Status InitializeBase(bool use_concatenated_input,
                                    const ComponentSpec &component_spec,
                                    VariableStore *variable_store,
                                    NetworkStateManager *network_state_manager,
                                    ExtensionManager *extension_manager);

  // Resets the fixed and linked embeddings in the |session_state| using its
  // network states and the |compute_session|.  Requires that InitializeBase()
  // was called.  If this was prepared for concatenation (see InitializeBase())
  // and if |concatenated_input| is non-null, points it at the concatenation of
  // the fixed and linked embeddings.  Otherwise, no concatenation occurs.  On
  // error, returns non-OK.
  tensorflow::Status EvaluateBase(SessionState *session_state,
                                  ComputeSession *compute_session,
                                  Vector<float> *concatenated_input) const;

  // Accessors.  All require that InitializeBase() was called.
  const FixedEmbeddingManager &fixed_embedding_manager() const;
  const LinkedEmbeddingManager &linked_embedding_manager() const;
  size_t num_actions() const { return num_actions_; }
  size_t concatenated_input_dim() const { return concatenated_input_dim_; }

 private:
  // Returns the concatenation of the fixed and linked embeddings in the
  // |seesion_state|.  Requires that |use_concatenated_input_| is true.
  Vector<float> ConcatenateInput(SessionState *session_state) const;

  // Managers for fixed and linked embeddings in this component.
  FixedEmbeddingManager fixed_embedding_manager_;
  LinkedEmbeddingManager linked_embedding_manager_;

  // Fixed and linked embeddings.
  SharedExtensionHandle<FixedEmbeddings> fixed_embeddings_handle_;
  SharedExtensionHandle<LinkedEmbeddings> linked_embeddings_handle_;

  // Number of actions supported by the transition system.
  size_t num_actions_ = 0;

  // Sum of dimensions of all fixed and linked embeddings.
  size_t concatenated_input_dim_ = 0;

  // Whether to concatenate the input embeddings.
  bool use_concatenated_input_ = false;

  // Handle of the vector that holds the concatenated input, or invalid if no
  // concatenation is required.
  LocalVectorHandle<float> concatenated_input_handle_;
};

// Implementation details below.

inline const FixedEmbeddingManager &NetworkUnitBase::fixed_embedding_manager()
    const {
  return fixed_embedding_manager_;
}

inline const LinkedEmbeddingManager &NetworkUnitBase::linked_embedding_manager()
    const {
  return linked_embedding_manager_;
}

}  // namespace runtime
}  // namespace dragnn
}  // namespace syntaxnet

#endif  // DRAGNN_RUNTIME_NETWORK_UNIT_BASE_H_
