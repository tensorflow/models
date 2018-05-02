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

#ifndef DRAGNN_RUNTIME_TEST_NETWORK_TEST_BASE_H_
#define DRAGNN_RUNTIME_TEST_NETWORK_TEST_BASE_H_

#include <stddef.h>
#include <functional>
#include <string>
#include <utility>
#include <vector>

#include "dragnn/core/test/mock_compute_session.h"
#include "dragnn/runtime/extensions.h"
#include "dragnn/runtime/math/types.h"
#include "dragnn/runtime/network_states.h"
#include "dragnn/runtime/session_state.h"
#include "dragnn/runtime/test/fake_variable_store.h"
#include "syntaxnet/base.h"
#include <gmock/gmock.h>
#include "tensorflow/core/platform/test.h"

namespace syntaxnet {
namespace dragnn {
namespace runtime {

// Base class for tests that depend on network structure.  Provides utils for
// adding/accessing network states and extracting features.
class NetworkTestBase : public ::testing::Test {
 protected:
  // Default component name for tests.
  static constexpr char kTestComponentName[] = "test_component";

  // A functor version of ComputeSession::GetTranslatedLinkFeatures().
  using GetTranslatedLinkFeaturesFunctor =
      std::function<std::vector<LinkFeatures>(const string &component_name,
                                              int channel_id)>;

  // A functor version of ComputeSession::GetInputFeatures().
  using GetInputFeaturesFunctor = std::function<int(
      const string &component_name,
      std::function<int32 *(int)> allocate_indices,
      std::function<int64 *(int)> allocate_ids,
      std::function<float *(int)> allocate_weights, int channel_id)>;

  // A feature to be extracted.
  struct Feature {
    // Creates a feature with index 0.
    Feature(int64 id, float weight) : Feature(0, id, weight) {}

    // Creates a fully-specified feature.
    Feature(int32 index, int64 id, float weight)
        : index(index), id(id), weight(weight) {}

    // Respectively appended to "indices", "ids", and "weights".
    const int32 index;
    const int64 id;
    const float weight;
  };

  // Discards test data structures.
  void TearDown() override;

  // Returns a functor that expects to be called with the |expected_channel_id|
  // and extracts the text-format LinkFeatures in |features_text|.  Useful for
  // mocking the behavior of the |compute_session_|.
  static GetTranslatedLinkFeaturesFunctor ExtractLinks(
      int expected_channel_id, const std::vector<string> &features_text);

  // Returns a functor that extracts the |features| and expects to be called
  // with the |expected_channel_id|.  Useful for mocking the behavior of the
  // |compute_session_|.
  static GetInputFeaturesFunctor ExtractFeatures(
      int expected_channel_id, const std::vector<Feature> &features);

  // Creates a vector or matrix with the |name| and dimensions, fills it with
  // the |fill_value|, and adds it to the |variable_store_|.
  void AddVectorVariable(const string &name, size_t dimension,
                         float fill_value);
  void AddMatrixVariable(const string &name, size_t num_rows,
                         size_t num_columns, float fill_value);

  // Creates an embedding matrix for the |channel_id| with the given dimensions,
  // fills it with the |fill_value|, and adds it to the |variable_store_|.
  void AddFixedEmbeddingMatrix(int channel_id, size_t vocabulary_size,
                               size_t embedding_dim, float fill_value);

  // Creates a linked weight matrix or out-of-bounds vector for the |channel_id|
  // with the given dimensions, fills it with the |fill_value|, and adds it to
  // the |variable_store_|.
  void AddLinkedWeightMatrix(int channel_id, size_t source_dim,
                             size_t embedding_dim, float fill_value);
  void AddLinkedOutOfBoundsVector(int channel_id, size_t embedding_dim,
                                  float fill_value);

  // Adds a component named |component_name| to the |network_state_manager_|.
  void AddComponent(const string &component_name);

  // Adds a float layer named |layer_name| to the current component of the
  // |network_state_manager_|.
  void AddLayer(const string &layer_name, size_t dimension);

  // As above, but for pairwise layers.
  void AddPairwiseLayer(const string &layer_name, size_t dimension);

  // Starts the next component of the |network_states_| and advances it by
  // |num_steps| steps.
  void StartComponent(size_t num_steps);

  // Returns the content of the layer named |layer_name| in the component named
  // |component_name|.
  MutableMatrix<float> GetLayer(const string &component_name,
                                const string &layer_name) const;

  // As above, but for pairwise layers.
  MutableMatrix<float> GetPairwiseLayer(const string &component_name,
                                        const string &layer_name) const;

  // Fills the layer named |layer_name| in the component named |component_name|
  // in the |network_states_| with the |fill_value|.
  void FillLayer(const string &component_name, const string &layer_name,
                 float fill_value) const;

  // Adds call expectations and return values to the control methods of the
  // |compute_session_| that execute a loop of |num_steps| transitions.
  void SetupTransitionLoop(size_t num_steps);

  // Expects that the |vector| has the given dimensions and is filled with the
  // |expected_value|.
  static void ExpectVector(Vector<float> vector, size_t dimension,
                           float expected_value);

  // Expects that the |matrix| has the given dimensions and is filled with the
  // |expected_value|.
  static void ExpectMatrix(Matrix<float> matrix, size_t num_rows,
                           size_t num_columns, float expected_value);

  FakeVariableStore variable_store_;
  NetworkStateManager network_state_manager_;
  ExtensionManager extension_manager_;
  ::testing::StrictMock<MockComputeSession> compute_session_;
  SessionState session_state_;
  NetworkStates &network_states_ = session_state_.network_states;
};

}  // namespace runtime
}  // namespace dragnn
}  // namespace syntaxnet

#endif  // DRAGNN_RUNTIME_TEST_NETWORK_TEST_BASE_H_
