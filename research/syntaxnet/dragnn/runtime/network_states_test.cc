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

#include "dragnn/runtime/network_states.h"

#include <stddef.h>
#include <string.h>
#include <string>
#include <vector>

#include "dragnn/core/test/generic.h"
#include "dragnn/runtime/alignment.h"
#include "dragnn/runtime/math/types.h"
#include "syntaxnet/base.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/test.h"

namespace syntaxnet {
namespace dragnn {
namespace runtime {
namespace {

// Expects that two objects have identical bit representations.
template <class T>
void ExpectBitwiseEqual(const T &object1, const T &object2) {
  EXPECT_EQ(memcmp(&object1, &object2, sizeof(T)), 0);
}

// Expects that the |matrix| has the given dimensions.
template <class T>
void ExpectDimensions(MutableMatrix<T> matrix, size_t num_rows,
                      size_t num_columns) {
  EXPECT_EQ(matrix.num_rows(), num_rows);
  EXPECT_EQ(matrix.num_columns(), num_columns);
}

// Sets the |vector| to |size| copies of the |value|.
template <class T>
void Fill(MutableVector<T> vector, size_t size, T value) {
  ASSERT_EQ(vector.size(), size);
  for (T &element : vector) element = value;
}

// Expects that the |vector| contains |size| copies of the |expected_value|.
template <class T>
void ExpectFilled(MutableVector<T> vector, size_t size, T expected_value) {
  ASSERT_EQ(vector.size(), size);
  for (const T element : vector) EXPECT_EQ(element, expected_value);
}

// Tests that NetworkStateManager can add a named component.
TEST(NetworkStateManagerTest, AddComponent) {
  NetworkStateManager manager;

  TF_EXPECT_OK(manager.AddComponent("foo/bar"));
  EXPECT_THAT(manager.AddComponent("foo/bar"),
              test::IsErrorWithSubstr("Component 'foo/bar' already exists"));

  // Empty component name is weird, but OK.
  TF_EXPECT_OK(manager.AddComponent(""));
  EXPECT_THAT(manager.AddComponent(""),
              test::IsErrorWithSubstr("Component '' already exists"));
}

// Tests that NetworkStateManager can add a named layer to the current
// component.
TEST(NetworkStateManagerTest, AddLayer) {
  NetworkStateManager manager;
  LayerHandle<float> unused_layer_handle;

  EXPECT_THAT(manager.AddLayer("layer", 1, &unused_layer_handle),
              test::IsErrorWithSubstr("No current component"));

  TF_EXPECT_OK(manager.AddComponent("component"));
  TF_EXPECT_OK(manager.AddLayer("layer", 2, &unused_layer_handle));

  EXPECT_THAT(manager.AddLayer("layer", 2, &unused_layer_handle),
              test::IsErrorWithSubstr(
                  "Layer 'layer' already exists in component 'component'"));
}

// Tests that NetworkStateManager can add a named pairwise layer to the current
// component.
TEST(NetworkStateManagerTest, AddLayerPairwise) {
  NetworkStateManager manager;
  PairwiseLayerHandle<float> unused_layer_handle;

  EXPECT_THAT(manager.AddLayer("layer", 1, &unused_layer_handle),
              test::IsErrorWithSubstr("No current component"));

  TF_EXPECT_OK(manager.AddComponent("component"));
  TF_EXPECT_OK(manager.AddLayer("layer", 2, &unused_layer_handle));

  EXPECT_THAT(manager.AddLayer("layer", 2, &unused_layer_handle),
              test::IsErrorWithSubstr(
                  "Layer 'layer' already exists in component 'component'"));
}

// Tests that NetworkStateManager can add an alias to an existing layer.  Also
// tests that layer and alias names are required to be unique.
TEST(NetworkStateManagerTest, AddLayerAlias) {
  NetworkStateManager manager;
  LayerHandle<float> unused_layer_handle;

  EXPECT_THAT(manager.AddLayerAlias("alias", "layer"),
              test::IsErrorWithSubstr("No current component"));

  TF_EXPECT_OK(manager.AddComponent("component"));
  EXPECT_THAT(
      manager.AddLayerAlias("alias", "layer"),
      test::IsErrorWithSubstr("Target layer 'layer' of alias 'alias' does not "
                              "exist in component 'component'"));

  TF_EXPECT_OK(manager.AddLayer("layer", 2, &unused_layer_handle));
  TF_EXPECT_OK(manager.AddLayerAlias("alias", "layer"));

  EXPECT_THAT(manager.AddLayerAlias("alias", "layer"),
              test::IsErrorWithSubstr(
                  "Alias 'alias' already exists in component 'component'"));

  EXPECT_THAT(
      manager.AddLayer("alias", 2, &unused_layer_handle),
      test::IsErrorWithSubstr("Layer 'alias' conflicts with an existing alias "
                              "in component 'component'"));

  TF_EXPECT_OK(manager.AddLayer("layer2", 2, &unused_layer_handle));
  EXPECT_THAT(
      manager.AddLayerAlias("layer2", "layer"),
      test::IsErrorWithSubstr("Alias 'layer2' conflicts with an existing layer "
                              "in component 'component'"));
}

// Tests that NetworkStateManager can add a local matrix or vector to the
// current component.
TEST(NetworkStateManagerTest, AddLocal) {
  NetworkStateManager manager;
  LocalVectorHandle<float> unused_local_vector_handle;
  LocalMatrixHandle<float> unused_local_matrix_handle;

  EXPECT_THAT(manager.AddLocal(11, &unused_local_matrix_handle),
              test::IsErrorWithSubstr("No current component"));

  TF_EXPECT_OK(manager.AddComponent("component"));
  TF_EXPECT_OK(manager.AddLocal(22, &unused_local_matrix_handle));
  TF_EXPECT_OK(manager.AddLocal(33, &unused_local_vector_handle));
}

// Tests that NetworkStateManager can look up existing layers or aliases, and
// fails on invalid layer or component names and for mismatched types.
TEST(NetworkStateManagerTest, LookupLayer) {
  NetworkStateManager manager;
  LayerHandle<char> char_handle;
  LayerHandle<int16> int16_handle;
  LayerHandle<uint16> uint16_handle;
  PairwiseLayerHandle<char> pairwise_char_handle;
  size_t dimension = 0;

  // Add some typed layers and aliases.
  TF_ASSERT_OK(manager.AddComponent("foo"));
  TF_ASSERT_OK(manager.AddLayer("char", 5, &char_handle));
  TF_ASSERT_OK(manager.AddLayer("int16", 7, &int16_handle));
  TF_ASSERT_OK(manager.AddLayerAlias("char_alias", "char"));
  TF_ASSERT_OK(manager.AddLayerAlias("int16_alias", "int16"));
  TF_ASSERT_OK(manager.AddComponent("bar"));
  TF_ASSERT_OK(manager.AddLayer("uint16", 11, &uint16_handle));
  TF_ASSERT_OK(manager.AddLayer("pairwise_char", 13, &pairwise_char_handle));
  TF_ASSERT_OK(manager.AddLayerAlias("uint16_alias", "uint16"));
  TF_ASSERT_OK(manager.AddLayerAlias("pairwise_char_alias", "pairwise_char"));

  // Try looking up unknown components.
  EXPECT_THAT(manager.LookupLayer("missing", "char", &dimension, &char_handle),
              test::IsErrorWithSubstr("Unknown component 'missing'"));
  EXPECT_THAT(manager.LookupLayer("baz", "float", &dimension, &char_handle),
              test::IsErrorWithSubstr("Unknown component 'baz'"));

  // Try looking up valid components but unknown layers.
  EXPECT_THAT(
      manager.LookupLayer("foo", "missing", &dimension, &char_handle),
      test::IsErrorWithSubstr("Unknown layer 'missing' in component 'foo'"));
  EXPECT_THAT(
      manager.LookupLayer("bar", "missing", &dimension, &char_handle),
      test::IsErrorWithSubstr("Unknown layer 'missing' in component 'bar'"));

  // Try looking up valid components and the names of layers or aliases in the
  // other components.
  EXPECT_THAT(
      manager.LookupLayer("foo", "uint16", &dimension, &uint16_handle),
      test::IsErrorWithSubstr("Unknown layer 'uint16' in component 'foo'"));
  EXPECT_THAT(
      manager.LookupLayer("foo", "uint16_alias", &dimension, &uint16_handle),
      test::IsErrorWithSubstr(
          "Unknown layer 'uint16_alias' in component 'foo'"));
  EXPECT_THAT(
      manager.LookupLayer("bar", "char", &dimension, &char_handle),
      test::IsErrorWithSubstr("Unknown layer 'char' in component 'bar'"));
  EXPECT_THAT(
      manager.LookupLayer("bar", "char_alias", &dimension, &char_handle),
      test::IsErrorWithSubstr("Unknown layer 'char_alias' in component 'bar'"));

  // Look up layers with incorrect types.
  EXPECT_THAT(
      manager.LookupLayer("foo", "char", &dimension, &int16_handle),
      test::IsErrorWithSubstr(
          "Layer 'char' in component 'foo' does not match its expected type"));
  EXPECT_THAT(
      manager.LookupLayer("foo", "char", &dimension, &uint16_handle),
      test::IsErrorWithSubstr(
          "Layer 'char' in component 'foo' does not match its expected type"));
  EXPECT_THAT(
      manager.LookupLayer("foo", "char", &dimension, &pairwise_char_handle),
      test::IsErrorWithSubstr("Layer 'char' in component 'foo' does not match "
                              "its expected OperandType"));

  EXPECT_THAT(
      manager.LookupLayer("foo", "int16", &dimension, &char_handle),
      test::IsErrorWithSubstr(
          "Layer 'int16' in component 'foo' does not match its expected type"));
  EXPECT_THAT(
      manager.LookupLayer("foo", "int16", &dimension, &uint16_handle),
      test::IsErrorWithSubstr(
          "Layer 'int16' in component 'foo' does not match its expected type"));
  EXPECT_THAT(
      manager.LookupLayer("foo", "int16", &dimension, &pairwise_char_handle),
      test::IsErrorWithSubstr(
          "Layer 'int16' in component 'foo' does not match its expected type"));

  EXPECT_THAT(manager.LookupLayer("bar", "uint16", &dimension, &char_handle),
              test::IsErrorWithSubstr("Layer 'uint16' in component 'bar' does "
                                      "not match its expected type"));
  EXPECT_THAT(manager.LookupLayer("bar", "uint16", &dimension, &int16_handle),
              test::IsErrorWithSubstr("Layer 'uint16' in component 'bar' does "
                                      "not match its expected type"));
  EXPECT_THAT(
      manager.LookupLayer("bar", "uint16", &dimension, &pairwise_char_handle),
      test::IsErrorWithSubstr("Layer 'uint16' in component 'bar' does "
                              "not match its expected type"));

  EXPECT_THAT(
      manager.LookupLayer("bar", "pairwise_char", &dimension, &char_handle),
      test::IsErrorWithSubstr("Layer 'pairwise_char' in component 'bar' does "
                              "not match its expected OperandType"));
  EXPECT_THAT(
      manager.LookupLayer("bar", "pairwise_char", &dimension, &int16_handle),
      test::IsErrorWithSubstr("Layer 'pairwise_char' in component 'bar' does "
                              "not match its expected type"));
  EXPECT_THAT(
      manager.LookupLayer("bar", "pairwise_char", &dimension, &uint16_handle),
      test::IsErrorWithSubstr("Layer 'pairwise_char' in component 'bar' does "
                              "not match its expected type"));

  // Look up layers properly, and check their dimensions.  Also verify that the
  // looked-up handles are identical to the original handles.
  LayerHandle<char> lookup_char_handle;
  LayerHandle<int16> lookup_int16_handle;
  LayerHandle<uint16> lookup_uint16_handle;
  PairwiseLayerHandle<char> lookup_pairwise_char_handle;
  TF_EXPECT_OK(
      manager.LookupLayer("foo", "char", &dimension, &lookup_char_handle));
  EXPECT_EQ(dimension, 5);
  ExpectBitwiseEqual(lookup_char_handle, char_handle);

  TF_EXPECT_OK(
      manager.LookupLayer("foo", "int16", &dimension, &lookup_int16_handle));
  EXPECT_EQ(dimension, 7);
  ExpectBitwiseEqual(lookup_int16_handle, int16_handle);

  TF_EXPECT_OK(
      manager.LookupLayer("bar", "uint16", &dimension, &lookup_uint16_handle));
  EXPECT_EQ(dimension, 11);
  ExpectBitwiseEqual(lookup_uint16_handle, uint16_handle);

  TF_EXPECT_OK(manager.LookupLayer("bar", "pairwise_char", &dimension,
                                   &lookup_pairwise_char_handle));
  EXPECT_EQ(dimension, 13);
  ExpectBitwiseEqual(lookup_pairwise_char_handle, pairwise_char_handle);
}

// Tests that NetworkStates cannot start components without a manager.
TEST(NetworkStatesTest, NoManager) {
  NetworkStates network_states;
  EXPECT_THAT(network_states.StartNextComponent(10),
              test::IsErrorWithSubstr("No manager"));
}

// Tests that NetworkStates cannot start components when the manager is empty.
TEST(NetworkStatesTest, EmptyManager) {
  NetworkStateManager empty_manager;

  NetworkStates network_states;
  network_states.Reset(&empty_manager);
  EXPECT_THAT(network_states.StartNextComponent(10),
              test::IsErrorWithSubstr("No next component"));
}

// Tests that NetworkStates can start the same number of components as were
// configured in its manager.
TEST(NetworkStatesTest, StartNextComponent) {
  NetworkStateManager manager;
  TF_EXPECT_OK(manager.AddComponent("foo"));
  TF_EXPECT_OK(manager.AddComponent("bar"));
  TF_EXPECT_OK(manager.AddComponent("baz"));

  NetworkStates network_states;
  network_states.Reset(&manager);

  TF_EXPECT_OK(network_states.StartNextComponent(10));
  TF_EXPECT_OK(network_states.StartNextComponent(11));
  TF_EXPECT_OK(network_states.StartNextComponent(12));

  EXPECT_THAT(network_states.StartNextComponent(13),
              test::IsErrorWithSubstr("No next component"));
}

// Tests that NetworkStates contains layers and locals whose dimensions match
// the configuration of its manager.
TEST(NetworkStatesTest, Dimensions) {
  NetworkStateManager manager;

  // The "foo" component has two layers and a local vector.
  LayerHandle<float> foo_hidden_handle;
  LocalVectorHandle<int16> foo_local_handle;
  PairwiseLayerHandle<float> foo_logits_handle;
  TF_ASSERT_OK(manager.AddComponent("foo"));
  TF_ASSERT_OK(manager.AddLayer("hidden", 10, &foo_hidden_handle));
  TF_ASSERT_OK(manager.AddLocal(20, &foo_local_handle));
  TF_ASSERT_OK(manager.AddLayer("logits", 30, &foo_logits_handle));

  // The "bar" component has one layer and a local matrix.
  LayerHandle<float> bar_logits_handle;
  LocalMatrixHandle<bool> bar_local_handle;
  TF_ASSERT_OK(manager.AddComponent("bar"));
  TF_ASSERT_OK(manager.AddLayer("logits", 40, &bar_logits_handle));
  TF_ASSERT_OK(manager.AddLocal(50, &bar_local_handle));

  // Initialize a NetworkStates and check its dimensions.  Note that matrices
  // start with 0 rows since there are 0 steps.
  NetworkStates network_states;
  network_states.Reset(&manager);
  TF_EXPECT_OK(network_states.StartNextComponent(13));
  ExpectDimensions(network_states.GetLayer(foo_hidden_handle), 0, 10);
  EXPECT_EQ(network_states.GetLocal(foo_local_handle).size(), 20);
  ExpectDimensions(network_states.GetLayer(foo_logits_handle), 0, 0);

  // Add some steps, and check that rows have been added to matrices, while
  // vectors are unaffected.
  network_states.AddSteps(19);
  ExpectDimensions(network_states.GetLayer(foo_hidden_handle), 19, 10);
  EXPECT_EQ(network_states.GetLocal(foo_local_handle).size(), 20);
  ExpectDimensions(network_states.GetLayer(foo_logits_handle), 19, 19 * 30);

  // Again for the next component.
  TF_EXPECT_OK(network_states.StartNextComponent(9));
  ExpectDimensions(network_states.GetLayer(bar_logits_handle), 0, 40);
  ExpectDimensions(network_states.GetLocal(bar_local_handle), 0, 50);

  // Add some steps, and check that rows have been added to matrices.
  network_states.AddSteps(25);
  ExpectDimensions(network_states.GetLayer(bar_logits_handle), 25, 40);
  ExpectDimensions(network_states.GetLocal(bar_local_handle), 25, 50);

  EXPECT_THAT(network_states.StartNextComponent(10),
              test::IsErrorWithSubstr("No next component"));

  // Check the layers of the first component.  They should still have the same
  // dimensions in spite of adding steps to the second component.
  ExpectDimensions(network_states.GetLayer(foo_hidden_handle), 19, 10);
  ExpectDimensions(network_states.GetLayer(foo_logits_handle), 19, 19 * 30);
}

// Tests that NetworkStates can be reused by resetting them repeatedly, possibly
// switching between different managers.
TEST(NetworkStatesTest, ResetWithDifferentManagers) {
  std::vector<NetworkStateManager> managers(10);
  std::vector<LayerHandle<int>> layer_handles(10);
  std::vector<PairwiseLayerHandle<int>> pairwise_layer_handles(10);
  std::vector<LocalVectorHandle<int>> vector_handles(10);
  std::vector<LocalMatrixHandle<double>> matrix_handles(10);
  for (int dim = 0; dim < 10; ++dim) {
    TF_ASSERT_OK(managers[dim].AddComponent("foo"));
    TF_ASSERT_OK(managers[dim].AddLayer(
        tensorflow::strings::StrCat("layer", dim), dim, &layer_handles[dim]));
    TF_ASSERT_OK(
        managers[dim].AddLayer(tensorflow::strings::StrCat("pairwise", dim),
                               dim, &pairwise_layer_handles[dim]));
    TF_ASSERT_OK(managers[dim].AddLocal(dim, &vector_handles[dim]));
    TF_ASSERT_OK(managers[dim].AddLocal(dim, &matrix_handles[dim]));
  }

  NetworkStates network_states;
  for (int trial = 0; trial < 10; ++trial) {
    for (int dim = 0; dim < 10; ++dim) {
      network_states.Reset(&managers[dim]);
      TF_ASSERT_OK(network_states.StartNextComponent(10));

      // Fill the vector local.
      Fill(network_states.GetLocal(vector_handles[dim]), dim,
           100 * trial + dim);

      // Check the vector local.
      ExpectFilled(network_states.GetLocal(vector_handles[dim]), dim,
                   100 * trial + dim);

      // Repeatedly add a step and fill it with values.
      for (int step = 0; step < 100; ++step) {
        network_states.AddStep();
        Fill(network_states.GetLayer(layer_handles[dim]).row(step), dim,
             1000 * trial + 100 * dim + step);
        Fill(network_states.GetLocal(matrix_handles[dim]).row(step), dim,
             9876.0 * trial + 100 * dim + step);
      }

      // Check that data from earlier steps is preserved across reallocations.
      for (int step = 0; step < 100; ++step) {
        ExpectFilled(network_states.GetLayer(layer_handles[dim]).row(step), dim,
                     1000 * trial + 100 * dim + step);
        ExpectFilled(network_states.GetLocal(matrix_handles[dim]).row(step),
                     dim, 9876.0 * trial + 100 * dim + step);
      }

      ExpectDimensions(network_states.GetLayer(pairwise_layer_handles[dim]),
                       100, 100 * dim);
    }
  }
}

// Tests that one NetworkStateManager can be shared simultaneously between
// multiple NetworkStates instances.
TEST(NetworkStatesTest, SharedManager) {
  const size_t kDim = 17;

  NetworkStateManager manager;
  LayerHandle<int> layer_handle;
  PairwiseLayerHandle<int> pairwise_layer_handle;
  LocalVectorHandle<int> vector_handle;
  LocalMatrixHandle<double> matrix_handle;
  TF_ASSERT_OK(manager.AddComponent("foo"));
  TF_ASSERT_OK(manager.AddLayer("layer", kDim, &layer_handle));
  TF_ASSERT_OK(manager.AddLayer("pairwise", kDim, &pairwise_layer_handle));
  TF_ASSERT_OK(manager.AddLocal(kDim, &vector_handle));
  TF_ASSERT_OK(manager.AddLocal(kDim, &matrix_handle));

  std::vector<NetworkStates> network_states_vec(10);
  for (NetworkStates &network_states : network_states_vec) {
    network_states.Reset(&manager);
    TF_ASSERT_OK(network_states.StartNextComponent(10));
  }

  // Fill all vectors.
  for (int trial = 0; trial < network_states_vec.size(); ++trial) {
    const NetworkStates &network_states = network_states_vec[trial];
    Fill(network_states.GetLocal(vector_handle), kDim, 3 * trial);
  }

  // Check all vectors.
  for (int trial = 0; trial < network_states_vec.size(); ++trial) {
    const NetworkStates &network_states = network_states_vec[trial];
    ExpectFilled(network_states.GetLocal(vector_handle), kDim, 3 * trial);
  }

  // Fill all matrices.  Interleave operations on the network states on each
  // step, so all network states are "active" at the same time.
  for (int step = 0; step < 100; ++step) {
    for (int trial = 0; trial < 10; ++trial) {
      NetworkStates &network_states = network_states_vec[trial];
      network_states.AddStep();
      Fill(network_states.GetLayer(layer_handle).row(step), kDim,
           999 * trial + step);
      Fill(network_states.GetLocal(matrix_handle).row(step), kDim,
           1234.0 * trial + step);

      ExpectDimensions(network_states.GetLayer(pairwise_layer_handle), step + 1,
                       kDim * (step + 1));
    }
  }

  // Check all matrices.
  for (int step = 0; step < 100; ++step) {
    for (int trial = 0; trial < 10; ++trial) {
      const NetworkStates &network_states = network_states_vec[trial];
      ExpectFilled(network_states.GetLayer(layer_handle).row(step), kDim,
                   999 * trial + step);
      ExpectFilled(network_states.GetLocal(matrix_handle).row(step), kDim,
                   1234.0 * trial + step);

      ExpectDimensions(network_states.GetLayer(pairwise_layer_handle), 100,
                       kDim * 100);
    }
  }
}

}  // namespace
}  // namespace runtime
}  // namespace dragnn
}  // namespace syntaxnet
