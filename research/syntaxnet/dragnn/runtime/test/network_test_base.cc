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

#include "dragnn/runtime/test/network_test_base.h"

#include "dragnn/protos/data.pb.h"
#include "dragnn/runtime/flexible_matrix_kernel.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/logging.h"

namespace syntaxnet {
namespace dragnn {
namespace runtime {
namespace {

using ::testing::InSequence;
using ::testing::Return;

// Fills the |matrix| with the |fill_value|.
void Fill(float fill_value, MutableMatrix<float> matrix) {
  for (size_t i = 0; i < matrix.num_rows(); ++i) {
    for (float &value : matrix.row(i)) value = fill_value;
  }
}

}  // namespace

constexpr char NetworkTestBase::kTestComponentName[];

void NetworkTestBase::TearDown() {
  // The state extensions may contain objects that cannot outlive the component,
  // so discard the extensions early.  This is not an issue in real-world usage,
  // as the Master calls destructors in the right order.
  session_state_.extensions = Extensions();
}

NetworkTestBase::GetInputFeaturesFunctor NetworkTestBase::ExtractFeatures(
    int expected_channel_id, const std::vector<Feature> &features) {
  return [=](const string &component_name,
             std::function<int32 *(int)> allocate_indices,
             std::function<int64 *(int)> allocate_ids,
             std::function<float *(int)> allocate_weights, int channel_id) {
    EXPECT_EQ(component_name, kTestComponentName);
    EXPECT_EQ(channel_id, expected_channel_id);
    const int num_features = features.size();
    int32 *indices = allocate_indices(num_features);
    int64 *ids = allocate_ids(num_features);
    float *weights = allocate_weights(num_features);
    for (int i = 0; i < num_features; ++i) {
      indices[i] = features[i].index;
      ids[i] = features[i].id;
      weights[i] = features[i].weight;
    }
    return num_features;
  };
}

NetworkTestBase::GetTranslatedLinkFeaturesFunctor NetworkTestBase::ExtractLinks(
    int expected_channel_id, const std::vector<string> &features_text) {
  std::vector<LinkFeatures> features;
  for (const string &text : features_text) {
    features.emplace_back();
    CHECK(TextFormat::ParseFromString(text, &features.back()));
  }
  return [=](const string &component_name, int channel_id) {
    EXPECT_EQ(component_name, kTestComponentName);
    EXPECT_EQ(channel_id, expected_channel_id);
    return features;
  };
}

void NetworkTestBase::AddVectorVariable(const string &name, size_t dimension,
                                        float fill_value) {
  const std::vector<float> row(dimension, fill_value);
  const std::vector<std::vector<float>> values(1, row);
  variable_store_.AddOrDie(name, values);
}

void NetworkTestBase::AddMatrixVariable(const string &name, size_t num_rows,
                                        size_t num_columns, float fill_value) {
  const std::vector<float> row(num_columns, fill_value);
  const std::vector<std::vector<float>> values(num_rows, row);
  variable_store_.AddOrDie(name, values);
}

void NetworkTestBase::AddFixedEmbeddingMatrix(int channel_id,
                                              size_t vocabulary_size,
                                              size_t embedding_dim,
                                              float fill_value) {
  const string name = tensorflow::strings::StrCat(
      kTestComponentName, "/fixed_embedding_matrix_", channel_id, "/trimmed");
  AddMatrixVariable(name, vocabulary_size, embedding_dim, fill_value);
}

void NetworkTestBase::AddLinkedWeightMatrix(int channel_id, size_t source_dim,
                                            size_t embedding_dim,
                                            float fill_value) {
  const string name = tensorflow::strings::StrCat(
      kTestComponentName, "/linked_embedding_matrix_", channel_id, "/weights",
      FlexibleMatrixKernel::kSuffix);
  AddMatrixVariable(name, embedding_dim, source_dim, fill_value);
}

void NetworkTestBase::AddLinkedOutOfBoundsVector(int channel_id,
                                                 size_t embedding_dim,
                                                 float fill_value) {
  const string name = tensorflow::strings::StrCat(kTestComponentName,
                                                  "/linked_embedding_matrix_",
                                                  channel_id, "/out_of_bounds");
  AddVectorVariable(name, embedding_dim, fill_value);
}

void NetworkTestBase::AddComponent(const string &component_name) {
  TF_ASSERT_OK(network_state_manager_.AddComponent(component_name));
}

void NetworkTestBase::AddLayer(const string &layer_name, size_t dimension) {
  LayerHandle<float> unused_layer_handle;
  TF_ASSERT_OK(network_state_manager_.AddLayer(layer_name, dimension,
                                               &unused_layer_handle));
}

void NetworkTestBase::AddPairwiseLayer(const string &layer_name,
                                       size_t dimension) {
  PairwiseLayerHandle<float> unused_layer_handle;
  TF_ASSERT_OK(network_state_manager_.AddLayer(layer_name, dimension,
                                               &unused_layer_handle));
}

void NetworkTestBase::StartComponent(size_t num_steps) {
  // The pre-allocation hint is arbitrary, but setting it to a small value
  // exercises reallocations.
  TF_ASSERT_OK(network_states_.StartNextComponent(5));
  for (size_t i = 0; i < num_steps; ++i) network_states_.AddStep();
}

MutableMatrix<float> NetworkTestBase::GetLayer(const string &component_name,
                                               const string &layer_name) const {
  size_t unused_dimension = 0;
  LayerHandle<float> handle;
  TF_CHECK_OK(network_state_manager_.LookupLayer(component_name, layer_name,
                                                 &unused_dimension, &handle));
  return network_states_.GetLayer(handle);
}

MutableMatrix<float> NetworkTestBase::GetPairwiseLayer(
    const string &component_name, const string &layer_name) const {
  size_t unused_dimension = 0;
  PairwiseLayerHandle<float> handle;
  TF_CHECK_OK(network_state_manager_.LookupLayer(component_name, layer_name,
                                                 &unused_dimension, &handle));
  return network_states_.GetLayer(handle);
}

void NetworkTestBase::FillLayer(const string &component_name,
                                const string &layer_name,
                                float fill_value) const {
  Fill(fill_value, GetLayer(component_name, layer_name));
}

void NetworkTestBase::SetupTransitionLoop(size_t num_steps) {
  // Return not terminal |num_steps| times, then return terminal.
  InSequence scoped;
  EXPECT_CALL(compute_session_, IsTerminal(kTestComponentName))
      .Times(num_steps)
      .WillRepeatedly(Return(false))
      .RetiresOnSaturation();
  EXPECT_CALL(compute_session_, IsTerminal(kTestComponentName))
      .WillOnce(Return(true));
}

void NetworkTestBase::ExpectVector(Vector<float> vector, size_t dimension,
                                   float expected_value) {
  ASSERT_EQ(vector.size(), dimension);
  for (const float value : vector) EXPECT_EQ(value, expected_value);
}

void NetworkTestBase::ExpectMatrix(Matrix<float> matrix, size_t num_rows,
                                   size_t num_columns, float expected_value) {
  ASSERT_EQ(matrix.num_rows(), num_rows);
  ASSERT_EQ(matrix.num_columns(), num_columns);
  for (size_t row = 0; row < num_rows; ++row) {
    ExpectVector(matrix.row(row), num_columns, expected_value);
  }
}

}  // namespace runtime
}  // namespace dragnn
}  // namespace syntaxnet
