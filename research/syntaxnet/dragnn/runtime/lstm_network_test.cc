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

#include "dragnn/core/test/generic.h"
#include "dragnn/protos/runtime.pb.h"
#include "dragnn/runtime/flexible_matrix_kernel.h"
#include "dragnn/runtime/lstm_cell/cell_function.h"
#include "dragnn/runtime/network_unit.h"
#include "dragnn/runtime/test/network_test_base.h"
#include "dragnn/runtime/variable_store.h"
#include <gmock/gmock.h>
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"

namespace syntaxnet {
namespace dragnn {
namespace runtime {
namespace {

using ::testing::Invoke;
using ::testing::_;

class LstmNetworkTest : public NetworkTestBase {
 protected:
  // Adds a blocked weight matrix with the |name| with the given dimensions and
  // |fill_value|.  If |is_flexible_matrix| is true, the variable is set up for
  // use by the FlexibleMatrixKernel.
  void AddWeights(const string &name, size_t input_dim, size_t output_dim,
                  float fill_value, bool is_flexible_matrix = false) {
    constexpr int kBatchSize = LstmCellFunction<>::kBatchSize;
    size_t output_padded =
        kBatchSize * ((output_dim + kBatchSize - 1) / kBatchSize);
    size_t num_views = (output_padded / kBatchSize) * input_dim;
    string var_name = tensorflow::strings::StrCat(
        kTestComponentName, "/", name,
        is_flexible_matrix ? FlexibleMatrixKernel::kSuffix
                           : "/matrix/blocked48");
    const std::vector<float> block(kBatchSize, fill_value);
    const std::vector<std::vector<float>> blocks(num_views, block);
    variable_store_.AddOrDie(
        var_name, blocks, VariableSpec::FORMAT_COLUMN_BLOCKED_ROW_MAJOR_MATRIX);
    variable_store_.SetBlockedDimensionOverride(
        var_name, {input_dim, output_padded, kBatchSize});
  }

  // Adds a bias vector with the |name_suffix| with the given dimensions and
  // |fill_value|.
  void AddBiases(const string &name, size_t dimension, float fill_value) {
    const string biases_name =
        tensorflow::strings::StrCat(kTestComponentName, "/", name);
    AddVectorVariable(biases_name, dimension, fill_value);
  }

  // Creates a network unit, initializes it based on the |component_spec_text|,
  // and evaluates it.  On error, returns non-OK.
  tensorflow::Status Run(const string &component_spec_text) {
    ComponentSpec component_spec;
    CHECK(TextFormat::ParseFromString(component_spec_text, &component_spec));
    component_spec.set_name(kTestComponentName);

    // Since LSTMNetwork uses the concatenated input, it is insensitive
    // to the particular fixed or linked embedding inputs.  For simplicity, the
    // tests use a trivial network structure and a single fixed embedding.
    AddComponent(kTestComponentName);

    TF_RETURN_IF_ERROR(
        NetworkUnit::CreateOrError("LSTMNetwork", &network_unit_));
    TF_RETURN_IF_ERROR(network_unit_->Initialize(
        component_spec, &variable_store_, &network_state_manager_,
        &extension_manager_));

    network_states_.Reset(&network_state_manager_);
    StartComponent(1);  // only evaluate the first step
    session_state_.extensions.Reset(&extension_manager_);

    TF_RETURN_IF_ERROR(
        network_unit_->Evaluate(0, &session_state_, &compute_session_));

    return tensorflow::Status::OK();
  }

  // Returns the activation vector of the first step of layer named |layer_name|
  // in the current component.
  Vector<float> GetActivations(const string &layer_name) const {
    Matrix<float> layer(GetLayer(kTestComponentName, layer_name));
    return layer.row(0);
  }

  std::unique_ptr<NetworkUnit> network_unit_;
};

// Tests that the LSTMNetwork does not produce logits when omit_logits is
// true, even if there are actions.
TEST_F(LstmNetworkTest, NoLogitsOrSoftmaxWhenOmitLogitsTrue) {
  constexpr size_t input_dim = 32;
  constexpr int kHiddenDim = LstmCellFunction<>::kBatchSize;
  const string kSpec = R"(fixed_feature {
                            vocabulary_size: 50
                            embedding_dim: 32
                            size: 1
                          }
                          network_unit {
                            parameters {
                              key: 'hidden_layer_sizes'
                              value: '48'
                            }
                            parameters {
                              key: 'omit_logits'
                              value: 'true'
                            }
                          }
                          num_actions: 10)";
  const float kEmbedding = 1.25;
  const float kFeature = 0.5;
  const float kWeight = 1.5;
  AddFixedEmbeddingMatrix(0, 50, input_dim, kEmbedding);

  // No "softmax" weights or biases.
  AddWeights("x_to_ico", input_dim, 3 * kHiddenDim, kWeight);
  AddWeights("h_to_ico", kHiddenDim, 3 * kHiddenDim, kWeight);
  AddWeights("c2i", kHiddenDim, kHiddenDim, kWeight);
  AddWeights("c2o", kHiddenDim, kHiddenDim, kWeight);
  AddBiases("ico_bias", 3 * kHiddenDim, kWeight);

  EXPECT_CALL(compute_session_, GetInputFeatures(_, _, _, _, _))
      .WillOnce(Invoke(ExtractFeatures(0, {{1, kFeature}})));

  TF_EXPECT_OK(Run(kSpec));

  // No specified logits layer.
  EXPECT_TRUE(network_unit_->GetLogitsName().empty());

  // No "logits" layer.
  size_t unused_dimension = 0;
  LayerHandle<float> unused_handle;
  EXPECT_THAT(
      network_state_manager_.LookupLayer(kTestComponentName, "logits",
                                         &unused_dimension, &unused_handle),
      test::IsErrorWithSubstr(
          "Unknown layer 'logits' in component 'test_component'"));
}

TEST_F(LstmNetworkTest, NormalOperationSmallHidden) {
  constexpr size_t input_dim = 32;
  constexpr int kHiddenDim = 8;
  constexpr int num_actions = 10;

  const string kSpec = R"(fixed_feature {
                            vocabulary_size: 50
                            embedding_dim: 32
                            size: 1
                          }
                          network_unit {
                            parameters {
                              key: 'hidden_layer_sizes'
                              value: '8'
                            }
                          }
                          num_actions: 10)";
  const float kEmbedding = 1.25;
  const float kFeature = 0.5;
  const float kWeight = 1.5;
  AddFixedEmbeddingMatrix(0, 50, input_dim, kEmbedding);

  // Same as above, with "softmax" weights and biases.
  AddWeights("x_to_ico", input_dim, 3 * kHiddenDim, kWeight);
  AddWeights("h_to_ico", kHiddenDim, 3 * kHiddenDim, kWeight);
  AddWeights("c2i", kHiddenDim, kHiddenDim, kWeight);
  AddWeights("c2o", kHiddenDim, kHiddenDim, kWeight);
  AddWeights("weights_softmax", kHiddenDim, num_actions, kWeight,
             /*is_flexible_matrix=*/true);
  AddBiases("ico_bias", 3 * kHiddenDim, kWeight);
  AddBiases("bias_softmax", num_actions, kWeight);

  EXPECT_CALL(compute_session_, GetInputFeatures(_, _, _, _, _))
      .WillOnce(Invoke(ExtractFeatures(0, {{1, kFeature}})));

  TF_EXPECT_OK(Run(kSpec));

  // Logits should exist.
  EXPECT_EQ(network_unit_->GetLogitsName(), "logits");

  // Logits dimension matches "num_actions" above. We don't test the values very
  // precisely here, and feel free to update if the cell function changes. Most
  // value tests should be in lstm_cell/cell_function_test.cc.

  Vector<float> logits = GetActivations("logits");
  EXPECT_EQ(logits.size(), num_actions);
  EXPECT_NEAR(logits[0], 10.6391, 0.1);
  for (int i = 1; i < 10; ++i) {
    EXPECT_EQ(logits[i], logits[0])
        << "With uniform weights, all logits should be equal.";
  }
}

TEST_F(LstmNetworkTest, ErrorWithTooSmallHidden) {
  constexpr size_t input_dim = 32;
  constexpr int kHiddenDim = 4;

  const string kSpec = R"(fixed_feature {
                            vocabulary_size: 50
                            embedding_dim: 32
                            size: 1
                          }
                          network_unit {
                            parameters {
                              key: 'hidden_layer_sizes'
                              value: '4'
                            }
                          }
                          num_actions: 0)";
  const float kEmbedding = 1.25;
  const float kWeight = 1.5;
  AddFixedEmbeddingMatrix(0, 50, input_dim, kEmbedding);

  // Same as above, with "softmax" weights and biases.
  AddWeights("x_to_ico", input_dim, 3 * kHiddenDim, kWeight);
  AddWeights("h_to_ico", kHiddenDim, 3 * kHiddenDim, kWeight);
  AddWeights("c2i", kHiddenDim, kHiddenDim, kWeight);
  AddWeights("c2o", kHiddenDim, kHiddenDim, kWeight);
  AddBiases("ico_bias", 3 * kHiddenDim, kWeight);

  EXPECT_THAT(
      Run(kSpec),
      test::IsErrorWithSubstr(
          "Expected hidden size (4) to be a multiple of the AVX width (8)"));
}

}  // namespace
}  // namespace runtime
}  // namespace dragnn
}  // namespace syntaxnet
