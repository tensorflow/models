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

#include <stddef.h>
#include <memory>
#include <string>
#include <vector>

#include "dragnn/core/test/generic.h"
#include "dragnn/protos/runtime.pb.h"
#include "dragnn/protos/spec.pb.h"
#include "dragnn/runtime/bulk_network_unit.h"
#include "dragnn/runtime/flexible_matrix_kernel.h"
#include "dragnn/runtime/lstm_cell/cell_function.h"
#include "dragnn/runtime/test/helpers.h"
#include "dragnn/runtime/test/network_test_base.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"

namespace syntaxnet {
namespace dragnn {
namespace runtime {
namespace {

constexpr size_t kNumSteps = 20;
constexpr size_t kNumActions = 10;
constexpr size_t kInputDim = 32;
constexpr size_t kHiddenDim = 8;

class BulkLSTMNetworkTest : public NetworkTestBase {
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

  // Initializes the |bulk_network_unit_| from the |component_spec_text|.  On
  // error, returns non-OK.
  tensorflow::Status Initialize(const string &component_spec_text) {
    ComponentSpec component_spec;
    CHECK(TextFormat::ParseFromString(component_spec_text, &component_spec));
    component_spec.set_name(kTestComponentName);

    AddComponent(kTestComponentName);

    TF_RETURN_IF_ERROR(
        BulkNetworkUnit::CreateOrError("BulkLSTMNetwork", &bulk_network_unit_));
    TF_RETURN_IF_ERROR(bulk_network_unit_->Initialize(
        component_spec, &variable_store_, &network_state_manager_,
        &extension_manager_));
    TF_RETURN_IF_ERROR(bulk_network_unit_->ValidateInputDimension(kInputDim));

    network_states_.Reset(&network_state_manager_);
    StartComponent(kNumSteps);
    session_state_.extensions.Reset(&extension_manager_);

    return tensorflow::Status::OK();
  }

  // Evaluates the |bulk_network_unit_| on the |inputs|.
  void Apply(const std::vector<std::vector<float>> &inputs) {
    UniqueMatrix<float> input_matrix(inputs);
    TF_ASSERT_OK(bulk_network_unit_->Evaluate(Matrix<float>(*input_matrix),
                                              &session_state_));
  }

  // Returns the logits matrix.
  Matrix<float> GetLogits() const {
    return Matrix<float>(GetLayer(kTestComponentName, "logits"));
  }

  std::unique_ptr<BulkNetworkUnit> bulk_network_unit_;
};

TEST_F(BulkLSTMNetworkTest, NormalOperation) {
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
  constexpr float kEmbedding = 1.25;
  constexpr float kWeight = 1.5;

  // Same as above, with "softmax" weights and biases.
  AddWeights("x_to_ico", kInputDim, 3 * kHiddenDim, kWeight);
  AddWeights("h_to_ico", kHiddenDim, 3 * kHiddenDim, kWeight);
  AddWeights("c2i", kHiddenDim, kHiddenDim, kWeight);
  AddWeights("c2o", kHiddenDim, kHiddenDim, kWeight);
  AddWeights("weights_softmax", kHiddenDim, kNumActions, kWeight,
             /*is_flexible_matrix=*/true);
  AddBiases("ico_bias", 3 * kHiddenDim, kWeight);
  AddBiases("bias_softmax", kNumActions, kWeight);

  TF_EXPECT_OK(Initialize(kSpec));

  // Logits should exist.
  EXPECT_EQ(bulk_network_unit_->GetLogitsName(), "logits");

  const std::vector<float> row(kInputDim, kEmbedding);
  const std::vector<std::vector<float>> rows(kNumSteps, row);
  Apply(rows);

  // Logits dimension matches "num_actions" above. We don't test the values very
  // precisely here, and feel free to update if the cell function changes. Most
  // value tests should be in lstm_cell/cell_function_test.cc.

  Matrix<float> logits = GetLogits();
  EXPECT_EQ(logits.num_rows(), kNumSteps);
  EXPECT_EQ(logits.num_columns(), kNumActions);
  EXPECT_NEAR(logits.row(0)[0], 10.6391, 0.1);
  for (int row = 0; row < logits.num_rows(); ++row) {
    for (const float value : logits.row(row)) {
      EXPECT_EQ(value, logits.row(0)[0])
          << "With uniform weights, all logits should be equal.";
    }
  }
}

}  // namespace
}  // namespace runtime
}  // namespace dragnn
}  // namespace syntaxnet
