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

#include "dragnn/runtime/lstm_network_kernel.h"

#include <stddef.h>
#include <string>
#include <vector>

#include "dragnn/core/test/generic.h"
#include "dragnn/protos/runtime.pb.h"
#include "dragnn/protos/spec.pb.h"
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

// Testing rig, parameterized on a bool that indicates whether the kernel is
// created in bulk mode.
class LSTMNetworkKernelTest : public NetworkTestBase,
                              public ::testing::WithParamInterface<bool> {
 protected:
  // Returns true if the |kernel_| was created in bulk mode.
  bool bulk() const { return GetParam(); }

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

  // Initializes the |kernel_| from the |component_spec_text|.  On error,
  // returns non-OK.
  tensorflow::Status Initialize(const string &component_spec_text) {
    ComponentSpec component_spec;
    CHECK(TextFormat::ParseFromString(component_spec_text, &component_spec));
    component_spec.set_name(kTestComponentName);

    // Since LSTMNetworkKernel uses the concatenated input, it is insensitive
    // to the particular fixed or linked embedding inputs.  For simplicity, the
    // tests use a trivial network structure and a single fixed embedding.
    AddComponent(kTestComponentName);

    TF_RETURN_IF_ERROR(kernel_.Initialize(component_spec, &variable_store_,
                                          &network_state_manager_,
                                          &extension_manager_));

    network_states_.Reset(&network_state_manager_);
    StartComponent(kNumSteps);
    session_state_.extensions.Reset(&extension_manager_);

    return tensorflow::Status::OK();
  }

  // Applies the |kernel_| to the |inputs|.
  void Apply(const std::vector<std::vector<float>> &inputs) {
    UniqueMatrix<float> input_matrix(inputs);
    if (bulk()) {
      TF_ASSERT_OK(
          kernel_.Apply(Matrix<float>(*input_matrix), &session_state_));
    } else {
      for (size_t step_index = 0; step_index < kNumSteps; ++step_index) {
        TF_ASSERT_OK(kernel_.Apply(step_index,
                                   Vector<float>(input_matrix->row(step_index)),
                                   &session_state_));
      }
    }
  }

  // Returns the logits matrix.
  Matrix<float> GetLogits() const {
    return Matrix<float>(GetLayer(kTestComponentName, "logits"));
  }

  LSTMNetworkKernel kernel_{bulk()};
};

INSTANTIATE_TEST_CASE_P(BulkMode, LSTMNetworkKernelTest, ::testing::Bool());

// Tests that the LSTMNetworkKernel does not produce logits when omit_logits is
// true, even if there are actions.
TEST_P(LSTMNetworkKernelTest, NoLogitsOrSoftmaxWhenOmitLogitsTrue) {
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
  constexpr float kEmbedding = 1.25;
  constexpr float kWeight = 1.5;

  // No "softmax" weights or biases.
  AddWeights("x_to_ico", input_dim, 3 * kHiddenDim, kWeight);
  AddWeights("h_to_ico", kHiddenDim, 3 * kHiddenDim, kWeight);
  AddWeights("c2i", kHiddenDim, kHiddenDim, kWeight);
  AddWeights("c2o", kHiddenDim, kHiddenDim, kWeight);
  AddBiases("ico_bias", 3 * kHiddenDim, kWeight);

  TF_ASSERT_OK(Initialize(kSpec));

  // No specified logits layer.
  EXPECT_TRUE(kernel_.GetLogitsName().empty());

  const std::vector<float> row(input_dim, kEmbedding);
  const std::vector<std::vector<float>> rows(kNumSteps, row);
  Apply(rows);

  // No "logits" layer.
  size_t unused_dimension = 0;
  LayerHandle<float> unused_handle;
  EXPECT_THAT(
      network_state_manager_.LookupLayer(kTestComponentName, "logits",
                                         &unused_dimension, &unused_handle),
      test::IsErrorWithSubstr(
          "Unknown layer 'logits' in component 'test_component'"));
}

TEST_P(LSTMNetworkKernelTest, NormalOperationSmallHidden) {
  constexpr size_t input_dim = 32;
  constexpr int kHiddenDim = 8;

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
  AddWeights("x_to_ico", input_dim, 3 * kHiddenDim, kWeight);
  AddWeights("h_to_ico", kHiddenDim, 3 * kHiddenDim, kWeight);
  AddWeights("c2i", kHiddenDim, kHiddenDim, kWeight);
  AddWeights("c2o", kHiddenDim, kHiddenDim, kWeight);
  AddWeights("weights_softmax", kHiddenDim, kNumActions, kWeight,
             /*is_flexible_matrix=*/true);
  AddBiases("ico_bias", 3 * kHiddenDim, kWeight);
  AddBiases("bias_softmax", kNumActions, kWeight);

  TF_EXPECT_OK(Initialize(kSpec));

  // Logits should exist.
  EXPECT_EQ(kernel_.GetLogitsName(), "logits");

  const std::vector<float> row(input_dim, kEmbedding);
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

TEST_P(LSTMNetworkKernelTest, ErrorWithTooSmallHidden) {
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
  constexpr float kEmbedding = 1.25;
  constexpr float kWeight = 1.5;
  AddFixedEmbeddingMatrix(0, 50, input_dim, kEmbedding);

  // Same as above, with "softmax" weights and biases.
  AddWeights("x_to_ico", input_dim, 3 * kHiddenDim, kWeight);
  AddWeights("h_to_ico", kHiddenDim, 3 * kHiddenDim, kWeight);
  AddWeights("c2i", kHiddenDim, kHiddenDim, kWeight);
  AddWeights("c2o", kHiddenDim, kHiddenDim, kWeight);
  AddBiases("ico_bias", 3 * kHiddenDim, kWeight);

  EXPECT_THAT(
      Initialize(kSpec),
      test::IsErrorWithSubstr(
          "Expected hidden size (4) to be a multiple of the AVX width (8)"));
}

}  // namespace
}  // namespace runtime
}  // namespace dragnn
}  // namespace syntaxnet
