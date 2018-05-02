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

#include "dragnn/runtime/feed_forward_network_layer.h"

#include <stddef.h>
#include <algorithm>
#include <string>

#include "dragnn/core/test/generic.h"
#include "dragnn/runtime/activation_functions.h"
#include "dragnn/runtime/flexible_matrix_kernel.h"
#include "dragnn/runtime/math/types.h"
#include "dragnn/runtime/network_states.h"
#include "dragnn/runtime/test/helpers.h"
#include "dragnn/runtime/test/network_test_base.h"
#include "dragnn/runtime/variable_store.h"
#include "syntaxnet/base.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/test.h"

namespace syntaxnet {
namespace dragnn {
namespace runtime {
namespace {

constexpr char kLayerName[] = "layer";
constexpr char kVariableSuffix[] = "suffix";

constexpr size_t kInputDim = 5;
constexpr size_t kLogitsDim = 3;
constexpr size_t kNumSteps = 4;

class FeedForwardNetworkLayerTest : public NetworkTestBase {
 protected:
  // Adds a weight matrix with the given dimensions and |fill_value|.
  void AddWeights(size_t num_rows, size_t num_columns, float fill_value) {
    const string weights_name = tensorflow::strings::StrCat(
        kTestComponentName, "/weights_", kVariableSuffix,
        FlexibleMatrixKernel::kSuffix);
    AddMatrixVariable(weights_name, num_columns, num_rows, fill_value);
  }

  // Adds a bias vector with the given dimensions and |fill_value|.
  void AddBiases(size_t dimension, float fill_value) {
    const string biases_name = tensorflow::strings::StrCat(
        kTestComponentName, "/bias_", kVariableSuffix);
    AddVectorVariable(biases_name, dimension, fill_value);
  }

  // Returns the result of initializing the |layer_| with the arguments.
  tensorflow::Status Initialize(
      ActivationFunction activation_function = ActivationFunction::kIdentity,
      size_t num_steps = kNumSteps) {
    if (!initialized_) {
      AddComponent(kTestComponentName);
      TF_RETURN_IF_ERROR(layer_.Initialize(
          kTestComponentName, kLayerName, kLogitsDim, activation_function,
          kVariableSuffix, &variable_store_, &network_state_manager_));
      initialized_ = true;
    }

    network_states_.Reset(&network_state_manager_);
    StartComponent(num_steps);
    return tensorflow::Status::OK();
  }

  // Applies the |layer_| to the |input| and returns the result.
  Vector<float> Apply(const std::vector<float> &input) {
    UniqueVector<float> input_vector(input);
    layer_.Apply(Vector<float>(*input_vector), network_states_,
                 /*step_index=*/0);
    return Vector<float>(GetLayer(kTestComponentName, kLayerName).row(0));
  }

  // Applies the |layer_| to the |inputs| and returns the result.
  Matrix<float> Apply(const std::vector<std::vector<float>> &inputs) {
    UniqueMatrix<float> input_matrix(inputs);
    layer_.Apply(Matrix<float>(*input_matrix), network_states_);
    return Matrix<float>(GetLayer(kTestComponentName, kLayerName));
  }

  bool initialized_ = false;
  FeedForwardNetworkLayer layer_;
};

// Tests that FeedForwardNetworkLayer fails when a weight matrix does not match
// the dimension of its output activations.
TEST_F(FeedForwardNetworkLayerTest, BadWeightRows) {
  AddWeights(kInputDim, kLogitsDim - 1 /* bad */, 1.0);
  AddBiases(kLogitsDim, 1.0);

  EXPECT_THAT(
      Initialize(),
      test::IsErrorWithSubstr(
          "Weight matrix shape should be output dimension plus padding"));
}

// Tests that FeedForwardNetworkLayer fails when a weight matrix does not match
// the dimension of its input activations.
TEST_F(FeedForwardNetworkLayerTest, BadWeightColumns) {
  AddWeights(kInputDim + 1 /* bad */, kLogitsDim, 1.0);
  AddBiases(kLogitsDim, 1.0);

  TF_ASSERT_OK(Initialize());

  size_t output_dim = 0;
  EXPECT_THAT(layer_.CheckInputDimAndGetOutputDim(kInputDim, &output_dim),
              test::IsErrorWithSubstr(
                  "Weight matrix shape does not match input dimension"));
}

// Tests that FeedForwardNetworkLayer fails when a bias vector does not match
// the dimension of its output activations.
TEST_F(FeedForwardNetworkLayerTest, BadBiasDimension) {
  AddWeights(kInputDim, kLogitsDim, 1.0);
  AddBiases(kLogitsDim + 1 /* bad */, 1.0);

  EXPECT_THAT(Initialize(),
              test::IsErrorWithSubstr(
                  "Bias vector shape does not match output dimension"));
}

// Tests that FeedForwardNetworkLayer can be used with identity activations.
TEST_F(FeedForwardNetworkLayerTest, IdentityActivations) {
  AddWeights(kInputDim, kLogitsDim, 1.0);
  AddBiases(kLogitsDim, 0.5);

  TF_ASSERT_OK(Initialize());

  size_t output_dim = 0;
  TF_ASSERT_OK(layer_.CheckInputDimAndGetOutputDim(kInputDim, &output_dim));
  EXPECT_EQ(output_dim, kLogitsDim);

  // 0.5 + 1 + 2 + 3 + 4 + 5 = 15.5
  std::vector<float> row = {1.0, 2.0, 3.0, 4.0, 5.0};
  ExpectVector(Apply(row), kLogitsDim, 15.5);
  ExpectMatrix(Apply(std::vector<std::vector<float>>(kNumSteps, row)),
               kNumSteps, kLogitsDim, 15.5);

  // 0.5 - 1 - 2 - 3 - 4 - 5 = -14.5
  row = {-1.0, -2.0, -3.0, -4.0, -5.0};
  ExpectVector(Apply(row), kLogitsDim, -14.5);
  ExpectMatrix(Apply(std::vector<std::vector<float>>(kNumSteps, row)),
               kNumSteps, kLogitsDim, -14.5);
}

// Tests that FeedForwardNetworkLayer can be used with ReLU activations.
TEST_F(FeedForwardNetworkLayerTest, ReluActivations) {
  AddWeights(kInputDim, kLogitsDim, 1.0);
  AddBiases(kLogitsDim, 0.5);

  TF_ASSERT_OK(Initialize(ActivationFunction::kRelu));

  size_t output_dim = 0;
  TF_ASSERT_OK(layer_.CheckInputDimAndGetOutputDim(kInputDim, &output_dim));
  EXPECT_EQ(output_dim, kLogitsDim);

  // max(0.0, 0.5 + 1 + 2 + 3 + 4 + 5) = 15.5
  std::vector<float> row = {1.0, 2.0, 3.0, 4.0, 5.0};
  ExpectVector(Apply(row), kLogitsDim, 15.5);
  ExpectMatrix(Apply(std::vector<std::vector<float>>(kNumSteps, row)),
               kNumSteps, kLogitsDim, 15.5);

  // max(0.0, 0.5 - 1 - 2 - 3 - 4 - 5) = 0.0
  row = {-1.0, -2.0, -3.0, -4.0, -5.0};
  ExpectVector(Apply(row), kLogitsDim, 0.0);
  ExpectMatrix(Apply(std::vector<std::vector<float>>(kNumSteps, row)),
               kNumSteps, kLogitsDim, 0.0);
}

// Make sure SGEMVV implementation is correct.
TEST_F(FeedForwardNetworkLayerTest, VaryingSizes) {
  AddWeights(kInputDim, kLogitsDim, 1.0);
  AddBiases(kLogitsDim, 0.5);

  std::vector<float> row1 = {1.0, 2.0, 3.0, 4.0, 5.0};  // relu(sum + b) = 15.5
  std::vector<float> row2 = {-1.0, -2.0, -3.0, -4.0, -5.0};  // result: 0
  std::vector<float> row3 = {1.0, -2.0, 3.0, -4.0, 5.0};     // result: 3.5

  // Zero-row computation.
  TF_ASSERT_OK(Initialize(ActivationFunction::kRelu, 0));
  Matrix<float> result = Apply(std::vector<std::vector<float>>());
  EXPECT_EQ(result.num_rows(), 0);

  // One-row computation.
  TF_ASSERT_OK(Initialize(ActivationFunction::kRelu, 1));
  result = Apply(std::vector<std::vector<float>>{row1});
  EXPECT_EQ(result.num_rows(), 1);
  ExpectVector(result.row(0), kLogitsDim, 15.5);

  // Two-row computation.
  TF_ASSERT_OK(Initialize(ActivationFunction::kRelu, 2));
  result = Apply({row1, row2});
  EXPECT_EQ(result.num_rows(), 2);
  ExpectVector(result.row(0), kLogitsDim, 15.5);
  ExpectVector(result.row(1), kLogitsDim, 0.0);

  // Three-row computation.
  TF_ASSERT_OK(Initialize(ActivationFunction::kRelu, 3));
  result = Apply({row1, row2, row3});
  EXPECT_EQ(result.num_rows(), 3);
  ExpectVector(result.row(0), kLogitsDim, 15.5);
  ExpectVector(result.row(1), kLogitsDim, 0.0);
  ExpectVector(result.row(2), kLogitsDim, 3.5);
}

}  // namespace
}  // namespace runtime
}  // namespace dragnn
}  // namespace syntaxnet
