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
#include <algorithm>
#include <memory>
#include <string>

#include "dragnn/core/test/generic.h"
#include "dragnn/protos/spec.pb.h"
#include "dragnn/runtime/bulk_network_unit.h"
#include "dragnn/runtime/flexible_matrix_kernel.h"
#include "dragnn/runtime/math/types.h"
#include "dragnn/runtime/network_states.h"
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

constexpr size_t kInputDim = 5;
constexpr size_t kLogitsDim = 3;
constexpr size_t kNumSteps = 4;
constexpr float kEmbedding = 1.25;

// Applies the ReLU activation to the |value|.
float Relu(float value) { return std::max(0.0f, value); }

class BulkFeedForwardNetworkTest : public NetworkTestBase {
 protected:
  // Adds a weight matrix with the |name_suffix| with the given dimensions and
  // |fill_value|.
  void AddWeights(const string &name_suffix, size_t num_rows,
                  size_t num_columns, float fill_value) {
    const string weights_name =
        tensorflow::strings::StrCat(kTestComponentName, "/weights_",
                                    name_suffix, FlexibleMatrixKernel::kSuffix);
    AddMatrixVariable(weights_name, num_columns, num_rows, fill_value);
  }

  // Adds a bias vector with the |name_suffix| with the given dimensions and
  // |fill_value|.
  void AddBiases(const string &name_suffix, size_t dimension,
                 float fill_value) {
    const string biases_name =
        tensorflow::strings::StrCat(kTestComponentName, "/bias_", name_suffix);
    AddVectorVariable(biases_name, dimension, fill_value);
  }

  // Creates a network unit, initializes it based on the |component_spec_text|,
  // and evaluates it.  On error, returns non-OK.
  tensorflow::Status Run(const string &component_spec_text) {
    ComponentSpec component_spec;
    CHECK(TextFormat::ParseFromString(component_spec_text, &component_spec));
    component_spec.set_name(kTestComponentName);
    AddComponent(kTestComponentName);

    TF_CHECK_OK(BulkNetworkUnit::CreateOrError("BulkFeedForwardNetwork",
                                               &bulk_network_unit_));
    TF_RETURN_IF_ERROR(bulk_network_unit_->Initialize(
        component_spec, &variable_store_, &network_state_manager_,
        &extension_manager_));

    size_t input_dimension = 0;
    for (const FixedFeatureChannel &channel : component_spec.fixed_feature()) {
      input_dimension += channel.embedding_dim();
    }
    TF_RETURN_IF_ERROR(
        bulk_network_unit_->ValidateInputDimension(input_dimension));

    network_states_.Reset(&network_state_manager_);
    StartComponent(kNumSteps);
    session_state_.extensions.Reset(&extension_manager_);

    const std::vector<float> row(kInputDim, kEmbedding);
    UniqueMatrix<float> input(std::vector<std::vector<float>>(kNumSteps, row));
    return bulk_network_unit_->Evaluate(Matrix<float>(*input), &session_state_);
  }

  // Returns the layer named |layer_name| in the current component.
  Matrix<float> GetActivations(const string &layer_name) const {
    return Matrix<float>(GetLayer(kTestComponentName, layer_name));
  }

  std::unique_ptr<BulkNetworkUnit> bulk_network_unit_;
};

// Tests that BulkFeedForwardNetwork fails when a weight matrix does not match
// the dimension of its output activations.
TEST_F(BulkFeedForwardNetworkTest, BadWeightRows) {
  const string kBadSpec = R"(fixed_feature {
                               vocabulary_size: 50
                               embedding_dim: 5
                               size: 1
                             }
                             num_actions: 3)";
  AddWeights("softmax", kInputDim, kLogitsDim - 1 /* bad */, 1.0);
  AddBiases("softmax", kLogitsDim, 1.0);

  EXPECT_THAT(
      Run(kBadSpec),
      test::IsErrorWithSubstr(
          "Weight matrix shape should be output dimension plus padding"));
}

// Tests that BulkFeedForwardNetwork fails when a weight matrix does not match
// the dimension of its input activations.
TEST_F(BulkFeedForwardNetworkTest, BadWeightColumns) {
  const string kBadSpec = R"(fixed_feature {
                               vocabulary_size: 50
                               embedding_dim: 5
                               size: 1
                             }
                             num_actions: 3)";
  AddWeights("softmax", kInputDim + 1 /* bad */, kLogitsDim, 1.0);
  AddBiases("softmax", kLogitsDim, 1.0);

  EXPECT_THAT(Run(kBadSpec),
              test::IsErrorWithSubstr(
                  "Weight matrix shape does not match input dimension"));
}

// Tests that BulkFeedForwardNetwork fails when a bias vector does not match the
// dimension of its output activations.
TEST_F(BulkFeedForwardNetworkTest, BadBiasDimension) {
  const string kBadSpec = R"(fixed_feature {
                               vocabulary_size: 50
                               embedding_dim: 5
                               size: 1
                             }
                             num_actions: 3)";
  AddWeights("softmax", kInputDim, kLogitsDim, 1.0);
  AddBiases("softmax", kLogitsDim + 1 /* bad */, 1.0);

  EXPECT_THAT(Run(kBadSpec),
              test::IsErrorWithSubstr(
                  "Bias vector shape does not match output dimension"));
}

// Tests that BulkFeedForwardNetwork fails when the value of the
// "layer_norm_input" option is not false.
TEST_F(BulkFeedForwardNetworkTest, UnsupportedLayerNormInputOption) {
  const string kBadSpec = R"(network_unit {
                               parameters {
                                 key: 'layer_norm_input'
                                 value: 'true'
                               }
                             })";

  EXPECT_THAT(Run(kBadSpec),
              test::IsErrorWithSubstr("Layer norm is not supported"));
}

// Tests that BulkFeedForwardNetwork fails when the value of the
// "layer_norm_hidden" option is not false.
TEST_F(BulkFeedForwardNetworkTest, UnsupportedLayerNormHiddenOption) {
  const string kBadSpec = R"(network_unit {
                               parameters {
                                 key: 'layer_norm_hidden'
                                 value: 'true'
                               }
                             })";

  EXPECT_THAT(Run(kBadSpec),
              test::IsErrorWithSubstr("Layer norm is not supported"));
}

// Tests that BulkFeedForwardNetwork fails when the value of the "nonlinearity"
// option is not "relu".
TEST_F(BulkFeedForwardNetworkTest, UnsupportedNonlinearityOption) {
  const string kBadSpec = R"(network_unit {
                               parameters {
                                 key: 'nonlinearity'
                                 value: 'elu'
                               }
                             })";

  EXPECT_THAT(Run(kBadSpec),
              test::IsErrorWithSubstr("Non-linearity is not supported"));
}

// Tests that BulkFeedForwardNetwork fails if there is a recurrent link.
TEST_F(BulkFeedForwardNetworkTest, UnsupportedRecurrentLink) {
  const string kBadSpec = R"(linked_feature {
                               source_component: 'test_component'
                             })";

  EXPECT_THAT(Run(kBadSpec),
              test::IsErrorWithSubstr(
                  "BulkFeedForwardNetwork forbids recurrent links"));
}

// Tests that the BulkFeedForwardNetwork works when there are no hidden layers,
// just a softmax that computes logits.
TEST_F(BulkFeedForwardNetworkTest, JustLogits) {
  const string kSpec = R"(fixed_feature {
                            vocabulary_size: 50
                            embedding_dim: 5
                            size: 1
                          }
                          num_actions: 3)";
  const float kWeight = 1.5;
  const float kBias = 0.75;
  AddWeights("softmax", kInputDim, kLogitsDim, kWeight);
  AddBiases("softmax", kLogitsDim, kBias);

  TF_ASSERT_OK(Run(kSpec));

  EXPECT_EQ("logits", bulk_network_unit_->GetLogitsName());
  ExpectMatrix(GetActivations("logits"), kNumSteps, kLogitsDim,
               kInputDim * kEmbedding * kWeight + kBias);
}

// Tests that the BulkFeedForwardNetwork works with multiple hidden layers as
// well as a softmax that computes logits.
TEST_F(BulkFeedForwardNetworkTest, MultiLayer) {
  const size_t kDims[] = {kInputDim, 4, 3, 2};
  const string kSpec = R"(fixed_feature {
                            vocabulary_size: 50
                            embedding_dim: 5
                            size: 1
                          }
                          network_unit {
                            parameters {
                              key: 'hidden_layer_sizes'
                              value: '4,3'
                            }
                          }
                          num_actions: 2)";
  const float kWeights[] = {-1.5, 1.0, 0.5};
  const float kBiases[] = {0.75, -0.5, -1.0};
  AddWeights("0", kDims[0], kDims[1], kWeights[0]);
  AddBiases("0", kDims[1], kBiases[0]);
  AddWeights("1", kDims[1], kDims[2], kWeights[1]);
  AddBiases("1", kDims[2], kBiases[1]);
  AddWeights("softmax", kDims[2], kDims[3], kWeights[2]);
  AddBiases("softmax", kDims[3], kBiases[2]);

  TF_ASSERT_OK(Run(kSpec));

  EXPECT_EQ("logits", bulk_network_unit_->GetLogitsName());
  float expected = Relu(kDims[0] * kWeights[0] + kBiases[0]);
  ExpectMatrix(GetActivations("layer_0"), kNumSteps, kDims[1], expected);
  expected = Relu(kDims[1] * expected * kWeights[1] + kBiases[1]);
  ExpectMatrix(GetActivations("layer_1"), kNumSteps, kDims[2], expected);
  ExpectMatrix(GetActivations("last_layer"), kNumSteps, kDims[2], expected);
  expected = kDims[2] * expected * kWeights[2] + kBiases[2];
  ExpectMatrix(GetActivations("logits"), kNumSteps, kDims[3], expected);
}

// Tests that the BulkFeedForwardNetwork does not produce logits and does not
// use the softmax variables when the component is deterministic.
TEST_F(BulkFeedForwardNetworkTest, NoLogitsOrSoftmaxWhenDeterministic) {
  const size_t kDims[] = {kInputDim, 4};
  const string kSpec = R"(num_actions: 1
                          fixed_feature {
                            vocabulary_size: 50
                            embedding_dim: 5
                            size: 1
                          }
                          network_unit {
                            parameters {
                              key: 'hidden_layer_sizes'
                              value: '4'
                            }
                          })";
  const float kWeight = -1.5;
  const float kBias = 0.75;

  // No "softmax" weights or biases.
  AddWeights("0", kDims[0], kDims[1], kWeight);
  AddBiases("0", kDims[1], kBias);

  TF_ASSERT_OK(Run(kSpec));

  // No specified logits layer.
  EXPECT_TRUE(bulk_network_unit_->GetLogitsName().empty());

  // No "logits" layer.
  size_t unused_dimension = 0;
  LayerHandle<float> unused_handle;
  EXPECT_THAT(
      network_state_manager_.LookupLayer(kTestComponentName, "logits",
                                         &unused_dimension, &unused_handle),
      test::IsErrorWithSubstr(
          "Unknown layer 'logits' in component 'test_component'"));

  // Hidden layer is still produced.
  const float kExpected = Relu(kDims[0] * kEmbedding * kWeight + kBias);
  ExpectMatrix(GetActivations("layer_0"), kNumSteps, kDims[1], kExpected);
  ExpectMatrix(GetActivations("last_layer"), kNumSteps, kDims[1], kExpected);
}

// Tests that the BulkFeedForwardNetwork does not produce logits when
// omit_logits is true, even if there are actions.
TEST_F(BulkFeedForwardNetworkTest, NoLogitsOrSoftmaxWhenOmitLogitsTrue) {
  const size_t kDims[] = {kInputDim, 4};
  const string kSpec = R"(fixed_feature {
                            vocabulary_size: 50
                            embedding_dim: 5
                            size: 1
                          }
                          network_unit {
                            parameters {
                              key: 'hidden_layer_sizes'
                              value: '4'
                            }
                            parameters {
                              key: 'omit_logits'
                              value: 'true'
                            }
                          }
                          num_actions: 10)";
  const float kWeight = 1.5;
  const float kBias = 0.75;

  // No "softmax" weights or biases.
  AddWeights("0", kDims[0], kDims[1], kWeight);
  AddBiases("0", kDims[1], kBias);

  TF_ASSERT_OK(Run(kSpec));

  // No specified logits layer.
  EXPECT_TRUE(bulk_network_unit_->GetLogitsName().empty());

  // No "logits" layer.
  size_t unused_dimension = 0;
  LayerHandle<float> unused_handle;
  EXPECT_THAT(
      network_state_manager_.LookupLayer(kTestComponentName, "logits",
                                         &unused_dimension, &unused_handle),
      test::IsErrorWithSubstr(
          "Unknown layer 'logits' in component 'test_component'"));

  // Hidden layer is still produced.
  const float kExpected = kDims[0] * kEmbedding * kWeight + kBias;
  ExpectMatrix(GetActivations("layer_0"), kNumSteps, kDims[1], kExpected);
  ExpectMatrix(GetActivations("last_layer"), kNumSteps, kDims[1], kExpected);
}

}  // namespace
}  // namespace runtime
}  // namespace dragnn
}  // namespace syntaxnet
