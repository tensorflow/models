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
#include "dragnn/runtime/flexible_matrix_kernel.h"
#include "dragnn/runtime/math/types.h"
#include "dragnn/runtime/network_states.h"
#include "dragnn/runtime/network_unit.h"
#include "dragnn/runtime/test/network_test_base.h"
#include "dragnn/runtime/variable_store.h"
#include "syntaxnet/base.h"
#include <gmock/gmock.h>
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/test.h"

namespace syntaxnet {
namespace dragnn {
namespace runtime {
namespace {

using ::testing::_;
using ::testing::Invoke;

// Applies the ReLU activation to the |value|.
float Relu(float value) { return std::max(0.0f, value); }

class FeedForwardNetworkTest : public NetworkTestBase {
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

    // Since FeedForwardNetwork uses the concatenated input, it is insensitive
    // to the particular fixed or linked embedding inputs.  For simplicity, the
    // tests use a trivial network structure and a single fixed embedding.
    AddComponent(kTestComponentName);

    TF_RETURN_IF_ERROR(
        NetworkUnit::CreateOrError("FeedForwardNetwork", &network_unit_));
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

// Tests that FeedForwardNetwork fails when a weight matrix does not match the
// dimension of its output activations.
TEST_F(FeedForwardNetworkTest, BadWeightRows) {
  const size_t kInputDim = 5;
  const size_t kLogitsDim = 3;
  const string kBadSpec = R"(fixed_feature {
                               vocabulary_size: 50
                               embedding_dim: 5
                               size: 1
                             }
                             num_actions: 3)";
  AddFixedEmbeddingMatrix(0, 50, kInputDim, 1.0);
  AddWeights("softmax", kInputDim, kLogitsDim - 1 /* bad */, 1.0);
  AddBiases("softmax", kLogitsDim, 1.0);

  EXPECT_THAT(
      Run(kBadSpec),
      test::IsErrorWithSubstr(
          "Weight matrix shape should be output dimension plus padding"));
}

// Tests that FeedForwardNetwork fails when a weight matrix does not match the
// dimension of its input activations.
TEST_F(FeedForwardNetworkTest, BadWeightColumns) {
  const size_t kInputDim = 5;
  const size_t kLogitsDim = 3;
  const string kBadSpec = R"(fixed_feature {
                               vocabulary_size: 50
                               embedding_dim: 5
                               size: 1
                             }
                             num_actions: 3)";
  AddFixedEmbeddingMatrix(0, 50, kInputDim, 1.0);
  AddWeights("softmax", kInputDim + 1 /* bad */, kLogitsDim, 1.0);
  AddBiases("softmax", kLogitsDim, 1.0);

  EXPECT_THAT(Run(kBadSpec),
              test::IsErrorWithSubstr(
                  "Weight matrix shape does not match input dimension"));
}

// Tests that FeedForwardNetwork fails when a bias vector does not match the
// dimension of its output activations.
TEST_F(FeedForwardNetworkTest, BadBiasDimension) {
  const size_t kInputDim = 5;
  const size_t kLogitsDim = 3;
  const string kBadSpec = R"(fixed_feature {
                               vocabulary_size: 50
                               embedding_dim: 5
                               size: 1
                             }
                             num_actions: 3)";
  AddFixedEmbeddingMatrix(0, 50, kInputDim, 1.0);
  AddWeights("softmax", kInputDim, kLogitsDim, 1.0);
  AddBiases("softmax", kLogitsDim + 1 /* bad */, 1.0);

  EXPECT_THAT(Run(kBadSpec),
              test::IsErrorWithSubstr(
                  "Bias vector shape does not match output dimension"));
}

// Tests that FeedForwardNetwork fails when the value of the "layer_norm_input"
// option is not false.
TEST_F(FeedForwardNetworkTest, UnsupportedLayerNormInputOption) {
  const string kBadSpec = R"(network_unit {
                               parameters {
                                 key: 'layer_norm_input'
                                 value: 'true'
                               }
                             })";

  EXPECT_THAT(Run(kBadSpec),
              test::IsErrorWithSubstr("Layer norm is not supported"));
}

// Tests that FeedForwardNetwork fails when the value of the "layer_norm_hidden"
// option is not false.
TEST_F(FeedForwardNetworkTest, UnsupportedLayerNormHiddenOption) {
  const string kBadSpec = R"(network_unit {
                               parameters {
                                 key: 'layer_norm_hidden'
                                 value: 'true'
                               }
                             })";

  EXPECT_THAT(Run(kBadSpec),
              test::IsErrorWithSubstr("Layer norm is not supported"));
}

// Tests that FeedForwardNetwork fails when the value of the "nonlinearity"
// option is not "relu".
TEST_F(FeedForwardNetworkTest, UnsupportedNonlinearityOption) {
  const string kBadSpec = R"(network_unit {
                               parameters {
                                 key: 'nonlinearity'
                                 value: 'elu'
                               }
                             })";

  EXPECT_THAT(Run(kBadSpec),
              test::IsErrorWithSubstr("Non-linearity is not supported"));
}

// Tests that the FeedForwardNetwork works when there are no hidden layers, just
// a softmax that computes logits.
TEST_F(FeedForwardNetworkTest, JustLogits) {
  const size_t kInputDim = 5;
  const size_t kLogitsDim = 3;
  const string kSpec = R"(fixed_feature {
                            vocabulary_size: 50
                            embedding_dim: 5
                            size: 1
                          }
                          num_actions: 3)";
  const float kEmbedding = 1.25;
  const float kFeature = 0.5;
  const float kWeight = 1.5;
  const float kBias = 0.75;
  AddFixedEmbeddingMatrix(0, 50, kInputDim, kEmbedding);
  AddWeights("softmax", kInputDim, kLogitsDim, kWeight);
  AddBiases("softmax", kLogitsDim, kBias);

  EXPECT_CALL(compute_session_, GetInputFeatures(_, _, _, _, _))
      .WillOnce(Invoke(ExtractFeatures(0, {{1, kFeature}})));

  TF_ASSERT_OK(Run(kSpec));

  EXPECT_EQ("logits", network_unit_->GetLogitsName());
  ExpectVector(GetActivations("logits"), kLogitsDim,
               kInputDim * kEmbedding * kFeature * kWeight + kBias);
}

// Tests that the FeedForwardNetwork works with multiple hidden layers as well
// as a softmax that computes logits.
TEST_F(FeedForwardNetworkTest, MultiLayer) {
  const size_t kDims[] = {5, 4, 3, 2};
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
  AddFixedEmbeddingMatrix(0, 50, 5, 1.0);
  AddWeights("0", kDims[0], kDims[1], kWeights[0]);
  AddBiases("0", kDims[1], kBiases[0]);
  AddWeights("1", kDims[1], kDims[2], kWeights[1]);
  AddBiases("1", kDims[2], kBiases[1]);
  AddWeights("softmax", kDims[2], kDims[3], kWeights[2]);
  AddBiases("softmax", kDims[3], kBiases[2]);

  EXPECT_CALL(compute_session_, GetInputFeatures(_, _, _, _, _))
      .WillOnce(Invoke(ExtractFeatures(0, {{1, 1.0}})));

  TF_ASSERT_OK(Run(kSpec));

  EXPECT_EQ("logits", network_unit_->GetLogitsName());
  float expected = Relu(kDims[0] * kWeights[0] + kBiases[0]);
  ExpectVector(GetActivations("layer_0"), kDims[1], expected);
  expected = Relu(kDims[1] * expected * kWeights[1] + kBiases[1]);
  ExpectVector(GetActivations("layer_1"), kDims[2], expected);
  ExpectVector(GetActivations("last_layer"), kDims[2], expected);
  expected = kDims[2] * expected * kWeights[2] + kBiases[2];
  ExpectVector(GetActivations("logits"), kDims[3], expected);
}

// Tests that the FeedForwardNetwork does not produce logits and does not use
// the softmax variables when the component is deterministic.
TEST_F(FeedForwardNetworkTest, NoLogitsOrSoftmaxWhenDeterministic) {
  const size_t kDims[] = {5, 4};
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
  const float kEmbedding = 1.25;
  const float kFeature = 0.5;
  const float kWeight = -1.5;
  const float kBias = 0.75;
  AddFixedEmbeddingMatrix(0, 50, kDims[0], kEmbedding);

  // No "softmax" weights or biases.
  AddWeights("0", kDims[0], kDims[1], kWeight);
  AddBiases("0", kDims[1], kBias);

  EXPECT_CALL(compute_session_, GetInputFeatures(_, _, _, _, _))
      .WillOnce(Invoke(ExtractFeatures(0, {{1, kFeature}})));

  TF_ASSERT_OK(Run(kSpec));

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

  // Hidden layer is still produced.
  const float kExpected =
      Relu(kDims[0] * kEmbedding * kFeature * kWeight + kBias);
  ExpectVector(GetActivations("layer_0"), kDims[1], kExpected);
  ExpectVector(GetActivations("last_layer"), kDims[1], kExpected);
}

// Tests that the FeedForwardNetwork does not produce logits when omit_logits is
// true, even if there are actions.
TEST_F(FeedForwardNetworkTest, NoLogitsOrSoftmaxWhenOmitLogitsTrue) {
  const size_t kDims[] = {5, 4};
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
  const float kEmbedding = 1.25;
  const float kFeature = 0.5;
  const float kWeight = 1.5;
  const float kBias = 0.75;
  AddFixedEmbeddingMatrix(0, 50, kDims[0], kEmbedding);

  // No "softmax" weights or biases.
  AddWeights("0", kDims[0], kDims[1], kWeight);
  AddBiases("0", kDims[1], kBias);

  EXPECT_CALL(compute_session_, GetInputFeatures(_, _, _, _, _))
      .WillOnce(Invoke(ExtractFeatures(0, {{1, kFeature}})));

  TF_ASSERT_OK(Run(kSpec));

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

  // Hidden layer is still produced.
  const float kExpected = kDims[0] * kEmbedding * kFeature * kWeight + kBias;
  ExpectVector(GetActivations("layer_0"), kDims[1], kExpected);
  ExpectVector(GetActivations("last_layer"), kDims[1], kExpected);
}

}  // namespace
}  // namespace runtime
}  // namespace dragnn
}  // namespace syntaxnet
