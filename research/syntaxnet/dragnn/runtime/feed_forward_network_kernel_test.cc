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

#include "dragnn/runtime/feed_forward_network_kernel.h"

#include <stddef.h>
#include <algorithm>
#include <memory>
#include <string>

#include "dragnn/core/test/generic.h"
#include "dragnn/protos/spec.pb.h"
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

class FeedForwardNetworkKernelTest : public NetworkTestBase {
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

  // Initializes the |kernel_| based on the |component_spec_text|.  On error,
  // returns non-OK.
  tensorflow::Status Initialize(const string &component_spec_text) {
    ComponentSpec component_spec;
    CHECK(TextFormat::ParseFromString(component_spec_text, &component_spec));
    component_spec.set_name(kTestComponentName);

    // Since FeedForwardNetwork uses the concatenated input, it is insensitive
    // to the particular fixed or linked embedding inputs.  For simplicity, the
    // tests use a trivial network structure and a single fixed embedding.
    AddComponent(kTestComponentName);

    TF_RETURN_IF_ERROR(kernel_.Initialize(component_spec, &variable_store_,
                                          &network_state_manager_));

    size_t input_dimension = 0;
    for (const FixedFeatureChannel &channel : component_spec.fixed_feature()) {
      input_dimension += channel.embedding_dim();
    }
    return kernel_.ValidateInputDimension(input_dimension);
  }

  FeedForwardNetworkKernel kernel_;
};

// Tests that FeedForwardNetworkKernel fails when a weight matrix does not match
// the dimension of its output activations.
TEST_F(FeedForwardNetworkKernelTest, BadWeightRows) {
  const size_t kInputDim = 5;
  const size_t kLogitsDim = 3;
  const string kBadSpec = R"(fixed_feature {
                               vocabulary_size: 50
                               embedding_dim: 5
                               size: 1
                             }
                             num_actions: 3)";
  AddWeights("softmax", kInputDim, kLogitsDim - 1 /* bad */, 1.0);
  AddBiases("softmax", kLogitsDim, 1.0);

  EXPECT_THAT(
      Initialize(kBadSpec),
      test::IsErrorWithSubstr(
          "Weight matrix shape should be output dimension plus padding"));
}

// Tests that FeedForwardNetworkKernel fails when a weight matrix does not match
// the dimension of its input activations.
TEST_F(FeedForwardNetworkKernelTest, BadWeightColumns) {
  const size_t kInputDim = 5;
  const size_t kLogitsDim = 3;
  const string kBadSpec = R"(fixed_feature {
                               vocabulary_size: 50
                               embedding_dim: 5
                               size: 1
                             }
                             num_actions: 3)";
  AddWeights("softmax", kInputDim + 1 /* bad */, kLogitsDim, 1.0);
  AddBiases("softmax", kLogitsDim, 1.0);

  EXPECT_THAT(Initialize(kBadSpec),
              test::IsErrorWithSubstr(
                  "Weight matrix shape does not match input dimension"));
}

// Tests that FeedForwardNetworkKernel fails when a bias vector does not match
// the dimension of its output activations.
TEST_F(FeedForwardNetworkKernelTest, BadBiasDimension) {
  const size_t kInputDim = 5;
  const size_t kLogitsDim = 3;
  const string kBadSpec = R"(fixed_feature {
                               vocabulary_size: 50
                               embedding_dim: 5
                               size: 1
                             }
                             num_actions: 3)";
  AddWeights("softmax", kInputDim, kLogitsDim, 1.0);
  AddBiases("softmax", kLogitsDim + 1 /* bad */, 1.0);

  EXPECT_THAT(Initialize(kBadSpec),
              test::IsErrorWithSubstr(
                  "Bias vector shape does not match output dimension"));
}

// Tests that FeedForwardNetworkKernel fails when the value of the
// "layer_norm_input" option is not false.
TEST_F(FeedForwardNetworkKernelTest, UnsupportedLayerNormInputOption) {
  const string kBadSpec = R"(network_unit {
                               parameters {
                                 key: 'layer_norm_input'
                                 value: 'true'
                               }
                             })";

  EXPECT_THAT(Initialize(kBadSpec),
              test::IsErrorWithSubstr("Layer norm is not supported"));
}

// Tests that FeedForwardNetworkKernel fails when the value of the
// "layer_norm_hidden" option is not false.
TEST_F(FeedForwardNetworkKernelTest, UnsupportedLayerNormHiddenOption) {
  const string kBadSpec = R"(network_unit {
                               parameters {
                                 key: 'layer_norm_hidden'
                                 value: 'true'
                               }
                             })";

  EXPECT_THAT(Initialize(kBadSpec),
              test::IsErrorWithSubstr("Layer norm is not supported"));
}

// Tests that FeedForwardNetworkKernel fails when the value of the
// "nonlinearity" option is not "relu".
TEST_F(FeedForwardNetworkKernelTest, UnsupportedNonlinearityOption) {
  const string kBadSpec = R"(network_unit {
                               parameters {
                                 key: 'nonlinearity'
                                 value: 'elu'
                               }
                             })";

  EXPECT_THAT(Initialize(kBadSpec),
              test::IsErrorWithSubstr("Non-linearity is not supported"));
}

// Tests that the FeedForwardNetworkKernel works when there are no hidden
// layers, just a softmax that computes logits.
TEST_F(FeedForwardNetworkKernelTest, JustLogits) {
  const size_t kInputDim = 5;
  const size_t kLogitsDim = 3;
  const string kSpec = R"(fixed_feature {
                            vocabulary_size: 50
                            embedding_dim: 5
                            size: 1
                          }
                          num_actions: 3)";
  AddWeights("softmax", kInputDim, kLogitsDim, 0.0);
  AddBiases("softmax", kLogitsDim, 0.0);

  TF_ASSERT_OK(Initialize(kSpec));

  EXPECT_EQ(kernel_.logits_name(), "logits");
  EXPECT_EQ(kernel_.layers().size(), 1);
}

// Tests that the FeedForwardNetworkKernel works with multiple hidden layers as
// well as a softmax that computes logits.
TEST_F(FeedForwardNetworkKernelTest, MultiLayer) {
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
  AddWeights("0", kDims[0], kDims[1], 0.0);
  AddBiases("0", kDims[1], 0.0);
  AddWeights("1", kDims[1], kDims[2], 0.0);
  AddBiases("1", kDims[2], 0.0);
  AddWeights("softmax", kDims[2], kDims[3], 0.0);
  AddBiases("softmax", kDims[3], 0.0);

  TF_ASSERT_OK(Initialize(kSpec));

  EXPECT_EQ(kernel_.logits_name(), "logits");
  EXPECT_EQ(kernel_.layers().size(), 3);
}

// Tests that the FeedForwardNetworkKernel does not produce logits and does not
// use the softmax variables when the component is deterministic.
TEST_F(FeedForwardNetworkKernelTest, NoLogitsOrSoftmaxWhenDeterministic) {
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

  // No "softmax" weights or biases.
  AddWeights("0", kDims[0], kDims[1], 0.0);
  AddBiases("0", kDims[1], 0.0);

  TF_ASSERT_OK(Initialize(kSpec));

  // No specified logits layer.
  EXPECT_TRUE(kernel_.logits_name().empty());
  EXPECT_EQ(kernel_.layers().size(), 1);
}

// Tests that the FeedForwardNetworkKernel does not produce logits when
// omit_logits is true, even if there are actions.
TEST_F(FeedForwardNetworkKernelTest, NoLogitsOrSoftmaxWhenOmitLogitsTrue) {
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

  // No "softmax" weights or biases.
  AddWeights("0", kDims[0], kDims[1], 0.0);
  AddBiases("0", kDims[1], 0.0);

  TF_ASSERT_OK(Initialize(kSpec));

  // No specified logits layer.
  EXPECT_TRUE(kernel_.logits_name().empty());
  EXPECT_EQ(kernel_.layers().size(), 1);
}

}  // namespace
}  // namespace runtime
}  // namespace dragnn
}  // namespace syntaxnet
