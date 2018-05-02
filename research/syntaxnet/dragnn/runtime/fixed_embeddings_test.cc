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

#include "dragnn/runtime/fixed_embeddings.h"

#include <string>
#include <utility>
#include <vector>

#include "dragnn/core/test/generic.h"
#include "dragnn/protos/spec.pb.h"
#include "dragnn/runtime/test/network_test_base.h"
#include "syntaxnet/base.h"
#include <gmock/gmock.h>
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/test.h"

namespace syntaxnet {
namespace dragnn {
namespace runtime {
namespace {

using ::testing::_;
using ::testing::Invoke;

// A working one-channel ComponentSpec.
const char kSingleSpec[] = R"(fixed_feature {
                                vocabulary_size: 11
                                embedding_dim: 35
                                size: 1
                              })";
const size_t kSingleRows = 11;
const size_t kSingleColumns = 35;

// A working multi-channel ComponentSpec.
const char kMultiSpec[] = R"(fixed_feature {
                               vocabulary_size: 13
                               embedding_dim: 11
                               size: 1
                             }
                             fixed_feature {
                               vocabulary_size: 19
                               embedding_dim: 17
                               size: 3
                             }
                             fixed_feature {
                               vocabulary_size: 29
                               embedding_dim: 23
                               size: 2
                             })";
const size_t kMultiRows[] = {13, 19, 29};
const size_t kMultiColumns[] = {11, 17, 23};
const size_t kMultiBases[] = {0, 1, 4};
const size_t kMultiSizes[] = {1, 3, 2};
const int kMultiNumChannels = 3;
const int kMultiNumEmbeddings = 6;

// A working one-channel ComponentSpec with non-embedded features.
const char kNonEmbeddedSpec[] = R"(fixed_feature {
                                     embedding_dim: -1
                                     size: 3
                                   })";

class FixedEmbeddingManagerTest : public NetworkTestBase {
 protected:
  // Resets the |manager_| and returns the result of Reset()-ing it using the
  // |component_spec_text|, |variable_store_|, and |network_state_manager_|.
  tensorflow::Status ResetManager(const string &component_spec_text) {
    ComponentSpec component_spec;
    CHECK(TextFormat::ParseFromString(component_spec_text, &component_spec));
    component_spec.set_name(kTestComponentName);

    AddComponent(kTestComponentName);
    return manager_.Reset(component_spec, &variable_store_,
                          &network_state_manager_);
  }

  FixedEmbeddingManager manager_;
};

// Tests that FixedEmbeddingManager is empty by default.
TEST_F(FixedEmbeddingManagerTest, EmptyByDefault) {
  EXPECT_EQ(manager_.num_channels(), 0);
  EXPECT_EQ(manager_.num_embeddings(), 0);
}

// Tests that FixedEmbeddingManager is empty when reset to an empty spec.
TEST_F(FixedEmbeddingManagerTest, EmptySpec) {
  TF_EXPECT_OK(ResetManager(""));

  EXPECT_EQ(manager_.component_name(), kTestComponentName);
  EXPECT_EQ(manager_.num_channels(), 0);
  EXPECT_EQ(manager_.num_embeddings(), 0);
}

// Tests that FixedEmbeddingManager produces the correct embedding dimension
// when configured with a single channel.
TEST_F(FixedEmbeddingManagerTest, OneChannel) {
  AddFixedEmbeddingMatrix(0, kSingleRows, kSingleColumns, 0.25);

  TF_EXPECT_OK(ResetManager(kSingleSpec));

  EXPECT_EQ(manager_.component_name(), kTestComponentName);
  EXPECT_EQ(manager_.num_channels(), 1);
  EXPECT_EQ(manager_.embedding_dim(0), kSingleColumns);
  EXPECT_EQ(manager_.num_embeddings(), 1);
  EXPECT_EQ(manager_.channel_base(0), 0);
  EXPECT_EQ(manager_.channel_size(0), 1);
  EXPECT_TRUE(manager_.is_embedded(0));
}

// Tests that FixedEmbeddingManager produces the correct embedding dimensions
// when configured with multiple channels.
TEST_F(FixedEmbeddingManagerTest, MultipleChannels) {
  for (int i = 0; i < kMultiNumChannels; ++i) {
    AddFixedEmbeddingMatrix(i, kMultiRows[i], kMultiColumns[i], -1.0);
  }

  TF_EXPECT_OK(ResetManager(kMultiSpec));

  EXPECT_EQ(manager_.component_name(), kTestComponentName);
  EXPECT_EQ(manager_.num_channels(), kMultiNumChannels);
  EXPECT_EQ(manager_.num_embeddings(), kMultiNumEmbeddings);
  for (int i = 0; i < kMultiNumChannels; ++i) {
    EXPECT_EQ(manager_.embedding_dim(i), kMultiColumns[i] * kMultiSizes[i]);
    EXPECT_EQ(manager_.channel_base(i), kMultiBases[i]);
    EXPECT_EQ(manager_.channel_size(i), kMultiSizes[i]);
    EXPECT_TRUE(manager_.is_embedded(i));
  }
}

// Tests that FixedEmbeddingManager works for non-embedded features.
TEST_F(FixedEmbeddingManagerTest, NonEmbeddedFeature) {
  TF_ASSERT_OK(ResetManager(kNonEmbeddedSpec));

  EXPECT_EQ(manager_.component_name(), kTestComponentName);
  EXPECT_EQ(manager_.num_channels(), 1);
  EXPECT_EQ(manager_.embedding_dim(0), 3);
  EXPECT_EQ(manager_.num_embeddings(), 3);
  EXPECT_EQ(manager_.channel_base(0), 0);
  EXPECT_EQ(manager_.channel_size(0), 3);
  EXPECT_FALSE(manager_.is_embedded(0));
}

// Tests that FixedEmbeddingManager fails when there are no embedding matrices.
TEST_F(FixedEmbeddingManagerTest, NoEmbeddingMatrices) {
  EXPECT_THAT(ResetManager(kSingleSpec),
              test::IsErrorWithSubstr("Unknown variable"));
}

// Tests that FixedEmbeddingManager fails when there are embedding matrices, but
// not for the right channel.
TEST_F(FixedEmbeddingManagerTest, MissingEmbeddingMatrix) {
  AddFixedEmbeddingMatrix(/* bad */ 1, kSingleRows, kSingleColumns, 0.25);

  EXPECT_THAT(ResetManager(kSingleSpec),
              test::IsErrorWithSubstr("Unknown variable"));
}

// Tests that FixedEmbeddingManager fails when the channel size is 0.
TEST_F(FixedEmbeddingManagerTest, InvalidChannelSize) {
  const string kBadSpec = R"(fixed_feature {
                               vocabulary_size: 8
                               embedding_dim: 16
                               size: 0  # bad
                             })";
  AddFixedEmbeddingMatrix(0, 8, 16, 0.25);

  EXPECT_THAT(ResetManager(kBadSpec),
              test::IsErrorWithSubstr("Invalid channel size"));
}

// Tests that the FixedEmbeddingManager fails when the embedding dimension does
// not match the embedding matrix.
TEST_F(FixedEmbeddingManagerTest, MismatchedEmbeddingDim) {
  const string kBadSpec = R"(fixed_feature {
                               vocabulary_size: 8
                               embedding_dim: 17  # bad
                               size: 1
                             })";
  AddFixedEmbeddingMatrix(0, 8, 16, 0.25);

  EXPECT_THAT(ResetManager(kBadSpec),
              test::IsErrorWithSubstr("ComponentSpec (17) and VariableStore "
                                      "(16) disagree on embedding dim"));
}

// Tests that the FixedEmbeddingManager fails when the vocabulary size does not
// match the embedding matrix.
TEST_F(FixedEmbeddingManagerTest, MismatchedVocabularySize) {
  const string kBadSpec = R"(fixed_feature {
                               vocabulary_size: 7  # bad
                               embedding_dim: 16
                               size: 1
                             })";
  AddFixedEmbeddingMatrix(0, 8, 16, 0.25);

  EXPECT_THAT(ResetManager(kBadSpec),
              test::IsErrorWithSubstr("ComponentSpec (7) and VariableStore "
                                      "(8) disagree on vocabulary size"));
}

class FixedEmbeddingsTest : public FixedEmbeddingManagerTest {
 protected:
  // Resets the |fixed_embeddings_| using the |manager_|, |network_states_|, and
  // |compute_session_|, and returns the resulting status.
  tensorflow::Status ResetFixedEmbeddings() {
    network_states_.Reset(&network_state_manager_);
    StartComponent(0);
    return fixed_embeddings_.Reset(&manager_, network_states_,
                                   &compute_session_);
  }

  // Returns a list of the expected size and value of each fixed embedding sum,
  // given that the channel-wise sums are the |channel_sums|.
  std::vector<std::pair<size_t, float>> ToEmbeddingSums(
      const std::vector<float> &channel_sums) {
    CHECK_EQ(channel_sums.size(), kMultiNumChannels);
    std::vector<std::pair<size_t, float>> expected_sums;
    for (int channel_id = 0; channel_id < kMultiNumChannels; ++channel_id) {
      for (int i = 0; i < kMultiSizes[channel_id]; ++i) {
        expected_sums.emplace_back(kMultiColumns[channel_id],
                                   channel_sums[channel_id]);
      }
    }
    return expected_sums;
  }

  // As above, but computes the channel sums as the product of |lhs| and |rhs|.
  std::vector<std::pair<size_t, float>> ToEmbeddingSums(
      const std::vector<float> &lhs, const std::vector<float> &rhs) {
    CHECK_EQ(lhs.size(), rhs.size());
    std::vector<float> channel_sums;
    for (int i = 0; i < lhs.size(); ++i) {
      channel_sums.push_back(lhs[i] * rhs[i]);
    }
    return ToEmbeddingSums(channel_sums);
  }

  FixedEmbeddings fixed_embeddings_;
};

// Tests that FixedEmbeddings is empty by default.
TEST_F(FixedEmbeddingsTest, EmptyByDefault) {
  EXPECT_EQ(fixed_embeddings_.num_embeddings(), 0);
}

// Tests that FixedEmbeddings is empty when reset with an empty manager.
TEST_F(FixedEmbeddingsTest, EmptyManager) {
  TF_ASSERT_OK(ResetManager(""));

  TF_ASSERT_OK(ResetFixedEmbeddings());
  EXPECT_EQ(fixed_embeddings_.num_embeddings(), 0);
}

// Tests that FixedEmbeddings produces a zero vector when no features are
// extracted.
TEST_F(FixedEmbeddingsTest, OneChannelNoFeatures) {
  AddFixedEmbeddingMatrix(0, kSingleRows, kSingleColumns, 0.5);
  TF_ASSERT_OK(ResetManager(kSingleSpec));

  EXPECT_CALL(compute_session_, GetInputFeatures(_, _, _, _, _))
      .WillOnce(Invoke(ExtractFeatures(0, {})));

  TF_ASSERT_OK(ResetFixedEmbeddings());
  ASSERT_EQ(fixed_embeddings_.num_embeddings(), 1);
  ExpectVector(fixed_embeddings_.embedding(0), kSingleColumns, 0.0);
}

// Tests that FixedEmbeddings produces a row of the embedding matrix when
// exactly one feature with weight=1 is extracted.
TEST_F(FixedEmbeddingsTest, OneChannelOneFeature) {
  AddFixedEmbeddingMatrix(0, kSingleRows, kSingleColumns, 0.125);
  TF_ASSERT_OK(ResetManager(kSingleSpec));

  EXPECT_CALL(compute_session_, GetInputFeatures(_, _, _, _, _))
      .WillOnce(Invoke(ExtractFeatures(0, {{kSingleRows - 1, 1.0}})));

  TF_ASSERT_OK(ResetFixedEmbeddings());
  ASSERT_EQ(fixed_embeddings_.num_embeddings(), 1);
  ExpectVector(fixed_embeddings_.embedding(0), kSingleColumns, 0.125);
}

// Tests that FixedEmbeddings produces a scaled row of the embedding matrix when
// exactly one feature with weight!=1 is extracted.
TEST_F(FixedEmbeddingsTest, OneChannelOneWeightedFeature) {
  AddFixedEmbeddingMatrix(0, kSingleRows, kSingleColumns, 0.5);
  TF_ASSERT_OK(ResetManager(kSingleSpec));

  EXPECT_CALL(compute_session_, GetInputFeatures(_, _, _, _, _))
      .WillOnce(Invoke(ExtractFeatures(0, {{0, -1.5}})));

  TF_ASSERT_OK(ResetFixedEmbeddings());
  ASSERT_EQ(fixed_embeddings_.num_embeddings(), 1);
  ExpectVector(fixed_embeddings_.embedding(0), kSingleColumns, -0.75);
}

// Tests that FixedEmbeddings produces a weighted embedding sum when multiple
// weighted features are extracted.
TEST_F(FixedEmbeddingsTest, OneChannelManyFeatures) {
  AddFixedEmbeddingMatrix(0, kSingleRows, kSingleColumns, 0.5);
  TF_ASSERT_OK(ResetManager(kSingleSpec));

  EXPECT_CALL(compute_session_, GetInputFeatures(_, _, _, _, _))
      .WillOnce(Invoke(ExtractFeatures(0, {{0, 1.0}, {1, -2.0}, {2, 4.0}})));
  const float kSum = 1.5;  // = 0.5 * (1.0 - 2.0 + 4.0)

  TF_ASSERT_OK(ResetFixedEmbeddings());
  ASSERT_EQ(fixed_embeddings_.num_embeddings(), 1);
  ExpectVector(fixed_embeddings_.embedding(0), kSingleColumns, kSum);
}

// Tests that FixedEmbeddings produces zero vectors for multiple channels that
// extract no features.
TEST_F(FixedEmbeddingsTest, ManyChannelsNoFeatures) {
  const std::vector<float> kValues = {0.0, 0.0, 0.0};
  for (int i = 0; i < kMultiNumChannels; ++i) {
    AddFixedEmbeddingMatrix(i, kMultiRows[i], kMultiColumns[i], 1.0);
  }
  TF_ASSERT_OK(ResetManager(kMultiSpec));

  EXPECT_CALL(compute_session_, GetInputFeatures(_, _, _, _, _))
      .WillOnce(Invoke(ExtractFeatures(0, {})))
      .WillOnce(Invoke(ExtractFeatures(1, {})))
      .WillOnce(Invoke(ExtractFeatures(2, {})));

  TF_ASSERT_OK(ResetFixedEmbeddings());
  const auto kSums = ToEmbeddingSums(kValues);
  ASSERT_EQ(fixed_embeddings_.num_embeddings(), kSums.size());
  for (int i = 0; i < kSums.size(); ++i) {
    ExpectVector(fixed_embeddings_.embedding(i), kSums[i].first,
                 kSums[i].second);
  }
}

// Tests that FixedEmbeddings produces rows of the embedding matrix for multiple
// channels that extract exactly one feature with weight=1.
TEST_F(FixedEmbeddingsTest, ManyChannelsOneFeature) {
  const std::vector<float> kValues = {1.0, -0.5, 0.75};
  ASSERT_EQ(kValues.size(), kMultiNumChannels);

  for (int i = 0; i < kMultiNumChannels; ++i) {
    AddFixedEmbeddingMatrix(i, kMultiRows[i], kMultiColumns[i], kValues[i]);
  }
  TF_ASSERT_OK(ResetManager(kMultiSpec));

  // NB: Sometimes the feature indices are extracted out-of-order.
  EXPECT_CALL(compute_session_, GetInputFeatures(_, _, _, _, _))
      .WillOnce(Invoke(ExtractFeatures(0, {{0, 10, 1.0}})))
      .WillOnce(Invoke(ExtractFeatures(1, {{1, 11, 1.0},  //
                                           {0, 11, 1.0},  //
                                           {2, 11, 1.0}})))
      .WillOnce(Invoke(ExtractFeatures(2, {{0, 12, 1.0},  //
                                           {1, 12, 1.0}})));

  TF_ASSERT_OK(ResetFixedEmbeddings());
  const auto kSums = ToEmbeddingSums(kValues);
  ASSERT_EQ(fixed_embeddings_.num_embeddings(), kSums.size());
  for (int i = 0; i < kSums.size(); ++i) {
    ExpectVector(fixed_embeddings_.embedding(i), kSums[i].first,
                 kSums[i].second);
  }
}

// Tests that FixedEmbeddings produces scaled rows of the embedding matrix for
// multiple channels that extract exactly one feature with weight!=1.
TEST_F(FixedEmbeddingsTest, ManyChannelsOneWeightedFeature) {
  const std::vector<float> kValues = {1.0, -0.5, 0.75};
  const std::vector<float> kFeatures = {1.25, 0.75, -1.5};
  ASSERT_EQ(kValues.size(), kMultiNumChannels);
  ASSERT_EQ(kFeatures.size(), kMultiNumChannels);

  for (int i = 0; i < kMultiNumChannels; ++i) {
    AddFixedEmbeddingMatrix(i, kMultiRows[i], kMultiColumns[i], kValues[i]);
  }
  TF_ASSERT_OK(ResetManager(kMultiSpec));

  // NB: Sometimes the feature indices are extracted out-of-order.
  EXPECT_CALL(compute_session_, GetInputFeatures(_, _, _, _, _))
      .WillOnce(Invoke(ExtractFeatures(0, {{0, 10, kFeatures[0]}})))
      .WillOnce(Invoke(ExtractFeatures(1, {{0, 11, kFeatures[1]},  //
                                           {1, 11, kFeatures[1]},  //
                                           {2, 11, kFeatures[1]}})))
      .WillOnce(Invoke(ExtractFeatures(2, {{1, 12, kFeatures[2]},  //
                                           {0, 12, kFeatures[2]}})));

  TF_ASSERT_OK(ResetFixedEmbeddings());
  const auto kSums = ToEmbeddingSums(kValues, kFeatures);
  ASSERT_EQ(fixed_embeddings_.num_embeddings(), kSums.size());
  for (int i = 0; i < kSums.size(); ++i) {
    ExpectVector(fixed_embeddings_.embedding(i), kSums[i].first,
                 kSums[i].second);
  }
}

// Tests that FixedEmbeddings produces weighted embedding sums for multiple
// channels that extract multiple weighted features.
TEST_F(FixedEmbeddingsTest, ManyChannelsManyFeatures) {
  const std::vector<float> kValues = {1.0, -0.5, 0.75};
  ASSERT_EQ(kValues.size(), kMultiNumChannels);

  for (int i = 0; i < kMultiNumChannels; ++i) {
    AddFixedEmbeddingMatrix(i, kMultiRows[i], kMultiColumns[i], kValues[i]);
  }
  TF_ASSERT_OK(ResetManager(kMultiSpec));

  // NB: Sometimes the feature indices are extracted out-of-order.
  EXPECT_CALL(compute_session_, GetInputFeatures(_, _, _, _, _))
      .WillOnce(Invoke(ExtractFeatures(0, {{0, 0, 1.0},   //
                                           {0, 1, -2.0},  //
                                           {0, 2, 9.0}})))
      .WillOnce(Invoke(ExtractFeatures(1, {{0, 0, 2.0},   //
                                           {1, 1, -4.0},  //
                                           {2, 2, 8.0},   //
                                           {1, 0, 2.0},   //
                                           {2, 1, -4.0},  //
                                           {0, 2, 8.0},   //
                                           {2, 0, 2.0},   //
                                           {0, 1, -4.0},  //
                                           {1, 2, 8.0}})))
      .WillOnce(Invoke(ExtractFeatures(2, {{0, 0, 3.0},   //
                                           {0, 1, -6.0},  //
                                           {0, 2, 7.0},   //
                                           {1, 2, 7.0},   //
                                           {1, 1, -6.0},  //
                                           {1, 0, 3.0}})));
  const std::vector<float> kFeatures = {1.0 - 2.0 + 9.0,
                                        2.0 - 4.0 + 8.0,
                                        3.0 - 6.0 + 7.0};
  ASSERT_EQ(kFeatures.size(), kMultiNumChannels);

  TF_ASSERT_OK(ResetFixedEmbeddings());
  const auto kSums = ToEmbeddingSums(kValues, kFeatures);
  ASSERT_EQ(fixed_embeddings_.num_embeddings(), kSums.size());
  for (int i = 0; i < kSums.size(); ++i) {
    ExpectVector(fixed_embeddings_.embedding(i), kSums[i].first,
                 kSums[i].second);
  }
}

// Tests that FixedEmbeddings produces feature IDs when configured with a
// non-embedded feature channel.
TEST_F(FixedEmbeddingsTest, NonEmbeddedFeature) {
  TF_ASSERT_OK(ResetManager(kNonEmbeddedSpec));

  // These feature values probe the boundaries of valid feature IDs.
  EXPECT_CALL(compute_session_, GetInputFeatures(_, _, _, _, _))
      .WillOnce(Invoke(ExtractFeatures(0, {{2, 2147483647, 1.0},  //
                                           {0, 0, 1.0},           //
                                           {1, 34, 1.0}})));

  TF_ASSERT_OK(ResetFixedEmbeddings());
  ASSERT_EQ(fixed_embeddings_.num_embeddings(), 3);
  ASSERT_EQ(fixed_embeddings_.ids(0).size(), 1);
  EXPECT_EQ(fixed_embeddings_.ids(0)[0], 0);
  ASSERT_EQ(fixed_embeddings_.ids(1).size(), 1);
  EXPECT_EQ(fixed_embeddings_.ids(1)[0], 34);
  ASSERT_EQ(fixed_embeddings_.ids(2).size(), 1);
  EXPECT_EQ(fixed_embeddings_.ids(2)[0], 2147483647);

  Vector<int32> ids;
  ids = network_states_.GetLocal(manager_.id_handle(0, 0));
  ASSERT_EQ(ids.size(), 1);
  EXPECT_EQ(ids[0], 0);
  ids = network_states_.GetLocal(manager_.id_handle(0, 1));
  ASSERT_EQ(ids.size(), 1);
  EXPECT_EQ(ids[0], 34);
  ids = network_states_.GetLocal(manager_.id_handle(0, 2));
  ASSERT_EQ(ids.size(), 1);
  EXPECT_EQ(ids[0], 2147483647);
}

// Tests that FixedEmbeddings fails if a feature ID has a negative ID.
TEST_F(FixedEmbeddingsTest, NonEmbeddedFeatureNegativeId) {
  TF_ASSERT_OK(ResetManager(kNonEmbeddedSpec));

  EXPECT_CALL(compute_session_, GetInputFeatures(_, _, _, _, _))
      .WillOnce(Invoke(ExtractFeatures(0, {{2, -1, 1.0},  //
                                           {0, 12, 1.0},  //
                                           {1, 34, 1.0}})));

  EXPECT_THAT(ResetFixedEmbeddings(),
              test::IsErrorWithSubstr(tensorflow::strings::StrCat(
                  "Component '", kTestComponentName,
                  "' channel 0 index 2: Invalid non-embedded feature ID -1")));
}

// Tests that FixedEmbeddings fails if a feature ID has an ID that is too large.
TEST_F(FixedEmbeddingsTest, NonEmbeddedFeatureIdTooLarge) {
  TF_ASSERT_OK(ResetManager(kNonEmbeddedSpec));

  EXPECT_CALL(compute_session_, GetInputFeatures(_, _, _, _, _))
      .WillOnce(Invoke(ExtractFeatures(0, {{2, 56, 1.0},          //
                                           {0, 2147483648, 1.0},  //
                                           {1, 34, 1.0}})));

  EXPECT_THAT(ResetFixedEmbeddings(),
              test::IsErrorWithSubstr(tensorflow::strings::StrCat(
                  "Component '", kTestComponentName,
                  "' channel 0 index 0: Invalid non-embedded feature ID "
                  "2147483648")));
}

// Tests that FixedEmbeddings fails if a feature weight is not 1.0.
TEST_F(FixedEmbeddingsTest, NonEmbeddedFeatureNonIdentityWeight) {
  TF_ASSERT_OK(ResetManager(kNonEmbeddedSpec));

  EXPECT_CALL(compute_session_, GetInputFeatures(_, _, _, _, _))
      .WillOnce(Invoke(ExtractFeatures(0, {{2, 56, 1.0},  //
                                           {0, 12, 1.0},  //
                                           {1, 34, 1.5}})));

  EXPECT_THAT(ResetFixedEmbeddings(),
              test::IsErrorWithSubstr(tensorflow::strings::StrCat(
                  "Component '", kTestComponentName,
                  "' channel 0 index 1: Invalid non-embedded feature weight "
                  "1.5 (expected 1.0)")));
}

// Tests that FixedEmbeddings fails if a feature ID is duplicated.
TEST_F(FixedEmbeddingsTest, NonEmbeddedFeatureDuplicateId) {
  TF_ASSERT_OK(ResetManager(kNonEmbeddedSpec));

  EXPECT_CALL(compute_session_, GetInputFeatures(_, _, _, _, _))
      .WillOnce(Invoke(ExtractFeatures(0, {{2, 56, 1.0},  //
                                           {2, 56, 1.0},  //
                                           {0, 12, 1.0},  //
                                           {1, 34, 1.0}})));

  EXPECT_THAT(
      ResetFixedEmbeddings(),
      test::IsErrorWithSubstr(tensorflow::strings::StrCat(
          "Component '", kTestComponentName,
          "' channel 0 index 2: Duplicate non-embedded feature ID 56")));
}

// Tests that FixedEmbeddings fails if a feature ID is missing.
TEST_F(FixedEmbeddingsTest, NonEmbeddedFeatureMissingId) {
  TF_ASSERT_OK(ResetManager(kNonEmbeddedSpec));

  EXPECT_CALL(compute_session_, GetInputFeatures(_, _, _, _, _))
      .WillOnce(Invoke(ExtractFeatures(0, {{2, 56, 1.0},  //
                                           {1, 34, 1.0}})));

  EXPECT_THAT(ResetFixedEmbeddings(),
              test::IsErrorWithSubstr(tensorflow::strings::StrCat(
                  "Component '", kTestComponentName,
                  "' channel 0 index 0: Missing non-embedded feature ID")));
}

}  // namespace
}  // namespace runtime
}  // namespace dragnn
}  // namespace syntaxnet
