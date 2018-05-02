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

#include "dragnn/runtime/sequence_features.h"

#include <string>
#include <vector>

#include "dragnn/core/input_batch_cache.h"
#include "dragnn/core/test/generic.h"
#include "dragnn/protos/spec.pb.h"
#include "dragnn/runtime/fixed_embeddings.h"
#include "dragnn/runtime/sequence_extractor.h"
#include "dragnn/runtime/test/network_test_base.h"
#include "syntaxnet/base.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/test.h"

namespace syntaxnet {
namespace dragnn {
namespace runtime {
namespace {

// Number of transition steps to take in each component in the network.
const size_t kNumSteps = 10;

// A working one-channel ComponentSpec.  This is intentionally identical to the
// first channel of |kMultiSpec|, so they can use the same embedding matrix.
const char kSingleSpec[] = R"(fixed_feature {
                                vocabulary_size: 13
                                embedding_dim: 11
                                size: 1
                              })";
const size_t kSingleRows = 13;
const size_t kSingleColumns = 11;
constexpr float kSingleValue = 1.25;

// A working multi-channel ComponentSpec.
const char kMultiSpec[] = R"(fixed_feature {
                               vocabulary_size: 13
                               embedding_dim: 11
                               size: 1
                             }
                             fixed_feature {
                               embedding_dim: -1
                               size: 1
                             }
                             fixed_feature {
                               embedding_dim: -1
                               size: 1
                             })";

// Fails to initialize.
class FailToInitialize : public SequenceExtractor {
 public:
  // Implements SequenceExtractor.
  bool Supports(const FixedFeatureChannel &,
                const ComponentSpec &component_spec) const override {
    LOG(FATAL) << "Should never be called.";
  }
  tensorflow::Status Initialize(const FixedFeatureChannel &,
                                const ComponentSpec &) override {
    return tensorflow::errors::Internal("No initialization for you!");
  }
  tensorflow::Status GetIds(InputBatchCache *,
                            std::vector<int32> *) const override {
    LOG(FATAL) << "Should never be called.";
  }
};

DRAGNN_RUNTIME_REGISTER_SEQUENCE_EXTRACTOR(FailToInitialize);

// Initializes OK, then fails to extract features.
class FailToGetIds : public FailToInitialize {
 public:
  // Implements SequenceExtractor.
  tensorflow::Status Initialize(const FixedFeatureChannel &,
                                const ComponentSpec &) override {
    return tensorflow::Status::OK();
  }
  tensorflow::Status GetIds(InputBatchCache *,
                            std::vector<int32> *) const override {
    return tensorflow::errors::Internal("No features for you!");
  }
};

DRAGNN_RUNTIME_REGISTER_SEQUENCE_EXTRACTOR(FailToGetIds);

// Initializes OK and extracts the previous step.
class ExtractPrevious : public FailToGetIds {
 public:
  // Implements SequenceExtractor.
  tensorflow::Status GetIds(InputBatchCache *,
                            std::vector<int32> *ids) const override {
    ids->resize(kNumSteps);
    for (int i = 0; i < kNumSteps; ++i) (*ids)[i] = i - 1;
    return tensorflow::Status::OK();
  }
};

DRAGNN_RUNTIME_REGISTER_SEQUENCE_EXTRACTOR(ExtractPrevious);

// Initializes OK but produces the wrong number of features.
class WrongNumberOfIds : public FailToGetIds {
 public:
  // Implements SequenceExtractor.
  tensorflow::Status GetIds(InputBatchCache *input,
                            std::vector<int32> *ids) const override {
    ids->resize(kNumSteps + 1);
    return tensorflow::Status::OK();
  }
};

DRAGNN_RUNTIME_REGISTER_SEQUENCE_EXTRACTOR(WrongNumberOfIds);

class SequenceFeatureManagerTest : public NetworkTestBase {
 protected:
  // Creates a SequenceFeatureManager and returns the result of Reset()-ing it
  // using the |component_spec_text|.
  tensorflow::Status ResetManager(
      const string &component_spec_text,
      const std::vector<string> &sequence_extractor_types) {
    ComponentSpec component_spec;
    CHECK(TextFormat::ParseFromString(component_spec_text, &component_spec));
    component_spec.set_name(kTestComponentName);

    AddFixedEmbeddingMatrix(0, kSingleRows, kSingleColumns, kSingleValue);
    AddComponent(kTestComponentName);

    TF_RETURN_IF_ERROR(fixed_embedding_manager_.Reset(
        component_spec, &variable_store_, &network_state_manager_));

    return manager_.Reset(&fixed_embedding_manager_, component_spec,
                          sequence_extractor_types);
  }

  FixedEmbeddingManager fixed_embedding_manager_;
  SequenceFeatureManager manager_;
};

// Tests that SequenceFeatureManager is empty by default.
TEST_F(SequenceFeatureManagerTest, EmptyByDefault) {
  EXPECT_EQ(manager_.num_channels(), 0);
}

// Tests that SequenceFeatureManager is empty when reset to an empty spec.
TEST_F(SequenceFeatureManagerTest, EmptySpec) {
  TF_EXPECT_OK(ResetManager("", {}));

  EXPECT_EQ(manager_.num_channels(), 0);
}

// Tests that SequenceFeatureManager works with a single channel.
TEST_F(SequenceFeatureManagerTest, OneChannel) {
  TF_EXPECT_OK(ResetManager(kSingleSpec, {"ExtractPrevious"}));

  EXPECT_EQ(manager_.num_channels(), 1);
}

// Tests that SequenceFeatureManager works with multiple channels.
TEST_F(SequenceFeatureManagerTest, MultipleChannels) {
  TF_EXPECT_OK(ResetManager(
      kMultiSpec, {"ExtractPrevious", "ExtractPrevious", "ExtractPrevious"}));

  EXPECT_EQ(manager_.num_channels(), 3);
}

// Tests that SequenceFeatureManager fails if the FixedEmbeddingManager and
// ComponentSpec are mismatched.
TEST_F(SequenceFeatureManagerTest, MismatchedFixedManagerAndComponentSpec) {
  ComponentSpec component_spec;
  CHECK(TextFormat::ParseFromString(kMultiSpec, &component_spec));
  component_spec.set_name(kTestComponentName);

  AddFixedEmbeddingMatrix(0, kSingleRows, kSingleColumns, kSingleValue);
  AddComponent(kTestComponentName);

  TF_ASSERT_OK(fixed_embedding_manager_.Reset(component_spec, &variable_store_,
                                               &network_state_manager_));

  // Remove one fixed feature, resulting in a mismatch.
  component_spec.mutable_fixed_feature()->RemoveLast();

  EXPECT_THAT(
      manager_.Reset(&fixed_embedding_manager_, component_spec,
                     {"ExtractPrevious", "ExtractPrevious", "ExtractPrevious"}),
      test::IsErrorWithSubstr("Channel mismatch between FixedEmbeddingManager "
                              "(3) and ComponentSpec (2)"));
}

// Tests that SequenceFeatureManager fails if the FixedEmbeddingManager and
// SequenceExtractors are mismatched.
TEST_F(SequenceFeatureManagerTest,
       MismatchedFixedManagerAndSequenceExtractors) {
  EXPECT_THAT(
      ResetManager(kMultiSpec, {"ExtractPrevious", "ExtractPrevious"}),
      test::IsErrorWithSubstr("Channel mismatch between FixedEmbeddingManager "
                              "(3) and SequenceExtractors (2)"));
}

// Tests that SequenceFeatureManager fails if a channel has multiple embeddings.
TEST_F(SequenceFeatureManagerTest, UnsupportedMultiEmbeddingChannel) {
  const string kBadSpec = R"(fixed_feature {
                               vocabulary_size: 13
                               embedding_dim: 11
                               size: 2  # bad
                             })";

  EXPECT_THAT(ResetManager(kBadSpec, {"ExtractPrevious"}),
              test::IsErrorWithSubstr(
                  "Multi-embedding fixed features are not supported"));
}

// Tests that SequenceFeatureManager fails if one of the SequenceExtractors
// fails to initialize.
TEST_F(SequenceFeatureManagerTest, FailToInitializeSequenceExtractor) {
  EXPECT_THAT(ResetManager(kMultiSpec, {"ExtractPrevious", "FailToInitialize",
                                        "ExtractPrevious"}),
              test::IsErrorWithSubstr("No initialization for you!"));
}

// Tests that SequenceFeatureManager is OK even if the SequenceExtractors would
// fail in GetIds().
TEST_F(SequenceFeatureManagerTest, ManagerDoesntCareAboutGetIds) {
  TF_EXPECT_OK(ResetManager(
      kMultiSpec, {"FailToGetIds", "FailToGetIds", "FailToGetIds"}));
}

class SequenceFeaturesTest : public SequenceFeatureManagerTest {
 protected:
  // Resets the |sequence_features_| on the |manager_| and |input_batch_cache_|
  // and returns the resulting status.
  tensorflow::Status ResetFeatures() {
    return sequence_features_.Reset(&manager_, &input_batch_cache_);
  }

  InputBatchCache input_batch_cache_;
  SequenceFeatures sequence_features_;
};

// Tests that SequenceFeatures is empty by default.
TEST_F(SequenceFeaturesTest, EmptyByDefault) {
  EXPECT_EQ(sequence_features_.num_channels(), 0);
  EXPECT_EQ(sequence_features_.num_steps(), 0);
}

// Tests that SequenceFeatures is empty when reset by an empty manager.
TEST_F(SequenceFeaturesTest, EmptyManager) {
  TF_ASSERT_OK(ResetManager("", {}));

  TF_EXPECT_OK(ResetFeatures());
  EXPECT_EQ(sequence_features_.num_channels(), 0);
  EXPECT_EQ(sequence_features_.num_steps(), 0);
}

// Tests that SequenceFeatures fails when one of the SequenceExtractors fails.
TEST_F(SequenceFeaturesTest, FailToGetIds) {
  TF_ASSERT_OK(ResetManager(
      kMultiSpec, {"ExtractPrevious", "ExtractPrevious", "FailToGetIds"}));

  EXPECT_THAT(ResetFeatures(), test::IsErrorWithSubstr("No features for you!"));
}

// Tests that SequenceFeatures fails when the SequenceExtractors produce
// different numbers of features.
TEST_F(SequenceFeaturesTest, MismatchedNumbersOfFeatures) {
  TF_ASSERT_OK(ResetManager(
      kMultiSpec, {"ExtractPrevious", "ExtractPrevious", "WrongNumberOfIds"}));

  EXPECT_THAT(ResetFeatures(), test::IsErrorWithSubstr(
                                   "Inconsistent feature sequence lengths at "
                                   "channel ID 2: got 11 but expected 10"));
}

// Tests that SequenceFeatures works as expected on one channel.
TEST_F(SequenceFeaturesTest, SingleChannel) {
  TF_ASSERT_OK(ResetManager(kSingleSpec, {"ExtractPrevious"}));

  TF_ASSERT_OK(ResetFeatures());
  ASSERT_EQ(sequence_features_.num_channels(), 1);
  ASSERT_EQ(sequence_features_.num_steps(), kNumSteps);

  // ExtractPrevious extracts -1 for the 0'th target index, which indicates a
  // missing ID and should be mapped to a zero vector.
  ExpectVector(sequence_features_.GetEmbedding(0, 0), kSingleColumns, 0.0);
  EXPECT_DEBUG_DEATH(sequence_features_.GetId(0, 0), "is_embedded");

  // The remaining feature IDs map to valid embedding rows.
  for (int i = 1; i < kNumSteps; ++i) {
    ExpectVector(sequence_features_.GetEmbedding(0, i), kSingleColumns,
                 kSingleValue);
    EXPECT_DEBUG_DEATH(sequence_features_.GetId(0, i), "is_embedded");
  }
}

// Tests that SequenceFeatures works as expected on multiple channels.
TEST_F(SequenceFeaturesTest, ManyChannels) {
  TF_ASSERT_OK(ResetManager(
      kMultiSpec, {"ExtractPrevious", "ExtractPrevious", "ExtractPrevious"}));

  TF_ASSERT_OK(ResetFeatures());
  ASSERT_EQ(sequence_features_.num_channels(), 3);
  ASSERT_EQ(sequence_features_.num_steps(), kNumSteps);

  // ExtractPrevious extracts -1 for the 0'th target index, which indicates a
  // missing ID and should be mapped to a zero vector.
  ExpectVector(sequence_features_.GetEmbedding(0, 0), kSingleColumns, 0.0);
  EXPECT_EQ(sequence_features_.GetId(1, 0), -1);
  EXPECT_EQ(sequence_features_.GetId(2, 0), -1);

  EXPECT_DEBUG_DEATH(sequence_features_.GetId(0, 0), "is_embedded");
  EXPECT_DEBUG_DEATH(sequence_features_.GetEmbedding(1, 0), "is_embedded");
  EXPECT_DEBUG_DEATH(sequence_features_.GetEmbedding(2, 0), "is_embedded");

  // The remaining features point to the previous item.
  for (int i = 1; i < kNumSteps; ++i) {
    ExpectVector(sequence_features_.GetEmbedding(0, i), kSingleColumns,
                 kSingleValue);
    EXPECT_EQ(sequence_features_.GetId(1, i), i - 1);
    EXPECT_EQ(sequence_features_.GetId(2, i), i - 1);

    EXPECT_DEBUG_DEATH(sequence_features_.GetId(0, i), "is_embedded");
    EXPECT_DEBUG_DEATH(sequence_features_.GetEmbedding(1, i), "is_embedded");
    EXPECT_DEBUG_DEATH(sequence_features_.GetEmbedding(2, i), "is_embedded");
  }
}

}  // namespace
}  // namespace runtime
}  // namespace dragnn
}  // namespace syntaxnet
