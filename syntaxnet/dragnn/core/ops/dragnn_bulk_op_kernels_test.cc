#include "dragnn/components/util/bulk_feature_extractor.h"
#include "dragnn/core/compute_session_pool.h"
#include "dragnn/core/resource_container.h"
#include "dragnn/core/test/mock_compute_session.h"

#include <gmock/gmock.h>
#include "tensorflow/core/framework/fake_input.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/kernels/ops_testutil.h"

namespace syntaxnet {
namespace dragnn {

using tensorflow::AllocatorAttributes;
using tensorflow::DT_FLOAT;
using tensorflow::DT_STRING;
using tensorflow::FrameAndIter;
using tensorflow::NodeDefBuilder;
using tensorflow::OpKernelContext;
using tensorflow::ResourceMgr;
using tensorflow::ScopedStepContainer;
using tensorflow::Status;
using tensorflow::TensorShape;
using tensorflow::checkpoint::TensorSliceReaderCacheWrapper;
using tensorflow::test::SetOutputAttrs;

using testing::Return;
using testing::_;

typedef ResourceContainer<ComputeSession> ComputeSessionResource;
typedef ResourceContainer<ComputeSessionPool> ComputeSessionPoolResource;

class DragnnBulkOpKernelsTest : public tensorflow::OpsTestBase {
 public:
  static const int kEmbeddingSize = 2;
  static const int kNumActions = 3;
  static const int kNumChannels = 2;
  static const int kNumIds = 8;
  static const int kNumItems = 3;
  static const int kNumSteps = 3;
  const string kComponentName = "TESTING_COMPONENT_NAME";

  MockComputeSession *GetMockSession() {
    TF_CHECK_OK(InitOp());

    // Set the input data.
    const string container_string = "container_str";
    const string id_string = "id_str";
    AddInputFromList<string>(TensorShape({2}), {container_string, id_string});

    // Reset the test context to ensure it's clean.
    ResetOpKernelContext();

    // Create a MockComputeSession and set expectations.
    std::unique_ptr<MockComputeSession> mock_session(new MockComputeSession());
    MockComputeSession *mock_session_ptr = mock_session.get();

    // Wrap the ComputeSessionResource and put it into the resource manager.
    TF_CHECK_OK(resource_mgr()->Create<ComputeSessionResource>(
        container_string, id_string,
        new ComputeSessionResource(std::move(mock_session))));
    return mock_session_ptr;
  }

  void ResetOpKernelContext() {
    params_.reset(new OpKernelContext::Params);
    params_->device = device_.get();
    params_->frame_iter = FrameAndIter(0, 0);
    params_->inputs = &inputs_;
    params_->op_kernel = kernel_.get();
    step_container_.reset(new ScopedStepContainer(0, [](const string &) {}));
    params_->step_container = step_container_.get();
    attrs_.clear();
    SetOutputAttrs(params_.get(), &attrs_);
    TensorSliceReaderCacheWrapper slice_reader_cache_wrapper;
    params_->slice_reader_cache = &slice_reader_cache_wrapper;
    params_->resource_manager = device_->resource_manager();
    context_.reset(new OpKernelContext(params_.get()));
  }

  Status RunOpKernelWithContext() {
    device_->Compute(kernel_.get(), context_.get());
    return context_->status();
  }

  // Accessor for the underlying resource manager.
  ResourceMgr *resource_mgr() { return params_->resource_manager; }
  /*
  // Returns a vector with dimensions: channel x batch x step.
  // For each item we return features for three steps:
  //   feature step 0: (5, 1)
  //   feature step 1: (5, 0.5), (6, 0.7)
  //   feature step 2: (3, 0.1), (7, [empty]) <- Default weight is 1.0.
  void ExpectFeatures(MockComputeSession *mock_session) {
    vector<FixedFeatures> feature_step_zero, feature_step_one, feature_step_two;
    for (int item = 0; item < kNumItems; ++item) {
      feature_step_zero.emplace_back();
      feature_step_zero.back().add_id(5);
      feature_step_zero.back().add_weight(1.0);
      feature_step_one.emplace_back();
      feature_step_one.back().add_id(5);
      feature_step_one.back().add_weight(0.5);
      feature_step_one.back().add_id(6);
      feature_step_one.back().add_weight(0.7);
      feature_step_two.emplace_back();
      feature_step_two.back().add_id(3);
      feature_step_two.back().add_weight(0.1);
      feature_step_two.back().add_id(7);
    }
    for (int channel = 0; channel < kNumChannels; ++channel) {
      EXPECT_CALL(*mock_session, GetInputFeatures(kComponentName, channel))
          .Times(3)
          .WillOnce(Return(feature_step_zero))
          .WillOnce(Return(feature_step_one))
          .WillOnce(Return(feature_step_two));
    }
  }

  // Returns a vector with dimensions: channel x batch x step.
  // For each item we return features for three steps with ids only:
  //   feature step 0: id=5
  //   feature step 1: id=6
  //   feature step 2: id=3
  void ExpectFeatureIds(MockComputeSession *mock_session) {
    vector<FixedFeatures> feature_step_zero, feature_step_one, feature_step_two;
    for (int item = 0; item < kNumItems; ++item) {
      feature_step_zero.emplace_back();
      feature_step_zero.back().add_id(5);
      feature_step_one.emplace_back();
      feature_step_one.back().add_id(6);
      feature_step_two.emplace_back();
      feature_step_two.back().add_id(3);
    }
    for (int channel = 0; channel < kNumChannels; ++channel) {
      EXPECT_CALL(*mock_session, GetInputFeatures(kComponentName, channel))
          .Times(3)
          .WillOnce(Return(feature_step_zero))
          .WillOnce(Return(feature_step_one))
          .WillOnce(Return(feature_step_two));
    }
  }
  */
  // This needs to maintain its existence throughout the compute call.
  std::vector<AllocatorAttributes> attrs_;
};

const int DragnnBulkOpKernelsTest::kEmbeddingSize;
const int DragnnBulkOpKernelsTest::kNumActions;
const int DragnnBulkOpKernelsTest::kNumChannels;
const int DragnnBulkOpKernelsTest::kNumIds;
const int DragnnBulkOpKernelsTest::kNumItems;
const int DragnnBulkOpKernelsTest::kNumSteps;

// The ExtractFixedFeatures op should return a set of fixed feature vectors
// as described below.
TEST_F(DragnnBulkOpKernelsTest, BulkFixedFeatures) {
  // Create and initialize the kernel under test.
  TF_ASSERT_OK(
      NodeDefBuilder("BulkFixedFeatures", "BulkFixedFeatures")
          .Attr("component", kComponentName)
          .Attr("num_channels", kNumChannels)
          .Input(FakeInput(DT_STRING))  // The handle for the ComputeSession.
          .Finalize(node_def()));

  MockComputeSession *mock_session = GetMockSession();
  const std::vector<int> expected_indices({0, 2, 1, 0, 1});
  const std::vector<int> expected_ids({5, 5, 6, 3, 7});
  const std::vector<float> expected_weights({1.0, 0.5, 0.7, 0.1, 1.0});

  // This function takes the allocator functions passed into GetBulkFF, uses
  // them to allocate a tensor, then fills that tensor based on channel.
  auto assigner_function = [=](string, const BulkFeatureExtractor &extractor) {
    constexpr int kFeatureCount = 3;
    constexpr int kTotalFeatures = 5;
    constexpr int kNumSteps = 3;
    for (int i = 0; i < kNumChannels; ++i) {
      // Allocate a new tensor set for every channel.
      int32 *indices =
          extractor.AllocateIndexMemory(i, kTotalFeatures * kNumSteps);
      int64 *ids = extractor.AllocateIdMemory(i, kTotalFeatures * kNumSteps);
      float *weights =
          extractor.AllocateWeightMemory(i, kTotalFeatures * kNumSteps);

      // Fill the tensor.
      int array_index = 0;
      for (int step = 0; step < kNumSteps; step++) {
        for (int j = 0; j < kTotalFeatures; ++j) {
          int offset = i + 1;
          indices[array_index] =
              (expected_indices[j] + step * kFeatureCount) * offset;
          ids[array_index] = expected_ids[j] * offset;
          weights[array_index] = expected_weights[j] * offset;
          ++array_index;
        }
      }
    }
    return kNumSteps;
  };

  EXPECT_CALL(*mock_session, BulkGetInputFeatures(kComponentName, _))
      .WillOnce(testing::Invoke(assigner_function));

  // Run the kernel.
  TF_EXPECT_OK(RunOpKernelWithContext());

  // Validate the outputs.
  // In this case, for every channel we should have:
  //   indices = [0  , 2  , 1  , 0  , 1  ]
  //             [3  , 5  , 4  , 3  , 4  ]
  //             [6  , 8  , 7  , 6  , 7  ]
  //   ids =     [5  , 5  , 6  , 3  , 7  ]
  //             [5  , 5  , 6  , 3  , 7  ]
  //             [5  , 5  , 6  , 3  , 7  ]
  //   weights = [1.0, 0.5, 0.7, 0.1, 1.0]
  //             [1.0, 0.5, 0.7, 0.1, 1.0]
  //             [1.0, 0.5, 0.7, 0.1, 1.0]

  for (int i = 0; i < kNumChannels * 3; ++i) {
    EXPECT_EQ(expected_indices.size() * kNumSteps,
              GetOutput(i + 1)->NumElements());
  }
  for (int channel = 0; channel < kNumChannels; ++channel) {
    LOG(INFO) << "Channel " << channel;
    for (int step = 0; step < kNumSteps; ++step) {
      for (int i = 0; i < expected_indices.size(); ++i) {
        const int j = i + step * expected_indices.size();

        // Note that the expectation on the indices changes per step, unlike the
        // expectation for ids and weights.
        int offset = channel + 1;
        EXPECT_EQ((expected_indices[i] + step * kNumItems) * offset,
                  GetOutput(channel + 1)->vec<int32>()(j));
        EXPECT_EQ(expected_ids[i] * offset,
                  GetOutput(kNumChannels + channel + 1)->vec<int64>()(j));
        EXPECT_EQ(expected_weights[i] * offset,
                  GetOutput(2 * kNumChannels + channel + 1)->vec<float>()(j));
      }
    }
  }
  EXPECT_EQ(kNumSteps, GetOutput(3 * kNumChannels + 1)->scalar<int32>()());
}

TEST_F(DragnnBulkOpKernelsTest, BulkFixedEmbeddings) {
  // Create and initialize the kernel under test.
  TF_ASSERT_OK(
      NodeDefBuilder("BulkFixedEmbeddings", "BulkFixedEmbeddings")
          .Attr("component", kComponentName)
          .Attr("num_channels", kNumChannels)
          .Input(FakeInput(DT_STRING))  // The handle for the ComputeSession.
          .Input(FakeInput(DT_FLOAT))   // Embedding matrices.
          .Finalize(node_def()));
  MockComputeSession *mock_session = GetMockSession();
  ComponentSpec spec;
  spec.set_name(kComponentName);
  auto chan0_spec = spec.add_fixed_feature();
  chan0_spec->set_size(2);
  auto chan1_spec = spec.add_fixed_feature();
  chan1_spec->set_size(1);
  EXPECT_CALL(*mock_session, Spec(kComponentName))
      .WillOnce(testing::ReturnRef(spec));

  EXPECT_CALL(*mock_session, BatchSize(kComponentName))
      .WillOnce(Return(kNumItems));

  const std::vector<int> feature_step_1({0, 1, 2, 1, 2, 2, 1, 0, 1, 0});
  const std::vector<int> feature_index_1({0, 0, 0, 0, 0, 1, 1, 1, 1, 1});
  const std::vector<int> feature_ids_1({5, 6, 3, 5, 7, 5, 6, 3, 5, 7});
  const std::vector<float> feature_weights_1(
      {1.0, 0.7, 0.1, 0.5, 1.0, 10, 7, 1, 5, 10});

  const std::vector<int> feature_step_2({0, 1, 2, 1, 2});
  const std::vector<int> feature_index_2({0, 0, 0, 0, 0});
  const std::vector<int> feature_ids_2({5, 6, 3, 5, 7});
  const std::vector<float> feature_weights_2({1.0, 0.7, 0.1, 0.5, 1.0});

  const std::vector<std::vector<int>> feature_steps_by_channel(
      {feature_step_1, feature_step_2});
  const std::vector<std::vector<int>> feature_index_by_channel(
      {feature_index_1, feature_index_2});
  const std::vector<std::vector<int>> feature_ids_by_channel(
      {feature_ids_1, feature_ids_2});
  const std::vector<std::vector<float>> feature_weights_by_channel(
      {feature_weights_1, feature_weights_2});

  // This function takes the allocator functions passed into GetBulkFF, uses
  // them to allocate a tensor, then fills that tensor based on channel.
  auto assigner_function = [=](string, const BulkFeatureExtractor &extractor) {
    constexpr int kNumElements = 3;
    constexpr int kNumSteps = 3;
    for (int i = 0; i < kNumChannels; ++i) {
      auto feature_step = feature_steps_by_channel.at(i);
      auto feature_index = feature_index_by_channel.at(i);
      auto feature_ids = feature_ids_by_channel.at(i);
      auto feature_weights = feature_weights_by_channel.at(i);

      // Allocate a new tensor set for every channel.
      int32 *indices =
          extractor.AllocateIndexMemory(i, kNumElements * feature_step.size());
      int64 *ids =
          extractor.AllocateIdMemory(i, kNumElements * feature_step.size());
      float *weights =
          extractor.AllocateWeightMemory(i, kNumElements * feature_step.size());

      // Fill the tensor.
      int array_index = 0;

      for (int element = 0; element < kNumElements; ++element) {
        for (int feature = 0; feature < feature_step.size(); ++feature) {
          indices[array_index] = extractor.GetIndex(
              kNumSteps, kNumElements, feature_index[feature], element,
              feature_step[feature]);
          ids[array_index] = feature_ids[feature];
          weights[array_index] = feature_weights[feature];
          ++array_index;
        }
      }
    }
    return kNumSteps;
  };

  EXPECT_CALL(*mock_session, BulkGetInputFeatures(kComponentName, _))
      .WillOnce(testing::Invoke(assigner_function));

  // Embedding matrices as additional inputs.
  // For channel 0, the embeddings are [id, 0].
  // For channel 1, the embeddings are [0, id].
  vector<float> embedding_matrix_a;
  vector<float> embedding_matrix_b;
  for (int id = 0; id < kNumIds; ++id) {
    embedding_matrix_a.push_back(id);
    embedding_matrix_a.push_back(0);
    embedding_matrix_b.push_back(0);
    embedding_matrix_b.push_back(id);
  }
  AddInputFromArray<float>(TensorShape({8, 2}), embedding_matrix_a);
  AddInputFromArray<float>(TensorShape({8, 2}), embedding_matrix_b);

  // Run the kernel.
  TF_EXPECT_OK(RunOpKernelWithContext());

  // Validate the outputs.
  // In this case we should have, for every item, these three steps:
  const vector<vector<float>> expected_embeddings = {{5.0, 0, 73, 0, 0, 5.0},
                                                     {6.7, 0, 67, 0, 0, 6.7},
                                                     {7.3, 0, 50, 0, 0, 7.3}};
  EXPECT_EQ(kNumSteps * kNumItems, GetOutput(1)->shape().dim_size(0));
  constexpr int kNumFeatures = 3;
  EXPECT_EQ(kNumFeatures * kEmbeddingSize, GetOutput(1)->shape().dim_size(1));
  for (int item = 0; item < kNumItems; ++item) {
    for (int step = 0; step < kNumSteps; ++step) {
      for (int col = 0; col < kNumChannels * kEmbeddingSize; ++col) {
        const int row = item * kNumSteps + step;
        EXPECT_EQ(expected_embeddings[step][col],
                  GetOutput(1)->matrix<float>()(row, col))
            << "step: " << step << ", row: " << row << ", col: " << col;
      }
    }
  }

  EXPECT_EQ(kNumSteps, GetOutput(2)->scalar<int32>()());
}

TEST_F(DragnnBulkOpKernelsTest, BulkFixedEmbeddingsWithPadding) {
  // Create and initialize the kernel under test.
  constexpr int kPaddedNumSteps = 5;
  constexpr int kPaddedBatchSize = 4;
  TF_ASSERT_OK(
      NodeDefBuilder("BulkFixedEmbeddings", "BulkFixedEmbeddings")
          .Attr("component", kComponentName)
          .Attr("num_channels", kNumChannels)
          .Attr("pad_to_steps", kPaddedNumSteps)
          .Attr("pad_to_batch", kPaddedBatchSize)
          .Input(FakeInput(DT_STRING))  // The handle for the ComputeSession.
          .Input(FakeInput(DT_FLOAT))   // Embedding matrices.
          .Finalize(node_def()));
  MockComputeSession *mock_session = GetMockSession();
  ComponentSpec spec;
  spec.set_name(kComponentName);
  auto chan0_spec = spec.add_fixed_feature();
  chan0_spec->set_size(2);
  auto chan1_spec = spec.add_fixed_feature();
  chan1_spec->set_size(1);
  EXPECT_CALL(*mock_session, Spec(kComponentName))
      .WillOnce(testing::ReturnRef(spec));

  EXPECT_CALL(*mock_session, BatchSize(kComponentName))
      .WillOnce(Return(kNumItems));

  const std::vector<int> feature_step_1({0, 1, 2, 1, 2, 2, 1, 0, 1, 0});
  const std::vector<int> feature_index_1({0, 0, 0, 0, 0, 1, 1, 1, 1, 1});
  const std::vector<int> feature_ids_1({5, 6, 3, 5, 7, 5, 6, 3, 5, 7});
  const std::vector<float> feature_weights_1(
      {1.0, 0.7, 0.1, 0.5, 1.0, 10, 7, 1, 5, 10});

  const std::vector<int> feature_step_2({0, 1, 2, 1, 2});
  const std::vector<int> feature_index_2({0, 0, 0, 0, 0});
  const std::vector<int> feature_ids_2({5, 6, 3, 5, 7});
  const std::vector<float> feature_weights_2({1.0, 0.7, 0.1, 0.5, 1.0});

  const std::vector<std::vector<int>> feature_steps_by_channel(
      {feature_step_1, feature_step_2});
  const std::vector<std::vector<int>> feature_index_by_channel(
      {feature_index_1, feature_index_2});
  const std::vector<std::vector<int>> feature_ids_by_channel(
      {feature_ids_1, feature_ids_2});
  const std::vector<std::vector<float>> feature_weights_by_channel(
      {feature_weights_1, feature_weights_2});

  // This function takes the allocator functions passed into GetBulkFF, uses
  // them to allocate a tensor, then fills that tensor based on channel.
  auto assigner_function = [=](string, const BulkFeatureExtractor &extractor) {
    constexpr int kNumElements = 3;
    constexpr int kNumSteps = 3;
    for (int i = 0; i < kNumChannels; ++i) {
      auto feature_step = feature_steps_by_channel.at(i);
      auto feature_index = feature_index_by_channel.at(i);
      auto feature_ids = feature_ids_by_channel.at(i);
      auto feature_weights = feature_weights_by_channel.at(i);

      // Allocate a new tensor set for every channel.
      int32 *indices =
          extractor.AllocateIndexMemory(i, kNumElements * feature_step.size());
      int64 *ids =
          extractor.AllocateIdMemory(i, kNumElements * feature_step.size());
      float *weights =
          extractor.AllocateWeightMemory(i, kNumElements * feature_step.size());

      // Fill the tensor.
      int array_index = 0;

      for (int element = 0; element < kNumElements; ++element) {
        for (int feature = 0; feature < feature_step.size(); ++feature) {
          indices[array_index] = extractor.GetIndex(
              kNumSteps, kNumElements, feature_index[feature], element,
              feature_step[feature]);
          ids[array_index] = feature_ids[feature];
          weights[array_index] = feature_weights[feature];
          ++array_index;
        }
      }
    }
    return kNumSteps;
  };

  EXPECT_CALL(*mock_session, BulkGetInputFeatures(kComponentName, _))
      .WillOnce(testing::Invoke(assigner_function));

  // Embedding matrices as additional inputs.
  // For channel 0, the embeddings are [id, 0].
  // For channel 1, the embeddings are [0, id].
  vector<float> embedding_matrix_a;
  vector<float> embedding_matrix_b;
  for (int id = 0; id < kNumIds; ++id) {
    embedding_matrix_a.push_back(id);
    embedding_matrix_a.push_back(0);
    embedding_matrix_b.push_back(0);
    embedding_matrix_b.push_back(id);
  }
  AddInputFromArray<float>(TensorShape({8, 2}), embedding_matrix_a);
  AddInputFromArray<float>(TensorShape({8, 2}), embedding_matrix_b);

  // Run the kernel.
  TF_EXPECT_OK(RunOpKernelWithContext());

  // Validate the outputs.
  // In this case we should have, for every item, these three steps:
  const vector<vector<float>> expected_embeddings = {{5.0, 0, 73, 0, 0, 5.0},
                                                     {6.7, 0, 67, 0, 0, 6.7},
                                                     {7.3, 0, 50, 0, 0, 7.3}};
  EXPECT_EQ(kPaddedNumSteps * kPaddedBatchSize,
            GetOutput(1)->shape().dim_size(0));

  constexpr int kNumFeatures = 3;
  EXPECT_EQ(kNumFeatures * kEmbeddingSize, GetOutput(1)->shape().dim_size(1));
  for (int item = 0; item < kNumItems; ++item) {
    for (int step = 0; step < kNumSteps; ++step) {
      for (int col = 0; col < kNumChannels * kEmbeddingSize; ++col) {
        const int row = item * kPaddedNumSteps + step;
        EXPECT_EQ(expected_embeddings[step][col],
                  GetOutput(1)->matrix<float>()(row, col))
            << "step: " << step << ", row: " << row << ", col: " << col;
      }
    }
  }

  EXPECT_EQ(kNumSteps, GetOutput(2)->scalar<int32>()());
}

TEST_F(DragnnBulkOpKernelsTest, BulkAdvanceFromOracle) {
  // Create and initialize the kernel under test.
  TF_ASSERT_OK(
      NodeDefBuilder("BulkAdvanceFromOracle", "BulkAdvanceFromOracle")
          .Attr("component", kComponentName)
          .Input(FakeInput(DT_STRING))  // The handle for the ComputeSession.
          .Finalize(node_def()));
  MockComputeSession *mock_session = GetMockSession();
  EXPECT_CALL(*mock_session, IsTerminal(kComponentName))
      .WillOnce(Return(false))
      .WillOnce(Return(false))
      .WillOnce(Return(false))
      .WillOnce(Return(true));
  EXPECT_CALL(*mock_session, AdvanceFromOracle(kComponentName))
      .Times(kNumSteps);
  const vector<vector<vector<int32>>> gold = {
      {{1}, {1}, {1}}, {{2}, {2}, {2}}, {{3}, {3}, {3}},
  };
  EXPECT_CALL(*mock_session, EmitOracleLabels(kComponentName))
      .WillOnce(Return(gold[0]))
      .WillOnce(Return(gold[1]))
      .WillOnce(Return(gold[2]));
  EXPECT_CALL(*mock_session, BeamSize(kComponentName)).WillOnce(Return(1));
  EXPECT_CALL(*mock_session, BatchSize(kComponentName))
      .WillOnce(Return(kNumItems));

  // Run the kernel.
  TF_EXPECT_OK(RunOpKernelWithContext());

  // Validate the outputs.
  // For every item we should have:
  const vector<int32> expected_gold = {1, 2, 3};
  EXPECT_EQ(kNumSteps * kNumItems, GetOutput(1)->NumElements());
  for (int item = 0; item < kNumItems; ++item) {
    for (int step = 0; step < kNumSteps; ++step) {
      EXPECT_EQ(expected_gold[step],
                GetOutput(1)->vec<int32>()(step + item * kNumSteps));
    }
  }
}

string ArrayToString(const float *array, const int size) {
  string str = "[ ";
  for (int i = 0; i < size; ++i) {
    str += tensorflow::strings::Printf("%.1f ", array[i]);
  }
  return str + "]";
}

MATCHER(CheckScoresAreConsecutiveIntegersDivTen, "") {
  const int size =
      DragnnBulkOpKernelsTest::kNumItems * DragnnBulkOpKernelsTest::kNumActions;
  for (int i(0), score(arg[0] * 10); i < size; ++i, ++score) {
    EXPECT_NEAR(score / 10.0f, arg[i], 1e-4)
        << "i: " << i << ", scores: " << ArrayToString(arg, size);
  }
  return true;
}

TEST_F(DragnnBulkOpKernelsTest, BulkAdvanceFromPrediction) {
  // Create and initialize the kernel under test.
  TF_ASSERT_OK(
      NodeDefBuilder("BulkAdvanceFromPrediction", "BulkAdvanceFromPrediction")
          .Attr("component", kComponentName)
          .Input(FakeInput(DT_STRING))  // The handle for the ComputeSession.
          .Input(FakeInput(DT_FLOAT))   // Prediction scores for advancing.
          .Finalize(node_def()));
  MockComputeSession *mock_session = GetMockSession();

  // Creates an input tensor such that each step will see a list of consecutive
  // integers divided by 10 as scores.
  vector<float> scores(kNumItems * kNumSteps * kNumActions);
  for (int step(0), cnt(0); step < kNumSteps; ++step) {
    for (int item = 0; item < kNumItems; ++item) {
      for (int action = 0; action < kNumActions; ++action, ++cnt) {
        scores[action + kNumActions * (step + item * kNumSteps)] = cnt / 10.0f;
      }
    }
  }
  AddInputFromArray<float>(TensorShape({kNumItems * kNumSteps, kNumActions}),
                           scores);

  EXPECT_CALL(*mock_session, BeamSize(kComponentName)).WillOnce(Return(1));
  EXPECT_CALL(*mock_session, BatchSize(kComponentName))
      .WillOnce(Return(kNumItems));
  EXPECT_CALL(*mock_session, IsTerminal(kComponentName))
      .Times(kNumSteps)
      .WillRepeatedly(Return(false));
  EXPECT_CALL(*mock_session,
              AdvanceFromPrediction(kComponentName,
                                    CheckScoresAreConsecutiveIntegersDivTen(),
                                    kNumItems * kNumActions))
      .Times(kNumSteps);

  // Run the kernel.
  TF_EXPECT_OK(RunOpKernelWithContext());
}

}  // namespace dragnn
}  // namespace syntaxnet
