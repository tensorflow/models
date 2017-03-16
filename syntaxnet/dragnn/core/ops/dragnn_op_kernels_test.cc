#include <functional>
#include <memory>
#include <vector>

#include "dragnn/core/compute_session.h"
#include "dragnn/core/compute_session_pool.h"
#include "dragnn/core/resource_container.h"
#include "dragnn/core/test/generic.h"
#include "dragnn/core/test/mock_compute_session.h"

#include <gmock/gmock.h>

#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/framework/control_flow.h"
#include "tensorflow/core/framework/fake_input.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/kernels/ops_testutil.h"
#include "tensorflow/core/kernels/ops_util.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"

namespace syntaxnet {
namespace dragnn {

using tensorflow::AllocatorAttributes;
using tensorflow::checkpoint::TensorSliceReaderCacheWrapper;
using tensorflow::DT_BOOL;
using tensorflow::DT_FLOAT;
using tensorflow::DT_STRING;
using tensorflow::DT_INT32;
using tensorflow::FrameAndIter;
using tensorflow::DataType;
using tensorflow::NodeDefBuilder;
using tensorflow::OpKernelContext;
using tensorflow::ResourceMgr;
using tensorflow::ScopedStepContainer;
using tensorflow::Status;
using tensorflow::test::SetOutputAttrs;
using tensorflow::TensorShape;

using testing::_;
using testing::ElementsAreArray;
using testing::Invoke;
using testing::Pointwise;
using testing::Return;

typedef ResourceContainer<ComputeSession> ComputeSessionResource;
typedef ResourceContainer<ComputeSessionPool> ComputeSessionPoolResource;

class DragnnOpKernelsTest : public tensorflow::OpsTestBase {
 public:
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

  // This needs to maintain its existence throughout the compute call.
  std::vector<AllocatorAttributes> attrs_;
};

// Helper function to build LinkFeatures.
LinkFeatures MakeFeatures(int batch_index, int beam_index, int step) {
  LinkFeatures features;
  features.set_batch_idx(batch_index);
  features.set_beam_idx(beam_index);
  features.set_step_idx(step);
  return features;
}

// The GetSessionOp should
// 1. create a ComputeSessionPool resource and store it in the ResourceMgr,
// 2. create a ComputeSession resource and store it in the ResourceMgr,
// 3. return the container and id strings in its output.
TEST_F(DragnnOpKernelsTest, GetSessionOpTest) {
  // Create a MasterSpec and GridPoint string to pass into the attrs for this
  // op.
  MasterSpec spec;
  spec.set_debug_tracing(true);
  string master_spec_str;
  spec.SerializeToString(&master_spec_str);

  GridPoint hyperparams;
  string hyperparams_str;
  hyperparams.SerializeToString(&hyperparams_str);

  // Create and initialize the kernel under test.
  TF_ASSERT_OK(
      NodeDefBuilder("get_session", "GetSession")
          .Attr("master_spec", master_spec_str)
          .Attr("grid_point", hyperparams_str)
          .Input(FakeInput(DT_STRING))  // The handle for the ComputeSession.
          .Finalize(node_def()));
  TF_ASSERT_OK(InitOp());

  // Set the input data.
  const string container_string = "container_str";
  AddInputFromList<string>(TensorShape({1}), {container_string});

  // Reset the test context to ensure it's clean.
  ResetOpKernelContext();

  // Run the kernel.
  TF_EXPECT_OK(RunOpKernelWithContext());

  // Expect that the 0th output contains two strings, and that the ResourceMgr
  // contains a ComputeSessionResource associated with those two strings.
  const string container_str = GetOutput(0)->vec<string>()(0);
  const string id_str = GetOutput(0)->vec<string>()(1);
  VLOG(2) << "container: " << container_str << " id: " << id_str;

  // The first compute session should have id "0".
  EXPECT_EQ("0", id_str);
  ComputeSessionResource *session_resource;
  TF_EXPECT_OK(resource_mgr()->Lookup<ComputeSessionResource>(
      container_str, id_str, &session_resource));

  // Expect that the ResourceMgr also contains a ComputeSessionPoolResource.
  const string pool_id_str = "pool";
  ComputeSessionPoolResource *pool_resource;
  TF_EXPECT_OK(resource_mgr()->Lookup<ComputeSessionPoolResource>(
      container_str, pool_id_str, &pool_resource));

  // Unref the managed resources so they get destroyed properly.
  session_resource->Unref();
  pool_resource->Unref();
}

// The GetSessionOp should take a session stored in the resource manager
// and return it to the ComputeSessionPool.
TEST_F(DragnnOpKernelsTest, ReleaseSessionOpTest) {
  // Create and initialize the kernel under test.
  TF_ASSERT_OK(
      NodeDefBuilder("release_session", "ReleaseSession")
          .Input(FakeInput(DT_STRING))  // The handle for the ComputeSession.
          .Finalize(node_def()));
  TF_ASSERT_OK(InitOp());

  // Set the input data.
  const string container_string = "container_str";
  const string id_string = "id_str";
  AddInputFromList<string>(TensorShape({2}), {container_string, id_string});

  // Reset the test context to ensure it's clean.
  ResetOpKernelContext();

  // Create a ComputeSessionPool.
  MasterSpec spec;
  GridPoint hyperparams;
  std::unique_ptr<ComputeSessionPool> pool(
      new ComputeSessionPool(spec, hyperparams));

  // Get an unowned pointer to the ComputeSessionPool before moving
  // the pool to the resource manager.
  ComputeSessionPool *pool_ptr = pool.get();
  TF_ASSERT_OK(resource_mgr()->Create<ComputeSessionPoolResource>(
      container_string, "pool",
      new ComputeSessionPoolResource(std::move(pool))));

  // Create a ComputeSession and move it to the resource manager.
  TF_ASSERT_OK(resource_mgr()->Create<ComputeSessionResource>(
      container_string, id_string,
      new ComputeSessionResource(pool_ptr->GetSession())));

  // At this point, the pool should report that it has one outstanding session.
  EXPECT_EQ(1, pool_ptr->num_outstanding_sessions());

  // Run the kernel.
  TF_EXPECT_OK(RunOpKernelWithContext());

  // At this point, the pool should report that it has no outstanding sessions.
  EXPECT_EQ(0, pool_ptr->num_outstanding_sessions());

  // The resource manager should no longer contain the session object.
  ComputeSessionResource *null_resource = nullptr;
  auto result = resource_mgr()->Lookup<ComputeSessionResource>(
      container_string, id_string, &null_resource);
  EXPECT_NE(Status::OK(), result);
  EXPECT_EQ(null_resource, nullptr);
}

// The AdvanceFromOracle op should call AdvanceFromOracle on the specified
// component name.
TEST_F(DragnnOpKernelsTest, AdvanceFromOracleOpTest) {
  // Create and initialize the kernel under test.
  const string component_name = "TESTING_COMPONENT_NAME";
  TF_ASSERT_OK(
      NodeDefBuilder("advance_from_oracle", "AdvanceFromOracle")
          .Attr("component", component_name)
          .Input(FakeInput(DT_STRING))  // The handle for the ComputeSession.
          .Finalize(node_def()));
  TF_ASSERT_OK(InitOp());

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
  TF_ASSERT_OK(resource_mgr()->Create<ComputeSessionResource>(
      container_string, id_string,
      new ComputeSessionResource(std::move(mock_session))));

  // Set expectations on the mock session.
  EXPECT_CALL(*mock_session_ptr, AdvanceFromOracle(component_name));

  // Run the kernel.
  TF_EXPECT_OK(RunOpKernelWithContext());
}

// The AdvanceFromPredicton op should call AdvanceFromPrediction on the
// specified component with the passed scores.
TEST_F(DragnnOpKernelsTest, AdvanceFromPredictionOpTest) {
  // Create and initialize the kernel under test.
  const string component_name = "TESTING_COMPONENT_NAME";
  TF_ASSERT_OK(
      NodeDefBuilder("advance_from_prediction", "AdvanceFromPrediction")
          .Attr("component", component_name)
          .Input(FakeInput(DT_STRING))  // The handle for the ComputeSession.
          .Input(FakeInput(DT_FLOAT))   // The prediction tensor.
          .Finalize(node_def()));
  TF_ASSERT_OK(InitOp());

  // Set the input data.
  const string container_string = "container_str";
  const string id_string = "id_str";
  AddInputFromList<string>(TensorShape({2}), {container_string, id_string});
  const std::vector<float> weights = {1.1, 2.2, 3.3, 4.4};
  AddInputFromArray<float>(TensorShape({2, 2}), weights);

  // Reset the test context to ensure it's clean.
  ResetOpKernelContext();

  // Create a MockComputeSession and set expectations.
  std::unique_ptr<MockComputeSession> mock_session(new MockComputeSession());
  MockComputeSession *mock_session_ptr = mock_session.get();

  // Wrap the ComputeSessionResource and put it into the resource manager.
  TF_ASSERT_OK(resource_mgr()->Create<ComputeSessionResource>(
      container_string, id_string,
      new ComputeSessionResource(std::move(mock_session))));

  // Set expectations on the mock session.
  auto validator_function = [weights](const string &component_name,
                                      const float score_matrix[],
                                      int score_matrix_length) {
    EXPECT_EQ(weights.size(), score_matrix_length);
    for (int i = 0; i < weights.size(); ++i) {
      EXPECT_EQ(weights[i], score_matrix[i]);
    }
  };
  EXPECT_CALL(*mock_session_ptr, AdvanceFromPrediction(component_name, _, _))
      .WillOnce(Invoke(validator_function));

  // Run the kernel.
  TF_EXPECT_OK(RunOpKernelWithContext());
}

// The ExtractFixedFeatures op should return a set of fixed feature vectors
// as described below.
TEST_F(DragnnOpKernelsTest, ExtractFixedFeaturesOpTest) {
  // Create and initialize the kernel under test.
  const string component_name = "TESTING_COMPONENT_NAME";
  constexpr int kChannelId = 78;
  TF_ASSERT_OK(
      NodeDefBuilder("advance_from_prediction", "ExtractFixedFeatures")
          .Attr("component", component_name)
          .Attr("channel_id", kChannelId)
          .Input(FakeInput(DT_STRING))  // The handle for the ComputeSession.
          .Finalize(node_def()));
  TF_ASSERT_OK(InitOp());

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
  TF_ASSERT_OK(resource_mgr()->Create<ComputeSessionResource>(
      container_string, id_string,
      new ComputeSessionResource(std::move(mock_session))));

  // If we have 3 features, for a given channel, we might have:
  //   feature a: (5, 1)
  //   feature b: (5, 0.5), (6, 0.7)
  //   feature c: (3, 0.1), (7, [empty]) <- Empty weights are equivalent to 1.0.
  // In this case:
  //   indices should look like  [0  , 1  , 1  , 2  , 2  ]
  //   ids should be             [5  , 5  , 6  , 3  , 7  ]
  //   weights should be         [1.0, 0.5, 0.7, 0.1, 1.0]
  const std::vector<int32> expected_indices({0, 1, 1, 2, 2});
  const std::vector<int64> expected_ids({5, 5, 6, 3, 7});
  const std::vector<float> expected_weights({1.0, 0.5, 0.7, 0.1, 1.0});

  auto assigner_function =
      [=](string, std::function<int32 *(int)> indices_allocator,
          std::function<int64 *(int)> ids_allocator,
          std::function<float *(int)> weights_allocator, int) {
        constexpr int kFeatureCount = 5;
        int32 *indices = indices_allocator(kFeatureCount);
        int64 *ids = ids_allocator(kFeatureCount);
        float *weights = weights_allocator(kFeatureCount);
        for (int i = 0; i < kFeatureCount; ++i) {
          indices[i] = expected_indices[i];
          ids[i] = expected_ids[i];
          weights[i] = expected_weights[i];
        }
        return kFeatureCount;
      };

  EXPECT_CALL(*mock_session_ptr,
              GetInputFeatures(component_name, _, _, _, kChannelId))
      .WillOnce(testing::Invoke(assigner_function));

  // Run the kernel.
  TF_EXPECT_OK(RunOpKernelWithContext());

  // Validate the outputs.
  EXPECT_EQ(expected_indices.size(), GetOutput(0)->NumElements());
  for (int i = 0; i < expected_indices.size(); ++i) {
    EXPECT_EQ(expected_indices[i], GetOutput(0)->vec<int32>()(i));
  }
  EXPECT_EQ(expected_ids.size(), GetOutput(1)->NumElements());
  for (int i = 0; i < expected_ids.size(); ++i) {
    EXPECT_EQ(expected_ids[i], GetOutput(1)->vec<int64>()(i));
  }
  EXPECT_EQ(expected_weights.size(), GetOutput(2)->NumElements());
  for (int i = 0; i < expected_weights.size(); ++i) {
    EXPECT_EQ(expected_weights[i], GetOutput(2)->vec<float>()(i));
  }
}

// The ExtractLinkFeatures op should return a set of linked feature vectors
// as described below.
TEST_F(DragnnOpKernelsTest, ExtractLinkFeaturesOpTest) {
  // TODO(googleuser): Is a 2-vector output the correct way to do this?
  // Why reshape instead of passing [batch, beam, index] or just
  // [batch,index] ?
  // Create and initialize the kernel under test.
  const string component_name = "TESTING_COMPONENT_NAME";
  constexpr int kChannelId = 3421;
  TF_ASSERT_OK(
      NodeDefBuilder("extract_link_features", "ExtractLinkFeatures")
          .Attr("component", component_name)
          .Attr("channel_id", kChannelId)
          .Input(FakeInput(DT_STRING))  // The handle for the ComputeSession.
          .Finalize(node_def()));
  TF_ASSERT_OK(InitOp());

  // Set the input data.
  const string container_string = "container_str";
  const string id_string = "id_str";
  AddInputFromList<string>(TensorShape({2}), {container_string, id_string});

  // Reset the test context to ensure it's clean.
  ResetOpKernelContext();

  std::unique_ptr<MockComputeSession> mock_session(new MockComputeSession());
  MockComputeSession *mock_session_ptr = mock_session.get();

  // This op will return link features in two flat arrays using batch-major
  // ordering. So, if we have a batch of 2 and a beam of 3, with data as follows
  // (note that the features are {batch,beam,step} and [] is 'empty')
  // batch 1 features: {{02,03,[]},{01,00,04},{08,06,01}}
  // batch 2 features: {{12,13,14},{11,12,-1},{18,16,20}}
  //
  // and a **source component** beam size of 5 should result in output tensors:
  // step_idx  (tensor 0): {-1,  4,  1, 14, -1, 20}
  // array_idx (tensor 1): { 0,  5, 46, 73,  0, 106}
  // (0 [step=-1]),(5=1*5+0),(46=8*5+6),(73=12*5+13),(0 [step=-1]),(96=18*5+16)
  constexpr int kSourceComponentBeamSize = 5;

  std::vector<LinkFeatures> features;
  features.push_back(MakeFeatures(2, 3, -1));
  features.back().clear_step_idx();  // step_idx is now empty.
  features.push_back(MakeFeatures(1, 0, 4));
  features.push_back(MakeFeatures(8, 6, 1));
  features.push_back(MakeFeatures(12, 13, 14));
  features.push_back(MakeFeatures(11, 12, -1));
  features.push_back(MakeFeatures(18, 16, 20));

  const std::vector<int> expected_step_idx({-1, 4, 1, 14, -1, 20});
  const std::vector<int> expected_array_idx({0, 5, 46, 73, 0, 106});

  EXPECT_CALL(*mock_session_ptr,
              SourceComponentBeamSize(component_name, kChannelId))
      .WillRepeatedly(Return(kSourceComponentBeamSize));
  EXPECT_CALL(*mock_session_ptr,
              GetTranslatedLinkFeatures(component_name, kChannelId))
      .WillOnce(Return(features));

  // Wrap the ComputeSessionResource and put it into the resource manager.
  TF_ASSERT_OK(resource_mgr()->Create<ComputeSessionResource>(
      container_string, id_string,
      new ComputeSessionResource(std::move(mock_session))));

  // Run the kernel.
  TF_EXPECT_OK(RunOpKernelWithContext());

  // Validate the outputs.
  EXPECT_EQ(expected_step_idx.size(), GetOutput(0)->NumElements());
  for (int i = 0; i < expected_step_idx.size(); ++i) {
    EXPECT_EQ(expected_step_idx[i], GetOutput(0)->vec<int32>()(i));
  }
  EXPECT_EQ(expected_array_idx.size(), GetOutput(1)->NumElements());
  for (int i = 0; i < expected_array_idx.size(); ++i) {
    EXPECT_EQ(expected_array_idx[i], GetOutput(1)->vec<int32>()(i));
  }
}

// The EmitOracleLabels op should return a set of oracle labels for all
// elements in all beams in all batches.
TEST_F(DragnnOpKernelsTest, EmitOracleLabelsOpTest) {
  // Create and initialize the kernel under test.
  const string component_name = "TESTING_COMPONENT_NAME";
  TF_ASSERT_OK(
      NodeDefBuilder("emit_oracle_labels", "EmitOracleLabels")
          .Attr("component", component_name)
          .Input(FakeInput(DT_STRING))  // The handle for the ComputeSession.
          .Finalize(node_def()));
  TF_ASSERT_OK(InitOp());

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
  TF_ASSERT_OK(resource_mgr()->Create<ComputeSessionResource>(
      container_string, id_string,
      new ComputeSessionResource(std::move(mock_session))));

  // The op should request the batch and beam size, then request the oracle
  // labels. They should be returned batch major, so:
  // batch 1 oracle labels: {1, 3, 5, 7}
  // batch 2 oracle labels: {2, 4, 6, 8}
  // should result in an output tensor as follows:
  // {1, 3, 5, 7, 2, 4, 6, 8}

  constexpr int kBatchSize = 2;
  constexpr int kBeamSize = 4;
  const std::vector<std::vector<int>> oracle_labels(
      {{1, 3, 5, 7}, {2, 4, 6, 8}});

  EXPECT_CALL(*mock_session_ptr, BatchSize(component_name))
      .WillRepeatedly(Return(kBatchSize));
  EXPECT_CALL(*mock_session_ptr, BeamSize(component_name))
      .WillRepeatedly(Return(kBeamSize));
  EXPECT_CALL(*mock_session_ptr, EmitOracleLabels(component_name))
      .WillOnce(Return(oracle_labels));

  const std::vector<int> expected_labels({1, 3, 5, 7, 2, 4, 6, 8});

  // Run the kernel.
  TF_EXPECT_OK(RunOpKernelWithContext());

  // Validate the outputs.
  EXPECT_EQ(expected_labels.size(), GetOutput(0)->NumElements());
  for (int i = 0; i < expected_labels.size(); ++i) {
    EXPECT_EQ(expected_labels[i], GetOutput(0)->vec<int32>()(i));
  }
}

// The EmitAllFinal op should return the result of IsTerminal(component_name).
TEST_F(DragnnOpKernelsTest, EmitAllFinalOpTest) {
  // Create and initialize the kernel under test.
  const string component_name = "TESTING_COMPONENT_NAME";
  TF_ASSERT_OK(
      NodeDefBuilder("emit_all_final", "EmitAllFinal")
          .Attr("component", component_name)
          .Input(FakeInput(DT_STRING))  // The handle for the ComputeSession.
          .Finalize(node_def()));
  TF_ASSERT_OK(InitOp());

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
  TF_ASSERT_OK(resource_mgr()->Create<ComputeSessionResource>(
      container_string, id_string,
      new ComputeSessionResource(std::move(mock_session))));

  // Set up mocks.
  constexpr bool kIsTerminal = true;
  EXPECT_CALL(*mock_session_ptr, IsTerminal(component_name))
      .WillOnce(Return(kIsTerminal));

  // Run the kernel.
  TF_EXPECT_OK(RunOpKernelWithContext());

  // Validate the outputs.
  EXPECT_EQ(1, GetOutput(0)->NumElements());
  EXPECT_EQ(kIsTerminal, GetOutput(0)->vec<bool>()(0));
}

// The InitComponent op should initialize the given component with the given
// beam size.
// TODO(googleuser): Should we just store the beam size somewhere in the
// ComputeSession?
TEST_F(DragnnOpKernelsTest, InitComponentDataOpTest) {
  // Create and initialize the kernel under test.
  const string component_name = "TESTING_COMPONENT_NAME";
  TF_ASSERT_OK(
      NodeDefBuilder("init_component_data", "InitComponentData")
          .Attr("component", component_name)
          .Input(FakeInput(DT_STRING))  // The handle for the ComputeSession.
          .Input(FakeInput(DT_INT32))   // The beam size.
          .Finalize(node_def()));
  TF_ASSERT_OK(InitOp());

  // Set the input data.
  const string container_string = "container_str";
  const string id_string = "id_str";
  AddInputFromList<string>(TensorShape({2}), {container_string, id_string});
  constexpr int32 kBeamSize = 9001;
  AddInputFromList<int32>(TensorShape({1}), {kBeamSize});

  // Reset the test context to ensure it's clean.
  ResetOpKernelContext();

  // Create a MockComputeSession and set expectations.
  std::unique_ptr<MockComputeSession> mock_session(new MockComputeSession());
  MockComputeSession *mock_session_ptr = mock_session.get();

  // Wrap the ComputeSessionResource and put it into the resource manager.
  TF_ASSERT_OK(resource_mgr()->Create<ComputeSessionResource>(
      container_string, id_string,
      new ComputeSessionResource(std::move(mock_session))));

  // Set up mocks.
  EXPECT_CALL(*mock_session_ptr,
              InitializeComponentData(component_name, kBeamSize));

  // Run the kernel.
  TF_EXPECT_OK(RunOpKernelWithContext());

  // Output should be the input handle.
  EXPECT_EQ(container_string, GetOutput(0)->vec<string>()(0));
  EXPECT_EQ(id_string, GetOutput(0)->vec<string>()(1));
}

// The BatchSize op should call BatchSize on the ComputeSession with the given
// component as argument.
TEST_F(DragnnOpKernelsTest, BatchSizeOpTest) {
  // Create and initialize the kernel under test.
  const string component_name = "TESTING_COMPONENT_NAME";
  TF_ASSERT_OK(
      NodeDefBuilder("batch_size", "BatchSize")
          .Attr("component", component_name)
          .Input(FakeInput(DT_STRING))  // The handle for the ComputeSession.
          .Finalize(node_def()));
  TF_ASSERT_OK(InitOp());

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
  TF_ASSERT_OK(resource_mgr()->Create<ComputeSessionResource>(
      container_string, id_string,
      new ComputeSessionResource(std::move(mock_session))));

  // Set up mocks.
  constexpr int kBatchSize = 8;
  EXPECT_CALL(*mock_session_ptr, BatchSize(component_name))
      .WillOnce(Return(kBatchSize));

  // Run the kernel.
  TF_EXPECT_OK(RunOpKernelWithContext());

  // Output should be the batch size.
  EXPECT_EQ(kBatchSize, GetOutput(0)->scalar<int>()());
}

// The AttachDataReader op should push the given vector of strings into the
// session.
TEST_F(DragnnOpKernelsTest, AttachDataReaderOpTest) {
  // Create and initialize the kernel under test.
  const string component_name = "TESTING_COMPONENT_NAME";
  TF_ASSERT_OK(
      NodeDefBuilder("attach_data_reader", "AttachDataReader")
          .Input(FakeInput(DT_STRING))  // The handle for the ComputeSession.
          .Input(FakeInput(DT_STRING))  // The data to pass to the session.
          .Finalize(node_def()));
  TF_ASSERT_OK(InitOp());

  // Set the input data.
  const string container_string = "container_str";
  const string id_string = "id_str";
  AddInputFromList<string>(TensorShape({2}), {container_string, id_string});

  const std::vector<string> data(
      {"one string", "two string", "red string", "blue string"});
  AddInputFromArray<string>(TensorShape({4}), data);

  // Reset the test context to ensure it's clean.
  ResetOpKernelContext();

  // Create a MockComputeSession and set expectations.
  std::unique_ptr<MockComputeSession> mock_session(new MockComputeSession());
  MockComputeSession *mock_session_ptr = mock_session.get();

  // Wrap the ComputeSessionResource and put it into the resource manager.
  TF_ASSERT_OK(resource_mgr()->Create<ComputeSessionResource>(
      container_string, id_string,
      new ComputeSessionResource(std::move(mock_session))));

  // Set up mocks.
  EXPECT_CALL(*mock_session_ptr, SetInputData(ElementsAreArray(data)));

  // Run the kernel.
  TF_EXPECT_OK(RunOpKernelWithContext());
}

// The SetTracingOp should pass its argument through to the underlying
// ComputeSession.
TEST_F(DragnnOpKernelsTest, SetTracingOpTest) {
  // Create and initialize the kernel under test.
  const string component_name = "TESTING_COMPONENT_NAME";
  TF_ASSERT_OK(
      NodeDefBuilder("set_tracing", "SetTracing")
          .Input(FakeInput(DT_STRING))  // The handle for the ComputeSession.
          .Input(FakeInput(DT_BOOL))    // The boolean to set tracing to.
          .Finalize(node_def()));
  TF_ASSERT_OK(InitOp());

  // Set the input data.
  const string container_string = "container_str";
  const string id_string = "id_str";
  AddInputFromList<string>(TensorShape({2}), {container_string, id_string});
  constexpr bool kSetTracing = true;
  AddInputFromList<bool>(TensorShape({1}), {kSetTracing});

  // Reset the test context to ensure it's clean.
  ResetOpKernelContext();

  // Create a MockComputeSession and set expectations.
  std::unique_ptr<MockComputeSession> mock_session(new MockComputeSession());
  MockComputeSession *mock_session_ptr = mock_session.get();

  // Wrap the ComputeSessionResource and put it into the resource manager.
  TF_ASSERT_OK(resource_mgr()->Create<ComputeSessionResource>(
      container_string, id_string,
      new ComputeSessionResource(std::move(mock_session))));

  // Set expectations on the mock session.
  EXPECT_CALL(*mock_session_ptr, SetTracing(kSetTracing));

  // Run the kernel.
  TF_EXPECT_OK(RunOpKernelWithContext());
}

// The WriteAnnotations op should call FinalizeData on the current component.
TEST_F(DragnnOpKernelsTest, WriteAnnotationsOpTest) {
  // Create and initialize the kernel under test.
  const string component_name = "TESTING_COMPONENT_NAME";
  TF_ASSERT_OK(
      NodeDefBuilder("write_annotations", "WriteAnnotations")
          .Attr("component", component_name)
          .Input(FakeInput(DT_STRING))  // The handle for the ComputeSession.
          .Finalize(node_def()));
  TF_ASSERT_OK(InitOp());

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
  TF_ASSERT_OK(resource_mgr()->Create<ComputeSessionResource>(
      container_string, id_string,
      new ComputeSessionResource(std::move(mock_session))));

  // Set expectations on the mock session.
  EXPECT_CALL(*mock_session_ptr, FinalizeData(component_name));

  // Run the kernel.
  TF_EXPECT_OK(RunOpKernelWithContext());
}

// The EmitAnnotations op should return a vector of annotated strings as
// described below.
TEST_F(DragnnOpKernelsTest, EmitAnnotationsOpTest) {
  // Create and initialize the kernel under test.
  const string component_name = "TESTING_COMPONENT_NAME";
  TF_ASSERT_OK(
      NodeDefBuilder("emit_annotations", "EmitAnnotations")
          .Attr("component", component_name)
          .Input(FakeInput(DT_STRING))  // The handle for the ComputeSession.
          .Finalize(node_def()));
  TF_ASSERT_OK(InitOp());

  // Set the input data.
  const string container_string = "container_str";
  const string id_string = "id_str";
  AddInputFromList<string>(TensorShape({2}), {container_string, id_string});

  // Reset the test context to ensure it's clean.
  ResetOpKernelContext();

  std::unique_ptr<MockComputeSession> mock_session(new MockComputeSession());
  MockComputeSession *mock_session_ptr = mock_session.get();

  constexpr int kBatchSize = 2;
  std::vector<string> predictions({"one", "two"});

  EXPECT_CALL(*mock_session_ptr, BatchSize(component_name))
      .WillRepeatedly(Return(kBatchSize));
  EXPECT_CALL(*mock_session_ptr, GetSerializedPredictions())
      .WillOnce(Return(predictions));

  // The output vector is batch_size.
  const std::vector<string> expected_output({"one", "two"});

  // Wrap the ComputeSessionResource and put it into the resource manager.
  TF_ASSERT_OK(resource_mgr()->Create<ComputeSessionResource>(
      container_string, id_string,
      new ComputeSessionResource(std::move(mock_session))));

  // Run the kernel.
  TF_EXPECT_OK(RunOpKernelWithContext());

  // Validate the outputs.
  EXPECT_EQ(expected_output.size(), GetOutput(0)->NumElements());
  for (int i = 0; i < expected_output.size(); ++i) {
    EXPECT_EQ(expected_output[i], GetOutput(0)->vec<string>()(i));
  }
}

// The GetComponentTrace op should return a vector of serialized trace protos.
TEST_F(DragnnOpKernelsTest, GetComponentTraceOpTest) {
  // Create and initialize the kernel under test.
  const string component_name = "TESTING_COMPONENT_NAME";
  TF_ASSERT_OK(
      NodeDefBuilder("get_component_trace", "GetComponentTrace")
          .Attr("component", component_name)
          .Input(FakeInput(DT_STRING))  // The handle for the ComputeSession.
          .Finalize(node_def()));
  TF_ASSERT_OK(InitOp());

  // Set the input data.
  const string container_string = "container_str";
  const string id_string = "id_str";
  AddInputFromList<string>(TensorShape({2}), {container_string, id_string});

  // Reset the test context to ensure it's clean.
  ResetOpKernelContext();

  std::unique_ptr<MockComputeSession> mock_session(new MockComputeSession());
  MockComputeSession *mock_session_ptr = mock_session.get();

  // This op will request a set of MasterTraces from GetTraceProtos(), then
  // return them.

  MasterTrace trace;
  auto component_trace = trace.add_component_trace();
  component_trace->set_name("arbitrary_component_name_for_html");
  auto component_trace_2 = trace.add_component_trace();
  component_trace_2->set_name("arbitrary_component_name_2_for_html");
  const std::vector<MasterTrace> master_traces({trace});

  EXPECT_CALL(*mock_session_ptr, GetTraceProtos())
      .WillOnce(Return(master_traces));

  // Wrap the ComputeSessionResource and put it into the resource manager.
  TF_ASSERT_OK(resource_mgr()->Create<ComputeSessionResource>(
      container_string, id_string,
      new ComputeSessionResource(std::move(mock_session))));

  // Run the kernel.
  TF_EXPECT_OK(RunOpKernelWithContext());

  // Validate the outputs.
  EXPECT_EQ(master_traces.size(), GetOutput(0)->NumElements());
  for (int i = 0; i < master_traces.size(); ++i) {
    string expected;
    master_traces.at(i).SerializeToString(&expected);
    EXPECT_EQ(expected, GetOutput(0)->vec<string>()(i));
  }
}

}  // namespace dragnn
}  // namespace syntaxnet
