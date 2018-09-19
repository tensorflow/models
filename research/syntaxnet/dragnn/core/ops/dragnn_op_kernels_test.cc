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

#include <functional>
#include <memory>
#include <vector>

#include "dragnn/core/component_registry.h"
#include "dragnn/core/compute_session.h"
#include "dragnn/core/compute_session_pool.h"
#include "dragnn/core/resource_container.h"
#include "dragnn/core/test/generic.h"
#include "dragnn/core/test/mock_compute_session.h"
#include "dragnn/core/util/label.h"

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
using tensorflow::DT_BOOL;
using tensorflow::DT_FLOAT;
using tensorflow::DT_INT32;
using tensorflow::DT_STRING;
using tensorflow::DataType;
using tensorflow::FrameAndIter;
using tensorflow::NodeDefBuilder;
using tensorflow::OpKernelContext;
using tensorflow::ResourceMgr;
using tensorflow::ScopedStepContainer;
using tensorflow::Status;
using tensorflow::TensorShape;
using tensorflow::checkpoint::TensorSliceReaderCacheWrapper;
using tensorflow::test::SetOutputAttrs;

using testing::ElementsAreArray;
using testing::Invoke;
using testing::Pointwise;
using testing::Return;
using testing::_;

typedef ResourceContainer<ComputeSession> ComputeSessionResource;
typedef ResourceContainer<ComputeSessionPool> ComputeSessionPoolResource;
typedef ResourceContainer<string> StringResource;

namespace {
const char kGlobalContainer[] = "__reserved_global_container";
const char kBasePathTag[] = "__reserved_asset_base_path";
const char kUnmanagedAssetDirectory[] = "assets.extra";
}  // namespace

// Define a test component to validate registered construction.
class TestComponent : public Component {
 public:
  TestComponent() {}
  void InitializeComponent(const ComponentSpec &spec) override {
    name_ = spec.name();
  }
  void InitializeData(
      const std::vector<std::vector<const TransitionState *>> &states,
      int max_beam_size, InputBatchCache *input_data) override {}
  void InitializeTracing() override {}
  void DisableTracing() override {}
  bool IsReady() const override { return true; }
  string Name() const override { return name_; }
  int BeamSize() const override { return 3; }
  int BatchSize() const override { return 1; }
  int StepsTaken(int batch_index) const override { return 0; }
  int GetBeamIndexAtStep(int step, int current_index,
                         int batch) const override {
    return 0;
  }
  int GetSourceBeamIndex(int current_index, int batch) const override {
    return 0;
  }
  bool AdvanceFromPrediction(const float *score_matrix, int num_items,
                             int num_actions) override {
    return true;
  }
  void AdvanceFromOracle() override {}
  bool IsTerminal() const override { return true; }
  std::function<int(int, int, int)> GetStepLookupFunction(
      const string &method) override {
    return nullptr;
  }
  std::vector<std::vector<const TransitionState *>> GetBeam() override {
    std::vector<std::vector<const TransitionState *>> states;
    return states;
  }
  int GetFixedFeatures(std::function<int32 *(int)> allocate_indices,
                       std::function<int64 *(int)> allocate_ids,
                       std::function<float *(int)> allocate_weights,
                       int channel_id) const override {
    return 0;
  }
  int BulkGetFixedFeatures(const BulkFeatureExtractor &extractor) override {
    return 0;
  }
  void BulkEmbedFixedFeatures(
      int batch_size_padding, int num_steps_padding, int output_array_size,
      const vector<const float *> &per_channel_embeddings,
      float *embedding_matrix) override {}
  void BulkEmbedDenseFixedFeatures(
      const vector<const float *> &per_channel_embeddings,
      float *embedding_output, int embedding_output_size,
      int *offset_array_output, int offset_array_size) override {}
  int BulkDenseFeatureSize() const override { return 0; }
  std::vector<LinkFeatures> GetRawLinkFeatures(int channel_id) const override {
    std::vector<LinkFeatures> ret;
    return ret;
  }
  std::vector<std::vector<std::vector<Label>>> GetOracleLabels()
      const override {
    std::vector<std::vector<std::vector<Label>>> ret;
    return ret;
  }
  void FinalizeData() override {}
  void ResetComponent() override {}

  std::vector<std::vector<ComponentTrace>> GetTraceProtos() const override {
    std::vector<std::vector<ComponentTrace>> ret;
    return ret;
  }
  void AddTranslatedLinkFeaturesToTrace(
      const std::vector<LinkFeatures> &features, int channel_id) override {}

  string name_;
};

REGISTER_DRAGNN_COMPONENT(TestComponent);

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

// The SetAssetDirectory op should
// 1. When given an asset path (foo/bar/baz/asset/thing), strip the path to
//    foo/bar/baz and add 'assets.extra' to it.
// 2. Store that path in the resource manager.
TEST_F(DragnnOpKernelsTest, SetAssetDirectoryTest) {
  // Create a MasterSpec and GridPoint string to pass into the attrs for this
  // op.
  const string new_asset_path = "new/directory/path/asset/master_spec";
  const string expected_asset_path =
      StrCat("new/directory/path/", kUnmanagedAssetDirectory);

  // Create and initialize the kernel under test.
  TF_ASSERT_OK(NodeDefBuilder("set_asset_directory", "SetAssetDirectory")
                   .Input(FakeInput(DT_STRING))  // The new asset path.
                   .Finalize(node_def()));
  TF_ASSERT_OK(InitOp());

  // Set the input data.
  AddInputFromList<string>(TensorShape({1}), {new_asset_path});

  // Reset the test context to ensure it's clean.
  ResetOpKernelContext();

  // Run the kernel.
  TF_EXPECT_OK(RunOpKernelWithContext());

  // Expect that the ResourceMgr contains a the correct string.
  StringResource *resource;
  TF_EXPECT_OK(resource_mgr()->Lookup<StringResource>(kGlobalContainer,
                                                      kBasePathTag, &resource));

  EXPECT_EQ(*resource->get(), expected_asset_path);

  resource->Unref();
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

// If an asset_base_path resource exists, the GetSession op should prepend
// that path to all paths in the MasterSpec before creating a session.
TEST_F(DragnnOpKernelsTest, GetSessionWithAssetBasePathTest) {
  // Create a MasterSpec and GridPoint string to pass into the attrs for this
  // op.
  const string new_asset_path = "new/base";
  MasterSpec spec;

  // The first component in the MasterSpec has one resource with one part.
  auto component_one = spec.add_component();
  auto backend_one = component_one->mutable_backend();
  backend_one->set_registered_name("TestComponent");
  component_one->add_resource()->add_part()->set_file_pattern(
      "path/to/an/asset.txt");
  const string expected_component_one_asset = "new/base/path/to/an/asset.txt";

  auto component_two = spec.add_component();
  auto backend_two = component_two->mutable_backend();
  backend_two->set_registered_name("TestComponent");

  // The second component's first resource has no assets.
  component_two->add_resource();

  // The second component's second resource has one part.
  vector<string> expected_component_two_assets;
  component_two->add_resource()->add_part()->set_file_pattern(
      "another/dir/with/an/asset.txt");
  expected_component_two_assets.push_back(
      "new/base/another/dir/with/an/asset.txt");

  // The second component's third resource has two parts.
  auto third_resource = component_two->add_resource();
  third_resource->add_part()->set_file_pattern(
      "another/dir/with/an/asset3.jif");
  expected_component_two_assets.push_back(
      "new/base/another/dir/with/an/asset3.jif");
  third_resource->add_part()->set_file_pattern(
      "another/dir/with/an/asset4.jif");
  expected_component_two_assets.push_back(
      "new/base/another/dir/with/an/asset4.jif");

  LOG(INFO) << spec.DebugString();

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

  // Create the string in the resource manager.
  std::unique_ptr<string> asset_path_ptr(new string(new_asset_path));

  TF_EXPECT_OK(resource_mgr()->Create<StringResource>(
      kGlobalContainer, kBasePathTag,
      new StringResource(std::move(asset_path_ptr))));

  // Run the kernel.
  TF_EXPECT_OK(RunOpKernelWithContext());

  // Expect that the ResourceMgr contains a ComputeSessionPoolResource.
  const string pool_id_str = "pool";
  ComputeSessionPoolResource *pool_resource;
  TF_EXPECT_OK(resource_mgr()->Lookup<ComputeSessionPoolResource>(
      container_string, pool_id_str, &pool_resource));

  // Validate that the master spec held by the pool has the new directory names.
  auto rewritten_spec = pool_resource->get()->GetSpec();
  EXPECT_EQ(rewritten_spec.component(0).resource(0).part(0).file_pattern(),
            expected_component_one_asset);
  EXPECT_EQ(rewritten_spec.component(1).resource(1).part(0).file_pattern(),
            expected_component_two_assets.at(0));
  EXPECT_EQ(rewritten_spec.component(1).resource(2).part(0).file_pattern(),
            expected_component_two_assets.at(1));
  EXPECT_EQ(rewritten_spec.component(1).resource(2).part(1).file_pattern(),
            expected_component_two_assets.at(2));

  // Unref the managed resources so they get destroyed properly.
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

// The GetSessionCounts op should report the number of sessions created and
// free.
TEST_F(DragnnOpKernelsTest, GetSessionCountsOpTest) {
  // Create and initialize the kernel under test.
  TF_ASSERT_OK(
      NodeDefBuilder("get_session_counts", "GetSessionCounts")
          .Input(FakeInput(DT_STRING))  // The handle for the ComputeSession.
          .Finalize(node_def()));
  TF_ASSERT_OK(InitOp());

  // Set the input data.
  const string container_string = "container_str";
  AddInputFromList<string>(TensorShape({1}), {container_string});

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

  // Create two ComputeSessions.
  auto session_one = pool_ptr->GetSession();
  auto session_two = pool_ptr->GetSession();

  // Retun one of them.
  pool_ptr->ReturnSession(std::move(session_two));

  // At this point, the pool should report that it has one outstanding session
  // and two sessions total.
  EXPECT_EQ(1, pool_ptr->num_outstanding_sessions());
  EXPECT_EQ(2, pool_ptr->num_unique_sessions());

  // Run the kernel.
  TF_EXPECT_OK(RunOpKernelWithContext());

  EXPECT_EQ(pool_ptr->num_unique_sessions(), GetOutput(0)->vec<int64>()(0));
  EXPECT_EQ(pool_ptr->num_outstanding_sessions(),
            GetOutput(0)->vec<int64>()(1));
}

// The RebatchDensor op should rebatch densors.
TEST_F(DragnnOpKernelsTest, RebatchDensorOpTest) {
  int sequence_length = 3;
  int pad_length = 2;
  TF_ASSERT_OK(NodeDefBuilder("rebatch_densor", "RebatchDensor")
                   .Attr("sequence_length", sequence_length)
                   .Attr("lr_padding", pad_length)
                   .Input(FakeInput(DT_FLOAT))  // The dense data tensor.
                   .Input(FakeInput(DT_INT32))  // The offsets tensor.
                   .Finalize(node_def()));
  TF_ASSERT_OK(InitOp());

  // Set the input data.
  const std::vector<float> weights = {
      // PASSAGE 1
      1.01, 1.02,  //
      1.04, 1.05,  //
      1.07, 1.08,  //
      1.10, 1.11,  //
      // PASSAGE 2
      2.01, 2.02,  //
      2.03, 2.04,  //
      2.05, 2.06,  //
      2.07, 2.08,  //
      2.09, 2.10,  //
      2.11, 2.12   //
  };
  AddInputFromArray<float>(TensorShape({10, 2}), weights);
  const std::vector<int> offsets = {0, 4, 10};
  AddInputFromArray<int>(TensorShape({3}), offsets);

  // Reset the test context to ensure it's clean.
  ResetOpKernelContext();

  // Run the kernel.
  TF_EXPECT_OK(RunOpKernelWithContext());

  // The first two embeddings in the 1st and 3rd output should be {0.0}
  // The first two embeddings in the 2nd output should be embeddings from token
  // 1 and 2 (so vector items 4 through 10).
  // The last 2 embeddings in row 1 should be from token 4, then 0s.
  // The last 4 embeddings in rows 2 and 3 should be 0.
  const std::vector<float> expected_weights = {
      // BATCH 0
      0.0, 0.0,    //
      0.0, 0.0,    //
      1.01, 1.02,  //
      1.04, 1.05,  //
      1.07, 1.08,  //
      1.10, 1.11,  //
      0.0, 0.0,    //
      // BATCH 1
      1.04, 1.05,  //
      1.07, 1.08,  //
      1.10, 1.11,  //
      0.0, 0.0,    //
      0.0, 0.0,    //
      0.0, 0.0,    //
      0.0, 0.0,    //
      // BATCH 2
      0.0, 0.0,    //
      0.0, 0.0,    //
      2.01, 2.02,  //
      2.03, 2.04,  //
      2.05, 2.06,  //
      2.07, 2.08,  //
      2.09, 2.10,  //
      // BATCH 3
      2.03, 2.04,  //
      2.05, 2.06,  //
      2.07, 2.08,  //
      2.09, 2.10,  //
      2.11, 2.12,  //
      0.0, 0.0,    //
      0.0, 0.0,    //
  };

  for (int i = 0; i < expected_weights.size(); ++i) {
    LOG(INFO) << GetOutput(0)->flat<float>()(i);
  }

  // The output should have dimensions {4, 7, 2}.
  EXPECT_EQ(4, GetOutput(0)->dim_size(0));
  EXPECT_EQ(7, GetOutput(0)->dim_size(1));
  EXPECT_EQ(2, GetOutput(0)->dim_size(2));

  // The output should match the expected tensor.
  for (int i = 0; i < expected_weights.size(); ++i) {
    EXPECT_EQ(expected_weights[i], GetOutput(0)->flat<float>()(i))
        << "Failed at index " << i;
  }

  // The offsets output shout have dimension {3}.
  EXPECT_EQ(4, GetOutput(1)->dim_size(0));
  std::vector<int> expected_indices = {0, 0, 1, 1};
  for (int i = 0; i < expected_indices.size(); ++i) {
    EXPECT_EQ(expected_indices[i], GetOutput(1)->flat<int32>()(i))
        << "Failed at index " << i;
  }
}

// Todo(me): write this
TEST_F(DragnnOpKernelsTest, UnbatchSubsequences) {
  TF_ASSERT_OK(NodeDefBuilder("unbatch_subsequences", "UnbatchSubsequences")
                   .Input(FakeInput(DT_FLOAT))  // The data tensor.
                   .Input(FakeInput(DT_INT32))  // The index tensor.
                   .Input(FakeInput(DT_INT32))  // The offsets tensor.
                   .Finalize(node_def()));
  TF_ASSERT_OK(InitOp());

  // Set the input data.
  const std::vector<float> input = {
      // BATCH 0
      1.01, 1.02,  //
      1.04, 1.05,  //
      1.07, 1.08,  //
      1.10, 1.11,  //
      1.12, 1.13,  //
      // BATCH 1
      1.14, 1.15,  //
      0.0, 0.0,    //
      0.0, 0.0,    //
      0.0, 0.0,    //
      0.0, 0.0,    //
      // BATCH 2
      2.01, 2.02,  //
      0.0, 0.0,    //
      0.0, 0.0,    //
      0.0, 0.0,    //
      0.0, 0.0,    //
      // BATCH 3
      3.01, 3.02,  //
      0.0, 0.0,    //
      0.0, 0.0,    //
      0.0, 0.0,    //
      0.0, 0.0     //
  };

  AddInputFromArray<float>(TensorShape({4, 1, 5, 2}), input);
  const std::vector<int> indices = {0, 0, 1, 2};
  AddInputFromArray<int>(TensorShape({4}), indices);
  const std::vector<int> offsets = {0, 6, 7, 8};
  AddInputFromArray<int>(TensorShape({4}), offsets);

  // Reset the test context to ensure it's clean.
  ResetOpKernelContext();

  // Run the kernel.
  TF_EXPECT_OK(RunOpKernelWithContext());

  // The first two embeddings in the 1st and 3rd output should be {0.0}
  // The first two embeddings in the 2nd output should be embeddings from token
  // 1 and 2 (so vector items 4 through 10).
  // The last 2 embeddings in row 1 should be from token 4, then 0s.
  // The last 4 embeddings in rows 2 and 3 should be 0.
  const std::vector<float> expected_weights = {
      // BATCH 0
      1.01, 1.02,  //
      1.04, 1.05,  //
      1.07, 1.08,  //
      1.10, 1.11,  //
      1.12, 1.13,  //
      1.14, 1.15,  //
      // BATCH 1
      2.01, 2.02,  //
      0.0, 0.0,    //
      0.0, 0.0,    //
      0.0, 0.0,    //
      0.0, 0.0,    //
      0.0, 0.0,    //
      // BATCH 2
      3.01, 3.02,  //
      0.0, 0.0,    //
      0.0, 0.0,    //
      0.0, 0.0,    //
      0.0, 0.0,    //
      0.0, 0.0     //
  };

  for (int i = 0; i < expected_weights.size(); ++i) {
    LOG(INFO) << GetOutput(0)->flat<float>()(i);
  }

  // The output should have dimensions {3, 7, 2}.
  EXPECT_EQ(3, GetOutput(0)->dim_size(0));
  EXPECT_EQ(6, GetOutput(0)->dim_size(1));
  EXPECT_EQ(2, GetOutput(0)->dim_size(2));

  // The output should match the expected tensor.
  for (int i = 0; i < expected_weights.size(); ++i) {
    EXPECT_EQ(expected_weights[i], GetOutput(0)->flat<float>()(i))
        << "Failed at index " << i;
  }
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
                                      const float *score_matrix, int num_items,
                                      int num_actions) {
    EXPECT_EQ(weights.size(), num_items * num_actions);
    for (int i = 0; i < weights.size(); ++i) {
      EXPECT_EQ(weights[i], score_matrix[i]);
    }
    return true;
  };
  EXPECT_CALL(*mock_session_ptr, AdvanceFromPrediction(component_name, _, _, _))
      .WillOnce(Invoke(validator_function));

  // Run the kernel.
  TF_EXPECT_OK(RunOpKernelWithContext());
}

// The AdvanceFromPredicton op should call AdvanceFromPrediction on the
// specified component with the passed scores. If it returns false, the op
// should not return OK.
TEST_F(DragnnOpKernelsTest, AdvanceFromPredictionFailureTest) {
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
                                      const float *score_matrix, int num_items,
                                      int num_actions) {
    EXPECT_EQ(weights.size(), num_items * num_actions);
    for (int i = 0; i < weights.size(); ++i) {
      EXPECT_EQ(weights[i], score_matrix[i]);
    }
    return true;
  };
  EXPECT_CALL(*mock_session_ptr, AdvanceFromPrediction(component_name, _, _, _))
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
  //   feature c: (3, 0.1), (7, [empty]) <- Empty weights are equivalent
  //   to 1.0.
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
  // ordering. So, if we have a batch of 2 and a beam of 3, with data as
  // follows (note that the features are {batch,beam,step} and [] is 'empty')
  // batch 1 features: {{02,03,[]},{01,00,04},{08,06,01}}
  // batch 2 features: {{12,13,14},{11,12,-1},{18,16,20}}
  //
  // and a **source component** beam size of 5 should result in output
  // tensors: step_idx  (tensor 0): {-1,  4,  1, 14, -1, 20} array_idx (tensor
  // 1): { 0,  5, 46, 73,  0, 106} (0
  // [step=-1]),(5=1*5+0),(46=8*5+6),(73=12*5+13),(0 [step=-1]),(96=18*5+16)
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

  // Vectors containing, respectively, label ids and the corresponding Labels.
  const std::vector<std::vector<std::vector<Label>>> oracle_labels(
      {{{{1, 1.f}}, {{3, 1.f}}, {{5, 1.f}}, {{7, 1.f}}},
       {{{2, 1.f}}, {{4, 1.f}}, {{6, 1.f}}, {{8, 1.f}}}});

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

// The EmitOracleLabelsAndProbabilities op returns vectors of instance
// indices, labels, and probabilities corresponding to the elements in the
// beams in the batch.
TEST_F(DragnnOpKernelsTest, EmitOracleLabelsAndProbabilitiesOpTest) {
  // Create and initialize the kernel under test.
  const string component_name = "TESTING_COMPONENT_NAME";
  TF_ASSERT_OK(
      NodeDefBuilder("emit_oracle_labels_and_probabilities",
                     "EmitOracleLabelsAndProbabilities")
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

  // The op should request the oracle labels, and probabilities. They should
  // be returned in batch major order, so if the label:probability pairs are:
  //   batch 1 oracle labels: {{1:0.6, 2:0.8}, {3:1.0}, {5:0.7}}
  //   batch 2 oracle labels: {{2:0.9}, {4:1.0}, {6:0.3, 8:0.6}}
  // then the resulting output tensors are:
  //   indices_output: {0, 0, 1, 2, 3, 4, 5, 5}
  //   label_output:   {1, 2, 3, 5, 2, 4, 6, 8}
  //   prob_output:    {0.6, 0.8, 1.0, 0.7, 0.9, 1.0, 0.3, 0.6}

  // Oracle labels along with their probabilities.
  const std::vector<std::vector<std::vector<Label>>> oracle_labels(
      {{{{1, 0.6}, {2, 0.8}}, {{3, 1.0}}, {{5, 0.7}}},
       {{{2, 0.9}}, {{4, 1.0}}, {{6, 0.3}, {8, 0.6}}}});

  EXPECT_CALL(*mock_session_ptr, EmitOracleLabels(component_name))
      .WillOnce(Return(oracle_labels));

  const std::vector<int> expected_indices({0, 0, 1, 2, 3, 4, 5, 5});
  const std::vector<int> expected_labels({1, 2, 3, 5, 2, 4, 6, 8});
  const std::vector<float> expected_probs(
      {0.6, 0.8, 1.0, 0.7, 0.9, 1.0, 0.3, 0.6});

  // Run the kernel.
  TF_EXPECT_OK(RunOpKernelWithContext());

  // Validate the outputs.
  EXPECT_EQ(expected_indices.size(), GetOutput(0)->NumElements());
  EXPECT_EQ(expected_labels.size(), GetOutput(1)->NumElements());
  EXPECT_EQ(expected_probs.size(), GetOutput(2)->NumElements());
  for (int i = 0; i < expected_indices.size(); ++i) {
    EXPECT_EQ(expected_indices[i], GetOutput(0)->vec<int32>()(i));
    EXPECT_EQ(expected_labels[i], GetOutput(1)->vec<int32>()(i));
    EXPECT_EQ(expected_probs[i], GetOutput(2)->vec<float>()(i));
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
