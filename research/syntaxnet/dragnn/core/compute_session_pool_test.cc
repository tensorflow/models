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

#include "dragnn/core/compute_session_pool.h"

#include <memory>

#include <gmock/gmock.h>

#include "dragnn/core/compute_session.h"
#include "dragnn/core/test/generic.h"
#include "dragnn/core/test/mock_component.h"
#include "dragnn/core/test/mock_compute_session.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/test.h"

namespace syntaxnet {
namespace dragnn {

using syntaxnet::test::EqualsProto;
using testing::Return;
using testing::Invoke;
using testing::MockFunction;

class ComputeSessionPoolTestPoolAccessor {
 public:
  static void SetComponentBuilder(
      ComputeSessionPool *pool,
      std::function<std::unique_ptr<Component>(const string &component_name,
                                               const string &backend_type)>
          component_builder_function) {
    pool->SetComponentBuilder(std::move(component_builder_function));
  }

  static void SetSessionBuilder(ComputeSessionPool *pool,
                                std::function<std::unique_ptr<ComputeSession>()>
                                    session_builder_function) {
    pool->SetComputeSessionBuilder(std::move(session_builder_function));
  }
};

TEST(ComputeSessionPoolTest, DefaultConstructorWorks) {
  MasterSpec spec;
  GridPoint hyperparams;
  ComputeSessionPool pool(spec, hyperparams);
  auto request = pool.GetSession();
  EXPECT_NE(request, nullptr);
}

TEST(ComputeSessionPoolTest, ComponentBuilderInjectionWorks) {
  MasterSpec spec;
  auto component = spec.add_component();
  component->set_name("test_component_name");
  auto backend = component->mutable_backend();
  backend->set_registered_name("arbitrary_component");
  GridPoint hyperparams;

  ComputeSessionPool pool(spec, hyperparams);

  // Set up a mock component builder.
  MockFunction<std::unique_ptr<Component>(const string &component_name,
                                          const string &backend_type)>
      mock_component_builder;
  auto mock_creation_function = [](string, string) {
    return std::unique_ptr<MockComponent>(new MockComponent());
  };
  EXPECT_CALL(mock_component_builder,
              Call("test_component_name", "arbitrary_component"))
      .WillOnce(Invoke(mock_creation_function));
  ComputeSessionPoolTestPoolAccessor::SetComponentBuilder(
      &pool, mock_component_builder.AsStdFunction());

  // Now, when the session is requested, the mock component builder should see
  // the expected call.
  auto request = pool.GetSession();
  EXPECT_NE(request, nullptr);
}

TEST(ComputeSessionPoolTest, CreatesNewSessionIfNoSessionsExist) {
  // We don't need to fill these for this test.
  MasterSpec spec;
  GridPoint hyperparams;
  ComputeSessionPool pool(spec, hyperparams);

  // Create a function that will track calls to the session builder.
  MockFunction<std::unique_ptr<ComputeSession>()> mock_session_builder;

  // Initialize expectations for a request for a ComputeSession.
  std::unique_ptr<MockComputeSession> session_one(new MockComputeSession());
  MockComputeSession *session_one_ptr = session_one.get();
  auto mock_creation_function = [&session_one]() {
    return std::move(session_one);
  };
  EXPECT_CALL(mock_session_builder, Call())
      .WillOnce(Invoke(mock_creation_function))
      .RetiresOnSaturation();
  EXPECT_CALL(*session_one_ptr,
              Init(EqualsProto(spec), EqualsProto(hyperparams)));

  // Initialize expectations for another request for a ComputeSession.
  std::unique_ptr<MockComputeSession> session_two(new MockComputeSession());
  MockComputeSession *session_two_ptr = session_two.get();
  auto mock_creation_function_two = [&session_two]() {
    return std::move(session_two);
  };
  EXPECT_CALL(mock_session_builder, Call())
      .WillOnce(Invoke(mock_creation_function_two))
      .RetiresOnSaturation();
  EXPECT_CALL(*session_two_ptr,
              Init(EqualsProto(spec), EqualsProto(hyperparams)));

  // Inject the function to the pool.
  ComputeSessionPoolTestPoolAccessor::SetSessionBuilder(
      &pool, mock_session_builder.AsStdFunction());

  // The first call will recieve the second session because of how the mocks go.
  auto first_request = pool.GetSession();
  EXPECT_EQ(first_request.get(), session_two_ptr);

  auto second_request = pool.GetSession();
  EXPECT_EQ(second_request.get(), session_one_ptr);
}

TEST(ComputeSessionPoolTest, ReusesAvailableSessions) {
  // We don't need to fill these for this test.
  MasterSpec spec;
  GridPoint hyperparams;
  ComputeSessionPool pool(spec, hyperparams);

  // Create a function that will track calls to the session builder.
  MockFunction<std::unique_ptr<ComputeSession>()> mock_session_builder;

  // Initialize expectations for a request for a ComputeSession.
  std::unique_ptr<MockComputeSession> session_one(new MockComputeSession());
  MockComputeSession *session_one_ptr = session_one.get();
  auto mock_creation_function = [&session_one]() {
    return std::move(session_one);
  };
  EXPECT_CALL(mock_session_builder, Call())
      .WillOnce(Invoke(mock_creation_function))
      .RetiresOnSaturation();
  EXPECT_CALL(*session_one_ptr,
              Init(EqualsProto(spec), EqualsProto(hyperparams)));

  // Initialize expectations for another request for a ComputeSession.
  std::unique_ptr<MockComputeSession> session_two(new MockComputeSession());
  MockComputeSession *session_two_ptr = session_two.get();
  auto mock_creation_function_two = [&session_two]() {
    return std::move(session_two);
  };
  EXPECT_CALL(mock_session_builder, Call())
      .WillOnce(Invoke(mock_creation_function_two))
      .RetiresOnSaturation();
  EXPECT_CALL(*session_two_ptr,
              Init(EqualsProto(spec), EqualsProto(hyperparams)));

  // Inject the function to the pool.
  ComputeSessionPoolTestPoolAccessor::SetSessionBuilder(
      &pool, mock_session_builder.AsStdFunction());

  // The first call will recieve the second session because of how the mocks go.
  auto first_request = pool.GetSession();
  EXPECT_EQ(1, pool.num_outstanding_sessions());
  EXPECT_EQ(first_request.get(), session_two_ptr);

  // Return the first pointer. After this, the second request should get that
  // pointer.
  EXPECT_CALL(*session_two_ptr, ResetSession());
  pool.ReturnSession(std::move(first_request));
  EXPECT_EQ(0, pool.num_outstanding_sessions());
  auto second_request = pool.GetSession();
  EXPECT_EQ(1, pool.num_outstanding_sessions());
  EXPECT_EQ(second_request.get(), session_two_ptr);

  // There are now no spare sessions, so the next session request should
  // create a second session.
  auto third_request = pool.GetSession();
  EXPECT_EQ(2, pool.num_outstanding_sessions());
  EXPECT_EQ(third_request.get(), session_one_ptr);
}

TEST(ComputeSessionPoolTest, AssignsUniqueIds) {
  MasterSpec spec;
  GridPoint hyperparams;
  ComputeSessionPool pool(spec, hyperparams);
  auto session = pool.GetSession();
  auto session_2 = pool.GetSession();
  EXPECT_NE(session->Id(), session_2->Id());
}

TEST(ComputeSessionPoolTest, SupportsMultithreadedAccess) {
  MasterSpec spec;
  GridPoint hyperparams;
  ComputeSessionPool pool(spec, hyperparams);

  std::vector<std::unique_ptr<tensorflow::Thread>> request_threads;
  constexpr int kNumThreadsToTest = 100;
  request_threads.reserve(kNumThreadsToTest);
  for (int i = 0; i < kNumThreadsToTest; ++i) {
    request_threads.push_back(std::unique_ptr<tensorflow::Thread>(
        tensorflow::Env::Default()->StartThread(
            tensorflow::ThreadOptions(), "thread",
            [this, &pool] { auto session = pool.GetSession(); })));
  }

  // Deleting a tensorflow::Thread blocks until the thread exits,
  // so clearing the vector blocks until all threads have exited.
  request_threads.clear();

  // Make sure all the threads got their session.
  EXPECT_EQ(kNumThreadsToTest, pool.num_outstanding_sessions());
}

}  // namespace dragnn
}  // namespace syntaxnet
