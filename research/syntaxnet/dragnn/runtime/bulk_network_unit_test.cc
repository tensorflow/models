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

#include "dragnn/runtime/bulk_network_unit.h"

#include <memory>
#include <string>

#include "dragnn/core/compute_session.h"
#include "dragnn/protos/spec.pb.h"
#include "dragnn/runtime/extensions.h"
#include "dragnn/runtime/math/types.h"
#include "dragnn/runtime/network_states.h"
#include "dragnn/runtime/session_state.h"
#include "dragnn/runtime/variable_store.h"
#include "syntaxnet/base.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"

namespace syntaxnet {
namespace dragnn {
namespace runtime {
namespace {

// Expects that the two pointers have the same address.
void ExpectSameAddress(const void *pointer1, const void *pointer2) {
  EXPECT_EQ(pointer1, pointer2);
}

// A trivial implementation for tests.
class BulkFooNetwork : public BulkNetworkUnit {
 public:
  // Implements BulkNetworkUnit.
  tensorflow::Status Initialize(const ComponentSpec &component_spec,
                                VariableStore *variable_store,
                                NetworkStateManager *network_state_manager,
                                ExtensionManager *extension_manager) override {
    return tensorflow::Status::OK();
  }
  tensorflow::Status ValidateInputDimension(size_t dimension) const override {
    return tensorflow::Status::OK();
  }
  string GetLogitsName() const override { return "foo_logits"; }
  tensorflow::Status Evaluate(Matrix<float> inputs,
                              SessionState *session_state) const override {
    return tensorflow::Status::OK();
  }
};

DRAGNN_RUNTIME_REGISTER_BULK_NETWORK_UNIT(BulkFooNetwork);

// Tests that BulkNetworkUnit::GetClassName() resolves names properly.
TEST(BulkNetworkUnitTest, GetClassName) {
  for (const string &registered_name :
       {"FooNetwork",
        "module.FooNetwork",
        "some.long.path.to.module.FooNetwork"}) {
    ComponentSpec component_spec;
    component_spec.mutable_network_unit()->set_registered_name(registered_name);
    EXPECT_EQ(BulkNetworkUnit::GetClassName(component_spec), "BulkFooNetwork");
  }
}

// Tests that BulkNetworkUnits can be created via the registry.
TEST(BulkNetworkUnitTest, CreateOrError) {
  std::unique_ptr<BulkNetworkUnit> foo;
  TF_ASSERT_OK(BulkNetworkUnit::CreateOrError("BulkFooNetwork", &foo));
  ASSERT_TRUE(foo != nullptr);
  ExpectSameAddress(dynamic_cast<BulkFooNetwork *>(foo.get()), foo.get());
  EXPECT_EQ(foo->GetLogitsName(), "foo_logits");
}

}  // namespace
}  // namespace runtime
}  // namespace dragnn
}  // namespace syntaxnet
