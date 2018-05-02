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

#include <memory>

#include "dragnn/core/test/generic.h"
#include "dragnn/protos/spec.pb.h"
#include "dragnn/runtime/component_transformation.h"
#include "dragnn/runtime/math/types.h"
#include "dragnn/runtime/sequence_extractor.h"
#include "dragnn/runtime/sequence_linker.h"
#include "dragnn/runtime/sequence_predictor.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"

namespace syntaxnet {
namespace dragnn {
namespace runtime {
namespace {

// Arbitrary supported component type.
constexpr char kSupportedComponentType[] = "MyelinDynamicComponent";

// Sequence-based version of the component type.
constexpr char kTransformedComponentType[] = "SequenceMyelinDynamicComponent";

// Trivial extractor that supports components named "supported".
class SupportIfNamedSupportedExtractor : public SequenceExtractor {
 public:
  // Implements SequenceExtractor.
  bool Supports(const FixedFeatureChannel &,
                const ComponentSpec &component_spec) const override {
    return component_spec.name() == "supported";
  }
  tensorflow::Status Initialize(const FixedFeatureChannel &,
                                const ComponentSpec &) override {
    return tensorflow::Status::OK();
  }
  tensorflow::Status GetIds(InputBatchCache *,
                            std::vector<int32> *) const override {
    return tensorflow::Status::OK();
  }
};

DRAGNN_RUNTIME_REGISTER_SEQUENCE_EXTRACTOR(SupportIfNamedSupportedExtractor);

// Trivial extractor that supports components if they have a resource.  This is
// used to generate a "multiple supported extractors" conflict.
class SupportIfHasResourcesExtractor : public SequenceExtractor {
 public:
  // Implements SequenceExtractor.
  bool Supports(const FixedFeatureChannel &,
                const ComponentSpec &component_spec) const override {
    return component_spec.resource_size() > 0;
  }
  tensorflow::Status Initialize(const FixedFeatureChannel &,
                                const ComponentSpec &) override {
    return tensorflow::Status::OK();
  }
  tensorflow::Status GetIds(InputBatchCache *,
                            std::vector<int32> *) const override {
    return tensorflow::Status::OK();
  }
};

DRAGNN_RUNTIME_REGISTER_SEQUENCE_EXTRACTOR(SupportIfHasResourcesExtractor);

// Trivial linker that supports components named "supported".
class SupportIfNamedSupportedLinker : public SequenceLinker {
 public:
  // Implements SequenceLinker.
  bool Supports(const LinkedFeatureChannel &,
                const ComponentSpec &component_spec) const override {
    return component_spec.name() == "supported";
  }
  tensorflow::Status Initialize(const LinkedFeatureChannel &,
                                const ComponentSpec &) override {
    return tensorflow::Status::OK();
  }
  tensorflow::Status GetLinks(size_t, InputBatchCache *,
                              std::vector<int32> *) const override {
    return tensorflow::Status::OK();
  }
};

DRAGNN_RUNTIME_REGISTER_SEQUENCE_LINKER(SupportIfNamedSupportedLinker);

// Trivial predictor that supports components named "supported".
class SupportIfNamedSupportedPredictor : public SequencePredictor {
 public:
  // Implements SequencePredictor.
  bool Supports(const ComponentSpec &component_spec) const override {
    return component_spec.name() == "supported";
  }
  tensorflow::Status Initialize(const ComponentSpec &) override {
    return tensorflow::Status::OK();
  }
  tensorflow::Status Predict(Matrix<float>, InputBatchCache *) const override {
    return tensorflow::Status::OK();
  }
};

DRAGNN_RUNTIME_REGISTER_SEQUENCE_PREDICTOR(SupportIfNamedSupportedPredictor);

// Returns a ComponentSpec that is supported by the transformer.
ComponentSpec MakeSupportedSpec() {
  ComponentSpec component_spec;
  component_spec.set_name("supported");
  component_spec.set_num_actions(10);
  component_spec.add_fixed_feature();
  component_spec.add_fixed_feature();
  component_spec.add_linked_feature();
  component_spec.add_linked_feature();
  component_spec.mutable_component_builder()->set_registered_name(
      kSupportedComponentType);
  return component_spec;
}

// Tests that a compatible spec is modified to use a new backend and component
// builder with SequenceExtractors, SequenceLinkers, and SequencePredictor.
TEST(SequenceComponentTransformerTest, Compatible) {
  ComponentSpec component_spec = MakeSupportedSpec();

  ComponentSpec modified_spec = component_spec;
  modified_spec.mutable_backend()->set_registered_name("SequenceBackend");
  modified_spec.mutable_component_builder()->set_registered_name(
      kTransformedComponentType);
  modified_spec.mutable_component_builder()->mutable_parameters()->insert(
      {"sequence_extractors",
       "SupportIfNamedSupportedExtractor,SupportIfNamedSupportedExtractor"});
  modified_spec.mutable_component_builder()->mutable_parameters()->insert(
      {"sequence_linkers",
       "SupportIfNamedSupportedLinker,SupportIfNamedSupportedLinker"});
  modified_spec.mutable_component_builder()->mutable_parameters()->insert(
      {"sequence_predictor", "SupportIfNamedSupportedPredictor"});

  TF_ASSERT_OK(ComponentTransformer::ApplyAll(&component_spec));
  EXPECT_THAT(component_spec, test::EqualsProto(modified_spec));
}

// Tests that a compatible deterministic spec is modified to use a new backend
// and component builder with SequenceExtractors and SequenceLinkers only.
TEST(SequenceComponentTransformerTest, CompatibleNoPredictor) {
  ComponentSpec component_spec = MakeSupportedSpec();
  component_spec.set_num_actions(1);

  ComponentSpec modified_spec = component_spec;
  modified_spec.mutable_backend()->set_registered_name("SequenceBackend");
  modified_spec.mutable_component_builder()->set_registered_name(
      kTransformedComponentType);
  modified_spec.mutable_component_builder()->mutable_parameters()->insert(
      {"sequence_extractors",
       "SupportIfNamedSupportedExtractor,SupportIfNamedSupportedExtractor"});
  modified_spec.mutable_component_builder()->mutable_parameters()->insert(
      {"sequence_linkers",
       "SupportIfNamedSupportedLinker,SupportIfNamedSupportedLinker"});
  modified_spec.mutable_component_builder()->mutable_parameters()->insert(
      {"sequence_predictor", ""});

  TF_ASSERT_OK(ComponentTransformer::ApplyAll(&component_spec));
  EXPECT_THAT(component_spec, test::EqualsProto(modified_spec));
}

// Tests that a ComponentSpec with no features is incompatible.
TEST(SequenceComponentTransformerTest, IncompatibleNoFeatures) {
  ComponentSpec component_spec = MakeSupportedSpec();
  component_spec.clear_fixed_feature();
  component_spec.clear_linked_feature();

  const ComponentSpec unchanged_spec = component_spec;
  TF_ASSERT_OK(ComponentTransformer::ApplyAll(&component_spec));
  EXPECT_THAT(component_spec, test::EqualsProto(unchanged_spec));
}

// Tests that a ComponentSpec with the wrong component builder is incompatible.
TEST(SequenceComponentTransformerTest, IncompatibleComponentBuilder) {
  ComponentSpec component_spec = MakeSupportedSpec();
  component_spec.mutable_component_builder()->set_registered_name("bad");

  const ComponentSpec unchanged_spec = component_spec;
  TF_ASSERT_OK(ComponentTransformer::ApplyAll(&component_spec));
  EXPECT_THAT(component_spec, test::EqualsProto(unchanged_spec));
}

// Tests that a ComponentSpec is incompatible if it is not supported by any
// SequenceExtractor.
TEST(SequenceComponentTransformerTest,
     IncompatibleNoSupportingSequenceExtractor) {
  ComponentSpec component_spec = MakeSupportedSpec();
  component_spec.set_name("bad");

  const ComponentSpec unchanged_spec = component_spec;
  TF_ASSERT_OK(ComponentTransformer::ApplyAll(&component_spec));
  EXPECT_THAT(component_spec, test::EqualsProto(unchanged_spec));
}

// Tests that a ComponentSpec fails if multiple SequenceExtractors support it.
TEST(SequenceComponentTransformerTest,
     FailIfMultipleSupportingSequenceExtractors) {
  ComponentSpec component_spec = MakeSupportedSpec();
  component_spec.add_resource();  // triggers SupportIfHasResourcesExtractor

  EXPECT_THAT(
      ComponentTransformer::ApplyAll(&component_spec),
      test::IsErrorWithSubstr("Multiple SequenceExtractors support channel"));
}

// Tests that a DynamicComponent is not upgraded if it is recurrent.
TEST(SequenceComponentTransformerTest, RecurrentDynamicComponent) {
  ComponentSpec component_spec = MakeSupportedSpec();
  component_spec.mutable_component_builder()->set_registered_name(
      "DynamicComponent");
  component_spec.mutable_linked_feature(0)->set_source_component(
      component_spec.name());

  const ComponentSpec unchanged_spec = component_spec;
  TF_ASSERT_OK(ComponentTransformer::ApplyAll(&component_spec));
  EXPECT_THAT(component_spec, test::EqualsProto(unchanged_spec));
}

// Tests that a DynamicComponent is upgraded to SequenceBulkDynamicComponent if
// it is non-recurrent.
TEST(SequenceComponentTransformerTest, NonRecurrentDynamicComponent) {
  ComponentSpec component_spec = MakeSupportedSpec();
  component_spec.mutable_component_builder()->set_registered_name(
      "DynamicComponent");

  ComponentSpec modified_spec = component_spec;
  modified_spec.mutable_backend()->set_registered_name("SequenceBackend");
  modified_spec.mutable_component_builder()->set_registered_name(
      "SequenceBulkDynamicComponent");
  modified_spec.mutable_component_builder()->mutable_parameters()->insert(
      {"sequence_extractors",
       "SupportIfNamedSupportedExtractor,SupportIfNamedSupportedExtractor"});
  modified_spec.mutable_component_builder()->mutable_parameters()->insert(
      {"sequence_linkers",
       "SupportIfNamedSupportedLinker,SupportIfNamedSupportedLinker"});
  modified_spec.mutable_component_builder()->mutable_parameters()->insert(
      {"sequence_predictor", "SupportIfNamedSupportedPredictor"});

  TF_ASSERT_OK(ComponentTransformer::ApplyAll(&component_spec));
  EXPECT_THAT(component_spec, test::EqualsProto(modified_spec));
}

}  // namespace
}  // namespace runtime
}  // namespace dragnn
}  // namespace syntaxnet
