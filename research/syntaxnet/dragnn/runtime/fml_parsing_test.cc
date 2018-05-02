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

#include "dragnn/runtime/fml_parsing.h"

#include <string>
#include <vector>

#include "dragnn/core/test/generic.h"
#include "syntaxnet/base.h"
#include "syntaxnet/feature_extractor.pb.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"

namespace syntaxnet {
namespace dragnn {
namespace runtime {
namespace {

// Attributes for the test.
struct TestAttributes : public FeatureFunctionAttributes {
  Optional<int32> foo{"foo", -1, this};
  Mandatory<float> bar{"bar", this};
};

// Tests that attributes can be parsed from a valid feature descriptor.
TEST(FeatureFunctionAttributesTest, ValidDescriptor) {
  FeatureFunctionDescriptor function;
  Parameter *parameter = function.add_parameter();
  parameter->set_name("bar");
  parameter->set_value("1.75");

  TestAttributes attributes;
  TF_ASSERT_OK(attributes.Reset(function));
  EXPECT_EQ(attributes.foo(), -1);
  EXPECT_EQ(attributes.bar(), 1.75);
}

// Tests that a feature chain can be parsed from valid FML, and the feature
// options can then be extracted as attributes.
TEST(ParseFeatureChainFmlTest, ValidFml) {
  FeatureFunctionDescriptor leaf;
  TF_ASSERT_OK(ParseFeatureChainFml("path.to.feature(foo=123,bar=-0.5)",
                                    {"path", "to", "feature"}, &leaf));

  TestAttributes attributes;
  TF_ASSERT_OK(attributes.Reset(leaf));
  EXPECT_EQ(attributes.foo(), 123);
  EXPECT_EQ(attributes.bar(), -0.5);
}

// Tests that an empty feature chain cannot be parsed.
TEST(ParseFeatureChainFmlTest, EmptyChain) {
  FeatureFunctionDescriptor leaf;
  EXPECT_THAT(ParseFeatureChainFml("foo", {}, &leaf),
              test::IsErrorWithSubstr("Empty chain of feature types"));
}

// Tests that empty FML cannot be parsed as a chain.
TEST(ParseFeatureChainFmlTest, EmptyFml) {
  FeatureFunctionDescriptor leaf;
  EXPECT_THAT(ParseFeatureChainFml("", {"foo"}, &leaf),
              test::IsErrorWithSubstr("Failed to parse feature chain"));
}

// Tests that feature chain parsing fails if the chain is too short.
TEST(ParseFeatureChainFmlTest, ChainTooShort) {
  FeatureFunctionDescriptor leaf;
  EXPECT_THAT(ParseFeatureChainFml("path.to.feature", {"path", "to"}, &leaf),
              test::IsErrorWithSubstr("Failed to parse feature chain"));
}

// Tests that feature chain parsing fails if the chain is too long.
TEST(ParseFeatureChainFmlTest, ChainTooLong) {
  FeatureFunctionDescriptor leaf;
  EXPECT_THAT(ParseFeatureChainFml("path.to", {"path", "to", "feature"}, &leaf),
              test::IsErrorWithSubstr("Failed to parse feature chain"));
}

// Tests that initial elements of the chain must match the specified types.
TEST(ParseFeatureChainFmlTest, WrongTypeInPrefix) {
  FeatureFunctionDescriptor leaf;
  EXPECT_THAT(
      ParseFeatureChainFml("path.to.feature", {"bad", "to", "feature"}, &leaf),
      test::IsErrorWithSubstr("Failed to parse feature chain"));
}

// Tests that the last feature in the chain must match the specified type.
TEST(ParseFeatureChainFmlTest, WrongTypeInLeaf) {
  FeatureFunctionDescriptor leaf;
  EXPECT_THAT(
      ParseFeatureChainFml("path.to.feature", {"path", "to", "bad"}, &leaf),
      test::IsErrorWithSubstr("Failed to parse feature chain"));
}

// Tests that initial elements of the chain cannot have an argument.
TEST(ParseFeatureChainFmlTest, ArgumentInPrefix) {
  FeatureFunctionDescriptor leaf;
  EXPECT_THAT(
      ParseFeatureChainFml("ok.bad(1).leaf", {"ok", "bad", "leaf"}, &leaf),
      test::IsErrorWithSubstr("Failed to parse feature chain"));
}

// Tests that initial elements of the chain cannot have an argument.
TEST(ParseFeatureChainFmlTest, OptionInPrefix) {
  FeatureFunctionDescriptor leaf;
  EXPECT_THAT(
      ParseFeatureChainFml("ok.bad(foo=1).leaf", {"ok", "bad", "leaf"}, &leaf),
      test::IsErrorWithSubstr("Failed to parse feature chain"));
}

}  // namespace
}  // namespace runtime
}  // namespace dragnn
}  // namespace syntaxnet
