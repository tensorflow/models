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

#include "dragnn/runtime/transition_system_traits.h"

#include <string>
#include <utility>

#include "dragnn/core/test/generic.h"
#include "dragnn/protos/spec.pb.h"
#include "syntaxnet/base.h"
#include "tensorflow/core/platform/test.h"

namespace syntaxnet {
namespace dragnn {
namespace runtime {
namespace {

// Returns a ComponentSpec that uses the |transition_system|, is configured to
// run left-to-right if |left_to_right| is true, and whose transition system
// predicts |num_actions| actions.
ComponentSpec MakeTestSpec(const string &transition_system, bool left_to_right,
                           int num_actions) {
  ComponentSpec component_spec;
  component_spec.set_num_actions(num_actions);
  component_spec.mutable_transition_system()->set_registered_name(
      transition_system);
  component_spec.mutable_transition_system()->mutable_parameters()->insert(
      {"left_to_right", left_to_right ? "true" : "false"});
  return component_spec;
}

// Tests that boolean values are case-insensitive.
TEST(TransitionSystemTraitsAttributeParsingTest, CaseInsensitiveBooleanValues) {
  ComponentSpec component_spec = MakeTestSpec("shift-only", false, 1);
  auto &parameters =
      *component_spec.mutable_transition_system()->mutable_parameters();

  for (const string &true_value : {"TRUE", "True"}) {
    parameters["left_to_right"] = true_value;
    TransitionSystemTraits traits(component_spec);
    EXPECT_TRUE(traits.is_left_to_right);
  }

  for (const string &false_value : {"FALSE", "False"}) {
    parameters["left_to_right"] = false_value;
    TransitionSystemTraits traits(component_spec);
    EXPECT_FALSE(traits.is_left_to_right);
  }
}

// Parameterized on (left-to-right, deterministic).
class TransitionSystemTraitsTest
    : public ::testing::TestWithParam<::testing::tuple<bool, bool>> {
 protected:
  // Returns the test parameters.
  bool left_to_right() const { return ::testing::get<0>(GetParam()); }
  bool deterministic() const { return ::testing::get<1>(GetParam()); }

  // Returns a ComponentSpec for the |transition_system|.
  ComponentSpec MakeSpec(const string &transition_system) {
    return MakeTestSpec(transition_system, left_to_right(),
                        deterministic() ? 1 : 10);
  }
};

INSTANTIATE_TEST_CASE_P(LeftToRightAndDeterministic, TransitionSystemTraitsTest,
                        ::testing::Combine(::testing::Bool(),
                                           ::testing::Bool()));

// Tests the traits of an unknown transition system.
TEST_P(TransitionSystemTraitsTest, Unknown) {
  TransitionSystemTraits traits(MakeSpec("unknown"));
  EXPECT_EQ(traits.is_deterministic, deterministic());
  EXPECT_FALSE(traits.is_sequential);
  EXPECT_EQ(traits.is_left_to_right, left_to_right());
  EXPECT_FALSE(traits.is_character_scale);
  EXPECT_FALSE(traits.is_token_scale);
}

// Tests the traits of the "char-shift-only" transition system.
TEST_P(TransitionSystemTraitsTest, CharShiftOnly) {
  TransitionSystemTraits traits(MakeSpec("char-shift-only"));
  EXPECT_EQ(traits.is_deterministic, deterministic());
  EXPECT_TRUE(traits.is_sequential);
  EXPECT_EQ(traits.is_left_to_right, left_to_right());
  EXPECT_TRUE(traits.is_character_scale);
  EXPECT_FALSE(traits.is_token_scale);
}

// Tests the traits of the "shift-only" transition system.
TEST_P(TransitionSystemTraitsTest, ShiftOnly) {
  TransitionSystemTraits traits(MakeSpec("shift-only"));
  EXPECT_EQ(traits.is_deterministic, deterministic());
  EXPECT_TRUE(traits.is_sequential);
  EXPECT_EQ(traits.is_left_to_right, left_to_right());
  EXPECT_FALSE(traits.is_character_scale);
  EXPECT_TRUE(traits.is_token_scale);
}

// Tests the traits of the "tagger" transition system.
TEST_P(TransitionSystemTraitsTest, Tagger) {
  TransitionSystemTraits traits(MakeSpec("tagger"));
  EXPECT_EQ(traits.is_deterministic, deterministic());
  EXPECT_TRUE(traits.is_sequential);
  EXPECT_EQ(traits.is_left_to_right, left_to_right());
  EXPECT_FALSE(traits.is_character_scale);
  EXPECT_TRUE(traits.is_token_scale);
}

// Tests the traits of the "morpher" transition system.
TEST_P(TransitionSystemTraitsTest, Morpher) {
  TransitionSystemTraits traits(MakeSpec("morpher"));
  EXPECT_EQ(traits.is_deterministic, deterministic());
  EXPECT_TRUE(traits.is_sequential);
  EXPECT_EQ(traits.is_left_to_right, left_to_right());
  EXPECT_FALSE(traits.is_character_scale);
  EXPECT_TRUE(traits.is_token_scale);
}

// Tests the traits of the "heads" transition system.
TEST_P(TransitionSystemTraitsTest, Heads) {
  TransitionSystemTraits traits(MakeSpec("heads"));
  EXPECT_EQ(traits.is_deterministic, deterministic());
  EXPECT_TRUE(traits.is_sequential);
  EXPECT_EQ(traits.is_left_to_right, left_to_right());
  EXPECT_FALSE(traits.is_character_scale);
  EXPECT_TRUE(traits.is_token_scale);
}

// Tests the traits of the "labels" transition system.
TEST_P(TransitionSystemTraitsTest, Labels) {
  TransitionSystemTraits traits(MakeSpec("labels"));
  EXPECT_EQ(traits.is_deterministic, deterministic());
  EXPECT_TRUE(traits.is_sequential);
  EXPECT_EQ(traits.is_left_to_right, left_to_right());
  EXPECT_FALSE(traits.is_character_scale);
  EXPECT_TRUE(traits.is_token_scale);
}

}  // namespace
}  // namespace runtime
}  // namespace dragnn
}  // namespace syntaxnet
