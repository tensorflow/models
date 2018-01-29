/* Copyright 2016 Google Inc. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "syntaxnet/generic_features.h"

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "syntaxnet/registry.h"
#include "syntaxnet/task_context.h"
#include <gmock/gmock.h>

namespace syntaxnet {

// Test feature extractor.
class TestFeatureExtractor : public FeatureExtractor<std::vector<int>, int> {};

// Registration macro.
#define REGISTER_TEST_FEATURE_FUNCTION(name, component)                     \
  REGISTER_SYNTAXNET_FEATURE_FUNCTION(TestFeatureExtractor::Function, name, \
                                      component)

// The registry must be declared in the global namespace.
REGISTER_SYNTAXNET_CLASS_REGISTRY("syntaxnet test feature function",
                                  syntaxnet::TestFeatureExtractor::Function);

typedef GenericFeatures<std::vector<int>, int> GenericTestFeatures;
REGISTER_SYNTAXNET_GENERIC_FEATURES(GenericTestFeatures);

class TestVectorFeatureFunction : public TestFeatureExtractor::Function {
 public:
  // Initializes the feature.
  void Init(TaskContext *context) override {
    int arg = argument();
    while (arg > 0) {
      offsets_.push_back(arg % 10);
      arg /= 10;
    }
    std::reverse(offsets_.begin(), offsets_.end());
    if (offsets_.empty()) offsets_.push_back(0);
    set_feature_type(new NumericFeatureType(name(), 10));
  }

  // Evaluates the feature.
  void Evaluate(const WorkspaceSet &workspace, const std::vector<int> &object,
                int focus, FeatureVector *features) const override {
    for (const uint32 offset : offsets_) {
      const uint32 index = focus + offset;
      if (index >= object.size()) continue;
      features->add(feature_type(), object[index]);
    }
  }

  // Returns the first extracted feature, if available.
  FeatureValue Compute(const WorkspaceSet &workspace,
                       const std::vector<int> &object, int focus,
                       const FeatureVector *fv) const override {
    CHECK_EQ(1, offsets_.size());
    FeatureVector features;
    Evaluate(workspace, object, focus, &features);
    return features.size() == 0 ? kNone : features.value(0);
  }

 private:
  // A list of offsets extracted from the feature's argument.
  std::vector<uint32> offsets_;
};

REGISTER_TEST_FEATURE_FUNCTION("f", TestVectorFeatureFunction);

class TestParityFeatureFunction : public TestFeatureExtractor::Function {
 public:
  // Initializes the feature.
  void Init(TaskContext *context) override {
    // "even" corresponds to feature value 0, "odd" to 1.
    enum ParityFeatureValue { EVEN, ODD };
    set_feature_type(
        new EnumFeatureType(name(), {{EVEN, "even"}, {ODD, "odd"}}));

    // Check the "offset" parameter.
    for (const auto &param : this->descriptor()->parameter()) {
      if (param.name() == "offset") {
        offset_ = std::stoi(param.value());
        CHECK(&offset_);
      }
    }
  }

  // Evaluates the feature.
  void Evaluate(const WorkspaceSet &workspace, const std::vector<int> &object,
                int focus, FeatureVector *features) const override {
    uint32 offset_focus = focus += offset_;
    if (offset_focus < object.size()) {
      features->add(feature_type(), object[offset_focus] & 1);
    }
  }

  // Returns the first extracted feature, if available.
  FeatureValue Compute(const WorkspaceSet &workspace,
                       const std::vector<int> &object, int focus,
                       const FeatureVector *fv) const override {
    FeatureVector features;
    Evaluate(workspace, object, focus, &features);
    return features.size() == 0 ? kNone : features.value(0);
  }

 private:
  int offset_ = 0;
};

REGISTER_TEST_FEATURE_FUNCTION("parity", TestParityFeatureFunction);

// Testing rig.
class GenericFeaturesTest : public ::testing::Test {
 public:
  // Deallocates test state.
  void TearDown() override {
    object_.reset();
    extractor_.reset();
    context_.reset();
  }

  // Initializes the test.
  void Init(const string &spec, const std::vector<int> &object) {
    context_.reset(new TaskContext());
    extractor_.reset(new TestFeatureExtractor());
    extractor_->Parse(spec);
    extractor_->Setup(context_.get());
    extractor_->Init(context_.get());
    object_.reset(new std::vector<int>(object));
  }

  // Tests extraction on the current object.
  void TestExtract(int focus, const string &feature_string) const {
    FeatureVector features;
    WorkspaceSet workspace;
    extractor_->Preprocess(&workspace, object_.get());
    extractor_->ExtractFeatures(workspace, *object_, focus, &features);
    EXPECT_EQ(feature_string, features.ToString());
  }

 private:
  // The task context for tests.
  std::unique_ptr<TaskContext> context_;

  // Feature extractor for tests.
  std::unique_ptr<TestFeatureExtractor> extractor_;

  // Object for tests.
  std::unique_ptr<std::vector<int> > object_;
};

TEST_F(GenericFeaturesTest, Singleton) {
  Init("f", {5, 3, 2, 4, 6});
  TestExtract(0, "[f=5]");
  TestExtract(1, "[f=3]");
  TestExtract(4, "[f=6]");
  TestExtract(5, "[]");
}

TEST_F(GenericFeaturesTest, TwoFeatures) {
  Init("f(0) f(1)", {5, 3, 2, 4, 6});
  TestExtract(0, "[f=5,f(1)=3]");
}

TEST_F(GenericFeaturesTest, Bias) {
  Init("bias", {0, 1});
  TestExtract(0, "[bias=ON]");
}

TEST_F(GenericFeaturesTest, Constant) {
  Init("constant(value=2)", {0, 1});

  TestExtract(0, "[constant(value=2)=2]");
}

TEST_F(GenericFeaturesTest, Equals) {
  Init("equals { f(0) f(1) }", {0, 1, 0});
  TestExtract(0, "[equals { f f(1) }=DIFFERENT]");
  Init("equals { f(0) f(2) }", {0, 1, 0});
  TestExtract(0, "[equals { f f(2) }=EQUAL]");
}

TEST_F(GenericFeaturesTest, Filter) {
  Init("filter(value=5).f", {3, 5});
  TestExtract(0, "[]");
  TestExtract(1, "[filter(value=5).f=ON]");

  // Check that we are actually parsing feature value names.
  Init("filter(value=odd).parity", {3, 4});
  TestExtract(0, "[filter(value=odd).parity=ON]");
  TestExtract(1, "[]");
  Init("filter(value=even).parity", {3, 4});
  TestExtract(0, "[]");
  TestExtract(1, "[filter(value=even).parity=ON]");
}

TEST_F(GenericFeaturesTest, Is) {
  Init("is(value=5).f", {3, 5});
  TestExtract(0, "[is(value=5).f=FALSE]");
  TestExtract(1, "[is(value=5).f=TRUE]");

  // Check that we are actually parsing feature value names.
  Init("is(value=odd).parity", {3, 4});
  TestExtract(0, "[is(value=odd).parity=TRUE]");
  TestExtract(1, "[is(value=odd).parity=FALSE]");
  Init("is(value=even).parity", {3, 4});
  TestExtract(0, "[is(value=even).parity=FALSE]");
  TestExtract(1, "[is(value=even).parity=TRUE]");
}

TEST_F(GenericFeaturesTest, Ignore) {
  Init("ignore(value=5).f", {3, 5});
  TestExtract(0, "[ignore(value=5).f=3]");
  TestExtract(1, "[]");

  // Check that we are actually parsing feature value names.
  Init("ignore(value=odd).parity", {3, 4});
  TestExtract(0, "[]");
  TestExtract(1, "[ignore(value=odd).parity=even]");
  Init("ignore(value=even).parity", {3, 4});
  TestExtract(0, "[ignore(value=even).parity=odd]");
  TestExtract(1, "[]");
}

TEST_F(GenericFeaturesTest, All) {
  Init("all { parity parity(offset=1) }", {2, 2});
  TestExtract(0, "[all { parity parity(offset=1) }=FALSE]");

  Init("all { parity parity(offset=1) }", {2, 3});
  TestExtract(0, "[all { parity parity(offset=1) }=FALSE]");

  Init("all { parity parity(offset=1) }", {3, 2});
  TestExtract(0, "[all { parity parity(offset=1) }=FALSE]");

  Init("all { parity parity(offset=1) }", {3, 3});
  TestExtract(0, "[all { parity parity(offset=1) }=TRUE]");
}

TEST_F(GenericFeaturesTest, Any) {
  Init("any { parity parity(offset=1) }", {2, 2});
  TestExtract(0, "[any { parity parity(offset=1) }=FALSE]");

  Init("any { parity parity(offset=1) }", {2, 3});
  TestExtract(0, "[any { parity parity(offset=1) }=TRUE]");

  Init("any { parity parity(offset=1) }", {3, 2});
  TestExtract(0, "[any { parity parity(offset=1) }=TRUE]");

  Init("any { parity parity(offset=1) }", {3, 3});
  TestExtract(0, "[any { parity parity(offset=1) }=TRUE]");
}

TEST_F(GenericFeaturesTest, Pair) {
  Init("pair { f(0) f(1) }", {5, 3, 2, 4, 6});
  TestExtract(0, "[pair { f f(1) }=(5,3)]");
}

TEST_F(GenericFeaturesTest, NestedPair) {
  Init("pair { pair { f(0) f(1) } pair { f(2) f(3) } }", {5, 3, 2, 4, 6});
  TestExtract(0, "[pair { pair { f f(1) } pair { f(2) f(3) } }=((5,3),(2,4))]");
}

TEST_F(GenericFeaturesTest, Triple) {
  Init("triple { f(0) f(1) f(2) }", {5, 3, 2, 4, 6});
  TestExtract(0, "[triple { f f(1) f(2) }=(5,3,2)]");
}

TEST_F(GenericFeaturesTest, Quad) {
  Init("quad { f(0) f(1) f(2) f(3) }", {5, 3, 2, 4, 6});
  TestExtract(0, "[quad { f f(1) f(2) f(3) }=(5,3,2,4)]");
}

TEST_F(GenericFeaturesTest, Quint) {
  Init("quint { f(0) f(1) f(2) f(3) f(4) }", {5, 3, 2, 4, 6});
  TestExtract(0, "[quint { f f(1) f(2) f(3) f(4) }=(5,3,2,4,6)]");
}

TEST_F(GenericFeaturesTest, Tuple) {
  Init("tuple { f(0) f(1) f(2) f(3) f(4) }", {5, 3, 2, 4, 6});
  TestExtract(0, "[tuple { f f(1) f(2) f(3) f(4) }=(5,3,2,4,6)]");
}

TEST_F(GenericFeaturesTest, Pairs) {
  Init("pairs { f(0) f(1) f(2) f(3) }", {0, 1, 2, 3, 4});
  TestExtract(0,
              "[pairs { f f(1) }=(0,1)"
              ",pairs { f f(2) }=(0,2)"
              ",pairs { f(1) f(2) }=(1,2)"
              ",pairs { f f(3) }=(0,3)"
              ",pairs { f(1) f(3) }=(1,3)"
              ",pairs { f(2) f(3) }=(2,3)]");
}

TEST_F(GenericFeaturesTest, PairsWithUnary) {
  Init("pairs(unary=true) { f(0) f(1) f(2) }", {0, 1, 2, 3, 4});
  TestExtract(0,
              "[pairs(unary=true).f=0"
              ",pairs(unary=true).f(1)=1"
              ",pairs(unary=true).f(2)=2"
              ",pairs(unary=true) { f f(1) }=(0,1)"
              ",pairs(unary=true) { f f(2) }=(0,2)"
              ",pairs(unary=true) { f(1) f(2) }=(1,2)]");
}

TEST_F(GenericFeaturesTest, Conjoin) {
  Init("conjoin { f(0) f(1) f(2) f(3) }", {0, 1, 2, 3, 4});
  TestExtract(0,
              "[conjoin { f f(1) }=(0,1)"
              ",conjoin { f f(2) }=(0,2)"
              ",conjoin { f f(3) }=(0,3)]");
}

TEST_F(GenericFeaturesTest, ConjoinWithUnary) {
  Init("conjoin(unary=true) { f(0) f(1) f(2) f(3) }", {0, 1, 2, 3, 4});
  TestExtract(0,
              "[conjoin(unary=true).f(1)=1"
              ",conjoin(unary=true) { f f(1) }=(0,1)"
              ",conjoin(unary=true).f(2)=2"
              ",conjoin(unary=true) { f f(2) }=(0,2)"
              ",conjoin(unary=true).f(3)=3"
              ",conjoin(unary=true) { f f(3) }=(0,3)]");
}

TEST_F(GenericFeaturesTest, SingletonMultiValue) {
  Init("f(12)", {0, 1, 2, 3, 4});
  TestExtract(0, "[f(12)=1,f(12)=2]");
}

TEST_F(GenericFeaturesTest, MultiPairOneSided) {
  Init("multipair { f(12) f(3) }", {0, 1, 2, 3, 4});
  TestExtract(0,
              "[multipair { f(12) f(3) }=(1,3)"
              ",multipair { f(12) f(3) }=(2,3)]");
}

TEST_F(GenericFeaturesTest, MultiPairTwoSided) {
  Init("multipair { f(12) f(34) }", {0, 1, 2, 3, 4});
  TestExtract(0,
              "[multipair { f(12) f(34) }=(1,3)"
              ",multipair { f(12) f(34) }=(1,4)"
              ",multipair { f(12) f(34) }=(2,3)"
              ",multipair { f(12) f(34) }=(2,4)]");
}

TEST_F(GenericFeaturesTest, MultiPairParallel) {
  Init("multipair(parallel=true) { f(12) f(34) }", {0, 1, 2, 3, 4});
  TestExtract(0,
              "[multipair(parallel=true) { f(12) f(34) }=(1,3)"
              ",multipair(parallel=true) { f(12) f(34) }=(2,4)]");
}

TEST_F(GenericFeaturesTest, MultiConjoinFirstOnly) {
  Init("multiconjoin { f(12) f(3) f(0) }", {0, 1, 2, 3, 4});
  TestExtract(0,
              "[multiconjoin { f(12) f(3) }=(1,3)"
              ",multiconjoin { f(12) f(3) }=(2,3)"
              ",multiconjoin { f(12) f }=(1,0)"
              ",multiconjoin { f(12) f }=(2,0)]");
}

TEST_F(GenericFeaturesTest, MultiConjoinFirstAndRest) {
  Init("multiconjoin { f(12) f(34) f(0) }", {0, 1, 2, 3, 4});
  TestExtract(0,
              "[multiconjoin { f(12) f(34) }=(1,3)"
              ",multiconjoin { f(12) f(34) }=(1,4)"
              ",multiconjoin { f(12) f(34) }=(2,3)"
              ",multiconjoin { f(12) f(34) }=(2,4)"
              ",multiconjoin { f(12) f }=(1,0)"
              ",multiconjoin { f(12) f }=(2,0)]");
}

}  // namespace syntaxnet
