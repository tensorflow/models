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

#include "dragnn/runtime/term_map_sequence_predictor.h"

#include <string>
#include <vector>

#include "dragnn/core/input_batch_cache.h"
#include "dragnn/core/test/generic.h"
#include "dragnn/protos/spec.pb.h"
#include "dragnn/runtime/math/types.h"
#include "dragnn/runtime/test/term_map_helpers.h"
#include "syntaxnet/base.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"

namespace syntaxnet {
namespace dragnn {
namespace runtime {
namespace {

constexpr char kResourceName[] = "term-map";
constexpr int kMinFrequency = 2;
constexpr int kMaxNumTerms = 0;  // no limit

// A subclass for tests.
class BasicTermMapSequencePredictor : public TermMapSequencePredictor {
 public:
  BasicTermMapSequencePredictor() : TermMapSequencePredictor(kResourceName) {}

  // Implements SequencePredictor.  These methods are never called, but must be
  // defined so we can instantiate the class.
  bool Supports(const ComponentSpec &) const override { return true; }
  tensorflow::Status Initialize(const ComponentSpec &) override {
    return tensorflow::Status::OK();
  }
  tensorflow::Status Predict(Matrix<float>, InputBatchCache *) const override {
    return tensorflow::Status::OK();
  }

  // Publicizes the TermFrequencyMap accessor.
  using TermMapSequencePredictor::term_map;
};

// Returns a ComponentSpec that contains a term map resource pointing at the
// |path|.
ComponentSpec MakeSpec(const string &path) {
  ComponentSpec component_spec;
  AddTermMapResource(kResourceName, path, &component_spec);
  return component_spec;
}

// Tests that a term map can be successfully read.
TEST(TermMapSequencePredictorTest, NormalOperation) {
  const string path = WriteTermMap({{"too-infrequent", kMinFrequency - 1},
                                    {"hello", kMinFrequency},
                                    {"world", kMinFrequency + 1}});
  const ComponentSpec spec = MakeSpec(path);

  BasicTermMapSequencePredictor predictor;
  ASSERT_TRUE(predictor.SupportsTermMap(spec));
  TF_ASSERT_OK(predictor.InitializeTermMap(spec, kMinFrequency, kMaxNumTerms));

  // NB: Terms are sorted by frequency.
  EXPECT_EQ(predictor.term_map().Size(), 2);
  EXPECT_EQ(predictor.term_map().LookupIndex("hello", -1), 1);
  EXPECT_EQ(predictor.term_map().LookupIndex("world", -1), 0);
  EXPECT_EQ(predictor.term_map().LookupIndex("unknown", -1), -1);
}

// Tests that SupportsTermMap() requires a resource with the proper name.
TEST(TermMapSequencePredictorTest, ResourceName) {
  const BasicTermMapSequencePredictor predictor;

  ComponentSpec spec = MakeSpec("/dev/null");
  ASSERT_TRUE(predictor.SupportsTermMap(spec));

  spec.mutable_resource(0)->set_name("whatever");
  EXPECT_FALSE(predictor.SupportsTermMap(spec));
}

// Tests that InitializeTermMap() fails if the term map cannot be found.
TEST(TermMapSequencePredictorTest, InitializeWithNoTermMap) {
  BasicTermMapSequencePredictor predictor;

  const ComponentSpec spec;
  EXPECT_THAT(predictor.InitializeTermMap(spec, kMinFrequency, kMaxNumTerms),
              test::IsErrorWithSubstr("No compatible resource"));
}

// Tests that InitializeTermMap() requires a proper term map file.
TEST(TermMapSequencePredictorTest, InvalidPath) {
  BasicTermMapSequencePredictor predictor;

  const ComponentSpec spec = MakeSpec("/some/bad/path");
  ASSERT_TRUE(predictor.SupportsTermMap(spec));
  EXPECT_DEATH(predictor.InitializeTermMap(spec, kMinFrequency, kMaxNumTerms)
                   .IgnoreError(),
               "/some/bad/path");
}

}  // namespace
}  // namespace runtime
}  // namespace dragnn
}  // namespace syntaxnet
