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

#include "dragnn/runtime/term_map_utils.h"

#include <string>

#include "dragnn/core/test/generic.h"
#include "dragnn/protos/spec.pb.h"
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
constexpr char kResourcePath[] = "/path/to/term-map";

// Returns a ComponentSpec with a term map resource named |kResourceName| that
// points at |kResourcePath|.
ComponentSpec MakeSpec() {
  ComponentSpec spec;
  AddTermMapResource(kResourceName, kResourcePath, &spec);
  return spec;
}

// Tests that a term map resource can be successfully read.
TEST(LookupTermMapResourcePathTest, Success) {
  const ComponentSpec spec = MakeSpec();

  const string *path = LookupTermMapResourcePath(kResourceName, spec);
  ASSERT_NE(path, nullptr);
  EXPECT_EQ(*path, kResourcePath);
}

// Tests that the returned path is null for an empty spec.
TEST(LookupTermMapResourcePathTest, EmptySpec) {
  const ComponentSpec spec;

  EXPECT_EQ(LookupTermMapResourcePath(kResourceName, spec), nullptr);
}

// Tests that the returned path is null for the wrong resource name.
TEST(LookupTermMapResourcePathTest, WrongName) {
  ComponentSpec spec = MakeSpec();

  spec.mutable_resource(0)->set_name("bad");
  EXPECT_EQ(LookupTermMapResourcePath(kResourceName, spec), nullptr);
}

// Tests that the returned path is null for the wrong number of parts.
TEST(LookupTermMapResourcePathTest, WrongNumberOfParts) {
  ComponentSpec spec = MakeSpec();

  spec.mutable_resource(0)->clear_part();
  EXPECT_EQ(LookupTermMapResourcePath(kResourceName, spec), nullptr);

  spec.mutable_resource(0)->add_part();
  spec.mutable_resource(0)->add_part();
  EXPECT_EQ(LookupTermMapResourcePath(kResourceName, spec), nullptr);
}

// Tests that the returned path is null for the wrong file format.
TEST(LookupTermMapResourcePathTest, WrongFileFormat) {
  ComponentSpec spec = MakeSpec();

  spec.mutable_resource(0)->mutable_part(0)->set_file_format("bad");
  EXPECT_EQ(LookupTermMapResourcePath(kResourceName, spec), nullptr);
}

// Tests that the returned path is null for the wrong record format.
TEST(LookupTermMapResourcePathTest, WrongRecordFormat) {
  ComponentSpec spec = MakeSpec();

  spec.mutable_resource(0)->mutable_part(0)->set_record_format("bad");
  EXPECT_EQ(LookupTermMapResourcePath(kResourceName, spec), nullptr);
}

// Tests that alternate record formats are accepted.
TEST(LookupTermMapResourcePathTest, SuccessWithAlternateRecordFormat) {
  ComponentSpec spec = MakeSpec();

  spec.mutable_resource(0)->mutable_part(0)->set_record_format(
      "TermFrequencyMap");
  const string *path = LookupTermMapResourcePath(kResourceName, spec);
  ASSERT_NE(path, nullptr);
  EXPECT_EQ(*path, kResourcePath);
}

// Tests that ParseTermMapFml() correctly parses term map feature options.
TEST(ParseTermMapFmlTest, Success) {
  int min_frequency = -1;
  int max_num_terms = -1;

  TF_ASSERT_OK(ParseTermMapFml("path.to.foo", {"path", "to", "foo"},
                               &min_frequency, &max_num_terms));
  EXPECT_EQ(min_frequency, 0);
  EXPECT_EQ(max_num_terms, 0);

  TF_ASSERT_OK(ParseTermMapFml("path.to.foo(min-freq=5)", {"path", "to", "foo"},
                               &min_frequency, &max_num_terms));
  EXPECT_EQ(min_frequency, 5);
  EXPECT_EQ(max_num_terms, 0);

  TF_ASSERT_OK(ParseTermMapFml("path.to.foo(max-num-terms=1000)",
                               {"path", "to", "foo"}, &min_frequency,
                               &max_num_terms));
  EXPECT_EQ(min_frequency, 0);
  EXPECT_EQ(max_num_terms, 1000);

  TF_ASSERT_OK(ParseTermMapFml("path.to.foo(min-freq=12,max-num-terms=3456)",
                               {"path", "to", "foo"}, &min_frequency,
                               &max_num_terms));
  EXPECT_EQ(min_frequency, 12);
  EXPECT_EQ(max_num_terms, 3456);
}

// Tests that ParseTermMapFml() tolerates a zero argument.
TEST(ParseTermMapFmlTest, SuccessWithZeroArgument) {
  int min_frequency = -1;
  int max_num_terms = -1;

  TF_ASSERT_OK(ParseTermMapFml("path.to.foo(0)", {"path", "to", "foo"},
                               &min_frequency, &max_num_terms));
  EXPECT_EQ(min_frequency, 0);
  EXPECT_EQ(max_num_terms, 0);

  TF_ASSERT_OK(ParseTermMapFml("path.to.foo(0,min-freq=5)",
                               {"path", "to", "foo"}, &min_frequency,
                               &max_num_terms));
  EXPECT_EQ(min_frequency, 5);
  EXPECT_EQ(max_num_terms, 0);

  TF_ASSERT_OK(ParseTermMapFml("path.to.foo(0,max-num-terms=1000)",
                               {"path", "to", "foo"}, &min_frequency,
                               &max_num_terms));
  EXPECT_EQ(min_frequency, 0);
  EXPECT_EQ(max_num_terms, 1000);

  TF_ASSERT_OK(ParseTermMapFml("path.to.foo(0,min-freq=12,max-num-terms=3456)",
                               {"path", "to", "foo"}, &min_frequency,
                               &max_num_terms));
  EXPECT_EQ(min_frequency, 12);
  EXPECT_EQ(max_num_terms, 3456);
}

// Tests that ParseTermMapFml() fails on a non-zero argument.
TEST(ParseTermMapFmlTest, NonZeroArgument) {
  int min_frequency = -1;
  int max_num_terms = -1;

  EXPECT_THAT(ParseTermMapFml("path.to.foo(1)", {"path", "to", "foo"},
                              &min_frequency, &max_num_terms),
              test::IsErrorWithSubstr(
                  "TermFrequencyMap-based feature should have no argument"));
  EXPECT_EQ(min_frequency, -1);
  EXPECT_EQ(max_num_terms, -1);
}

// Tests that ParseTermMapFml() fails on an unknown feature option.
TEST(ParseTermMapFmlTest, UnknownOption) {
  int min_frequency = -1;
  int max_num_terms = -1;

  EXPECT_THAT(ParseTermMapFml("path.to.foo(unknown=1)", {"path", "to", "foo"},
                              &min_frequency, &max_num_terms),
              test::IsErrorWithSubstr("Unknown attribute"));
  EXPECT_EQ(min_frequency, -1);
  EXPECT_EQ(max_num_terms, -1);
}

}  // namespace
}  // namespace runtime
}  // namespace dragnn
}  // namespace syntaxnet
