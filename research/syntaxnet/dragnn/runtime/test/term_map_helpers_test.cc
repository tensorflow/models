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

#include "dragnn/runtime/test/term_map_helpers.h"

#include <map>
#include <string>

#include "dragnn/core/test/generic.h"
#include "dragnn/protos/spec.pb.h"
#include "syntaxnet/base.h"
#include "syntaxnet/term_frequency_map.h"
#include "tensorflow/core/platform/test.h"

namespace syntaxnet {
namespace dragnn {
namespace runtime {
namespace {

// Tests that a term map can be successfully written and read.
TEST(TermMapHelpersTest, WriteTermMap) {
  const string path = WriteTermMap({{"hello", 1}, {"world", 2}});
  TermFrequencyMap term_map(path, 0, 0);

  // Terms are sorted by descending frequency, so "world" has index 0.
  EXPECT_EQ(term_map.Size(), 2);
  EXPECT_EQ(term_map.LookupIndex("hello", -1), 1);
  EXPECT_EQ(term_map.LookupIndex("world", -1), 0);
  EXPECT_EQ(term_map.LookupIndex("unknown", -1), -1);
}

// Tests that a term map resource can be added to a ComponentSpec.
TEST(TermMapHelpersTest, AddTermMapResource) {
  ComponentSpec component_spec;
  AddTermMapResource("foo-map", "/foo/bar/baz", &component_spec);

  ComponentSpec expected_spec;
  CHECK(TextFormat::ParseFromString(
      "resource { name:'foo-map' "
      "part { file_format:'text' file_pattern:'/foo/bar/baz' } }",
      &expected_spec));

  EXPECT_THAT(component_spec, test::EqualsProto(expected_spec));
}

}  // namespace
}  // namespace runtime
}  // namespace dragnn
}  // namespace syntaxnet
