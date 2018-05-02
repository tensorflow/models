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

#include "dragnn/runtime/attributes.h"

#include <map>
#include <set>
#include <string>
#include <vector>

#include "dragnn/core/test/generic.h"
#include "syntaxnet/base.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/protobuf.h"
#include "tensorflow/core/platform/test.h"

namespace syntaxnet {
namespace dragnn {
namespace runtime {
namespace {

// Returns the attribute mapping equivalent of the |std_map|.
Attributes::Mapping MakeMapping(const std::map<string, string> &std_map) {
  Attributes::Mapping mapping;
  for (const auto &it : std_map) mapping[it.first] = it.second;
  return mapping;
}

// Returns a mapping with all attributes explicitly set.
Attributes::Mapping GetFullySpecifiedMapping() {
  return MakeMapping({{"some_string", "explicit"},
                      {"some_bool", "true"},
                      {"some_int32", "987"},
                      {"some_int64", "654321"},
                      {"some_size_t", "7777777"},
                      {"some_float", "0.25"},
                      {"some_intvec", "2,3,5,7,11,13"},
                      {"some_strvec", "a,bc,def"}});
}

// A set of optional attributes.
struct OptionalAttributes : public Attributes {
  Optional<string> some_string{"some_string", "default", this};
  Optional<bool> some_bool{"some_bool", false, this};
  Optional<int32> some_int32{"some_int32", 32, this};
  Optional<int64> some_int64{"some_int64", 64, this};
  Optional<size_t> some_size_t{"some_size_t", 999, this};
  Optional<float> some_float{"some_float", -1.5, this};
  Optional<std::vector<int32>> some_intvec{"some_intvec", {}, this};
  Optional<std::vector<string>> some_strvec{"some_strvec", {"x", "y"}, this};
};

// Tests that attributes take their default values when they are not explicitly
// specified.
TEST(OptionalAttributesTest, Defaulted) {
  Attributes::Mapping mapping;
  OptionalAttributes attributes;
  TF_ASSERT_OK(attributes.Reset(mapping));
  EXPECT_EQ(attributes.some_string(), "default");
  EXPECT_FALSE(attributes.some_bool());
  EXPECT_EQ(attributes.some_int32(), 32);
  EXPECT_EQ(attributes.some_int64(), 64);
  EXPECT_EQ(attributes.some_size_t(), 999);
  EXPECT_EQ(attributes.some_float(), -1.5);
  EXPECT_EQ(attributes.some_intvec(), std::vector<int32>());
  EXPECT_EQ(attributes.some_strvec(), std::vector<string>({"x", "y"}));
}

// Tests that attributes can be overridden to explicitly-specified values.
TEST(OptionalAttributesTest, FullySpecified) {
  OptionalAttributes attributes;
  TF_ASSERT_OK(attributes.Reset(GetFullySpecifiedMapping()));
  EXPECT_EQ(attributes.some_string(), "explicit");
  EXPECT_TRUE(attributes.some_bool());
  EXPECT_EQ(attributes.some_int32(), 987);
  EXPECT_EQ(attributes.some_int64(), 654321);
  EXPECT_EQ(attributes.some_size_t(), 7777777);
  EXPECT_EQ(attributes.some_float(), 0.25);
  EXPECT_EQ(attributes.some_intvec(), std::vector<int32>({2, 3, 5, 7, 11, 13}));
  EXPECT_EQ(attributes.some_strvec(), std::vector<string>({"a", "bc", "def"}));
}

// Tests that attribute parsing fails for an unknown name.
TEST(OptionalAttributesTest, UnknownName) {
  const Attributes::Mapping mapping = MakeMapping({{"unknown", "##BAD##"}});
  OptionalAttributes attributes;
  EXPECT_THAT(attributes.Reset(mapping),
              test::IsErrorWithSubstr("Unknown attribute"));
}

// Tests that attribute parsing fails for malformed bool values.
TEST(OptionalAttributesTest, BadBool) {
  for (const string &value :
       {" true", "true ", "tr ue", "arst", "1", "t", "y", "yes", " false",
        "false ", "fa lse", "oien", "0", "f", "n", "no"}) {
    const Attributes::Mapping mapping = MakeMapping({{"some_bool", value}});
    OptionalAttributes attributes;
    EXPECT_THAT(attributes.Reset(mapping),
                test::IsErrorWithSubstr("Attribute can't be parsed as bool"));
  }
}

// Tests that attribute parsing works for well-formed bool values.
TEST(OptionalAttributesTest, GoodBool) {
  for (const string &value : {"true", "TRUE", "True", "tRuE"}) {
    const Attributes::Mapping mapping = MakeMapping({{"some_bool", value}});
    OptionalAttributes attributes;
    TF_ASSERT_OK(attributes.Reset(mapping));
    EXPECT_TRUE(attributes.some_bool());
  }

  for (const string &value : {"false", "FALSE", "False", "fAlSe"}) {
    const Attributes::Mapping mapping = MakeMapping({{"some_bool", value}});
    OptionalAttributes attributes;
    TF_ASSERT_OK(attributes.Reset(mapping));
    EXPECT_FALSE(attributes.some_bool());
  }
}

// Tests that attribute parsing fails for malformed int32 values.
TEST(OptionalAttributesTest, BadInt32) {
  for (const string &value : {"hello", "true", "1.0", "inf", "nan"}) {
    const Attributes::Mapping mapping = MakeMapping({{"some_int32", value}});
    OptionalAttributes attributes;
    EXPECT_THAT(attributes.Reset(mapping),
                test::IsErrorWithSubstr("Attribute can't be parsed as int32"));
  }
}

// Tests that attribute parsing fails for malformed int64 values.
TEST(OptionalAttributesTest, BadInt64) {
  for (const string &value : {"hello", "true", "1.0", "inf", "nan"}) {
    const Attributes::Mapping mapping = MakeMapping({{"some_int64", value}});
    OptionalAttributes attributes;
    EXPECT_THAT(attributes.Reset(mapping),
                test::IsErrorWithSubstr("Attribute can't be parsed as int64"));
  }
}

// Tests that attribute parsing fails for malformed size_t values.
TEST(OptionalAttributesTest, BadSizeT) {
  for (const string &value :
       {"hello", "true", "1.0", "inf", "nan", "-1.0", "-123"}) {
    const Attributes::Mapping mapping = MakeMapping({{"some_size_t", value}});
    OptionalAttributes attributes;
    EXPECT_THAT(attributes.Reset(mapping),
                test::IsErrorWithSubstr("Attribute can't be parsed as size_t"));
  }
}

// Tests that attribute parsing fails for malformed floats.
TEST(OptionalAttributesTest, BadFloat) {
  for (const string &value : {"hello", "true"}) {
    const Attributes::Mapping mapping = MakeMapping({{"some_float", value}});
    OptionalAttributes attributes;
    EXPECT_THAT(attributes.Reset(mapping),
                test::IsErrorWithSubstr("Attribute can't be parsed as float"));
  }
}

// Tests that attribute parsing fails for malformed std::vector<int32> values.
TEST(OptionalAttributesTest, BadIntVector) {
  for (const string &value :
       {"hello", "true", "1.0", "inf", "nan", "true,false", "foo,bar,baz"}) {
    const Attributes::Mapping mapping = MakeMapping({{"some_intvec", value}});
    OptionalAttributes attributes;
    EXPECT_THAT(attributes.Reset(mapping),
                test::IsErrorWithSubstr("Attribute can't be parsed as int32"));
  }
}

// A set of mandatory attributes.
struct MandatoryAttributes : public Attributes {
  Mandatory<string> some_string{"some_string", this};
  Mandatory<bool> some_bool{"some_bool", this};
  Mandatory<int32> some_int32{"some_int32", this};
  Mandatory<int64> some_int64{"some_int64", this};
  Mandatory<size_t> some_size_t{"some_size_t", this};
  Mandatory<float> some_float{"some_float", this};
  Mandatory<std::vector<int32>> some_intvec{"some_intvec", this};
  Mandatory<std::vector<string>> some_strvec{"some_strvec", this};
};

// Tests that attribute parsing works when all mandatory attributes are
// explicitly specified.
TEST(MandatoryAttributesTest, FullySpecified) {
  MandatoryAttributes attributes;
  TF_ASSERT_OK(attributes.Reset(GetFullySpecifiedMapping()));
  EXPECT_EQ(attributes.some_string(), "explicit");
  EXPECT_TRUE(attributes.some_bool());
  EXPECT_EQ(attributes.some_int32(), 987);
  EXPECT_EQ(attributes.some_int64(), 654321);
  EXPECT_EQ(attributes.some_size_t(), 7777777);
  EXPECT_EQ(attributes.some_float(), 0.25);
  EXPECT_EQ(attributes.some_intvec(), std::vector<int32>({2, 3, 5, 7, 11, 13}));
  EXPECT_EQ(attributes.some_strvec(), std::vector<string>({"a", "bc", "def"}));
}

// Tests that attribute parsing fails when even one mandatory attribute is not
// explicitly specified.
TEST(MandatoryAttributesTest, MissingAttribute) {
  for (const auto &it : GetFullySpecifiedMapping()) {
    const string &name = it.first;
    Attributes::Mapping mapping = GetFullySpecifiedMapping();
    CHECK_EQ(mapping.erase(name), 1);

    MandatoryAttributes attributes;
    EXPECT_THAT(attributes.Reset(mapping),
                test::IsErrorWithSubstr("Missing mandatory attributes"));
  }
}

// A set of ignored attributes.
struct IgnoredAttributes : public Attributes {
  Ignored foo{"foo", this};
  Ignored bar{"bar", this};
  Ignored baz{"baz", this};
};

// Tests that ignored attributes are not mandatory.
TEST(IgnoredAttributesTest, NotMandatory) {
  const Attributes::Mapping mapping;
  IgnoredAttributes attributes;
  TF_ASSERT_OK(attributes.Reset(mapping));
}

// Tests that attribute parsing consumes ignored names.
TEST(IgnoredAttributesTest, IgnoredName) {
  const Attributes::Mapping mapping =
      MakeMapping({{"foo", "blah"}, {"bar", "123"}, {"baz", "   "}});
  IgnoredAttributes attributes;
  TF_ASSERT_OK(attributes.Reset(mapping));
}

// Tests that attribute parsing still fails for unknown names.
TEST(IgnoredAttributesTest, UnknownName) {
  const Attributes::Mapping mapping = MakeMapping(
      {{"foo", "blah"}, {"bar", "123"}, {"baz", "   "}, {"unknown", ""}});
  IgnoredAttributes attributes;
  EXPECT_THAT(attributes.Reset(mapping),
              test::IsErrorWithSubstr("Unknown attribute"));
}

}  // namespace
}  // namespace runtime
}  // namespace dragnn
}  // namespace syntaxnet
