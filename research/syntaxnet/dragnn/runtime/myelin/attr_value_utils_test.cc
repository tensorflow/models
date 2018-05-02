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

// NB: These tests don't assert on dtypes, shapes, or tensors, because those are
// just calls to TF library functions.  (I.e., don't test someone else's API).

#include "dragnn/runtime/myelin/attr_value_utils.h"

#include <string>

#include "dragnn/core/test/generic.h"
#include "syntaxnet/base.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/lib/strings/numbers.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/test.h"

namespace syntaxnet {
namespace dragnn {
namespace runtime {
namespace {

// Tests that singular attributes are stringified correctly.
TEST(AttrValueToStringTest, Singular) {
  {
    tensorflow::AttrValue attr_value;
    attr_value.set_s("foo");
    EXPECT_EQ(AttrValueToString(attr_value), "\"foo\"");
  }

  {
    tensorflow::AttrValue attr_value;
    attr_value.set_i(123);
    EXPECT_EQ(AttrValueToString(attr_value), "123");
  }

  {
    tensorflow::AttrValue attr_value;
    attr_value.set_f(-1.5);
    EXPECT_EQ(AttrValueToString(attr_value), "-1.5");
  }

  {
    tensorflow::AttrValue attr_value;
    attr_value.set_b(false);
    EXPECT_EQ(AttrValueToString(attr_value), "false");
    attr_value.set_b(true);
    EXPECT_EQ(AttrValueToString(attr_value), "true");
  }
}

// Tests that list attributes are stringified correctly.
TEST(AttrValueToStringTest, List) {
  {
    tensorflow::AttrValue attr_value;
    attr_value.mutable_list()->add_s("foo");
    attr_value.mutable_list()->add_s("bar");
    attr_value.mutable_list()->add_s("baz");
    EXPECT_EQ(AttrValueToString(attr_value), "[\"foo\", \"bar\", \"baz\"]");
  }

  {
    tensorflow::AttrValue attr_value;
    attr_value.mutable_list()->add_i(123);
    attr_value.mutable_list()->add_i(-45);
    attr_value.mutable_list()->add_i(6789);
    EXPECT_EQ(AttrValueToString(attr_value), "[123, -45, 6789]");
  }

  {
    tensorflow::AttrValue attr_value;
    attr_value.mutable_list()->add_f(-1.5);
    attr_value.mutable_list()->add_f(0.25);
    attr_value.mutable_list()->add_f(3.5);
    EXPECT_EQ(AttrValueToString(attr_value), "[-1.5, 0.25, 3.5]");
  }

  {
    tensorflow::AttrValue attr_value;
    attr_value.mutable_list()->add_b(false);
    attr_value.mutable_list()->add_b(true);
    attr_value.mutable_list()->add_b(false);
    EXPECT_EQ(AttrValueToString(attr_value), "[false, true, false]");
  }
}

}  // namespace
}  // namespace runtime
}  // namespace dragnn
}  // namespace syntaxnet
