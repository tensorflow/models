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

#include "syntaxnet/fml_parser.h"

#include <string>
#include <vector>

#include "syntaxnet/base.h"
#include "syntaxnet/feature_extractor.pb.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/platform/test.h"

namespace syntaxnet {
namespace {

// Returns the list of lines in the |text|.  Also strips trailing whitespace
// from each line, since the FML generator sometimes appends trailing spaces.
std::vector<string> LinesOf(const string &text) {
  std::vector<string> lines = tensorflow::str_util::Split(
      text, "\n", tensorflow::str_util::SkipEmpty());
  for (string &line : lines) {
    tensorflow::str_util::StripTrailingWhitespace(&line);
  }
  return lines;
}

// Tests that a single function can be round-trip converted from FML to
// descriptor protos and back to FML.
TEST(FMLParserTest, RoundTripSingleFunction) {
  FeatureExtractorDescriptor extractor;
  FMLParser().Parse("offset(1).input.token.word(min-freq=10)", &extractor);

  EXPECT_EQ(LinesOf(AsFML(extractor)),
            LinesOf("offset(1).input.token.word(min-freq=\"10\")"));

  // Also check each individual feature function.
  EXPECT_EQ(AsFML(extractor.feature(0)),
            "offset(1).input.token.word(min-freq=\"10\")");
  EXPECT_EQ(AsFML(extractor.feature(0).feature(0)),
            "input.token.word(min-freq=\"10\")");
  EXPECT_EQ(AsFML(extractor.feature(0).feature(0).feature(0)),
            "token.word(min-freq=\"10\")");
  EXPECT_EQ(AsFML(extractor.feature(0).feature(0).feature(0).feature(0)),
            "word(min-freq=\"10\")");
}

// Tests that a set of functions can be round-trip converted from FML to
// descriptor protos and back to FML.
TEST(FMLParserTest, RoundTripMultipleFunctions) {
  FeatureExtractorDescriptor extractor;
  FMLParser().Parse(R"(offset(1).word(max-num-terms=987)
                       input { tag(outside=false) label }
                       pairs { stack.tag input.tag input.child(-1).label })",
                    &extractor);

  // Note that AsFML() adds quotes to all feature option values.
  EXPECT_EQ(LinesOf(AsFML(extractor)),
            LinesOf("offset(1).word(max-num-terms=\"987\")\n"
                    "input { tag(outside=\"false\") label }\n"
                    "pairs { stack.tag input.tag input.child(-1).label }"));
}

}  // namespace
}  // namespace syntaxnet
