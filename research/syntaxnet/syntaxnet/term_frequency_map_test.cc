/* Copyright 2017 Google Inc. All Rights Reserved.

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

#include "syntaxnet/term_frequency_map.h"

#include "syntaxnet/base.h"
#include <gmock/gmock.h>
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/test.h"

namespace syntaxnet {
namespace {

// Matches an error status whose message matches |substr|.
MATCHER(IsError, string(negation ? "isn't" : "is") + " an error Status") {
  return !arg.ok();
}

// Matches an error status whose message matches |substr|.
MATCHER_P(IsErrorWithSubstr, substr,
          string(negation ? "isn't" : "is") +
          " an error Status whose message matches the substring '" +
          ::testing::PrintToString(substr) + "'") {
  return !arg.ok() && arg.error_message().find(substr) != string::npos;
}

// Writes the |content| to a temporary file and returns its path.
string AsTempFile(const string &content) {
  static int counter = 0;
  const string basename = tensorflow::strings::StrCat("temp_", counter++);
  const string path =
      tensorflow::io::JoinPath(tensorflow::testing::TmpDir(), basename);
  TF_CHECK_OK(
      tensorflow::WriteStringToFile(tensorflow::Env::Default(), path, content));
  return path;
}

// Tests that TermFrequencyMap::TryLoad() fails on an invalid path.
TEST(TermFrequencyMapTest, TryLoadInvalidPath) {
  const string kInvalidPath = "/some/invalid/path";

  TermFrequencyMap term_map;
  EXPECT_THAT(term_map.TryLoad(kInvalidPath, 0, 0), IsError());
}

// Tests that TermFrequencyMap::TryLoad() fails on an empty file.
TEST(TermFrequencyMapTest, TryLoadEmptyFile) {
  const string path = AsTempFile("");

  TermFrequencyMap term_map;
  EXPECT_THAT(term_map.TryLoad(path, 0, 0), IsError());
}

// Tests that TermFrequencyMap::TryLoad() fails if the term count in the first
// line is not parsable as an integer.
TEST(TermFrequencyMapTest, TryLoadFileWithMalformedCount) {
  const string path = AsTempFile("asdf");

  TermFrequencyMap term_map;
  EXPECT_THAT(term_map.TryLoad(path, 0, 0),
              IsErrorWithSubstr(tensorflow::strings::StrCat(
                  path, ":0: Unable to parse term map size")));
}

// Tests that TermFrequencyMap::TryLoad() fails if the term count in the first
// line is negative.
TEST(TermFrequencyMapTest, TryLoadFileWithNegativeCount) {
  const string path = AsTempFile("-1");

  TermFrequencyMap term_map;
  EXPECT_THAT(term_map.TryLoad(path, 0, 0),
              IsErrorWithSubstr(tensorflow::strings::StrCat(
                  path, ":0: Invalid term map size: -1")));
}

// Tests that TermFrequencyMap::TryLoad() is OK if there are no terms.
TEST(TermFrequencyMapTest, TryLoadFileWithNoTerms) {
  const string path = AsTempFile("0");

  TermFrequencyMap term_map;
  TF_ASSERT_OK(term_map.TryLoad(path, 0, 0));

  EXPECT_EQ(term_map.Size(), 0);
}

// Tests that TermFrequencyMap::TryLoad() fails if there is a malformed line.
TEST(TermFrequencyMapTest, TryLoadFileWithMalformedLine) {
  const string path = AsTempFile(
      "2\n"
      "valid term with spaces 1\n"
      "bad term\n");

  TermFrequencyMap term_map;
  EXPECT_THAT(
      term_map.TryLoad(path, 0, 0),
      IsErrorWithSubstr(tensorflow::strings::StrCat(
          path, ":2: Couldn't split term and frequency in line: bad term")));
}

// Tests that TermFrequencyMap::TryLoad() fails if there is an empty term.
TEST(TermFrequencyMapTest, TryLoadFileWithEmptyTerm) {
  const string path = AsTempFile(
      "2\n"
      " 1\n"
      "some_term 1\n");

  TermFrequencyMap term_map;
  EXPECT_THAT(term_map.TryLoad(path, 0, 0),
              IsErrorWithSubstr(
                  tensorflow::strings::StrCat(path, ":1: Invalid empty term")));
}

// Tests that TermFrequencyMap::TryLoad() fails if there is a term with zero
// frequency.
TEST(TermFrequencyMapTest, TryLoadFileWithZeroFrequency) {
  const string path = AsTempFile(
      "2\n"
      "good_term 1\n"
      "bad_term 0\n");

  TermFrequencyMap term_map;
  EXPECT_THAT(term_map.TryLoad(path, 0, 0),
              IsErrorWithSubstr(tensorflow::strings::StrCat(
                  path, ":2: Invalid frequency: term=bad_term frequency=0")));
}

// Tests that TermFrequencyMap::TryLoad() fails if terms are not in descending
// order of frequency.
TEST(TermFrequencyMapTest, TryLoadFileWithOutOfOrderTerms) {
  const string path = AsTempFile(
      "2\n"
      "good_term 1\n"
      "bad_term 2\n");

  TermFrequencyMap term_map;
  EXPECT_THAT(
      term_map.TryLoad(path, 0, 0),
      IsErrorWithSubstr(tensorflow::strings::StrCat(
          path, ":2: Non-descending frequencies: current=2 previous=1")));
}

// Tests that TermFrequencyMap::TryLoad() fails if there are duplicate terms.
TEST(TermFrequencyMapTest, TryLoadFileWithDuplicateTerms) {
  const string path = AsTempFile(
      "2\n"
      "duplicate 1\n"
      "duplicate 1\n");

  TermFrequencyMap term_map;
  EXPECT_THAT(term_map.TryLoad(path, 0, 0),
              IsErrorWithSubstr(tensorflow::strings::StrCat(
                  path, ":2: Duplicate term: duplicate")));
}

// Tests that TermFrequencyMap contains the specified terms and frequencies.
TEST(TermFrequencyMapTest, LoadAndCheckContents) {
  const string path = AsTempFile(
      "3\n"
      "foo 100\n"
      "bar 10\n"
      "baz 1\n");

  TermFrequencyMap term_map;
  TF_ASSERT_OK(term_map.TryLoad(path, 0, 0));

  EXPECT_EQ(term_map.Size(), 3);
  EXPECT_EQ(term_map.GetTerm(0), "foo");
  EXPECT_EQ(term_map.GetTerm(1), "bar");
  EXPECT_EQ(term_map.GetTerm(2), "baz");
  EXPECT_EQ(term_map.GetFrequency(0), 100);
  EXPECT_EQ(term_map.GetFrequency(1), 10);
  EXPECT_EQ(term_map.GetFrequency(2), 1);
}

}  // namespace
}  // namespace syntaxnet
