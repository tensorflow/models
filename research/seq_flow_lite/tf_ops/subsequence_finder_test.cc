/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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
#include "tf_ops/subsequence_finder.h"  // seq_flow_lite

#include <gmock/gmock.h>
#include <gtest/gtest.h>

namespace seq_flow_lite {
namespace {

using ::testing::UnorderedElementsAre;

TEST(SubsequenceFinderTest, SingleExists) {
  SubsequenceFinder subsequence_finder(3);
  subsequence_finder.AddSubsequence("ab cd", 0);

  EXPECT_THAT(subsequence_finder.FindSubsequences("abcd"),
              UnorderedElementsAre(0));

  EXPECT_THAT(subsequence_finder.FindSubsequences("ab012cd"),
              UnorderedElementsAre(0));

  EXPECT_THAT(subsequence_finder.FindSubsequences("AB CD"),
              UnorderedElementsAre(0));
}

TEST(SubsequenceFinderTest, SingleNotExists) {
  SubsequenceFinder subsequence_finder(3);
  subsequence_finder.AddSubsequence("ab cd", 0);

  EXPECT_THAT(subsequence_finder.FindSubsequences("a bcd"),
              UnorderedElementsAre());

  EXPECT_THAT(subsequence_finder.FindSubsequences("ab0123cd"),
              UnorderedElementsAre());

  EXPECT_THAT(subsequence_finder.FindSubsequences("abdc"),
              UnorderedElementsAre());
}

TEST(SubsequenceFinderTest, Multiple) {
  SubsequenceFinder subsequence_finder(3);
  subsequence_finder.AddSubsequence("a b c d", 0);
  subsequence_finder.AddSubsequence("q r s", 2);
  subsequence_finder.AddSubsequence("b c d e", 4);

  EXPECT_THAT(subsequence_finder.FindSubsequences("a__b__c__d__e"),
              UnorderedElementsAre(0, 4));

  EXPECT_THAT(subsequence_finder.FindSubsequences("aqbrcsd"),
              UnorderedElementsAre(0, 2));

  EXPECT_THAT(subsequence_finder.FindSubsequences("b q c r d s e"),
              UnorderedElementsAre(2, 4));
}

TEST(SubsequenceFinderTest, Utf8) {
  SubsequenceFinder subsequence_finder(3);
  subsequence_finder.AddSubsequence("一二 三四 五六", 0);

  EXPECT_THAT(subsequence_finder.FindSubsequences("一二おはよ三四こんに五六"),
              UnorderedElementsAre(0));

  EXPECT_THAT(subsequence_finder.FindSubsequences("一二三 四五六"),
              UnorderedElementsAre());
}

}  // namespace
}  // namespace seq_flow_lite
