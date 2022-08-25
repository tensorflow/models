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
#include <cstdint>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/framework/types.proto.h"
#include "tensorflow/core/kernels/ops_testutil.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status_test_util.h"

namespace seq_flow_lite {
namespace {

using ::tensorflow::DT_FLOAT;
using ::tensorflow::DT_INT32;
using ::tensorflow::DT_INT64;
using ::tensorflow::DT_STRING;
using ::tensorflow::NodeDefBuilder;
using ::tensorflow::OpsTestBase;
using ::tensorflow::Tensor;
using ::tensorflow::TensorShape;
using ::tensorflow::errors::InvalidArgument;
using ::tensorflow::test::ExpectTensorEqual;
using ::tensorflow::test::FillValues;

class SkipgramDenylistOpTest : public OpsTestBase {};

TEST_F(SkipgramDenylistOpTest, Correct) {
  TF_ASSERT_OK(NodeDefBuilder("test_op", "SkipgramDenylist")
                   .Input({"input", 0, DT_STRING})
                   .Attr("max_skip_size", 1)
                   .Attr("denylist", {"a b c"})
                   .Attr("denylist_category", {1})
                   .Attr("categories", 2)
                   .Attr("negative_categories", 1)
                   .Finalize(node_def()));
  TF_ASSERT_OK(InitOp());

  AddInputFromArray<::tensorflow::tstring>(TensorShape({2}),
                                           {"q a q b q c q", "q a b q q c"});

  TF_ASSERT_OK(RunOpKernel());
  const Tensor& output = *GetOutput(0);

  Tensor expected(allocator(), DT_FLOAT, TensorShape({2, 2}));
  FillValues<float>(&expected, {0.0, 1.0, 1.0, 0.0});
  ExpectTensorEqual<float>(expected, output);
}

TEST_F(SkipgramDenylistOpTest, Prefix) {
  TF_ASSERT_OK(NodeDefBuilder("test_op", "SkipgramDenylist")
                   .Input({"input", 0, DT_STRING})
                   .Attr("max_skip_size", 1)
                   .Attr("denylist", {"a b.* c"})
                   .Attr("denylist_category", {1})
                   .Attr("categories", 2)
                   .Attr("negative_categories", 1)
                   .Finalize(node_def()));
  TF_ASSERT_OK(InitOp());

  AddInputFromArray<::tensorflow::tstring>(TensorShape({2}),
                                           {"q a q bq q c q", "q a bq q q c"});

  TF_ASSERT_OK(RunOpKernel());
  const Tensor& output = *GetOutput(0);

  Tensor expected(allocator(), DT_FLOAT, TensorShape({2, 2}));
  FillValues<float>(&expected, {0.0, 1.0, 1.0, 0.0});
  ExpectTensorEqual<float>(expected, output);
}

TEST_F(SkipgramDenylistOpTest, ZeroCategories) {
  TF_ASSERT_OK(NodeDefBuilder("test_op", "SkipgramDenylist")
                   .Input({"input", 0, DT_STRING})
                   .Attr("max_skip_size", 1)
                   .Attr("denylist", {"a b c"})
                   .Attr("denylist_category", {1})
                   .Attr("categories", 0)
                   .Attr("negative_categories", 0)
                   .Finalize(node_def()));
  EXPECT_EQ(InitOp(),
            InvalidArgument("Number of categories (0) must be positive."));
}

TEST_F(SkipgramDenylistOpTest, NegativeCategoriesLessThanZero) {
  TF_ASSERT_OK(NodeDefBuilder("test_op", "SkipgramDenylist")
                   .Input({"input", 0, DT_STRING})
                   .Attr("max_skip_size", 1)
                   .Attr("denylist", {"a b c"})
                   .Attr("denylist_category", {1})
                   .Attr("categories", 1)
                   .Attr("negative_categories", -1)
                   .Finalize(node_def()));
  EXPECT_EQ(InitOp(),
            InvalidArgument(
                "Number of negative_categories (-1) must be non-negative."));
}

TEST_F(SkipgramDenylistOpTest, CategoriesEqualNegativeCategories) {
  TF_ASSERT_OK(NodeDefBuilder("test_op", "SkipgramDenylist")
                   .Input({"input", 0, DT_STRING})
                   .Attr("max_skip_size", 1)
                   .Attr("denylist", {"a b c"})
                   .Attr("denylist_category", {1})
                   .Attr("categories", 1)
                   .Attr("negative_categories", 1)
                   .Finalize(node_def()));
  EXPECT_EQ(InitOp(),
            InvalidArgument("Number of categories (1) must be greater than the "
                            "number of negative_categories (1)."));
}

class SubsequenceDenylistOpTest : public OpsTestBase {};

TEST_F(SubsequenceDenylistOpTest, Correct) {
  TF_ASSERT_OK(NodeDefBuilder("test_op", "SubsequenceDenylist")
                   .Input({"input", 0, DT_STRING})
                   .Attr("max_skip_size", 1)
                   .Attr("denylist", {"a b c"})
                   .Attr("denylist_category", {1})
                   .Attr("categories", 2)
                   .Attr("negative_categories", 1)
                   .Finalize(node_def()));
  TF_ASSERT_OK(InitOp());

  AddInputFromArray<::tensorflow::tstring>(TensorShape({2}),
                                           {"qaqbqcq", "qabqqc"});

  TF_ASSERT_OK(RunOpKernel());
  const Tensor& output = *GetOutput(0);

  Tensor expected(allocator(), DT_FLOAT, TensorShape({2, 2}));
  FillValues<float>(&expected, {0.0, 1.0, 1.0, 0.0});
  ExpectTensorEqual<float>(expected, output);
}

TEST_F(SubsequenceDenylistOpTest, ZeroCategories) {
  TF_ASSERT_OK(NodeDefBuilder("test_op", "SubsequenceDenylist")
                   .Input({"input", 0, DT_STRING})
                   .Attr("max_skip_size", 1)
                   .Attr("denylist", {"a b c"})
                   .Attr("denylist_category", {1})
                   .Attr("categories", 0)
                   .Attr("negative_categories", 0)
                   .Finalize(node_def()));
  EXPECT_EQ(InitOp(),
            InvalidArgument("Number of categories (0) must be positive."));
}

TEST_F(SubsequenceDenylistOpTest, NegativeCategoriesLessThanZero) {
  TF_ASSERT_OK(NodeDefBuilder("test_op", "SubsequenceDenylist")
                   .Input({"input", 0, DT_STRING})
                   .Attr("max_skip_size", 1)
                   .Attr("denylist", {"a b c"})
                   .Attr("denylist_category", {1})
                   .Attr("categories", 1)
                   .Attr("negative_categories", -1)
                   .Finalize(node_def()));
  EXPECT_EQ(InitOp(),
            InvalidArgument(
                "Number of negative_categories (-1) must be non-negative."));
}

TEST_F(SubsequenceDenylistOpTest, CategoriesEqualNegativeCategories) {
  TF_ASSERT_OK(NodeDefBuilder("test_op", "SubsequenceDenylist")
                   .Input({"input", 0, DT_STRING})
                   .Attr("max_skip_size", 1)
                   .Attr("denylist", {"a b c"})
                   .Attr("denylist_category", {1})
                   .Attr("categories", 1)
                   .Attr("negative_categories", 1)
                   .Finalize(node_def()));
  EXPECT_EQ(InitOp(),
            InvalidArgument("Number of categories (1) must be greater than the "
                            "number of negative_categories (1)."));
}

class TokenizedDenylistOpTest : public OpsTestBase {};

TEST_F(TokenizedDenylistOpTest, CorrectInt64TokenCount) {
  TF_ASSERT_OK(NodeDefBuilder("test_op", "TokenizedDenylist")
                   .Input({"input", 0, DT_STRING})
                   .Input({"token_count", 0, DT_INT64})
                   .Attr("max_skip_size", 1)
                   .Attr("denylist", {"a b c"})
                   .Attr("denylist_category", {1})
                   .Attr("categories", 2)
                   .Attr("negative_categories", 1)
                   .Finalize(node_def()));
  TF_ASSERT_OK(InitOp());

  AddInputFromArray<::tensorflow::tstring>(
      TensorShape({2, 7}), {"q", "a", "q", "b", "q", "c", "q",  //
                            "q", "a", "b", "q", "q", "c", ""});
  AddInputFromArray<int64_t>(TensorShape({2}), {7, 6});

  TF_ASSERT_OK(RunOpKernel());
  const Tensor& output = *GetOutput(0);

  Tensor expected(allocator(), DT_FLOAT, TensorShape({2, 2}));
  FillValues<float>(&expected, {0.0, 1.0, 1.0, 0.0});
  ExpectTensorEqual<float>(expected, output);
}

TEST_F(TokenizedDenylistOpTest, CorrectInt32TokenCount) {
  TF_ASSERT_OK(NodeDefBuilder("test_op", "TokenizedDenylist")
                   .Input({"input", 0, DT_STRING})
                   .Input({"token_count", 0, DT_INT32})
                   .Attr("max_skip_size", 1)
                   .Attr("denylist", {"a b c"})
                   .Attr("denylist_category", {1})
                   .Attr("categories", 2)
                   .Attr("negative_categories", 1)
                   .Finalize(node_def()));
  TF_ASSERT_OK(InitOp());

  AddInputFromArray<::tensorflow::tstring>(
      TensorShape({2, 7}), {"q", "a", "q", "b", "q", "c", "q",  //
                            "q", "a", "b", "q", "q", "c", ""});
  AddInputFromArray<int32_t>(TensorShape({2}), {7, 6});

  TF_ASSERT_OK(RunOpKernel());
  const Tensor& output = *GetOutput(0);

  Tensor expected(allocator(), DT_FLOAT, TensorShape({2, 2}));
  FillValues<float>(&expected, {0.0, 1.0, 1.0, 0.0});
  ExpectTensorEqual<float>(expected, output);
}

TEST_F(TokenizedDenylistOpTest, ZeroCategories) {
  TF_ASSERT_OK(NodeDefBuilder("test_op", "TokenizedDenylist")
                   .Input({"input", 0, DT_STRING})
                   .Input({"token_count", 0, DT_INT64})
                   .Attr("max_skip_size", 1)
                   .Attr("denylist", {"a b c"})
                   .Attr("denylist_category", {1})
                   .Attr("categories", 0)
                   .Attr("negative_categories", 0)
                   .Finalize(node_def()));
  EXPECT_EQ(InitOp(),
            InvalidArgument("Number of categories (0) must be positive."));
}

TEST_F(TokenizedDenylistOpTest, NegativeCategoriesLessThanZero) {
  TF_ASSERT_OK(NodeDefBuilder("test_op", "TokenizedDenylist")
                   .Input({"input", 0, DT_STRING})
                   .Input({"token_count", 0, DT_INT64})
                   .Attr("max_skip_size", 1)
                   .Attr("denylist", {"a b c"})
                   .Attr("denylist_category", {1})
                   .Attr("categories", 1)
                   .Attr("negative_categories", -1)
                   .Finalize(node_def()));
  EXPECT_EQ(InitOp(),
            InvalidArgument(
                "Number of negative_categories (-1) must be non-negative."));
}

TEST_F(TokenizedDenylistOpTest, CategoriesEqualNegativeCategories) {
  TF_ASSERT_OK(NodeDefBuilder("test_op", "TokenizedDenylist")
                   .Input({"input", 0, DT_STRING})
                   .Input({"token_count", 0, DT_INT64})
                   .Attr("max_skip_size", 1)
                   .Attr("denylist", {"a b c"})
                   .Attr("denylist_category", {1})
                   .Attr("categories", 1)
                   .Attr("negative_categories", 1)
                   .Finalize(node_def()));
  EXPECT_EQ(InitOp(),
            InvalidArgument("Number of categories (1) must be greater than the "
                            "number of negative_categories (1)."));
}

}  // namespace
}  // namespace seq_flow_lite
