/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/shape_inference_testutil.h"
#include "tensorflow/core/kernels/ops_testutil.h"
#include "tensorflow/core/kernels/ops_util.h"

namespace {

using ::tensorflow::DT_INT32;
using ::tensorflow::DT_STRING;
using ::tensorflow::int32;
using ::tensorflow::NodeDefBuilder;
using ::tensorflow::OpsTestBase;
using ::tensorflow::Tensor;
using ::tensorflow::TensorShape;

class SequenceStringProjectionOpV2Test : public OpsTestBase {
 protected:
  bool FeatureMatches(const Tensor& output, int i1, int j1, int i2, int j2) {
    bool all_matches = true;
    auto output_tensor = output.tensor<float, 3>();
    for (int k = 0; k < output.dim_size(2); ++k) {
      all_matches &= (output_tensor(i1, j1, k) == output_tensor(i2, j2, k));
    }
    return all_matches;
  }
  bool FeatureIsZero(const Tensor& output, int i, int j) {
    auto output_tensor = output.tensor<float, 3>();
    bool all_zeros = true;
    for (int k = 0; k < output.dim_size(2); ++k) {
      all_zeros &= (output_tensor(i, j, k) == 0.0f);
    }
    return all_zeros;
  }
};

TEST_F(SequenceStringProjectionOpV2Test, TestOutput) {
  TF_ASSERT_OK(NodeDefBuilder("test_op", "SequenceStringProjectionV2")
                   .Input({"input", 1, DT_STRING})
                   .Input({"sequence_length", 1, DT_INT32})
                   .Attr("vocabulary", "abcdefghijklmnopqrstuvwxyz")
                   .Attr("feature_size", 16)
                   .Finalize(node_def()));
  TF_ASSERT_OK(InitOp());

  AddInputFromArray<::tensorflow::tstring>(
      TensorShape({2, 8, 1}),
      {"hello", "world", "147", "dog", "xyz", "abc", "efg", "hij", "quick",
       "hel1lo", "123", "jumped", "over", "the", "lazy", "dog"});

  AddInputFromArray<int32>(TensorShape({3, 1}), {9, 0, 9});

  EXPECT_EQ(RunOpKernel().error_message(),
            "`input` must be a matrix, got shape: [2,8,1]");

  auto old = *mutable_input(0).tensor;
  *mutable_input(0).tensor = Tensor(DT_STRING, TensorShape({2, 8}));
  (*mutable_input(0).tensor).flat<::tensorflow::tstring>() =
      old.flat<::tensorflow::tstring>();

  EXPECT_EQ(RunOpKernel().error_message(),
            "`sequence_length` must be a vector, got shape: [3,1]");

  *mutable_input(1).tensor = Tensor(DT_INT32, TensorShape({3}));

  EXPECT_EQ(RunOpKernel().error_message(),
            "`sequence_length` should have batch size number of elements, got "
            "size 3, batch size is 2");

  *mutable_input(1).tensor = Tensor(DT_INT32, TensorShape({2}));
  (*mutable_input(1).tensor).flat<int32>()(0) = 9;
  (*mutable_input(1).tensor).flat<int32>()(1) = 0;

  EXPECT_EQ(
      RunOpKernel().error_message(),
      "`sequence_length` should have values less than or equal to max_seq_len");

  (*mutable_input(1).tensor).flat<int32>()(0) = 4;

  EXPECT_EQ(RunOpKernel().error_message(),
            "`sequence_length` should have values greater than 0");

  (*mutable_input(1).tensor).flat<int32>()(1) = 8;

  TF_EXPECT_OK(RunOpKernel());

  const Tensor& output = *GetOutput(0);

  // First checks dimensions.
  ASSERT_EQ(output.dims(), 3);
  EXPECT_EQ(output.dim_size(0), 2);   // Batch size
  EXPECT_EQ(output.dim_size(1), 8);   // Max sequence length
  EXPECT_EQ(output.dim_size(2), 16);  // Feature size

  EXPECT_FALSE(FeatureMatches(output, 0, 0, 1, 0));  // hello != quick.
  EXPECT_FALSE(FeatureMatches(output, 0, 1, 1, 1));  // world != hello.
  EXPECT_TRUE(FeatureMatches(output, 0, 0, 1, 1));   // hello == hel1lo.
  EXPECT_TRUE(FeatureMatches(output, 0, 2, 1, 2));   // 147 == 123 (oov values).
  EXPECT_TRUE(FeatureMatches(output, 0, 3, 1, 7));   // dog == dog.
  // Check zero padding for first sentence.
  for (int i = 4; i < 8; ++i) {
    EXPECT_TRUE(FeatureIsZero(output, 0, i));
  }
}

TEST_F(SequenceStringProjectionOpV2Test, TestOutputBoS) {
  TF_ASSERT_OK(NodeDefBuilder("test_op", "SequenceStringProjectionV2")
                   .Input({"input", 1, DT_STRING})
                   .Input({"sequence_length", 1, DT_INT32})
                   .Attr("add_bos_tag", true)
                   .Attr("vocabulary", "abcdefghijklmnopqrstuvwxyz")
                   .Attr("feature_size", 16)
                   .Finalize(node_def()));
  TF_ASSERT_OK(InitOp());

  AddInputFromArray<::tensorflow::tstring>(
      TensorShape({2, 8}),
      {"hello", "world", "147", "dog", "", "", "", "", "quick", "hel1lo", "123",
       "jumped", "over", "the", "lazy", "dog"});

  AddInputFromArray<int32>(TensorShape({2}), {4, 8});

  TF_ASSERT_OK(RunOpKernel());
  const Tensor& output = *GetOutput(0);

  // First checks dimensions.
  ASSERT_EQ(output.dims(), 3);
  EXPECT_EQ(output.dim_size(0), 2);   // Batch size
  EXPECT_EQ(output.dim_size(1), 9);   // Max sequence length
  EXPECT_EQ(output.dim_size(2), 16);  // Feature size

  EXPECT_TRUE(FeatureMatches(output, 0, 0, 1, 0));   // <bos> == <bos>.
  EXPECT_FALSE(FeatureMatches(output, 0, 1, 1, 1));  // hello != quick.
  EXPECT_FALSE(FeatureMatches(output, 0, 2, 1, 2));  // world != hello.
  EXPECT_TRUE(FeatureMatches(output, 0, 1, 1, 2));   // hello == hel1lo.
  EXPECT_TRUE(FeatureMatches(output, 0, 3, 1, 3));   // 147 == 123 (oov values).
  EXPECT_TRUE(FeatureMatches(output, 0, 4, 1, 8));   // dog == dog.
  // Check zero padding for first sentence.
  for (int i = 5; i < 9; ++i) {
    EXPECT_TRUE(FeatureIsZero(output, 0, i));
  }
}

TEST_F(SequenceStringProjectionOpV2Test, TestOutputEoS) {
  TF_ASSERT_OK(NodeDefBuilder("test_op", "SequenceStringProjectionV2")
                   .Input({"input", 1, DT_STRING})
                   .Input({"sequence_length", 1, DT_INT32})
                   .Attr("add_eos_tag", true)
                   .Attr("vocabulary", "abcdefghijklmnopqrstuvwxyz")
                   .Attr("feature_size", 16)
                   .Finalize(node_def()));
  TF_ASSERT_OK(InitOp());

  AddInputFromArray<::tensorflow::tstring>(
      TensorShape({2, 8}),
      {"hello", "world", "147", "dog", "", "", "", "", "quick", "hel1lo", "123",
       "jumped", "over", "the", "lazy", "dog"});

  AddInputFromArray<int32>(TensorShape({2}), {4, 8});

  TF_ASSERT_OK(RunOpKernel());
  const Tensor& output = *GetOutput(0);

  // First checks dimensions.
  ASSERT_EQ(output.dims(), 3);
  EXPECT_EQ(output.dim_size(0), 2);   // Batch size
  EXPECT_EQ(output.dim_size(1), 9);   // Max sequence length
  EXPECT_EQ(output.dim_size(2), 16);  // Feature size

  EXPECT_FALSE(FeatureMatches(output, 0, 0, 1, 0));  // hello != quick.
  EXPECT_FALSE(FeatureMatches(output, 0, 1, 1, 1));  // world != hello.
  EXPECT_TRUE(FeatureMatches(output, 0, 0, 1, 1));   // hello == hel1lo.
  EXPECT_TRUE(FeatureMatches(output, 0, 2, 1, 2));   // 147 == 123 (oov values).
  EXPECT_TRUE(FeatureMatches(output, 0, 3, 1, 7));   // dog == dog.
  EXPECT_TRUE(FeatureMatches(output, 0, 4, 1, 8));   // <bos> == <bos>.
  // Check zero padding for first sentence.
  for (int i = 5; i < 9; ++i) {
    EXPECT_TRUE(FeatureIsZero(output, 0, i));
  }
}

TEST_F(SequenceStringProjectionOpV2Test, TestOutputBoSEoS) {
  TF_ASSERT_OK(NodeDefBuilder("test_op", "SequenceStringProjectionV2")
                   .Input({"input", 1, DT_STRING})
                   .Input({"sequence_length", 1, DT_INT32})
                   .Attr("add_bos_tag", true)
                   .Attr("add_eos_tag", true)
                   .Attr("vocabulary", "abcdefghijklmnopqrstuvwxyz.")
                   .Attr("feature_size", 16)
                   .Finalize(node_def()));
  TF_ASSERT_OK(InitOp());

  AddInputFromArray<::tensorflow::tstring>(
      TensorShape({2, 8}),
      {"hello", "world", "147", "dog", "...", "..", "", "", "quick", "hel1lo",
       "123", "jumped", "over", "the", "lazy", "dog"});

  AddInputFromArray<int32>(TensorShape({2}), {6, 8});

  TF_ASSERT_OK(RunOpKernel());
  const Tensor& output = *GetOutput(0);

  // First checks dimensions.
  ASSERT_EQ(output.dims(), 3);
  EXPECT_EQ(output.dim_size(0), 2);   // Batch size
  EXPECT_EQ(output.dim_size(1), 10);  // Max sequence length
  EXPECT_EQ(output.dim_size(2), 16);  // Feature size

  EXPECT_TRUE(FeatureMatches(output, 0, 0, 1, 0));   // <bos> == <bos>.
  EXPECT_FALSE(FeatureMatches(output, 0, 1, 1, 1));  // hello != quick.
  EXPECT_FALSE(FeatureMatches(output, 0, 2, 1, 2));  // world != hello.
  EXPECT_TRUE(FeatureMatches(output, 0, 1, 1, 2));   // hello == hel1lo.
  EXPECT_TRUE(FeatureMatches(output, 0, 3, 1, 3));   // 147 == 123 (oov values).
  EXPECT_TRUE(FeatureMatches(output, 0, 4, 1, 8));   // dog == dog.
  EXPECT_TRUE(FeatureMatches(output, 0, 7, 1, 9));   // <eos> == <eos>.
  // Check for default normalize_repetition=false
  EXPECT_FALSE(FeatureMatches(output, 0, 4, 0, 5));  // ... != ..
  // Check zero padding for first sentence.
  for (int i = 8; i < 10; ++i) {
    EXPECT_TRUE(FeatureIsZero(output, 0, i));
  }
}

TEST_F(SequenceStringProjectionOpV2Test, TestOutputNormalize) {
  TF_ASSERT_OK(NodeDefBuilder("test_op", "SequenceStringProjectionV2")
                   .Input({"input", 1, DT_STRING})
                   .Input({"sequence_length", 1, DT_INT32})
                   .Attr("normalize_repetition", true)
                   .Attr("vocabulary", "abcdefghijklmnopqrstuvwxyz.")
                   .Attr("feature_size", 16)
                   .Finalize(node_def()));
  TF_ASSERT_OK(InitOp());

  AddInputFromArray<::tensorflow::tstring>(
      TensorShape({2, 8}),
      {"hello", "world", "..", "....", "", "", "", "", "quick", "hel1lo", "123",
       "jumped", "over", "...", ".....", "dog"});

  AddInputFromArray<int32>(TensorShape({2}), {4, 8});

  TF_ASSERT_OK(RunOpKernel());
  const Tensor& output = *GetOutput(0);

  // First checks dimensions.
  ASSERT_EQ(output.dims(), 3);
  EXPECT_EQ(output.dim_size(0), 2);   // Batch size
  EXPECT_EQ(output.dim_size(1), 8);   // Max sequence length
  EXPECT_EQ(output.dim_size(2), 16);  // Feature size

  EXPECT_TRUE(FeatureMatches(output, 0, 2, 0, 3));  // .. == ....
  EXPECT_TRUE(FeatureMatches(output, 1, 5, 1, 6));  // ... == ..
  EXPECT_TRUE(FeatureMatches(output, 0, 3, 1, 6));  // .... == ...
  // Check zero padding for first sentence.
  for (int i = 4; i < 8; ++i) {
    EXPECT_TRUE(FeatureIsZero(output, 0, i));
  }
}

}  // namespace

int main(int argc, char** argv) {
  // On Linux, add: absl::SetFlag(&FLAGS_logtostderr, true);
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
