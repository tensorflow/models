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
#include <string>

#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/framework/fake_input.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/shape_inference_testutil.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/kernels/ops_testutil.h"
#include "tensorflow/core/kernels/ops_util.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/core/status_test_util.h"

using ::tensorflow::DT_STRING;
using ::tensorflow::NodeDefBuilder;
using ::tensorflow::OpsTestBase;
using ::tensorflow::Tensor;
using ::tensorflow::TensorShape;

class SequenceStringProjectionOpTest : public OpsTestBase {
 protected:
  const float* FeatureIndex(const Tensor& output, int i, int j) {
    return &output.flat<float>()((i * output.dim_size(2) * output.dim_size(1)) +
                                 (j * output.dim_size(2)));
  }
  bool FeatureMatches(const Tensor& output, int i1, int j1, int i2, int j2) {
    const float* feature1 = FeatureIndex(output, i1, j1);
    const float* feature2 = FeatureIndex(output, i2, j2);
    bool all_matches = true;
    for (int i = 0; i < output.dim_size(2); ++i) {
      all_matches &= (feature1[i] == feature2[i]);
    }
    return all_matches;
  }
  void FeatureIsZero(const Tensor& output, int i, int j) {
    const float* feature = FeatureIndex(output, i, j);
    bool all_zeros = true;
    for (int i = 0; i < output.dim_size(2); ++i) {
      all_zeros &= (feature[i] == 0.0f);
    }
    EXPECT_TRUE(all_zeros);
  }
};

TEST_F(SequenceStringProjectionOpTest, TestOutputEoS) {
  TF_ASSERT_OK(NodeDefBuilder("test_op", "SequenceStringProjection")
                   .Input({"input", 1, DT_STRING})
                   .Attr("distortion_probability", 0.0f)
                   .Attr("vocabulary", "abcdefghijklmnopqrstuvwxyz")
                   .Attr("feature_size", 16)
                   .Finalize(node_def()));
  TF_ASSERT_OK(InitOp());

  AddInputFromArray<::tensorflow::tstring>(
      TensorShape({2}),
      {"hello world 147 dog", "quick hel1lo 123 jumped over the lazy dog"});

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
  EXPECT_TRUE(FeatureMatches(output, 0, 4, 1, 8));   // <eos> == <eos>.
  // Check zero padding for features after "<eos>" for first sentence.
  for (int i = 5; i < 9; ++i) {
    FeatureIsZero(output, 0, i);
  }

  const Tensor& bag_of_chars = *GetOutput(1);
  ASSERT_EQ(bag_of_chars.dims(), 1);
  EXPECT_EQ(bag_of_chars.dim_size(0), 1);  // Dummy output

  const Tensor& sequence_length = *GetOutput(2);
  EXPECT_EQ(sequence_length.dim_size(0), 2);  // Batch size
  EXPECT_EQ(sequence_length.flat<float>()(0), 5);
  EXPECT_EQ(sequence_length.flat<float>()(1), 9);
}

TEST_F(SequenceStringProjectionOpTest, TestOutputBoSEoS) {
  TF_ASSERT_OK(NodeDefBuilder("test_op", "SequenceStringProjection")
                   .Input({"input", 1, DT_STRING})
                   .Attr("add_bos_tag", true)
                   .Attr("distortion_probability", 0.0f)
                   .Attr("vocabulary", "abcdefghijklmnopqrstuvwxyz")
                   .Attr("feature_size", 16)
                   .Finalize(node_def()));
  TF_ASSERT_OK(InitOp());

  AddInputFromArray<::tensorflow::tstring>(
      TensorShape({2}),
      {"hello world 147 dog", "quick hel1lo 123 jumped over the lazy dog"});

  TF_ASSERT_OK(RunOpKernel());
  const Tensor& output = *GetOutput(0);

  // First checks dimensions.
  ASSERT_EQ(output.dims(), 3);
  EXPECT_EQ(output.dim_size(0), 2);   // Batch size
  EXPECT_EQ(output.dim_size(1), 10);  // Max sequence length
  EXPECT_EQ(output.dim_size(2), 16);  // Feature size

  EXPECT_TRUE(FeatureMatches(output, 0, 0, 1, 0));   // bos == bos.
  EXPECT_FALSE(FeatureMatches(output, 0, 1, 1, 1));  // hello != quick.
  EXPECT_FALSE(FeatureMatches(output, 0, 2, 1, 2));  // world != hello.
  EXPECT_TRUE(FeatureMatches(output, 0, 1, 1, 2));   // hello == hel1lo.
  EXPECT_TRUE(FeatureMatches(output, 0, 3, 1, 3));   // 147 == 123 (oov values).
  EXPECT_TRUE(FeatureMatches(output, 0, 4, 1, 8));   // dog == dog.
  EXPECT_TRUE(FeatureMatches(output, 0, 5, 1, 9));   // <eos> == <eos>.
  // Check zero padding for features after "<eos>" for first sentence.
  for (int i = 6; i < 10; ++i) {
    FeatureIsZero(output, 0, i);
  }

  const Tensor& bag_of_chars = *GetOutput(1);
  ASSERT_EQ(bag_of_chars.dims(), 1);
  EXPECT_EQ(bag_of_chars.dim_size(0), 1);  // Dummy output

  const Tensor& sequence_length = *GetOutput(2);
  EXPECT_EQ(sequence_length.dim_size(0), 2);  // Batch size
  EXPECT_EQ(sequence_length.flat<float>()(0), 6);
  EXPECT_EQ(sequence_length.flat<float>()(1), 10);
}

TEST_F(SequenceStringProjectionOpTest, TestOutputBoS) {
  TF_ASSERT_OK(NodeDefBuilder("test_op", "SequenceStringProjection")
                   .Input({"input", 1, DT_STRING})
                   .Attr("add_bos_tag", true)
                   .Attr("add_eos_tag", false)
                   .Attr("distortion_probability", 0.0f)
                   .Attr("vocabulary", "abcdefghijklmnopqrstuvwxyz")
                   .Attr("feature_size", 16)
                   .Finalize(node_def()));
  TF_ASSERT_OK(InitOp());

  AddInputFromArray<::tensorflow::tstring>(
      TensorShape({2}),
      {"hello world 147 dog", "quick hel1lo 123 jumped over the lazy dog"});

  TF_ASSERT_OK(RunOpKernel());
  const Tensor& output = *GetOutput(0);

  // First checks dimensions.
  ASSERT_EQ(output.dims(), 3);
  EXPECT_EQ(output.dim_size(0), 2);   // Batch size
  EXPECT_EQ(output.dim_size(1), 9);   // Max sequence length
  EXPECT_EQ(output.dim_size(2), 16);  // Feature size

  EXPECT_TRUE(FeatureMatches(output, 0, 0, 1, 0));   // bos == bos.
  EXPECT_FALSE(FeatureMatches(output, 0, 1, 1, 1));  // hello != quick.
  EXPECT_FALSE(FeatureMatches(output, 0, 2, 1, 2));  // world != hello.
  EXPECT_TRUE(FeatureMatches(output, 0, 1, 1, 2));   // hello == hel1lo.
  EXPECT_TRUE(FeatureMatches(output, 0, 3, 1, 3));   // 147 == 123 (oov values).
  EXPECT_TRUE(FeatureMatches(output, 0, 4, 1, 8));   // dog == dog.
  // Check zero padding for features after "<eos>" for first sentence.
  for (int i = 6; i < 9; ++i) {
    FeatureIsZero(output, 0, i);
  }

  const Tensor& bag_of_chars = *GetOutput(1);
  ASSERT_EQ(bag_of_chars.dims(), 1);
  EXPECT_EQ(bag_of_chars.dim_size(0), 1);  // Dummy output

  const Tensor& sequence_length = *GetOutput(2);
  EXPECT_EQ(sequence_length.dim_size(0), 2);  // Batch size
  EXPECT_EQ(sequence_length.flat<float>()(0), 5);
  EXPECT_EQ(sequence_length.flat<float>()(1), 9);
}

TEST_F(SequenceStringProjectionOpTest, TestOutput) {
  TF_ASSERT_OK(NodeDefBuilder("test_op", "SequenceStringProjection")
                   .Input({"input", 1, DT_STRING})
                   .Attr("add_bos_tag", false)
                   .Attr("add_eos_tag", false)
                   .Attr("distortion_probability", 0.0f)
                   .Attr("vocabulary", "abcdefghijklmnopqrstuvwxyz")
                   .Attr("feature_size", 16)
                   .Finalize(node_def()));
  TF_ASSERT_OK(InitOp());

  AddInputFromArray<::tensorflow::tstring>(
      TensorShape({2}),
      {"hello world 147 dog", "quick hel1lo 123 jumped over the lazy dog"});

  TF_ASSERT_OK(RunOpKernel());
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
  // Check zero padding for features after "<eos>" for first sentence.
  for (int i = 5; i < 8; ++i) {
    FeatureIsZero(output, 0, i);
  }

  const Tensor& bag_of_chars = *GetOutput(1);
  ASSERT_EQ(bag_of_chars.dims(), 1);
  EXPECT_EQ(bag_of_chars.dim_size(0), 1);  // Dummy output

  const Tensor& sequence_length = *GetOutput(2);
  EXPECT_EQ(sequence_length.dim_size(0), 2);  // Batch size
  EXPECT_EQ(sequence_length.flat<float>()(0), 4);
  EXPECT_EQ(sequence_length.flat<float>()(1), 8);
}

TEST_F(SequenceStringProjectionOpTest, DocSize) {
  TF_ASSERT_OK(NodeDefBuilder("test_op", "SequenceStringProjection")
                   .Input({"input", 1, DT_STRING})
                   .Attr("distortion_probability", 0.0f)
                   .Attr("vocabulary", "abcdefghijklmnopqrstuvwxyz")
                   .Attr("doc_size_levels", 4)
                   .Attr("feature_size", 16)
                   .Finalize(node_def()));
  TF_ASSERT_OK(InitOp());

  AddInputFromArray<::tensorflow::tstring>(
      TensorShape({4}), {"dog", "dog dog", "dog dog dog dog", "dog"});

  TF_ASSERT_OK(RunOpKernel());
  const Tensor& output = *GetOutput(0);

  // First checks dimensions.
  ASSERT_EQ(output.dims(), 3);
  EXPECT_EQ(output.dim_size(0), 4);   // Batch size
  EXPECT_EQ(output.dim_size(1), 5);   // Max sequence length
  EXPECT_EQ(output.dim_size(2), 16);  // Feature size

  EXPECT_TRUE(FeatureMatches(output, 0, 0, 3, 0));   // dog(0) == dog(3).
  EXPECT_TRUE(FeatureMatches(output, 0, 1, 3, 1));   // EOS == EOS.
  EXPECT_FALSE(FeatureMatches(output, 0, 0, 1, 0));  // dog(0) != dog(1).
  EXPECT_FALSE(FeatureMatches(output, 0, 0, 2, 0));  // dog(0) != dog(2).

  const Tensor& sequence_length = *GetOutput(2);
  EXPECT_EQ(sequence_length.dim_size(0), 4);  // Batch size
  EXPECT_EQ(sequence_length.flat<float>()(0), 2);
  EXPECT_EQ(sequence_length.flat<float>()(1), 3);
  EXPECT_EQ(sequence_length.flat<float>()(2), 5);
  EXPECT_EQ(sequence_length.flat<float>()(3), 2);
}

TEST_F(SequenceStringProjectionOpTest, WordNovelty) {
  TF_ASSERT_OK(NodeDefBuilder("test_op", "SequenceStringProjection")
                   .Input({"input", 1, DT_STRING})
                   .Attr("distortion_probability", 0.0f)
                   .Attr("vocabulary", "abcdefghijklmnopqrstuvwxyz")
                   .Attr("word_novelty_bits", 3)
                   .Attr("feature_size", 16)
                   .Finalize(node_def()));
  TF_ASSERT_OK(InitOp());

  AddInputFromArray<::tensorflow::tstring>(
      TensorShape({2}), {"dog", "dog dog dog dog dog dog dog dog dog dog"});

  TF_ASSERT_OK(RunOpKernel());
  const Tensor& output = *GetOutput(0);

  // First checks dimensions.
  ASSERT_EQ(output.dims(), 3);
  EXPECT_EQ(output.dim_size(0), 2);   // Batch size
  EXPECT_EQ(output.dim_size(1), 11);  // Max sequence length
  EXPECT_EQ(output.dim_size(2), 16);  // Feature size

  EXPECT_TRUE(FeatureMatches(output, 0, 0, 1, 0));   // dog(0) == dog(0).
  EXPECT_TRUE(FeatureMatches(output, 0, 1, 1, 10));  // EOS == EOS.
  for (int i = 0; i < 8; ++i) {
    for (int j = i + 1; j < 8; ++j) {
      EXPECT_FALSE(FeatureMatches(output, 1, i, 1, j));  // dog(i) != dog(j).
    }
  }
  // Check novel word feature saturates after 9 steps
  EXPECT_TRUE(FeatureMatches(output, 1, 8, 1, 9));  // dog(8) == dog(9).
  // Check zero padding for features after "<eos>" for first sentence.
  for (int i = 2; i < 11; ++i) {
    FeatureIsZero(output, 0, i);
  }

  const Tensor& bag_of_chars = *GetOutput(1);
  ASSERT_EQ(bag_of_chars.dims(), 1);
  EXPECT_EQ(bag_of_chars.dim_size(0), 1);  // Dummy output

  const Tensor& sequence_length = *GetOutput(2);
  EXPECT_EQ(sequence_length.dim_size(0), 2);  // Batch size
  EXPECT_EQ(sequence_length.flat<float>()(0), 2);
  EXPECT_EQ(sequence_length.flat<float>()(1), 11);
}

TEST_F(SequenceStringProjectionOpTest, TestMaxSplits) {
  TF_ASSERT_OK(NodeDefBuilder("test_op", "SequenceStringProjection")
                   .Input({"input", 1, DT_STRING})
                   .Attr("vocabulary", "abcdefghijklmnopqrstuvwxyz")
                   .Attr("max_splits", 3)
                   .Attr("feature_size", 16)
                   .Finalize(node_def()));
  TF_ASSERT_OK(InitOp());

  AddInputFromArray<::tensorflow::tstring>(
      TensorShape({2}),
      {"hello world 147 dog", "quick hel1lo 123 jumped over the lazy dog"});

  TF_ASSERT_OK(RunOpKernel());
  const Tensor& output = *GetOutput(0);

  // First checks dimensions.
  ASSERT_EQ(output.dims(), 3);
  EXPECT_EQ(output.dim_size(0), 2);   // Batch size
  EXPECT_EQ(output.dim_size(1), 4);   // Max sequence length
  EXPECT_EQ(output.dim_size(2), 16);  // Feature size

  EXPECT_FALSE(FeatureMatches(output, 0, 0, 1, 0));  // hello != quick.
  EXPECT_FALSE(FeatureMatches(output, 0, 1, 1, 1));  // world != hello.
  EXPECT_TRUE(FeatureMatches(output, 0, 0, 1, 1));   // hello == hel1lo.
  EXPECT_TRUE(FeatureMatches(output, 0, 2, 1, 2));   // 147 == 123 (oov values).
  EXPECT_TRUE(FeatureMatches(output, 0, 3, 1, 3));   // <eos> == <eos>.

  const Tensor& bag_of_chars = *GetOutput(1);
  ASSERT_EQ(bag_of_chars.dims(), 1);
  EXPECT_EQ(bag_of_chars.dim_size(0), 1);  // Dummy output

  const Tensor& sequence_length = *GetOutput(2);
  EXPECT_EQ(sequence_length.dim_size(0), 2);  // Batch size
  EXPECT_EQ(sequence_length.flat<float>()(0), 4);
  EXPECT_EQ(sequence_length.flat<float>()(1), 4);
}

TEST_F(SequenceStringProjectionOpTest, TestNoEOS) {
  TF_ASSERT_OK(NodeDefBuilder("test_op", "SequenceStringProjection")
                   .Input({"input", 1, DT_STRING})
                   .Attr("vocabulary", "abcdefghijklmnopqrstuvwxyz")
                   .Attr("add_eos_tag", false)
                   .Attr("feature_size", 16)
                   .Finalize(node_def()));
  TF_ASSERT_OK(InitOp());

  AddInputFromArray<::tensorflow::tstring>(
      TensorShape({2}), {"hello world 147 dog", "quick hel1lo 123"});

  TF_ASSERT_OK(RunOpKernel());
  const Tensor& output = *GetOutput(0);

  // First checks dimensions.
  ASSERT_EQ(output.dims(), 3);
  EXPECT_EQ(output.dim_size(0), 2);   // Batch size
  EXPECT_EQ(output.dim_size(1), 4);   // Max sequence length
  EXPECT_EQ(output.dim_size(2), 16);  // Feature size

  EXPECT_FALSE(FeatureMatches(output, 0, 0, 1, 0));  // hello != quick.
  EXPECT_FALSE(FeatureMatches(output, 0, 1, 1, 1));  // world != hel1lo.
  EXPECT_TRUE(FeatureMatches(output, 0, 0, 1, 1));   // hello == hel1lo.
  EXPECT_TRUE(FeatureMatches(output, 0, 2, 1, 2));   // 147 == 123 (oov values).

  const Tensor& bag_of_chars = *GetOutput(1);
  ASSERT_EQ(bag_of_chars.dims(), 1);
  EXPECT_EQ(bag_of_chars.dim_size(0), 1);  // Dummy output

  const Tensor& sequence_length = *GetOutput(2);
  EXPECT_EQ(sequence_length.dim_size(0), 2);  // Batch size
  EXPECT_EQ(sequence_length.flat<float>()(0), 4);
  EXPECT_EQ(sequence_length.flat<float>()(1), 3);
}

TEST_F(SequenceStringProjectionOpTest, TestNonSpaceMaxSplit) {
  TF_ASSERT_OK(NodeDefBuilder("test_op", "SequenceStringProjection")
                   .Input({"input", 1, DT_STRING})
                   .Attr("vocabulary", "abcdefghijklmnopqrstuvwxyz⺁")
                   .Attr("split_on_space", false)
                   .Attr("max_splits", 4)
                   .Attr("feature_size", 16)
                   .Finalize(node_def()));
  TF_ASSERT_OK(InitOp());

  AddInputFromArray<::tensorflow::tstring>(TensorShape({2}),
                                           {"hel  world", "⺁leh1ho"});

  TF_ASSERT_OK(RunOpKernel());
  const Tensor& output = *GetOutput(0);

  // First checks dimensions.
  ASSERT_EQ(output.dims(), 3);
  EXPECT_EQ(output.dim_size(0), 2);   // Batch size
  EXPECT_EQ(output.dim_size(1), 5);   // Max sequence length
  EXPECT_EQ(output.dim_size(2), 16);  // Feature size

  EXPECT_FALSE(FeatureMatches(output, 0, 0, 1, 1));  // h != l.
  EXPECT_TRUE(FeatureMatches(output, 0, 1, 1, 2));   // e == e.
  EXPECT_TRUE(FeatureMatches(output, 0, 0, 1, 3));   // h == h.
  EXPECT_TRUE(FeatureMatches(output, 0, 4, 1, 4));   // <eos> == <eos>.

  const Tensor& bag_of_chars = *GetOutput(1);
  ASSERT_EQ(bag_of_chars.dims(), 1);
  EXPECT_EQ(bag_of_chars.dim_size(0), 1);  // Dummy output

  const Tensor& sequence_length = *GetOutput(2);
  EXPECT_EQ(sequence_length.dim_size(0), 2);  // Batch size
  EXPECT_EQ(sequence_length.flat<float>()(0), 5);
  EXPECT_EQ(sequence_length.flat<float>()(1), 5);
}

TEST_F(SequenceStringProjectionOpTest, TestNonSpace) {
  TF_ASSERT_OK(NodeDefBuilder("test_op", "SequenceStringProjection")
                   .Input({"input", 1, DT_STRING})
                   .Attr("vocabulary", "abcdefghijklmnopqrstuvwxyz⺁")
                   .Attr("split_on_space", false)
                   .Attr("feature_size", 16)
                   .Finalize(node_def()));
  TF_ASSERT_OK(InitOp());

  AddInputFromArray<::tensorflow::tstring>(TensorShape({2}),
                                           {"hello world", "leh1ho⺁"});

  TF_ASSERT_OK(RunOpKernel());
  const Tensor& output = *GetOutput(0);

  // First checks dimensions.
  ASSERT_EQ(output.dims(), 3);
  EXPECT_EQ(output.dim_size(0), 2);   // Batch size
  EXPECT_EQ(output.dim_size(1), 12);  // Max sequence length
  EXPECT_EQ(output.dim_size(2), 16);  // Feature size

  EXPECT_FALSE(FeatureMatches(output, 0, 0, 1, 0));  // h != l.
  EXPECT_TRUE(FeatureMatches(output, 0, 1, 1, 1));   // e == e.
  EXPECT_TRUE(FeatureMatches(output, 0, 0, 1, 2));   // h == h.
  EXPECT_TRUE(FeatureMatches(output, 0, 5, 1, 3));   // oov == oov.
  EXPECT_TRUE(FeatureMatches(output, 0, 11, 1, 7));  // <eos> == <eos>.
  // Check zero padding for features after "<eos>" for first sentence.
  for (int i = 8; i < 12; ++i) {
    FeatureIsZero(output, 1, i);
  }

  const Tensor& bag_of_chars = *GetOutput(1);
  ASSERT_EQ(bag_of_chars.dims(), 1);
  EXPECT_EQ(bag_of_chars.dim_size(0), 1);  // Dummy output

  const Tensor& sequence_length = *GetOutput(2);
  EXPECT_EQ(sequence_length.dim_size(0), 2);  // Batch size
  EXPECT_EQ(sequence_length.flat<float>()(0), 12);
  EXPECT_EQ(sequence_length.flat<float>()(1), 8);
}

TEST_F(SequenceStringProjectionOpTest, TestNonSpaceExcludeNonAlpha) {
  TF_ASSERT_OK(NodeDefBuilder("test_op", "SequenceStringProjection")
                   .Input({"input", 1, DT_STRING})
                   .Attr("vocabulary", "")
                   .Attr("split_on_space", false)
                   .Attr("exclude_nonalphaspace_unicodes", true)
                   .Attr("feature_size", 16)
                   .Finalize(node_def()));
  TF_ASSERT_OK(InitOp());

  AddInputFromArray<::tensorflow::tstring>(TensorShape({2}),
                                           {"hello world4", "leh1ho⺁"});

  TF_ASSERT_OK(RunOpKernel());
  const Tensor& output = *GetOutput(0);

  // First checks dimensions.
  ASSERT_EQ(output.dims(), 3);
  EXPECT_EQ(output.dim_size(0), 2);   // Batch size
  EXPECT_EQ(output.dim_size(1), 13);  // Max sequence length
  EXPECT_EQ(output.dim_size(2), 16);  // Feature size

  EXPECT_TRUE(FeatureMatches(
      output, 0, 11, 1,
      3));  // 1 == 4 (nonalpha and space mapped to same feature).
  EXPECT_TRUE(FeatureMatches(output, 0, 12, 1, 7));  // <eos> == <eos>.
  // Check zero padding for features after "<eos>" for first sentence.
  for (int i = 8; i < 13; ++i) {
    FeatureIsZero(output, 1, i);
  }

  const Tensor& bag_of_chars = *GetOutput(1);
  ASSERT_EQ(bag_of_chars.dims(), 1);
  EXPECT_EQ(bag_of_chars.dim_size(0), 1);  // Dummy output

  const Tensor& sequence_length = *GetOutput(2);
  EXPECT_EQ(sequence_length.dim_size(0), 2);  // Batch size
  EXPECT_EQ(sequence_length.flat<float>()(0), 13);
  EXPECT_EQ(sequence_length.flat<float>()(1), 8);
}

TEST_F(SequenceStringProjectionOpTest, TestEmptyVocabulary) {
  TF_ASSERT_OK(NodeDefBuilder("test_op", "SequenceStringProjection")
                   .Input({"input", 1, DT_STRING})
                   .Attr("feature_size", 16)
                   .Finalize(node_def()));
  TF_ASSERT_OK(InitOp());

  AddInputFromArray<::tensorflow::tstring>(
      TensorShape({2}),
      {"hello world 147 dog", "quick hel1lo 123 jumped over the lazy dog"});

  TF_ASSERT_OK(RunOpKernel());
  const Tensor& output = *GetOutput(0);

  // First checks dimensions.
  ASSERT_EQ(output.dims(), 3);
  EXPECT_EQ(output.dim_size(0), 2);   // Batch size
  EXPECT_EQ(output.dim_size(1), 9);   // Max sequence length
  EXPECT_EQ(output.dim_size(2), 16);  // Feature size

  EXPECT_FALSE(FeatureMatches(output, 0, 0, 1, 0));  // hello != quick.
  EXPECT_FALSE(FeatureMatches(output, 0, 1, 1, 1));  // world != hello.
  EXPECT_FALSE(FeatureMatches(output, 0, 0, 1, 1));  // hello != hel1lo.
  EXPECT_FALSE(FeatureMatches(output, 0, 2, 1, 2));  // 147 != 123 (no oov).
  EXPECT_TRUE(FeatureMatches(output, 0, 3, 1, 7));   // dog == dog.
  EXPECT_TRUE(FeatureMatches(output, 0, 4, 1, 8));   // <eos> == <eos>.
  // Check zero padding for features after "<eos>" for first sentence.
  for (int i = 5; i < 9; ++i) {
    FeatureIsZero(output, 0, i);
  }

  const Tensor& bag_of_chars = *GetOutput(1);
  ASSERT_EQ(bag_of_chars.dims(), 1);
  EXPECT_EQ(bag_of_chars.dim_size(0), 1);  // Dummy output

  const Tensor& sequence_length = *GetOutput(2);
  EXPECT_EQ(sequence_length.dim_size(0), 2);  // Batch size
  EXPECT_EQ(sequence_length.flat<float>()(0), 5);
  EXPECT_EQ(sequence_length.flat<float>()(1), 9);
}

TEST_F(SequenceStringProjectionOpTest, Normalization) {
  TF_ASSERT_OK(NodeDefBuilder("test_op", "SequenceStringProjection")
                   .Input({"input", 1, DT_STRING})
                   .Attr("feature_size", 16)
                   .Attr("normalize_repetition", true)
                   .Attr("token_separators", " ")
                   .Finalize(node_def()));
  TF_ASSERT_OK(InitOp());

  AddInputFromArray<::tensorflow::tstring>(TensorShape({2}),
                                           {"hello 147 .....", "hello 147 .."});

  TF_ASSERT_OK(RunOpKernel());
  const Tensor& output = *GetOutput(0);

  // First checks dimensions.
  ASSERT_EQ(output.dims(), 3);
  EXPECT_EQ(output.dim_size(0), 2);   // Batch size
  EXPECT_EQ(output.dim_size(1), 4);   // Max sequence length
  EXPECT_EQ(output.dim_size(2), 16);  // Feature size

  EXPECT_TRUE(FeatureMatches(output, 0, 0, 1, 0));  // hello = hello
  EXPECT_TRUE(FeatureMatches(output, 0, 1, 1, 1));  // 147 = 147
  EXPECT_TRUE(FeatureMatches(output, 0, 2, 1, 2));  // ..... = ..
  EXPECT_TRUE(FeatureMatches(output, 0, 3, 1, 3));  // <eos> == <eos>

  const Tensor& bag_of_chars = *GetOutput(1);
  ASSERT_EQ(bag_of_chars.dims(), 1);
  EXPECT_EQ(bag_of_chars.dim_size(0), 1);  // Dummy output

  const Tensor& sequence_length = *GetOutput(2);
  EXPECT_EQ(sequence_length.dim_size(0), 2);  // Batch size
  EXPECT_EQ(sequence_length.flat<float>()(0), 4);
  EXPECT_EQ(sequence_length.flat<float>()(1), 4);
}

int main(int argc, char** argv) {
  // On Linux, add: absl::SetFlag(&FLAGS_logtostderr, true);
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
