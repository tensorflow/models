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

#include "dragnn/runtime/trained_model_variable_store.h"

#include <stddef.h>
#include <string>
#include <vector>

#include "dragnn/core/test/generic.h"
#include "dragnn/protos/runtime.pb.h"
#include "dragnn/runtime/math/avx_vector_array.h"
#include "dragnn/runtime/math/float16_types.h"
#include "syntaxnet/base.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/test.h"

namespace syntaxnet {
namespace dragnn {
namespace runtime {
namespace {

class TrainedModelVariableStoreTest : public ::testing::Test {
 protected:
  // Computes a value that accesses all bytes in the |view| or |area|.  Useful
  // for checking that a piece of memory is accessible.
  size_t SumBytes(AlignedView view) {
    size_t sum = 0;
    for (size_t i = 0; i < view.size(); ++i) sum += view.data()[i];
    return sum;
  }
  size_t SumBytes(AlignedArea area) {
    size_t sum = 0;
    for (size_t i = 0; i < area.num_views(); ++i) sum += SumBytes(area.view(i));
    return sum;
  }

  // Returns the name of a tensor containing the blocked version of
  // |kVariableName|, with the given |block_size|.
  string GetBlockedVariableName(int block_size) const {
    return tensorflow::strings::StrCat(kVariableNamePrefix, "/matrix/blocked",
                                       block_size, "/ExponentialMovingAverage");
  }

  // Same as above, but returns the name of the bfloat16 variable.
  string GetBfloat16VariableName(int block_size) const {
    return tensorflow::strings::StrCat(kVariableNamePrefix, "/matrix/blocked",
                                       block_size,
                                       "/bfloat16/ExponentialMovingAverage");
  }

  // Path to a saved model file for tests.  Expected to contain:
  //   * A tf.float32 variable named |kVariableName| with shape
  //     [|kVariableRows|, |kVariableColumns|].
  //   * A variable named |kUnsupportedTypeVariableName| whose type is not
  //     supported by the implementation.
  //   * A variable named |kLowRankVariableName| whose rank is < 2.
  const string kSavedModelDir = tensorflow::io::JoinPath(
      test::GetTestDataPrefix(), "dragnn/runtime/testdata/rnn_tagger");

  // A valid variable name in the test model and its dimensions.
  const string kVariableNamePrefix = "tagger/weights_0";
  const string kVariableName = tensorflow::strings::StrCat(
      kVariableNamePrefix, "/ExponentialMovingAverage");
  const size_t kVariableRows = 160;
  const size_t kVariableColumns = 64;

  // A variable with unsupported type; this variable is tf.int32.
  const string kUnsupportedTypeVariableName = "tagger/step";

  // A variable whose rank is < 2; this is a scalar.
  const string kLowRankVariableName = "tagger/bias_1";

  // Variable store for tests.
  TrainedModelVariableStore store_;
};

// Tests that TrainedModelVariableStore can be initialized from a valid model.
TEST_F(TrainedModelVariableStoreTest, ResetValid) {
  TF_EXPECT_OK(store_.Reset(kSavedModelDir));
}

// Tests that TrainedModelVariableStore fails on a valid directory that doesn't
// actually contain a TF saved model, but can be re-Reset() on valid files.
TEST_F(TrainedModelVariableStoreTest, ResetInvalidDirectoryThenValid) {
  EXPECT_FALSE(store_.Reset("/tmp").ok());
  TF_EXPECT_OK(store_.Reset(kSavedModelDir));
}

// Tests that TrainedModelVariableStore fails on a non-directory, but can be
// re-Reset() on valid files.
TEST_F(TrainedModelVariableStoreTest, ResetNotADirectoryThenValid) {
  EXPECT_FALSE(store_.Reset("/dev/null").ok());
  TF_EXPECT_OK(store_.Reset(kSavedModelDir));
}

// Tests that TrainedModelVariableStore fails with missing files node scope, but
// can be re-Reset() on valid files.
TEST_F(TrainedModelVariableStoreTest, ResetMissingDirectoryThenValid) {
  EXPECT_FALSE(store_.Reset("/missing/model/dir").ok());
  TF_EXPECT_OK(store_.Reset(kSavedModelDir));
}

// Tests that TrainedModelVariableStore can only be closed once, and only after
// it is has been initialized.
TEST_F(TrainedModelVariableStoreTest, Close) {
  EXPECT_THAT(store_.Close(),
              test::IsErrorWithSubstr("TF Session is not active"));

  TF_ASSERT_OK(store_.Reset(kSavedModelDir));
  TF_EXPECT_OK(store_.Close());

  EXPECT_THAT(store_.Close(),
              test::IsErrorWithSubstr("TF Session is not active"));
}

// Tests that TrainedModelVariableStore can look up flat variables.
TEST_F(TrainedModelVariableStoreTest, LookupFlat) {
  AlignedArea area;
  std::vector<size_t> dimensions;

  // Fail to look up a valid name before initialization.
  EXPECT_THAT(store_.Lookup(kVariableName, VariableSpec::FORMAT_FLAT,
                            &dimensions, &area),
              test::IsErrorWithSubstr("TF Session is not active"));
  EXPECT_TRUE(area.empty());  // not modified

  // Repeating the failed lookup should still fail.
  EXPECT_THAT(store_.Lookup(kVariableName, VariableSpec::FORMAT_FLAT,
                            &dimensions, &area),
              test::IsErrorWithSubstr("TF Session is not active"));
  EXPECT_TRUE(area.empty());  // not modified

  // Fail to look up an invalid name after initialization.
  TF_ASSERT_OK(store_.Reset(kSavedModelDir));
  EXPECT_FALSE(
      store_
          .Lookup("invalid/name", VariableSpec::FORMAT_FLAT, &dimensions, &area)
          .ok());
  EXPECT_TRUE(area.empty());  // not modified

  // Successfully look up a valid name.
  TF_ASSERT_OK(store_.Lookup(kVariableName, VariableSpec::FORMAT_FLAT,
                             &dimensions, &area));
  EXPECT_FALSE(area.empty());  // modified
  EXPECT_EQ(area.num_views(), 1);
  EXPECT_EQ(area.view_size(), kVariableRows * kVariableColumns * sizeof(float));

  // Try looking up the same name again.
  area = AlignedArea();
  TF_ASSERT_OK(store_.Lookup(kVariableName, VariableSpec::FORMAT_FLAT,
                             &dimensions, &area));
  EXPECT_EQ(area.num_views(), 1);
  EXPECT_EQ(area.view_size(), kVariableRows * kVariableColumns * sizeof(float));

  // Check that the area can be accessed even after the |store| is closed.
  TF_EXPECT_OK(store_.Close());
  LOG(INFO) << "Logging to prevent elision by optimizer: " << SumBytes(area);
}

// Tests that TrainedModelVariableStore can look up row-major matrix variables.
TEST_F(TrainedModelVariableStoreTest, LookupRowMajorMatrix) {
  AlignedArea area;
  std::vector<size_t> dimensions;

  // Fail to look up a valid name before initialization.
  EXPECT_THAT(
      store_.Lookup(kVariableName, VariableSpec::FORMAT_ROW_MAJOR_MATRIX,
                    &dimensions, &area),
      test::IsErrorWithSubstr("TF Session is not active"));
  EXPECT_TRUE(area.empty());  // not modified

  // Repeating the failed lookup should still fail.
  EXPECT_THAT(
      store_.Lookup(kVariableName, VariableSpec::FORMAT_ROW_MAJOR_MATRIX,
                    &dimensions, &area),
      test::IsErrorWithSubstr("TF Session is not active"));
  EXPECT_TRUE(area.empty());  // not modified

  // Fail to look up an invalid name after initialization.
  TF_ASSERT_OK(store_.Reset(kSavedModelDir));
  EXPECT_FALSE(store_
                   .Lookup("invalid/name",
                           VariableSpec::FORMAT_ROW_MAJOR_MATRIX, &dimensions,
                           &area)
                   .ok());
  EXPECT_TRUE(area.empty());  // not modified

  // Successfully look up a valid name.
  TF_ASSERT_OK(store_.Lookup(kVariableName,
                             VariableSpec::FORMAT_ROW_MAJOR_MATRIX, &dimensions,
                             &area));
  ASSERT_FALSE(area.empty());  // modified
  EXPECT_EQ(dimensions, std::vector<size_t>({kVariableRows, kVariableColumns}));
  EXPECT_EQ(area.num_views(), kVariableRows);
  EXPECT_EQ(area.view_size(), kVariableColumns * sizeof(float));

  // Try looking up the same name again.
  area = AlignedArea();
  dimensions.clear();
  TF_ASSERT_OK(store_.Lookup(kVariableName,
                             VariableSpec::FORMAT_ROW_MAJOR_MATRIX, &dimensions,
                             &area));
  EXPECT_EQ(dimensions, std::vector<size_t>({kVariableRows, kVariableColumns}));
  EXPECT_EQ(area.num_views(), kVariableRows);
  EXPECT_EQ(area.view_size(), kVariableColumns * sizeof(float));

  // Check that the area can be accessed even after the |store| is closed.
  TF_EXPECT_OK(store_.Close());
  LOG(INFO) << "Logging to prevent elision by optimizer: " << SumBytes(area);
}

// Tests that the same contents can be retrieved in various formats, and that
// the content is the same asides from rearrangement.
TEST_F(TrainedModelVariableStoreTest, CompareFormats) {
  Vector<float> flat;
  Matrix<float> row_major_matrix;

  TF_ASSERT_OK(store_.Reset(kSavedModelDir));
  TF_ASSERT_OK(store_.Lookup(kVariableName, &flat));
  TF_ASSERT_OK(store_.Lookup(kVariableName, &row_major_matrix));

  ASSERT_EQ(flat.size(),
            row_major_matrix.num_rows() * row_major_matrix.num_columns());
  for (size_t flat_index = 0, row = 0; row < row_major_matrix.num_rows();
       ++row) {
    for (size_t column = 0; column < row_major_matrix.num_columns();
         ++column, ++flat_index) {
      EXPECT_EQ(row_major_matrix.row(row)[column], flat[flat_index]);
    }
  }
}

// Tests that TrainedModelVariableStore fails to retrieve a variable of an
// unsupported type.
TEST_F(TrainedModelVariableStoreTest, LookupUnsupportedType) {
  AlignedArea area;
  std::vector<size_t> dimensions;

  TF_ASSERT_OK(store_.Reset(kSavedModelDir));
  EXPECT_THAT(store_.Lookup(kUnsupportedTypeVariableName,
                            VariableSpec::FORMAT_FLAT, &dimensions, &area),
              test::IsErrorWithSubstr("Data type not supported"));
}

// Tests that TrainedModelVariableStore fails to retrieve a variable of an
// unsupported type.
TEST_F(TrainedModelVariableStoreTest, LookupUnknownFormat) {
  AlignedArea area;
  std::vector<size_t> dimensions;

  TF_ASSERT_OK(store_.Reset(kSavedModelDir));
  EXPECT_THAT(store_.Lookup(kVariableName, VariableSpec::FORMAT_UNKNOWN,
                            &dimensions, &area),
              test::IsErrorWithSubstr("Unknown variable format"));
}

// Tests that TrainedModelVariableStore fails to look up a variable without
// sufficient structure as an matrix.
TEST_F(TrainedModelVariableStoreTest, LookupInsufficientRank) {
  AlignedArea area;
  std::vector<size_t> dimensions;

  TF_ASSERT_OK(store_.Reset(kSavedModelDir));
  EXPECT_THAT(
      store_.Lookup(kLowRankVariableName, VariableSpec::FORMAT_ROW_MAJOR_MATRIX,
                    &dimensions, &area),
      test::IsErrorWithSubstr("Tensor must be rank >= 2"));
}

// Tests that TrainedModelVariableStore produces column-blocked row-major
// matrices with the same content as the non-blocked version. Checks that
// bfloat16 matrices are a permuted version of blocked matrices.
TEST_F(TrainedModelVariableStoreTest, ColumnBlockedComparison) {
  const int kBlockSize = 32;
  const string kBlockedVariableName = GetBlockedVariableName(kBlockSize);
  const string kBfloat16VariableName = GetBfloat16VariableName(kBlockSize);

  Matrix<float> plain_matrix;
  BlockedMatrix<float> matrix;
  BlockedMatrix<TruncatedFloat16> bfloat16_matrix;

  TF_ASSERT_OK(store_.Reset(kSavedModelDir));
  TF_ASSERT_OK(store_.Lookup(kVariableName, &plain_matrix));
  TF_ASSERT_OK(store_.Lookup(kBlockedVariableName, &matrix));
  TF_ASSERT_OK(store_.Lookup(kBfloat16VariableName, &bfloat16_matrix));

  ASSERT_EQ(matrix.num_rows(), kVariableRows);
  ASSERT_EQ(matrix.num_columns(), kVariableColumns);
  ASSERT_EQ(matrix.block_size(), kBlockSize);

  // Compare the content of the plain matrix with the blocked version.
  for (int column = 0; column < matrix.num_columns(); ++column) {
    const int column_block_index = column / kBlockSize;
    const int index_in_block = column % kBlockSize;
    for (int row = 0; row < matrix.num_rows(); ++row) {
      const int block_index = column_block_index * matrix.num_rows() + row;
      Vector<float> block = matrix.vector(block_index);
      EXPECT_EQ(block[index_in_block], plain_matrix.row(row)[column]);
    }
  }

  // Compare bfloat16-encoded values with float32 values.
  ASSERT_EQ(matrix.num_vectors(), bfloat16_matrix.num_vectors());
  ASSERT_EQ(matrix.block_size(), bfloat16_matrix.block_size());
  ASSERT_EQ(matrix.num_rows(), bfloat16_matrix.num_rows());
  ASSERT_EQ(matrix.num_columns(), bfloat16_matrix.num_columns());
  for (int vector = 0; vector < matrix.num_vectors(); ++vector) {
    const auto &matrix_vector = matrix.vector(vector);
    const auto &bfloat16_vector = bfloat16_matrix.vector(vector);
    for (int i = 0; i < matrix.block_size(); ++i) {
      int permuted = FastUnpackPermutation(i);
      const float matrix_value = matrix_vector[i];
      const float bfloat16_value = bfloat16_vector[permuted].DebugToFloat();
      EXPECT_NEAR(matrix_value, bfloat16_value, 5e-3);
    }
  }
}

// Tests that TrainedModelVariableStore overwrites the dimension vector passed
// to Lookup().
TEST_F(TrainedModelVariableStoreTest, OverwritesDimensions) {
  const int kBlockSize = 32;
  const string kBlockedVariableName = GetBlockedVariableName(kBlockSize);

  TF_ASSERT_OK(store_.Reset(kSavedModelDir));

  std::vector<VariableSpec::Format> formats{
      VariableSpec::FORMAT_FLAT, VariableSpec::FORMAT_ROW_MAJOR_MATRIX,
      VariableSpec::FORMAT_COLUMN_BLOCKED_ROW_MAJOR_MATRIX};
  for (const auto &format : formats) {
    std::vector<size_t> dimensions;
    dimensions.push_back(1234);
    AlignedArea area;
    TF_ASSERT_OK(
        store_.Lookup(kBlockedVariableName, format, &dimensions, &area));
    EXPECT_NE(dimensions[0], 1234);

    std::vector<size_t> expected_dimensions;
    switch (format) {
      case VariableSpec::FORMAT_UNKNOWN:
        LOG(FATAL) << "Invalid format";

      case VariableSpec::FORMAT_FLAT:
        expected_dimensions = {kVariableRows * kVariableColumns};
        break;

      case VariableSpec::FORMAT_ROW_MAJOR_MATRIX:
        // NB: We're fetching the rank-3 "/matrix/blockedNN" version and then
        // reshaping into a matrix, so the dimensions are not the same as the
        // plain matrix.
        expected_dimensions = {kVariableRows * kVariableColumns / kBlockSize,
                               kBlockSize};
        break;

      case VariableSpec::FORMAT_COLUMN_BLOCKED_ROW_MAJOR_MATRIX:
        expected_dimensions = {kVariableRows, kVariableColumns, kBlockSize};
        break;
    }

    EXPECT_EQ(dimensions, expected_dimensions);
  }
}

}  // namespace
}  // namespace runtime
}  // namespace dragnn
}  // namespace syntaxnet
