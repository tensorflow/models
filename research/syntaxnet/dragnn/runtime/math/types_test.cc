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

#include "dragnn/runtime/math/types.h"

#include <stddef.h>
#include <string.h>
#include <set>

#include "dragnn/core/test/generic.h"
#include "dragnn/runtime/alignment.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"

namespace syntaxnet {
namespace dragnn {
namespace runtime {
namespace {

// Creates a pointer that is be invalid. This is useful for creating proxy areas
// for testing, whose real data should never be accessed. We manually tested
// that if this pointer is dereferenced, a segmentation fault will be thrown.
char *InvalidAlignedPointer() {
  return reinterpret_cast<char *>(3 * internal::kAlignmentBytes);
}

// Expects that two pointers point to the same address.
void ExpectSameAddress(const void *ptr1, const void *ptr2) {
  EXPECT_EQ(ptr1, ptr2);
}

template <class A, class B>
bool StructsEqual(const A &a, const B &b) {
  static_assert(sizeof(A) == sizeof(B),
                "StructsEqual must be given structs of the same size.");
  return memcmp(&a, &b, sizeof(A)) == 0;
}

// Tests that (Mutable)Vector is empty by default.
TEST(VectorTest, EmptyByDefault) {
  const Vector<int> vector1;
  EXPECT_EQ(vector1.size(), 0);
  EXPECT_TRUE(vector1.empty());

  const MutableVector<int> vector2;
  EXPECT_EQ(vector2.size(), 0);
  EXPECT_TRUE(vector2.empty());
}

// Tests that (Mutable)Vector can be initialized from a view.
TEST(VectorTest, ConstructFromView) {
  MutableAlignedView view;
  char *ptr = InvalidAlignedPointer();
  TF_ASSERT_OK(view.Reset(ptr, 10 * sizeof(int)));

  const Vector<int> vector1(view);
  ExpectSameAddress(vector1.data(), ptr);
  EXPECT_EQ(vector1.size(), 10);
  EXPECT_FALSE(vector1.empty());

  const MutableVector<int> vector2(view);
  ExpectSameAddress(vector2.data(), ptr);
  EXPECT_EQ(vector2.size(), 10);
  EXPECT_FALSE(vector2.empty());
}

// Tests that (Mutable)Vector can be initialized from a prefix of a view.
TEST(VectorTest, ConstructFromViewPrefix) {
  MutableAlignedView view;
  char *ptr = InvalidAlignedPointer();
  TF_ASSERT_OK(view.Reset(ptr, 10 * sizeof(int)));

  // Use a prefix of 3 of the 10 available ints in the |view|.
  const Vector<int> vector1(view, 3);
  ExpectSameAddress(vector1.data(), ptr);
  EXPECT_EQ(vector1.size(), 3);
  EXPECT_FALSE(vector1.empty());

  // Use a prefix of 5 of the 10 available ints in the |view|.
  const MutableVector<int> vector2(view, 5);
  ExpectSameAddress(vector2.data(), ptr);
  EXPECT_EQ(vector2.size(), 5);
  EXPECT_FALSE(vector2.empty());
}

// Tests that (Mutable)Vector supports copy-construction and assignment with
// shallow-copy semantics, and reinterprets from T* to const T*.
TEST(VectorTest, CopyAndAssign) {
  MutableAlignedView view;
  char *ptr = InvalidAlignedPointer();
  TF_ASSERT_OK(view.Reset(ptr, 10 * sizeof(int)));

  const MutableVector<int> vector1(view);

  // Copy-construct from another vector.
  MutableVector<int> vector2(vector1);
  ExpectSameAddress(vector2.data(), ptr);
  EXPECT_EQ(vector2.size(), 10);
  EXPECT_FALSE(vector2.empty());

  // Assign from an empty vector, effectively clearing it.
  vector2 = MutableVector<int>();
  EXPECT_EQ(vector2.size(), 0);
  EXPECT_TRUE(vector2.empty());

  // Assign from the original vector.
  vector2 = vector1;
  ExpectSameAddress(vector2.data(), ptr);
  EXPECT_EQ(vector2.size(), 10);
  EXPECT_FALSE(vector2.empty());

  // Copy-construct from another vector.  Note that this reinterprets type.
  Vector<int> vector3(vector1);
  ExpectSameAddress(vector3.data(), ptr);
  EXPECT_EQ(vector3.size(), 10);
  EXPECT_FALSE(vector3.empty());

  // Assign from an empty vector, effectively clearing it.
  vector3 = Vector<int>();
  EXPECT_EQ(vector3.size(), 0);
  EXPECT_TRUE(vector3.empty());

  // Assign from another vector.  Note that this reinterprets type.
  vector3 = vector2;
  ExpectSameAddress(vector3.data(), ptr);
  EXPECT_EQ(vector3.size(), 10);
  EXPECT_FALSE(vector3.empty());
}

// Tests that (Mutable)Vector supports access via operator[].
TEST(VectorTest, Subscript) {
  UniqueAlignedArray array;
  array.Reset(10 * sizeof(float));

  // Write into a mutable vector.
  const MutableVector<float> mutable_vector(array.view());
  ASSERT_EQ(mutable_vector.size(), 10);
  for (int i = 0; i < 10; ++i) mutable_vector[i] = i;

  // Read from a const vector that points at the same values.
  const Vector<float> const_vector(array.view());
  ASSERT_EQ(const_vector.size(), 10);
  for (int i = 0; i < 10; ++i) EXPECT_EQ(const_vector[i], i);
}

// Tests the subsequence operator.
TEST(VectorTest, Subsequence) {
  // Debug checks will fail if either of the constructed vectors is not aligned.
  constexpr int numAlignedFloats = internal::kAlignmentBytes / sizeof(float);

  UniqueAlignedArray array;
  array.Reset(2 * numAlignedFloats * sizeof(float));

  // Write into a mutable vector.
  const MutableVector<float> mutable_vector(array.view());
  for (int i = 0; i < 2 * numAlignedFloats; ++i) mutable_vector[i] = i;

  // Subscript beginning.
  Vector<float> first_half(mutable_vector.Subsequence(0, numAlignedFloats));
  ASSERT_EQ(first_half.size(), numAlignedFloats);
  for (int i = 0; i < numAlignedFloats; ++i) {
    EXPECT_EQ(first_half[i], i);
  }

  // Subscript end.
  Vector<float> second_half(
      mutable_vector.Subsequence(numAlignedFloats, numAlignedFloats));
  ASSERT_EQ(second_half.size(), numAlignedFloats);
  for (int i = 0; i < numAlignedFloats; ++i) {
    EXPECT_EQ(second_half[i], i + numAlignedFloats);
  }
}

// Tests that (Mutable)Vector supports access via range-based for loops.
TEST(VectorTest, RangeBasedFor) {
  UniqueAlignedArray array;
  array.Reset(10 * sizeof(float));

  // Write into a mutable vector.
  const MutableVector<float> mutable_vector(array.view());
  ASSERT_EQ(mutable_vector.size(), 10);
  float counter = 0.0;
  for (float &value : mutable_vector) value = counter++;

  // Read from a const vector that points at the same values.
  const Vector<float> const_vector(array.view());
  ASSERT_EQ(const_vector.size(), 10);
  counter = 0.0;
  for (const float &value : const_vector) EXPECT_EQ(value, counter++);
}

// Tests that (Mutable)Matrix is empty by default.
TEST(MatrixTest, EmptyByDefault) {
  const Matrix<int> matrix1;
  EXPECT_EQ(matrix1.num_rows(), 0);
  EXPECT_EQ(matrix1.num_columns(), 0);
  EXPECT_EQ(matrix1.row_stride(), 0);

  const MutableMatrix<int> matrix2;
  EXPECT_EQ(matrix2.num_rows(), 0);
  EXPECT_EQ(matrix2.num_columns(), 0);
  EXPECT_EQ(matrix2.row_stride(), 0);
}

// Tests that (Mutable)Matrix can be constructed from an area.
TEST(MatrixTest, ConstructFromArea) {
  MutableAlignedView view;
  char *ptr = InvalidAlignedPointer();
  const size_t kNumRows = 11;
  const size_t kNumColumns = 13;
  const size_t kRowBytes = kNumColumns * sizeof(int);
  const size_t kRowStride = PadToAlignment(kRowBytes) / sizeof(int);
  const size_t bytes = ComputeAlignedAreaSize(kNumRows, kRowBytes);
  TF_ASSERT_OK(view.Reset(ptr, bytes));

  MutableAlignedArea area;
  TF_ASSERT_OK(area.Reset(view, kNumRows, kRowBytes));

  const Matrix<int> matrix1(area);
  EXPECT_EQ(matrix1.num_rows(), kNumRows);
  EXPECT_EQ(matrix1.num_columns(), kNumColumns);
  EXPECT_EQ(matrix1.row_stride(), kRowStride);
  ExpectSameAddress(matrix1.row(0).data(), ptr);
  ExpectSameAddress(matrix1.data(), ptr);

  const MutableMatrix<int> matrix2(area);
  EXPECT_EQ(matrix2.num_rows(), kNumRows);
  EXPECT_EQ(matrix2.num_columns(), kNumColumns);
  EXPECT_EQ(matrix2.row_stride(), kRowStride);
  ExpectSameAddress(matrix2.row(0).data(), ptr);
  ExpectSameAddress(matrix2.data(), ptr);
}

// Tests that (Mutable)Matrix supports copy-construction and assignment with
// shallow-copy semantics, and reinterprets from T* to const T*.
TEST(MatrixTest, CopyAndAssign) {
  MutableAlignedView view;
  char *ptr = InvalidAlignedPointer();
  const size_t kNumRows = 11;
  const size_t kNumColumns = 13;
  const size_t kRowBytes = kNumColumns * sizeof(int);
  const size_t kRowStride = PadToAlignment(kRowBytes) / sizeof(int);
  const size_t bytes = ComputeAlignedAreaSize(kNumRows, kRowBytes);
  TF_ASSERT_OK(view.Reset(ptr, bytes));

  MutableAlignedArea area;
  TF_ASSERT_OK(area.Reset(view, kNumRows, kRowBytes));

  const MutableMatrix<int> matrix1(area);
  EXPECT_EQ(matrix1.num_rows(), kNumRows);
  EXPECT_EQ(matrix1.num_columns(), kNumColumns);
  EXPECT_EQ(matrix1.row_stride(), kRowStride);
  ExpectSameAddress(matrix1.row(0).data(), ptr);
  ExpectSameAddress(matrix1.data(), ptr);

  // Copy-construct from another matrix.
  MutableMatrix<int> matrix2(matrix1);
  EXPECT_EQ(matrix2.num_rows(), kNumRows);
  EXPECT_EQ(matrix2.num_columns(), kNumColumns);
  EXPECT_EQ(matrix2.row_stride(), kRowStride);
  ExpectSameAddress(matrix2.row(0).data(), ptr);
  ExpectSameAddress(matrix2.data(), ptr);

  // Assign from an empty matrix, effectively clearing it.
  matrix2 = MutableMatrix<int>();
  EXPECT_EQ(matrix2.num_rows(), 0);
  EXPECT_EQ(matrix2.num_columns(), 0);
  EXPECT_EQ(matrix2.row_stride(), 0);

  // Assign from the original matrix.
  matrix2 = matrix1;
  EXPECT_EQ(matrix2.num_rows(), kNumRows);
  EXPECT_EQ(matrix2.num_columns(), kNumColumns);
  EXPECT_EQ(matrix2.row_stride(), kRowStride);
  ExpectSameAddress(matrix2.row(0).data(), ptr);
  ExpectSameAddress(matrix2.data(), ptr);

  // Copy-construct from another matrix.  Note that this reinterprets type.
  Matrix<int> matrix3(matrix2);
  EXPECT_EQ(matrix3.num_rows(), kNumRows);
  EXPECT_EQ(matrix3.num_columns(), kNumColumns);
  EXPECT_EQ(matrix3.row_stride(), kRowStride);
  ExpectSameAddress(matrix3.row(0).data(), ptr);
  ExpectSameAddress(matrix3.data(), ptr);

  // Assign from an empty matrix, effectively clearing it.
  matrix3 = Matrix<int>();
  EXPECT_EQ(matrix3.num_rows(), 0);
  EXPECT_EQ(matrix3.num_columns(), 0);
  EXPECT_EQ(matrix3.row_stride(), 0);

  // Assign from the original matrix.  Note that this reinterprets type.
  matrix3 = matrix1;
  EXPECT_EQ(matrix3.num_rows(), kNumRows);
  EXPECT_EQ(matrix3.num_columns(), kNumColumns);
  EXPECT_EQ(matrix3.row_stride(), kRowStride);
  ExpectSameAddress(matrix3.row(0).data(), ptr);
  ExpectSameAddress(matrix3.data(), ptr);
}

// Tests that (Mutable)Matrix supports row access.
TEST(MatrixTest, Rows) {
  const size_t kNumRows = 11;
  const size_t kNumColumns = 13;
  const size_t bytes =
      ComputeAlignedAreaSize(kNumRows, kNumColumns * sizeof(float));
  UniqueAlignedArray array;
  array.Reset(bytes);

  MutableAlignedArea area;
  TF_ASSERT_OK(area.Reset(array.view(), kNumRows, kNumColumns * sizeof(float)));

  // Write to a mutable matrix.
  const MutableMatrix<float> mutable_matrix(area);
  ASSERT_EQ(mutable_matrix.num_rows(), kNumRows);
  ASSERT_EQ(mutable_matrix.num_columns(), kNumColumns);
  for (size_t row = 0; row < kNumRows; ++row) {
    for (size_t column = 0; column < kNumColumns; ++column) {
      mutable_matrix.row(row)[column] = row * 1000.0 + column;
    }
  }

  // Read from a const matrix that points at the same values.
  const Matrix<float> const_matrix(area);
  ASSERT_EQ(const_matrix.num_rows(), kNumRows);
  ASSERT_EQ(const_matrix.num_columns(), kNumColumns);
  for (size_t row = 0; row < kNumRows; ++row) {
    for (size_t column = 0; column < kNumColumns; ++column) {
      EXPECT_EQ(const_matrix.row(row)[column], row * 1000.0 + column);
    }
  }
}

TEST(MatrixTest, MatrixFromVector) {
  for (int cols = 0; cols < 100; ++cols) {
    MutableAlignedView view;
    char *ptr = InvalidAlignedPointer();
    TF_ASSERT_OK(view.Reset(ptr, cols * sizeof(int)));
    const MutableVector<int> vector(view);
    const MutableMatrix<int> matrix(vector);
    ASSERT_EQ(matrix.row(0).data(), vector.data());
    ExpectSameAddress(matrix.data(), vector.data());
    ASSERT_EQ(matrix.num_rows(), 1);
    ASSERT_EQ(matrix.num_columns(), vector.size());
  }
}

template <class MatrixType>
class BlockedMatrixTest : public ::testing::Test {};

typedef ::testing::Types<
    BlockedMatrix<float, BlockedMatrixFormat::kRowBlockedColumnMajor>,
    BlockedMatrix<float, BlockedMatrixFormat::kColumnBlockedRowMajor>,
    BlockedMatrix<int64, BlockedMatrixFormat::kRowBlockedColumnMajor>,
    BlockedMatrix<int64, BlockedMatrixFormat::kColumnBlockedRowMajor>>
    BlockedRowAndColumnTypes;
TYPED_TEST_CASE(BlockedMatrixTest, BlockedRowAndColumnTypes);

TYPED_TEST(BlockedMatrixTest, PaddingNotAllowed) {
  MutableAlignedView view;
  MutableAlignedArea area;
  constexpr size_t kNumRows = 10;
  constexpr size_t kNumColumns = 10;
  constexpr size_t kBlockSize = 5;
  constexpr size_t kNumViews = (kNumRows * kNumColumns) / kBlockSize;
  constexpr size_t kBlockSizeBytes =
      kBlockSize * sizeof(typename TypeParam::ElementType);
  const size_t bytes = ComputeAlignedAreaSize(kNumViews, kBlockSizeBytes);
  TF_ASSERT_OK(view.Reset(InvalidAlignedPointer(), bytes));
  TF_ASSERT_OK(area.Reset(view, kNumViews, kBlockSizeBytes));

  // 5 is usually relatively prime to the alignment size, but you may have to
  // update this test if kAlignmentBytes changes.
  ASSERT_NE(PadToAlignment(kBlockSizeBytes), kBlockSizeBytes);

  TypeParam matrix;
  EXPECT_THAT(matrix.Reset(area, kNumRows, kNumColumns),
              test::IsErrorWithSubstr(
                  "Padding is not supported for blocked matrix formats."));
}

// Tests accessors, and the size of matrices after allocation.
TYPED_TEST(BlockedMatrixTest, Accessors) {
  MutableAlignedView view;
  MutableAlignedArea area;
  char *ptr = InvalidAlignedPointer();
  constexpr size_t kNumRows = 48;
  constexpr size_t kNumColumns = 24;
  constexpr size_t kBlockSize = 8;
  constexpr size_t kNumViews = (kNumRows * kNumColumns) / kBlockSize;
  constexpr size_t kBlockSizeBytes =
      kBlockSize * sizeof(typename TypeParam::ElementType);
  const size_t bytes = ComputeAlignedAreaSize(kNumViews, kBlockSizeBytes);
  TF_ASSERT_OK(view.Reset(ptr, bytes));
  TF_ASSERT_OK(area.Reset(view, kNumViews, kBlockSizeBytes));

  TypeParam matrix;

  // If the view size is wrong, it should fail.
  EXPECT_THAT(
      matrix.Reset(area, kNumRows + 1, kNumColumns),
      test::IsErrorWithSubstr("Area has 144 views, but should have 147"));

  // If the blocking scheme cannot divide the matrix evenly, an error should
  // be raised. The choice of 12 and 96 is a bit non-trivial: they are numbers
  // that conveniently result in the correct area (so other errors won't be
  // raised), but an incompatible division of the vectors.
  if (TypeParam::IsRowBlocked()) {
    EXPECT_THAT(
        matrix.Reset(area, 12, 96),
        test::IsErrorWithSubstr("row-blocked matrix has major dimension 12 "
                                "which is not divisible by the block "
                                "size, 8"));
  } else {
    EXPECT_THAT(
        matrix.Reset(area, 96, 12),
        test::IsErrorWithSubstr("column-blocked matrix has major dimension "
                                "12 which is not divisible by the block "
                                "size, 8"));
  }

  TF_EXPECT_OK(matrix.Reset(area, kNumRows, kNumColumns));

  EXPECT_EQ(matrix.vector(0).data(),
            reinterpret_cast<typename TypeParam::ElementType *>(ptr));
  EXPECT_EQ(matrix.num_rows(), kNumRows);
  EXPECT_EQ(matrix.num_columns(), kNumColumns);
  EXPECT_EQ(matrix.block_size(), kBlockSize);
  EXPECT_EQ(matrix.num_vectors(), kNumViews);
}

TYPED_TEST(BlockedMatrixTest, CopyCastTranspose) {
  MutableAlignedView view;
  MutableAlignedArea area;
  constexpr size_t kNumRows = 48;
  constexpr size_t kNumColumns = 24;
  constexpr size_t kBlockSize = 8;
  constexpr size_t kNumViews = (kNumRows * kNumColumns) / kBlockSize;
  constexpr size_t kBlockSizeBytes =
      kBlockSize * sizeof(typename TypeParam::ElementType);
  const size_t bytes = ComputeAlignedAreaSize(kNumViews, kBlockSizeBytes);
  TF_ASSERT_OK(view.Reset(InvalidAlignedPointer(), bytes));
  TF_ASSERT_OK(area.Reset(view, kNumViews, kBlockSizeBytes));

  TypeParam matrix;
  TF_EXPECT_OK(matrix.Reset(area, kNumRows, kNumColumns));

  // Test both copying and casting to const.
  TypeParam matrix_copy = matrix;
  auto readonly = matrix.AsConst();
  EXPECT_TRUE(StructsEqual(matrix, matrix_copy));
  EXPECT_TRUE(StructsEqual(matrix, readonly));
  for (int i = 0; i < kNumViews; ++i) {
    EXPECT_EQ(matrix.vector(i).data(), matrix_copy.vector(i).data());
    EXPECT_EQ(matrix.vector(i).data(), readonly.vector(i).data());
  }

  // Transpose matrix.
  auto transposed = matrix.Transpose();
  auto readonly_transposed = readonly.Transpose();
  EXPECT_FALSE(StructsEqual(matrix, transposed));
  EXPECT_FALSE(StructsEqual(readonly, readonly_transposed));
  EXPECT_TRUE(StructsEqual(transposed, readonly_transposed));
}

}  // namespace
}  // namespace runtime
}  // namespace dragnn
}  // namespace syntaxnet
