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

// Mathematical types.

#ifndef DRAGNN_RUNTIME_MATH_TYPES_H_
#define DRAGNN_RUNTIME_MATH_TYPES_H_

#include <stddef.h>
#include <limits>

#include "dragnn/runtime/alignment.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/logging.h"

namespace syntaxnet {
namespace dragnn {
namespace runtime {

// Blocked matrix formats, for fast inference routines.
enum class BlockedMatrixFormat {
  // Represents a row-blocked block-column-major matrix. In other words, first
  // split a matrix M into
  //
  // [ M_1
  //   ...
  //   M_m ]
  //
  // sub-matrices, where each M_i is a `block_size x n` sub-matrix. Then each
  // M_i is formatted in column-major order, and the sub-matrices' data is
  // concatenated together.
  kRowBlockedColumnMajor,

  // Represents a column-blocked block-row-major matrix. This is the
  // transpose of the above. A matrix M is split into
  //
  // [ M_1 ... M_n ]
  //
  // sub-matrices, where each M_i is a `m x block_size` sub-matrix. Then each
  // M_i is formatted in row-major order, and the sub-matrices' data is
  // concatenated together.
  kColumnBlockedRowMajor,
};

namespace internal {

// An aligned vector of values.  Do not use this class directly, instead use
// (Mutable)Vector below.
template <class T>
class VectorImpl {
 public:
  static_assert(IsAlignable<T>(), "T must be alignable");

  // Creates an empty vector.
  VectorImpl() = default;

  // Points this at the |view|, which must be evenly divisible into Ts.
  template <class Byte>
  explicit VectorImpl(AlignedViewImpl<Byte> view);

  // Points this at a prefix of the |view| containing |size| Ts.  The |view|
  // must span at least |size| * sizeof(T) bytes.
  template <class Byte>
  VectorImpl(AlignedViewImpl<Byte> view, size_t size);

  // Points this at the same values as |that|, possibly reinterpreting type.
  template <class U>
  explicit VectorImpl(VectorImpl<U> that);
  template <class U>
  VectorImpl &operator=(VectorImpl<U> that);

  // Enables range-based for loops.
  T *begin() const { return data(); }
  T *end() const { return begin() + size(); }

  // Accessors.
  T *data() const { return data_; }
  size_t size() const { return size_; }
  bool empty() const { return size() == 0; }
  T &operator[](size_t index) const;

  // Gets a sub-vector starting at |start| with |size| elements.
  VectorImpl<T> Subsequence(size_t start, size_t size) const;

 private:
  template <class U>
  friend class MatrixImpl;
  template <class U, BlockedMatrixFormat format>
  friend class BlockedMatrixImpl;

  // Points this at [|data|,|data|+|size|), bypassing alignment checks.
  VectorImpl(T *data, size_t size);

  // Pointer to the start of the vector.
  T *data_ = nullptr;

  // Number of values in the vector.
  size_t size_ = 0;
};

// Returns the format corresponding to the transpose of the |format|.
constexpr BlockedMatrixFormat TransposeBlockedMatrixFormat(
    BlockedMatrixFormat format);

// A row-major matrix where each row or column is aligned.  Do not use this
// class directly, instead use (Mutable)Matrix below.
template <class T>
class MatrixImpl {
 public:
  static_assert(IsAlignable<T>(), "T must be alignable");

  // Creates an empty matrix.
  MatrixImpl() = default;

  // Points each row of this matrix at the corresponding sub-view of the |area|.
  // Each view in the |area| must be evenly divisible into Ts.
  template <class Byte>
  explicit MatrixImpl(AlignedAreaImpl<Byte> area);

  // Creates a matrix from a single vector. Assumes that the vector's stride is
  // the minimum alignment padding.
  explicit MatrixImpl(VectorImpl<T> single_vector);

  // Points this at the same values as |that|.
  template <class U>
  explicit MatrixImpl(MatrixImpl<U> that);
  template <class U>
  MatrixImpl &operator=(MatrixImpl<U> that);

  // Accessors.
  T *data() const { return data_; }
  size_t num_rows() const { return num_rows_; }
  size_t num_columns() const { return num_columns_; }
  size_t row_stride() const { return row_stride_; }
  VectorImpl<T> row(size_t index) const;

 private:
  template <class U>
  friend class MatrixImpl;

  // Pointer to the start of the matrix.
  T *data_ = nullptr;

  // Number of rows and columns in the matrix.
  size_t num_rows_ = 0;
  size_t num_columns_ = 0;

  // Distance between the starts of consecutive rows.
  size_t row_stride_ = 0;
};

// Blocked matrix representation. See BlockedMatrixFormat for details.
template <class T, BlockedMatrixFormat format>
class BlockedMatrixImpl {
 public:
  static_assert(IsAlignable<T>(), "T must be alignable");

  // These aliases allow templated code to reach back in and get template
  // parameters, like std::vector<T>::iterator::value aliases.
  using ElementType = T;
  static constexpr bool IsRowBlocked() {
    return format == BlockedMatrixFormat::kRowBlockedColumnMajor;
  }

  // Creates an empty matrix.
  BlockedMatrixImpl() = default;

  // Creates a copy of this matrix, using the same values (underlying area), but
  // possibly re-interpreting the type. The new type U must be the same size,
  // and `T *` must be implictly convertible to `U *` (usually just adding
  // "const" qualifiers, but theoretically it could be a superclass).
  template <class U>
  explicit BlockedMatrixImpl(BlockedMatrixImpl<U, format> that);
  template <class U>
  BlockedMatrixImpl &operator=(BlockedMatrixImpl<U, format> that);

  // Creates a new view that's const-qualified, in particular converting
  // MutableBlockedMatrix to BlockedMatrix.
  BlockedMatrixImpl<const T, format> AsConst() const {
    return BlockedMatrixImpl<const T, format>(*this);
  }

  // Initializes the matrix. Raises errors if the matrix dimensions are
  // incompatible with the underlying area, namely if the number of views in
  // `area` do not cover the whole matrix, and also if the matrix cannot be
  // blocked according to (template parameter) `format`.
  //
  // Further, because this class is used for (delicate / specialized) optimized
  // inference routines, it is also required that no padding is present, i.e.
  // that the block size is divisible by kAlignmentBytes (currently 32).
  template <class Byte>
  tensorflow::Status Reset(AlignedAreaImpl<Byte> area, size_t num_rows,
                           size_t num_columns);

  // Returns the transpose of this.
  BlockedMatrixImpl<T, TransposeBlockedMatrixFormat(format)> Transpose() const;

  // Accessors.
  size_t num_rows() const { return num_rows_; }
  size_t num_columns() const { return num_columns_; }
  size_t block_size() const { return block_size_; }
  size_t num_vectors() const { return num_vectors_; }
  VectorImpl<T> vector(size_t index) const;

 private:
  template <class U, BlockedMatrixFormat other_format>
  friend class BlockedMatrixImpl;

  // This is the same as calling Reset(), except the area is not checked.
  template <class Byte>
  explicit BlockedMatrixImpl(AlignedAreaImpl<Byte> area, int num_rows,
                             int num_columns);

  // Pointer to the start of the matrix.
  T *data_ = nullptr;

  // Number of rows and columns in the matrix. Unlike MatrixImpl, there is no
  // API for directly accessing rows and columns, but it's necessary for any
  // algorithm (e.g. matrix-vector multiplication) to know the logical shape.
  size_t num_rows_ = 0;
  size_t num_columns_ = 0;

  size_t block_size_ = 0;   // in T's
  size_t num_vectors_ = 0;  // = num_rows * num_columns / block_size
};

}  // namespace internal

// Public aliases; use these.
template <class T>
using Vector = internal::VectorImpl<const T>;
template <class T>
using Matrix = internal::MatrixImpl<const T>;
template <class T, BlockedMatrixFormat format =
                       BlockedMatrixFormat::kColumnBlockedRowMajor>
using BlockedMatrix = internal::BlockedMatrixImpl<const T, format>;
template <class T>
using MutableVector = internal::VectorImpl<T>;
template <class T>
using MutableMatrix = internal::MatrixImpl<T>;
template <class T, BlockedMatrixFormat format =
                       BlockedMatrixFormat::kColumnBlockedRowMajor>
using MutableBlockedMatrix = internal::BlockedMatrixImpl<T, format>;

// Implementation details below.

namespace internal {

template <class T>
template <class Byte>
VectorImpl<T>::VectorImpl(AlignedViewImpl<Byte> view)
    : data_(reinterpret_cast<T *>(view.data())),
      size_(view.size() / sizeof(T)) {
  DCHECK_EQ(view.size() % sizeof(T), 0);
}

template <class T>
template <class Byte>
VectorImpl<T>::VectorImpl(AlignedViewImpl<Byte> view, size_t size)
    : data_(reinterpret_cast<T *>(view.data())), size_(size) {
  DCHECK_LE(size * sizeof(T), view.size());
}

template <class T>
template <class U>
VectorImpl<T>::VectorImpl(VectorImpl<U> that)
    : data_(that.data()), size_(that.size()) {
  static_assert(sizeof(T) == sizeof(U), "T and U must be the same size");
}

template <class T>
template <class U>
VectorImpl<T> &VectorImpl<T>::operator=(VectorImpl<U> that) {
  static_assert(sizeof(T) == sizeof(U), "T and U must be the same size");
  data_ = that.data();
  size_ = that.size();
  return *this;
}

template <class T>
T &VectorImpl<T>::operator[](size_t index) const {
  DCHECK_LT(index, size());
  return data_[index];
}

template <class T>
VectorImpl<T>::VectorImpl(T *data, size_t size) : data_(data), size_(size) {
  TF_DCHECK_OK(OkIfAligned(data));
}

template <class T>
VectorImpl<T> VectorImpl<T>::Subsequence(size_t start, size_t size) const {
  DCHECK_LE(start + size, size_);
  return VectorImpl<T>(&data_[start], size);
}

constexpr BlockedMatrixFormat TransposeBlockedMatrixFormat(
    BlockedMatrixFormat format) {
  return format == BlockedMatrixFormat::kRowBlockedColumnMajor
             ? BlockedMatrixFormat::kColumnBlockedRowMajor
             : BlockedMatrixFormat::kRowBlockedColumnMajor;
}

template <class T>
MatrixImpl<T>::MatrixImpl(VectorImpl<T> single_vector)
    : data_(single_vector.data()),
      num_rows_(1),
      num_columns_(single_vector.size()),
      row_stride_(PadToAlignment(single_vector.size() * sizeof(T)) /
                  sizeof(T)) {}

template <class T>
template <class Byte>
MatrixImpl<T>::MatrixImpl(AlignedAreaImpl<Byte> area)
    : data_(reinterpret_cast<T *>(area.data())),
      num_rows_(area.num_views()),
      num_columns_(area.view_size() / sizeof(T)),
      row_stride_(area.view_stride() / sizeof(T)) {
  DCHECK_EQ(area.view_size() % sizeof(T), 0);
  DCHECK_EQ(area.view_stride() % sizeof(T), 0);
}

template <class T>
template <class U>
MatrixImpl<T>::MatrixImpl(MatrixImpl<U> that)
    : data_(that.data_),
      num_rows_(that.num_rows()),
      num_columns_(that.num_columns()),
      row_stride_(that.row_stride_) {
  static_assert(sizeof(T) == sizeof(U), "T and U must be the same size");
}

template <class T>
template <class U>
MatrixImpl<T> &MatrixImpl<T>::operator=(MatrixImpl<U> that) {
  static_assert(sizeof(T) == sizeof(U), "T and U must be the same size");
  data_ = that.data_;
  num_rows_ = that.num_rows();
  num_columns_ = that.num_columns();
  row_stride_ = that.row_stride_;
  return *this;
}

template <class T>
VectorImpl<T> MatrixImpl<T>::row(size_t index) const {
  DCHECK_LT(index, num_rows());

  // Note that |row_stride_|, not |num_columns_|, determines the start of the
  // row.  The former is aligned and may stride over a wider span than normal
  // when this is a "slice" of a larger matrix.
  return VectorImpl<T>(data_ + row_stride_ * index, num_columns());
}

template <class T, BlockedMatrixFormat format>
template <class U>
BlockedMatrixImpl<T, format>::BlockedMatrixImpl(
    BlockedMatrixImpl<U, format> that)
    : data_(that.data_),
      num_rows_(that.num_rows()),
      num_columns_(that.num_columns()),
      block_size_(that.block_size()),
      num_vectors_(that.num_vectors()) {
  static_assert(sizeof(T) == sizeof(U), "T and U must be the same size");
}

template <class T, BlockedMatrixFormat format>
template <class U>
BlockedMatrixImpl<T, format> &BlockedMatrixImpl<T, format>::operator=(
    BlockedMatrixImpl<U, format> that) {
  static_assert(sizeof(T) == sizeof(U), "T and U must be the same size");
  data_ = that.data_;
  num_rows_ = that.num_rows();
  num_columns_ = that.num_columns();
  block_size_ = that.block_size();
  num_vectors_ = that.num_vectors();
  return *this;
}

template <class T, BlockedMatrixFormat format>
template <class Byte>
tensorflow::Status BlockedMatrixImpl<T, format>::Reset(
    AlignedAreaImpl<Byte> area, size_t num_rows, size_t num_columns) {
  data_ = reinterpret_cast<T *>(area.view(0).data());
  num_rows_ = num_rows;
  num_columns_ = num_columns;
  block_size_ = area.view_size() / sizeof(T);
  num_vectors_ = num_rows * num_columns / block_size_;

  if (area.view_stride() != area.view_size()) {
    return tensorflow::errors::InvalidArgument(
        "Padding is not supported for blocked matrix formats. Underlying area "
        "has size ",
        area.view_size(), " which is padded to stride ", area.view_stride(),
        ".");
  }
  if (area.view_size() % sizeof(T) != 0) {
    return tensorflow::errors::InvalidArgument(
        "View size ", area.view_size(),
        " is not a multiple of the templated type's size, ", sizeof(T));
  }
  if (num_vectors_ != area.num_views()) {
    return tensorflow::errors::InvalidArgument("Area has ", area.num_views(),
                                               " views, but should have ",
                                               num_vectors_);
  }

  // The block dimension must divide rows or columns evenly.
  size_t divided_dimension = IsRowBlocked() ? num_rows : num_columns;
  if (divided_dimension % block_size_ != 0) {
    return tensorflow::errors::InvalidArgument(
        IsRowBlocked() ? "row" : "column",
        "-blocked matrix has major dimension ", divided_dimension,
        " which is not divisible by the block size, ", block_size_);
  }

  return tensorflow::Status::OK();
}

template <class T, BlockedMatrixFormat format>
VectorImpl<T> BlockedMatrixImpl<T, format>::vector(size_t index) const {
  DCHECK_LT(index, num_vectors_);
  return VectorImpl<T>(data_ + block_size_ * index, block_size_);
}

template <class T, BlockedMatrixFormat format>
BlockedMatrixImpl<T, TransposeBlockedMatrixFormat(format)>
BlockedMatrixImpl<T, format>::Transpose() const {
  BlockedMatrixImpl<T, TransposeBlockedMatrixFormat(format)> result;
  result.data_ = data_;
  result.num_columns_ = num_rows_;
  result.num_rows_ = num_columns_;
  result.block_size_ = block_size_;
  result.num_vectors_ = num_vectors_;
  return result;
}

}  // namespace internal
}  // namespace runtime
}  // namespace dragnn
}  // namespace syntaxnet

#endif  // DRAGNN_RUNTIME_MATH_TYPES_H_
