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

// Utils for working with aligned memory blocks.  The DRAGNN runtime requires
// aligned memory for use in vectorized math.  Do not rely on any particular
// value of the alignment requirement, because it will vary over time and in
// different build configurations.

#ifndef DRAGNN_RUNTIME_ALIGNMENT_H_
#define DRAGNN_RUNTIME_ALIGNMENT_H_

#include <stddef.h>
#include <stdint.h>
#include <string.h>
#include <memory>
#include <type_traits>
#include <vector>

#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/platform/logging.h"

// This is a type that has some private methods (so non-POD), but is known to be
// trivially-deconstructable. Ergo we add some special handling so
// IsAlignable<bfloat16> returns true.
namespace tensorflow {
struct bfloat16;
}

namespace syntaxnet {
namespace dragnn {
namespace runtime {

// Returns true if |T| can be used in an aligned memory block.
template <class T>
constexpr bool IsAlignable();

// Returns OK iff the |pointer| satisfies the alignment requirement.
tensorflow::Status OkIfAligned(const void *pointer);

// Returns the next alignment boundary at or after the |byte_offset|.
size_t PadToAlignment(size_t byte_offset);

// As above, but for pointers.
template <class T>
T *PadToAlignment(T *pointer);

// Returns the number of bytes required to store a sequence of |num_arrays|
// aligned arrays of |array_size| bytes, including alignment padding.  See
// (Mutable)AlignedArea below.
size_t ComputeAlignedAreaSize(size_t num_arrays, size_t array_size);

// Returns the number of bytes required to store a sequence of byte arrays of
// the given |sizes|, including alignment padding after each array.
size_t ComputeTotalBytesWithAlignmentPadding(const std::vector<size_t> &sizes);

// Forward-declared for friendship below.
class Operands;
class UniqueAlignedArray;
enum class BlockedMatrixFormat;

namespace internal {

// A non-owning view of an aligned byte array.  Templated so const and mutable
// versions can share implementation.  Do not use this class directly, instead
// use (Mutable)AlignedView below.
template <class Byte>
class AlignedViewImpl {
 public:
  static_assert(sizeof(Byte) == 1, "Byte must be byte-sized");

  // Creates an empty view.
  AlignedViewImpl() = default;

  // Points this at the same bytes as |that|, possibly reinterpreting type.
  template <class OtherByte>
  explicit AlignedViewImpl(AlignedViewImpl<OtherByte> that);
  template <class OtherByte>
  AlignedViewImpl &operator=(AlignedViewImpl<OtherByte> that);

  // Points this at [|data|,|data|+|size|).  On error, returns non-OK and
  // modifies nothing.
  tensorflow::Status Reset(Byte *data, size_t size);

  // Splits this into a list of |views| of the |sizes|, possibly reinterpreting
  // type.  The |views| need not completely cover all bytes of this.  Requires
  // that this spans ComputeTotalBytesWithAlignmentPadding(|sizes|) bytes.  On
  // error, returns non-OK and modifies nothing.
  template <class OtherByte>
  tensorflow::Status Split(
      const std::vector<size_t> &sizes,
      std::vector<AlignedViewImpl<OtherByte>> *views) const;

  // Accessors.
  Byte *data() const { return data_; }
  size_t size() const { return size_; }
  bool empty() const { return size() == 0; }

 private:
  template <class OtherByte>
  friend class AlignedViewImpl;
  template <class OtherByte>
  friend class AlignedAreaImpl;
  friend Operands;
  friend UniqueAlignedArray;

  // Directly creates an aligned view, bypassing alignment checks.
  AlignedViewImpl(Byte *data, size_t size);

  // Pointer to the start of the view.
  Byte *data_ = nullptr;

  // Number of bytes in the view.
  size_t size_ = 0;
};

// A non-owning view of an aligned, 2-dimensional byte array.  Templated so
// const and mutable versons can share implementation.  Do not use this class
// directly, instead use (Mutable)AlignedArea below.
template <class Byte>
class AlignedAreaImpl {
 public:
  static_assert(sizeof(Byte) == 1, "Byte must be byte-sized");

  // Creates an empty area.
  AlignedAreaImpl() = default;

  // Points this at the same bytes as |that|, possibly reinterpreting type.
  template <class OtherByte>
  explicit AlignedAreaImpl(AlignedAreaImpl<OtherByte> that);
  template <class OtherByte>
  AlignedAreaImpl &operator=(AlignedAreaImpl<OtherByte> that);

  // Resets this to a sequence of |num_views| aligned sub-views of the |view|,
  // each |view_size| bytes wide.  The first sub-view covers [0,|view_size|) of
  // |view|, and each subsequent sub-view starts at the next alignment boundary.
  // Requires that |view| spans ComputeAlignedAreaSize(|num_views|,|view_size|)
  // bytes or more.  On error, returns non-OK and modifies nothing.
  template <class OtherByte>
  tensorflow::Status Reset(AlignedViewImpl<OtherByte> view, size_t num_views,
                           size_t view_size);

  // Accessors.
  AlignedViewImpl<Byte> view(size_t index) const;
  Byte *data() const { return data_; }
  size_t num_views() const { return num_views_; }
  size_t view_size() const { return view_size_; }
  size_t view_stride() const { return view_stride_; }
  bool empty() const { return num_views() == 0; }

 private:
  template <class OtherByte>
  friend class AlignedAreaImpl;
  friend Operands;

  // Directly creates an aligned view, bypassing alignment checks.
  AlignedAreaImpl(Byte *data, size_t num_views, size_t view_size,
                  size_t view_stride);

  // Pointer to the start of the first view.
  Byte *data_ = nullptr;

  // Number of views in the area.
  size_t num_views_ = 0;

  // Size of each view in bytes, excluding alignment padding.
  size_t view_size_ = 0;

  // Number of bytes between the starts of consecutive views.  NB: This is not
  // necessarily equal to PadToAlignment(|view_size_|).
  size_t view_stride_ = 0;
};

}  // namespace internal

// Public aliases; use these.
using AlignedView = internal::AlignedViewImpl<const char>;
using AlignedArea = internal::AlignedAreaImpl<const char>;
using MutableAlignedView = internal::AlignedViewImpl<char>;
using MutableAlignedArea = internal::AlignedAreaImpl<char>;

// A uniquely-owned aligned byte array.
class UniqueAlignedArray {
 public:
  // Creates an empty byte array.
  UniqueAlignedArray() = default;

  // Reallocates this to |new_size| bytes, and discards the current byte array.
  // Contents are uninitialized.
  void Reset(size_t new_size);

  // Like Reset(), but only reallocates if |new_size| is more than the current
  // capacity.  NB: Does not preserve current content when reallocation occurs;
  // use Resize() if that is desired.
  void Reserve(size_t new_size);

  // Resizes this to contain |new_size| bytes, preserving current content.  If
  // |new_size| exceeds the current size, the added bytes are uninitialized.  If
  // |new_size| exceeds the current capacity, reallocates, and copies current
  // content.  Returns true if reallocation occurred.
  bool Resize(size_t new_size);

  // Returns the aligned byte array.
  MutableAlignedView view() const { return view_; }

 private:
  // Underlying byte array, which is padded for alignment.
  std::unique_ptr<char[]> padded_array_;

  // Size of the aligned portion of |padded_array_|.
  size_t capacity_ = 0;

  // Active range of the |storage_|.
  MutableAlignedView view_;
};

// Implementation details below.

namespace internal {

// Required alignment for memory blocks.  Only the runtime framework should use
// this; otherwise, DO NOT access or otherwise depend on this value.
enum : size_t { kAlignmentBytes = 32 };

}  // namespace internal

template <class T>
constexpr bool IsAlignable() {
  // Either T is divisible into alignment windows, or an alignment window is
  // divisible into Ts.  Likewise for T's alignment requirement.  Finally, T
  // must be POD because we won't call its constructor or destructor.
  return (sizeof(T) % internal::kAlignmentBytes == 0 ||
          internal::kAlignmentBytes % sizeof(T) == 0) &&
         (alignof(T) % internal::kAlignmentBytes == 0 ||
          internal::kAlignmentBytes % alignof(T) == 0) &&
         (std::is_pod<T>::value ||
          std::is_same<T, tensorflow::bfloat16>::value);
}

inline tensorflow::Status OkIfAligned(const void *pointer) {
  const uintptr_t address = reinterpret_cast<uintptr_t>(pointer);
  if (address % internal::kAlignmentBytes != 0) {
    return tensorflow::errors::InvalidArgument(
        "Pointer fails alignment requirement: ", address, " vs required ",
        internal::kAlignmentBytes);
  }
  return tensorflow::Status::OK();
}

inline size_t PadToAlignment(size_t byte_offset) {
  // Round up to the next alignment boundary by incrementing by a certain amount
  // and then rounding down.  Note that the bitmask clears the low-order bits of
  // the offset, effectively rounding down to the previous alignment boundary.
  return (byte_offset + internal::kAlignmentBytes - 1) &
         ~(internal::kAlignmentBytes - 1);
}

template <class T>
T *PadToAlignment(T *pointer) {
  static_assert(IsAlignable<T>(), "T is not alignable");
  uintptr_t address = reinterpret_cast<uintptr_t>(pointer);
  address = (address + internal::kAlignmentBytes - 1) &
            ~(internal::kAlignmentBytes - 1);
  return reinterpret_cast<T *>(address);
}

inline size_t ComputeAlignedAreaSize(size_t num_arrays, size_t array_size) {
  return num_arrays * PadToAlignment(array_size);
}

inline size_t ComputeTotalBytesWithAlignmentPadding(
    const std::vector<size_t> &sizes) {
  size_t total = 0;
  for (const size_t size : sizes) total += PadToAlignment(size);
  return total;
}

namespace internal {

template <class Byte>
template <class OtherByte>
AlignedViewImpl<Byte>::AlignedViewImpl(AlignedViewImpl<OtherByte> that)
    : data_(reinterpret_cast<Byte *>(that.data())), size_(that.size()) {}

template <class Byte>
template <class OtherByte>
AlignedViewImpl<Byte> &AlignedViewImpl<Byte>::operator=(
    AlignedViewImpl<OtherByte> that) {
  data_ = reinterpret_cast<Byte *>(that.data());
  size_ = that.size();
  return *this;
}

template <class Byte>
tensorflow::Status AlignedViewImpl<Byte>::Reset(Byte *data, size_t size) {
  TF_RETURN_IF_ERROR(OkIfAligned(data));

  // Success; make modifications.
  data_ = data;
  size_ = size;
  return tensorflow::Status::OK();
}

template <class Byte>
template <class OtherByte>
tensorflow::Status AlignedViewImpl<Byte>::Split(
    const std::vector<size_t> &sizes,
    std::vector<AlignedViewImpl<OtherByte>> *views) const {
  const size_t total_bytes = ComputeTotalBytesWithAlignmentPadding(sizes);
  if (size() < total_bytes) {
    return tensorflow::errors::InvalidArgument(
        "View is too small to be split into sizes [",
        tensorflow::str_util::Join(sizes, ", "), "]: need ", total_bytes,
        " bytes but have ", size(), " bytes");
  }

  // Success; make modifications.
  views->clear();
  views->reserve(sizes.size());
  Byte *base = data();
  for (const size_t size : sizes) {
    views->push_back(AlignedViewImpl<OtherByte>(base, size));
    base = PadToAlignment(base + size);
  }
  DCHECK_EQ(base - data(), total_bytes);

  return tensorflow::Status::OK();
}

template <class Byte>
AlignedViewImpl<Byte>::AlignedViewImpl(Byte *data, size_t size)
    : data_(data), size_(size) {
  TF_DCHECK_OK(OkIfAligned(data_));
}

template <class Byte>
template <class OtherByte>
AlignedAreaImpl<Byte>::AlignedAreaImpl(AlignedAreaImpl<OtherByte> that)
    : data_(reinterpret_cast<Byte *>(that.data_)),
      num_views_(that.num_views()),
      view_size_(that.view_size()),
      view_stride_(that.view_stride_) {}

template <class Byte>
template <class OtherByte>
AlignedAreaImpl<Byte> &AlignedAreaImpl<Byte>::operator=(
    AlignedAreaImpl<OtherByte> that) {
  data_ = reinterpret_cast<Byte *>(that.data_);
  num_views_ = that.num_views();
  view_size_ = that.view_size();
  view_stride_ = that.view_stride_;
  return *this;
}

template <class Byte>
template <class OtherByte>
tensorflow::Status AlignedAreaImpl<Byte>::Reset(AlignedViewImpl<OtherByte> view,
                                                size_t num_views,
                                                size_t view_size) {
  const size_t total_bytes = ComputeAlignedAreaSize(num_views, view_size);
  if (view.size() < total_bytes) {
    return tensorflow::errors::InvalidArgument(
        "View is too small for area of ", num_views, " views of ", view_size,
        " bytes: need ", total_bytes, " bytes but got ", view.size(), " bytes");
  }

  // Success; make modifications.
  data_ = reinterpret_cast<Byte *>(view.data());
  num_views_ = num_views;
  view_size_ = view_size;
  view_stride_ = PadToAlignment(view_size_);
  return tensorflow::Status::OK();
}

template <class Byte>
AlignedViewImpl<Byte> AlignedAreaImpl<Byte>::view(size_t index) const {
  DCHECK_LT(index, num_views());
  return AlignedViewImpl<Byte>(data_ + view_stride_ * index, view_size_);
}

template <class Byte>
AlignedAreaImpl<Byte>::AlignedAreaImpl(Byte *data, size_t num_views,
                                       size_t view_size, size_t view_stride)
    : data_(data),
      num_views_(num_views),
      view_size_(view_size),
      view_stride_(view_stride) {
  TF_DCHECK_OK(OkIfAligned(data_));
  TF_DCHECK_OK(OkIfAligned(static_cast<const char *>(nullptr) + view_stride_));
}

}  // namespace internal

inline void UniqueAlignedArray::Reset(size_t new_size) {
  // Pad the |new_size| to the next alignment boundary, so the final bytes of
  // the array are still in a full alignment window.  E.g., if we resize to 48
  // bytes with 32-byte alignment, then we allocate 64 bytes so the final 16
  // bytes are still part of a full 32-byte alignment window.
  const size_t aligned_size = PadToAlignment(new_size);

  // To obtain an aligned address, allocate a sufficiently-padded byte array and
  // find an aligned address near the start of the block.
  //
  // TODO(googleuser): Alternatively, we could use library functions such as
  // memalign(), posix_memalign(), or aligned_alloc(), but those may not be
  // present on all platforms.  Consider adding some #ifs to allow use of those
  // library functions when available.
  padded_array_.reset(new char[aligned_size + internal::kAlignmentBytes - 1]);
  capacity_ = aligned_size;
  view_.size_ = new_size;
  view_.data_ = PadToAlignment(padded_array_.get());
  TF_DCHECK_OK(OkIfAligned(view_.data_));
}

inline void UniqueAlignedArray::Reserve(size_t new_size) {
  if (new_size > capacity_) {
    Reset(new_size);
  } else {
    view_.size_ = new_size;
  }
}

inline bool UniqueAlignedArray::Resize(size_t new_size) {
  // Avoid reallocation, if possible.
  if (new_size <= capacity_) {
    view_.size_ = new_size;
    return false;
  }

  // Reallocate and copy.  Extend the life of the old array until it is copied.
  //
  // Note: C realloc() can extend a byte array in place (i.e., without copying).
  // Unfortunately, there is no aligned version of realloc().  Moreover, adding
  // alignment padding could cause double-copying: first, when realloc() copies
  // the data to the new buffer, and second, if the amount of padding required
  // at the new address is not the same as before.
  const std::unique_ptr<char[]> old_array = std::move(padded_array_);
  const MutableAlignedView old_view = view_;
  Reset(2 * new_size);
  memcpy(view_.data(), old_view.data(), old_view.size());
  view_.size_ = new_size;
  return true;
}

}  // namespace runtime
}  // namespace dragnn
}  // namespace syntaxnet

#endif  // DRAGNN_RUNTIME_ALIGNMENT_H_
