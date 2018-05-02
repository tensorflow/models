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

#include "dragnn/runtime/alignment.h"

#include <stddef.h>
#include <string>
#include <vector>

#include "dragnn/core/test/generic.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"

namespace syntaxnet {
namespace dragnn {
namespace runtime {
namespace {

static_assert(internal::kAlignmentBytes >= 4, "alignment too small");

// Expects that two pointers have the same address.
void ExpectSameAddress(const void *pointer1, const void *pointer2) {
  EXPECT_EQ(pointer1, pointer2);
}

// Tests that standard scalar types are alignable.
TEST(IsAlignableTest, Alignable) {
  EXPECT_TRUE(IsAlignable<char>());
  EXPECT_TRUE(IsAlignable<float>());
  EXPECT_TRUE(IsAlignable<double>());
}

// Tests that objects of odd sizes are not alignable.
TEST(IsAlignableTest, NotAlignable) {
  EXPECT_FALSE(IsAlignable<char[3]>());
  EXPECT_FALSE(IsAlignable<char[7]>());
  EXPECT_FALSE(IsAlignable<char[7919]>());
}

// Tests that OkIfAligned() returns OK on aligned pointers.
TEST(OkIfAlignedTest, Aligned) {
  const char *ptr = nullptr;
  TF_EXPECT_OK(OkIfAligned(ptr));
  ptr += internal::kAlignmentBytes;
  TF_EXPECT_OK(OkIfAligned(ptr));
  ptr += 123 * internal::kAlignmentBytes;
  TF_EXPECT_OK(OkIfAligned(ptr));
}

// Tests that OkIfAligned() returns non-OK on misaligned pointers.
TEST(OkIfAlignedTest, NotAligned) {
  const char *ptr = nullptr;
  EXPECT_THAT(OkIfAligned(ptr + 1),
              test::IsErrorWithSubstr("Pointer fails alignment requirement"));
  EXPECT_THAT(OkIfAligned(ptr + 23),
              test::IsErrorWithSubstr("Pointer fails alignment requirement"));
}

// Tests that any window of |internal::kAlignmentBytes| bytes contains exactly
// one aligned address.
TEST(OkIfAlignedTest, OnePerAlignmentWindow) {
  // Note that |bytes| does not necessarily start at an aligned address.  Even
  // so, it is still guaranteed to contain exactly one aligned address, in the
  // same sense that any sequence of 10 consecutive integers contains exactly
  // one whose decimal representation ends in '0'.  This property is exploited
  // in UniqueAlignedArray::Reset().
  const string bytes(internal::kAlignmentBytes, ' ');
  int num_ok = 0;
  for (int i = 0; i < bytes.size(); ++i) {
    if (OkIfAligned(bytes.data() + i).ok()) ++num_ok;
  }
  EXPECT_EQ(num_ok, 1);
}

// Tests that PadToAlignment() produces an aligned byte offset.
TEST(PadToAlignmentTest, Offset) {
  EXPECT_EQ(PadToAlignment(0), 0);
  EXPECT_EQ(PadToAlignment(1), internal::kAlignmentBytes);
  EXPECT_EQ(PadToAlignment(internal::kAlignmentBytes + 1),
            2 * internal::kAlignmentBytes);
  EXPECT_EQ(PadToAlignment(99 * internal::kAlignmentBytes + 3),
            100 * internal::kAlignmentBytes);
}

// Tests that PadToAlignment() produces an aligned pointer.
TEST(PadToAlignmentTest, Pointer) {
  const string bytes = "hello";
  TF_EXPECT_OK(OkIfAligned(PadToAlignment(bytes.data())));
  const std::vector<float> reals(10);
  TF_EXPECT_OK(OkIfAligned(PadToAlignment(reals.data())));
}

// Tests that ComputeAlignedAreaSize() calculates the correct size.
TEST(ComputeAlignedAreaSizeTest, Basic) {
  EXPECT_EQ(ComputeAlignedAreaSize(0, 0), 0);
  EXPECT_EQ(ComputeAlignedAreaSize(0, 1), 0);
  EXPECT_EQ(ComputeAlignedAreaSize(1, 0), 0);
  EXPECT_EQ(ComputeAlignedAreaSize(1, 1), internal::kAlignmentBytes);
  EXPECT_EQ(ComputeAlignedAreaSize(1, internal::kAlignmentBytes),
            internal::kAlignmentBytes);
  EXPECT_EQ(ComputeAlignedAreaSize(3, internal::kAlignmentBytes + 1),
            6 * internal::kAlignmentBytes);
  EXPECT_EQ(ComputeAlignedAreaSize(11, internal::kAlignmentBytes - 1),
            11 * internal::kAlignmentBytes);
  EXPECT_EQ(ComputeAlignedAreaSize(7, internal::kAlignmentBytes),
            7 * internal::kAlignmentBytes);
}

// Tests that ComputeTotalBytesWithAlignmentPadding() calculates the correct
// total size.
TEST(ComputeTotalBytesWithAlignmentPaddingTest, DifferentSizes) {
  EXPECT_EQ(ComputeTotalBytesWithAlignmentPadding({}), 0);
  EXPECT_EQ(ComputeTotalBytesWithAlignmentPadding({0}), 0);
  EXPECT_EQ(ComputeTotalBytesWithAlignmentPadding({0, 0, 0}), 0);

  EXPECT_EQ(ComputeTotalBytesWithAlignmentPadding({1}),
            internal::kAlignmentBytes);
  EXPECT_EQ(ComputeTotalBytesWithAlignmentPadding({1, 1, 1}),
            3 * internal::kAlignmentBytes);
  EXPECT_EQ(ComputeTotalBytesWithAlignmentPadding(
                {1, internal::kAlignmentBytes, internal::kAlignmentBytes + 1}),
            4 * internal::kAlignmentBytes);

  std::vector<size_t> sizes;
  for (size_t i = 1; i <= internal::kAlignmentBytes; ++i) sizes.push_back(i);
  EXPECT_EQ(ComputeTotalBytesWithAlignmentPadding(sizes),
            internal::kAlignmentBytes * internal::kAlignmentBytes);
}

// Tests that ComputeTotalBytesWithAlignmentPadding() is equivalent to
// ComputeAlignedAreaSize() when all sizes are equal.
TEST(ComputeTotalBytesWithAlignmentPaddingTest, AllSameSize) {
  EXPECT_EQ(ComputeTotalBytesWithAlignmentPadding({1, 1, 1, 1}),
            ComputeAlignedAreaSize(4, 1));
  EXPECT_EQ(ComputeTotalBytesWithAlignmentPadding({7, 7, 7, 7, 7, 7}),
            ComputeAlignedAreaSize(6, 7));
  EXPECT_EQ(ComputeTotalBytesWithAlignmentPadding({77, 77, 77}),
            ComputeAlignedAreaSize(3, 77));
}

// Tests that UniqueAlignedArray is empty by default.
TEST(UniqueAlignedArrayTest, EmptyByDefault) {
  UniqueAlignedArray array;
  EXPECT_EQ(array.view().size(), 0);
  EXPECT_TRUE(array.view().empty());
}

// Tests that UniqueAlignedArray::Reset() always reallocates.
TEST(UniqueAlignedArrayTest, Reset) {
  UniqueAlignedArray array;

  // Reset to non-empty.
  array.Reset(10);
  const MutableAlignedView view1 = array.view();
  TF_EXPECT_OK(OkIfAligned(view1.data()));
  EXPECT_EQ(view1.size(), 10);

  // Calling view() again should return the same byte array.
  const MutableAlignedView view2 = array.view();
  ExpectSameAddress(view2.data(), view1.data());
  EXPECT_EQ(view2.size(), view1.size());

  // Reset to a different size.
  array.Reset(33);
  const MutableAlignedView view3 = array.view();
  TF_EXPECT_OK(OkIfAligned(view3.data()));
  EXPECT_EQ(view3.size(), 33);
}

// Tests that UniqueAlignedArray::Reset() reallocates when growing.
TEST(UniqueAlignedArrayTest, Reserve) {
  UniqueAlignedArray array;

  // Reset to non-empty.
  array.Reserve(20);
  const MutableAlignedView view1 = array.view();
  TF_EXPECT_OK(OkIfAligned(view1.data()));
  EXPECT_EQ(view1.size(), 20);

  // Shrink to a smaller size; should not reallocate.
  array.Reserve(7);
  const MutableAlignedView view2 = array.view();
  ExpectSameAddress(view2.data(), view1.data());
  EXPECT_EQ(view2.size(), 7);

  // Grow but still remain within capacity; should not reallocate.
  array.Reserve(14);
  const MutableAlignedView view3 = array.view();
  ExpectSameAddress(view3.data(), view1.data());
  EXPECT_EQ(view3.size(), 14);
}

// Tests that UniqueAlignedArray::Resize() reallocates when growing and
// preserves existing contents.
TEST(UniqueAlignedArrayTest, Resize) {
  UniqueAlignedArray array;

  // Resize to non-empty.
  EXPECT_TRUE(array.Resize(10));
  const MutableAlignedView view1 = array.view();
  TF_EXPECT_OK(OkIfAligned(view1.data()));
  EXPECT_EQ(view1.size(), 10);

  // Write some stuff.
  for (int i = 0; i < 10; ++i) view1.data()[i] = '1';

  // Resize to a larger size.
  EXPECT_TRUE(array.Resize(33));
  const MutableAlignedView view2 = array.view();
  TF_EXPECT_OK(OkIfAligned(view2.data()));
  EXPECT_EQ(view2.size(), 33);

  // Check that content was preserved.
  for (int i = 0; i < 10; ++i) EXPECT_EQ(view2.data()[i], '1');

  // Append more stuff.
  for (int i = 10; i < 33; ++i) view2.data()[i] = '2';

  // Resize to a smaller size.
  EXPECT_FALSE(array.Resize(15));
  const MutableAlignedView view3 = array.view();
  TF_EXPECT_OK(OkIfAligned(view3.data()));
  ExpectSameAddress(view3.data(), view2.data());
  EXPECT_EQ(view3.size(), 15);

  // Check that content was preserved.
  for (int i = 0; i < 10; ++i) EXPECT_EQ(view3.data()[i], '1');
  for (int i = 10; i < 15; ++i) EXPECT_EQ(view3.data()[i], '2');

  // Overwrite with new stuff.
  for (int i = 0; i < 15; ++i) view3.data()[i] = '3';

  // Resize to a larger size, but still below capacity.
  EXPECT_FALSE(array.Resize(20));
  const MutableAlignedView view4 = array.view();
  TF_EXPECT_OK(OkIfAligned(view4.data()));
  ExpectSameAddress(view4.data(), view2.data());
  EXPECT_EQ(view4.size(), 20);

  // Check that content was preserved.
  for (int i = 0; i < 15; ++i) EXPECT_EQ(view4.data()[i], '3');
}

// Tests that (Mutable)AlignedView is empty by default.
TEST(AlignedViewTest, EmptyByDefault) {
  AlignedView view1;
  EXPECT_EQ(view1.size(), 0);
  EXPECT_TRUE(view1.empty());

  MutableAlignedView view2;
  EXPECT_EQ(view2.size(), 0);
  EXPECT_TRUE(view2.empty());
}

// Tests that (Mutable)AlignedView::Reset() works on aligned pointers.
TEST(AlignedViewTest, ResetValid) {
  char *pointer = nullptr;
  pointer += 3 * internal::kAlignmentBytes;

  AlignedView view1;
  TF_EXPECT_OK(view1.Reset(pointer, 100));
  ExpectSameAddress(view1.data(), pointer);
  EXPECT_EQ(view1.size(), 100);
  EXPECT_FALSE(view1.empty());

  MutableAlignedView view2;
  TF_EXPECT_OK(view2.Reset(pointer, 100));
  ExpectSameAddress(view2.data(), pointer);
  EXPECT_EQ(view2.size(), 100);
  EXPECT_FALSE(view2.empty());
}

// Tests that (Mutable)AlignedView::Reset() fails on misaligned pointers.
TEST(AlignedViewTest, ResetInvalid) {
  char *pointer = nullptr;
  ++pointer;  // not aligned

  AlignedView view1;
  EXPECT_THAT(view1.Reset(pointer, 10),
              test::IsErrorWithSubstr("Pointer fails alignment requirement"));

  MutableAlignedView view2;
  EXPECT_THAT(view2.Reset(pointer, 10),
              test::IsErrorWithSubstr("Pointer fails alignment requirement"));
}

// Tests that (Mutable)AlignedView::Reset() can empty the view.
TEST(AlignedViewTest, ResetEmpty) {
  char *pointer = nullptr;
  pointer += 11 * internal::kAlignmentBytes;

  // First point to a non-empty byte array.
  AlignedView view1;
  TF_EXPECT_OK(view1.Reset(pointer, 100));
  ExpectSameAddress(view1.data(), pointer);
  EXPECT_EQ(view1.size(), 100);
  EXPECT_FALSE(view1.empty());

  // Then reset to empty.
  TF_EXPECT_OK(view1.Reset(pointer, 0));
  EXPECT_EQ(view1.size(), 0);
  EXPECT_TRUE(view1.empty());

  // First point to a non-empty byte array.
  MutableAlignedView view2;
  TF_EXPECT_OK(view2.Reset(pointer, 100));
  ExpectSameAddress(view2.data(), pointer);
  EXPECT_EQ(view2.size(), 100);
  EXPECT_FALSE(view2.empty());

  // Then reset to empty.
  TF_EXPECT_OK(view2.Reset(pointer, 0));
  EXPECT_EQ(view2.size(), 0);
  EXPECT_TRUE(view2.empty());
}

// Tests that (Mutable)AlignedView supports copy-construction and assignment
// with shallow-copy semantics, and reinterprets from char* to const char*.
TEST(AlignedViewTest, CopyAndAssign) {
  char *pointer1 = nullptr;
  pointer1 += 3 * internal::kAlignmentBytes;
  const char *pointer2 = nullptr;
  pointer2 += 7 * internal::kAlignmentBytes;

  MutableAlignedView view1;
  TF_ASSERT_OK(view1.Reset(pointer1, 100));
  AlignedView view2;
  TF_ASSERT_OK(view2.Reset(pointer2, 200));

  MutableAlignedView view3(view1);
  ExpectSameAddress(view3.data(), pointer1);
  EXPECT_EQ(view3.size(), 100);
  EXPECT_FALSE(view3.empty());

  view3 = MutableAlignedView();
  EXPECT_EQ(view3.size(), 0);
  EXPECT_TRUE(view3.empty());

  view3 = view1;
  ExpectSameAddress(view3.data(), pointer1);
  EXPECT_EQ(view3.size(), 100);
  EXPECT_FALSE(view3.empty());

  AlignedView view4(view1);  // reinterprets type
  ExpectSameAddress(view4.data(), pointer1);
  EXPECT_EQ(view4.size(), 100);
  EXPECT_FALSE(view4.empty());

  view4 = AlignedView();
  EXPECT_EQ(view4.size(), 0);
  EXPECT_TRUE(view4.empty());

  view4 = view2;
  ExpectSameAddress(view4.data(), pointer2);
  EXPECT_EQ(view4.size(), 200);
  EXPECT_FALSE(view4.empty());

  view4 = view1;  // reinterprets type
  ExpectSameAddress(view4.data(), pointer1);
  EXPECT_EQ(view4.size(), 100);
  EXPECT_FALSE(view4.empty());

  view4 = MutableAlignedView();  // reinterprets type
  EXPECT_EQ(view4.size(), 0);
  EXPECT_TRUE(view4.empty());
}

// Tests that AlignedView can split itself into sub-views with specified sizes.
TEST(AlignedViewTest, SplitConst) {
  const std::vector<size_t> sizes = {1, internal::kAlignmentBytes,
                                     internal::kAlignmentBytes + 1, 1, 123};
  const size_t total_bytes = ComputeTotalBytesWithAlignmentPadding(sizes);

  AlignedView view;
  TF_ASSERT_OK(view.Reset(nullptr, total_bytes));

  std::vector<AlignedView> views(100);  // will be resized
  TF_ASSERT_OK(view.Split(sizes, &views));
  ASSERT_EQ(views.size(), 5);

  const char *base = view.data();
  ExpectSameAddress(views[0].data(), base);
  EXPECT_EQ(views[0].size(), 1);

  base += internal::kAlignmentBytes;
  ExpectSameAddress(views[1].data(), base);
  EXPECT_EQ(views[1].size(), internal::kAlignmentBytes);

  base += internal::kAlignmentBytes;
  ExpectSameAddress(views[2].data(), base);
  EXPECT_EQ(views[2].size(), internal::kAlignmentBytes + 1);

  base += 2 * internal::kAlignmentBytes;
  ExpectSameAddress(views[3].data(), base);
  EXPECT_EQ(views[3].size(), 1);

  base += internal::kAlignmentBytes;
  ExpectSameAddress(views[4].data(), base);
  EXPECT_EQ(views[4].size(), 123);
}

// Tests that MutableAlignedView can split itself into sub-views with specified
// sizes, and reinterprets from char* to const char*.
TEST(AlignedViewTest, SplitMutable) {
  const std::vector<size_t> sizes = {1, internal::kAlignmentBytes,
                                     internal::kAlignmentBytes + 1, 1, 123};
  const size_t total_bytes = ComputeTotalBytesWithAlignmentPadding(sizes);

  // Also add some padding to check that we can split part of the view.
  MutableAlignedView view;
  TF_ASSERT_OK(view.Reset(nullptr, total_bytes + 10));

  std::vector<AlignedView> const_views(99);  // will be resized
  std::vector<MutableAlignedView> mutable_views(2);  // will be resized
  TF_ASSERT_OK(view.Split(sizes, &const_views));
  TF_ASSERT_OK(view.Split(sizes, &mutable_views));
  ASSERT_EQ(const_views.size(), 5);
  ASSERT_EQ(mutable_views.size(), 5);

  const char *base = view.data();
  ExpectSameAddress(const_views[0].data(), base);
  ExpectSameAddress(mutable_views[0].data(), base);
  EXPECT_EQ(const_views[0].size(), 1);
  EXPECT_EQ(mutable_views[0].size(), 1);

  base += internal::kAlignmentBytes;
  ExpectSameAddress(const_views[1].data(), base);
  ExpectSameAddress(mutable_views[1].data(), base);
  EXPECT_EQ(const_views[1].size(), internal::kAlignmentBytes);
  EXPECT_EQ(mutable_views[1].size(), internal::kAlignmentBytes);

  base += internal::kAlignmentBytes;
  ExpectSameAddress(const_views[2].data(), base);
  ExpectSameAddress(mutable_views[2].data(), base);
  EXPECT_EQ(const_views[2].size(), internal::kAlignmentBytes + 1);
  EXPECT_EQ(mutable_views[2].size(), internal::kAlignmentBytes + 1);

  base += 2 * internal::kAlignmentBytes;
  ExpectSameAddress(const_views[3].data(), base);
  ExpectSameAddress(mutable_views[3].data(), base);
  EXPECT_EQ(const_views[3].size(), 1);
  EXPECT_EQ(mutable_views[3].size(), 1);

  base += internal::kAlignmentBytes;
  ExpectSameAddress(const_views[4].data(), base);
  ExpectSameAddress(mutable_views[4].data(), base);
  EXPECT_EQ(const_views[4].size(), 123);
  EXPECT_EQ(mutable_views[4].size(), 123);
}

TEST(AlignedViewTest, SplitTooSmall) {
  const std::vector<size_t> sizes = {1, internal::kAlignmentBytes,
                                     internal::kAlignmentBytes + 1, 1, 123};
  const size_t total_bytes = ComputeTotalBytesWithAlignmentPadding(sizes);

  // Make the view just a bit too small.
  MutableAlignedView view;
  TF_ASSERT_OK(view.Reset(nullptr, total_bytes - 1));

  std::vector<MutableAlignedView> views;
  EXPECT_THAT(view.Split(sizes, &views),
              test::IsErrorWithSubstr("View is too small to be split"));
}

// Tests that (Mutable)AlignedArea is empty by default.
TEST(AlignedAreaTest, EmptyByDefault) {
  AlignedArea area1;
  EXPECT_EQ(area1.num_views(), 0);
  EXPECT_EQ(area1.view_size(), 0);
  EXPECT_TRUE(area1.empty());

  MutableAlignedArea area2;
  EXPECT_EQ(area2.num_views(), 0);
  EXPECT_EQ(area2.view_size(), 0);
  EXPECT_TRUE(area2.empty());
}

// Tests that (Mutable)AlignedArea::Reset() can initialize to a single view.
TEST(AlignedAreaTest, ResetSingleton) {
  const char *pointer1 = nullptr;
  pointer1 += 3 * internal::kAlignmentBytes;
  char *pointer2 = nullptr;
  pointer2 += 7 * internal::kAlignmentBytes;

  AlignedView view1;
  TF_ASSERT_OK(view1.Reset(pointer1, internal::kAlignmentBytes));

  MutableAlignedView view2;
  TF_ASSERT_OK(view2.Reset(pointer2, internal::kAlignmentBytes + 1));

  AlignedArea area1;
  TF_ASSERT_OK(area1.Reset(view1, 1, 1));
  EXPECT_EQ(area1.num_views(), 1);
  EXPECT_EQ(area1.view_size(), 1);
  EXPECT_FALSE(area1.empty());
  ExpectSameAddress(area1.view(0).data(), pointer1);
  EXPECT_EQ(area1.view(0).size(), 1);

  TF_ASSERT_OK(area1.Reset(view2, 1, 2));
  EXPECT_EQ(area1.num_views(), 1);
  EXPECT_EQ(area1.view_size(), 2);
  EXPECT_FALSE(area1.empty());
  ExpectSameAddress(area1.view(0).data(), pointer2);
  EXPECT_EQ(area1.view(0).size(), 2);

  TF_ASSERT_OK(area1.Reset(view2, 1, 1));
  EXPECT_EQ(area1.num_views(), 1);
  EXPECT_EQ(area1.view_size(), 1);
  EXPECT_FALSE(area1.empty());
  ExpectSameAddress(area1.view(0).data(), pointer2);
  EXPECT_EQ(area1.view(0).size(), 1);

  MutableAlignedArea area2;
  TF_ASSERT_OK(area2.Reset(view2, 1, 2));
  EXPECT_EQ(area2.num_views(), 1);
  EXPECT_EQ(area2.view_size(), 2);
  EXPECT_FALSE(area2.empty());
  ExpectSameAddress(area2.view(0).data(), pointer2);
  EXPECT_EQ(area2.view(0).size(), 2);

  TF_ASSERT_OK(area2.Reset(view2, 1, 1));
  EXPECT_EQ(area2.num_views(), 1);
  EXPECT_EQ(area2.view_size(), 1);
  EXPECT_FALSE(area2.empty());
  ExpectSameAddress(area2.view(0).data(), pointer2);
  EXPECT_EQ(area2.view(0).size(), 1);
}

// Tests that (Mutable)AlignedArea::Reset() can initialize to a sequence of
// multiple views.
TEST(AlignedAreaTest, ResetMultiple) {
  const char *pointer1 = nullptr;
  pointer1 += 3 * internal::kAlignmentBytes;
  char *pointer2 = nullptr;
  pointer2 += 7 * internal::kAlignmentBytes;

  AlignedView view1;
  TF_ASSERT_OK(view1.Reset(pointer1, 11 * internal::kAlignmentBytes));

  MutableAlignedView view2;
  TF_ASSERT_OK(view2.Reset(pointer2, 2 * internal::kAlignmentBytes));

  AlignedArea area1;
  TF_ASSERT_OK(area1.Reset(view1, 11, 1));
  EXPECT_EQ(area1.num_views(), 11);
  EXPECT_EQ(area1.view_size(), 1);
  EXPECT_FALSE(area1.empty());
  for (int i = 0; i < area1.num_views(); ++i) {
    ExpectSameAddress(area1.view(i).data(),
                      pointer1 + internal::kAlignmentBytes * i);
    EXPECT_EQ(area1.view(i).size(), 1);
  }

  TF_ASSERT_OK(area1.Reset(view1, 10, internal::kAlignmentBytes));
  EXPECT_EQ(area1.num_views(), 10);
  EXPECT_EQ(area1.view_size(), internal::kAlignmentBytes);
  EXPECT_FALSE(area1.empty());
  for (int i = 0; i < area1.num_views(); ++i) {
    ExpectSameAddress(area1.view(i).data(),
                      pointer1 + internal::kAlignmentBytes * i);
    EXPECT_EQ(area1.view(i).size(), internal::kAlignmentBytes);
  }

  TF_ASSERT_OK(area1.Reset(view2, 2, 2));
  EXPECT_EQ(area1.num_views(), 2);
  EXPECT_EQ(area1.view_size(), 2);
  EXPECT_FALSE(area1.empty());
  for (int i = 0; i < area1.num_views(); ++i) {
    ExpectSameAddress(area1.view(i).data(),
                      pointer2 + internal::kAlignmentBytes * i);
    EXPECT_EQ(area1.view(i).size(), 2);
  }

  MutableAlignedArea area2;
  TF_ASSERT_OK(area2.Reset(view2, 2, internal::kAlignmentBytes));
  EXPECT_EQ(area2.num_views(), 2);
  EXPECT_EQ(area2.view_size(), internal::kAlignmentBytes);
  EXPECT_FALSE(area2.empty());
  for (int i = 0; i < area2.num_views(); ++i) {
    ExpectSameAddress(area2.view(i).data(),
                      pointer2 + internal::kAlignmentBytes * i);
    EXPECT_EQ(area2.view(i).size(), internal::kAlignmentBytes);
  }
}

// Tests that (Mutable)AlignedArea::Reset() fails when the view being split into
// sub-views is too small.
TEST(AlignedAreaTest, ResetInvalid) {
  AlignedView view1;
  TF_ASSERT_OK(view1.Reset(nullptr, 11 * internal::kAlignmentBytes));

  MutableAlignedView view2;
  TF_ASSERT_OK(view2.Reset(nullptr, 2 * internal::kAlignmentBytes));

  // View size larger than available view.
  AlignedArea area;
  EXPECT_THAT(area.Reset(view1, 1, 11 * internal::kAlignmentBytes + 1),
              test::IsErrorWithSubstr("View is too small for area"));
  TF_ASSERT_OK(area.Reset(view1, 11, 1));
  EXPECT_THAT(area.Reset(view2, 1, 2 * internal::kAlignmentBytes + 1),
              test::IsErrorWithSubstr("View is too small for area"));
  TF_ASSERT_OK(area.Reset(view2, 2, 1));

  // Total size larger than available view.
  EXPECT_THAT(area.Reset(view1, 12, 1),
              test::IsErrorWithSubstr("View is too small for area"));
  TF_ASSERT_OK(area.Reset(view1, 11, 1));
  EXPECT_THAT(area.Reset(view1, 4, 2 * internal::kAlignmentBytes + 1),
              test::IsErrorWithSubstr("View is too small for area"));
  TF_ASSERT_OK(area.Reset(view1, 11, 1));
  EXPECT_THAT(area.Reset(view1, 3, 3 * internal::kAlignmentBytes + 1),
              test::IsErrorWithSubstr("View is too small for area"));
  TF_ASSERT_OK(area.Reset(view1, 11, 1));
  EXPECT_THAT(area.Reset(view1, 2, 5 * internal::kAlignmentBytes + 1),
              test::IsErrorWithSubstr("View is too small for area"));
  TF_ASSERT_OK(area.Reset(view1, 11, 1));
  EXPECT_THAT(area.Reset(view2, 3, 1),
              test::IsErrorWithSubstr("View is too small for area"));
  TF_ASSERT_OK(area.Reset(view2, 2, 1));
  EXPECT_THAT(area.Reset(view2, 2, internal::kAlignmentBytes + 1),
              test::IsErrorWithSubstr("View is too small for area"));
  TF_ASSERT_OK(area.Reset(view2, 2, 1));
}

// Tests that (Mutable)AlignedView::Reset() can empty the area.
TEST(AlignedAreaTest, ResetEmpty) {
  AlignedView view1;
  TF_ASSERT_OK(view1.Reset(nullptr, 11 * internal::kAlignmentBytes));

  MutableAlignedView view2;
  TF_ASSERT_OK(view2.Reset(nullptr, 2 * internal::kAlignmentBytes));

  // First point to a non-empty byte array, then clear.
  AlignedArea area1;
  TF_ASSERT_OK(area1.Reset(view1, 11, 1));
  TF_ASSERT_OK(area1.Reset(view1, 0, 0));
  EXPECT_EQ(area1.num_views(), 0);
  EXPECT_EQ(area1.view_size(), 0);
  EXPECT_TRUE(area1.empty());

  TF_ASSERT_OK(area1.Reset(view2, 2, 1));
  TF_ASSERT_OK(area1.Reset(view2, 0, 100));
  EXPECT_EQ(area1.num_views(), 0);
  EXPECT_EQ(area1.view_size(), 100);
  EXPECT_TRUE(area1.empty());

  TF_ASSERT_OK(area1.Reset(view2, 2, 1));
  TF_ASSERT_OK(area1.Reset(MutableAlignedView(), 0, 1));
  EXPECT_EQ(area1.num_views(), 0);
  EXPECT_EQ(area1.view_size(), 1);
  EXPECT_TRUE(area1.empty());

  MutableAlignedArea area2;
  TF_ASSERT_OK(area2.Reset(view2, 2, 1));
  TF_ASSERT_OK(area2.Reset(view2, 0, 0));
  EXPECT_EQ(area2.num_views(), 0);
  EXPECT_EQ(area2.view_size(), 0);
  EXPECT_TRUE(area2.empty());

  TF_ASSERT_OK(area2.Reset(view2, 2, 1));
  TF_ASSERT_OK(area2.Reset(view2, 0, 100));
  EXPECT_EQ(area2.num_views(), 0);
  EXPECT_EQ(area2.view_size(), 100);
  EXPECT_TRUE(area2.empty());

  TF_ASSERT_OK(area2.Reset(view2, 2, 1));
  TF_ASSERT_OK(area2.Reset(MutableAlignedView(), 0, 1));
  EXPECT_EQ(area2.num_views(), 0);
  EXPECT_EQ(area2.view_size(), 1);
  EXPECT_TRUE(area2.empty());
}

// Tests that (Mutable)AlignedArea supports copy-construction and assignment
// with shallow-copy semantics, and reinterprets from char* to const char*.
TEST(AlignedAreaTest, CopyAndAssign) {
  char *pointer1 = nullptr;
  pointer1 += 3 * internal::kAlignmentBytes;
  const char *pointer2 = nullptr;
  pointer2 += 7 * internal::kAlignmentBytes;

  MutableAlignedView view1;
  TF_ASSERT_OK(view1.Reset(pointer1, ComputeAlignedAreaSize(1, 5)));
  AlignedView view2;
  TF_ASSERT_OK(view2.Reset(pointer2, ComputeAlignedAreaSize(2, 77)));

  MutableAlignedArea area1;
  TF_ASSERT_OK(area1.Reset(view1, 1, 5));
  AlignedArea area2;
  TF_ASSERT_OK(area2.Reset(view2, 2, 77));

  MutableAlignedArea area3(area1);
  EXPECT_EQ(area3.num_views(), 1);
  EXPECT_EQ(area3.view_size(), 5);
  EXPECT_FALSE(area3.empty());
  ExpectSameAddress(area3.view(0).data(), pointer1);
  EXPECT_EQ(area3.view(0).size(), 5);

  area3 = MutableAlignedArea();
  EXPECT_EQ(area3.num_views(), 0);
  EXPECT_EQ(area3.view_size(), 0);
  EXPECT_TRUE(area3.empty());

  area3 = area1;
  EXPECT_EQ(area3.num_views(), 1);
  EXPECT_EQ(area3.view_size(), 5);
  EXPECT_FALSE(area3.empty());
  ExpectSameAddress(area3.view(0).data(), pointer1);
  EXPECT_EQ(area3.view(0).size(), 5);

  AlignedArea area4(area1);  // reinterprets type
  EXPECT_EQ(area4.num_views(), 1);
  EXPECT_EQ(area4.view_size(), 5);
  EXPECT_FALSE(area4.empty());
  ExpectSameAddress(area4.view(0).data(), pointer1);
  EXPECT_EQ(area4.view(0).size(), 5);

  area4 = AlignedArea();
  EXPECT_EQ(area4.num_views(), 0);
  EXPECT_EQ(area4.view_size(), 0);
  EXPECT_TRUE(area4.empty());

  area4 = area2;
  EXPECT_EQ(area4.num_views(), 2);
  EXPECT_EQ(area4.view_size(), 77);
  EXPECT_FALSE(area4.empty());
  ExpectSameAddress(area4.view(0).data(), pointer2);
  EXPECT_EQ(area4.view(0).size(), 77);
  ExpectSameAddress(area4.view(1).data(), PadToAlignment(pointer2 + 77));
  EXPECT_EQ(area4.view(1).size(), 77);

  area4 = area1;  // reinterprets type
  EXPECT_EQ(area4.num_views(), 1);
  EXPECT_EQ(area4.view_size(), 5);
  EXPECT_FALSE(area4.empty());
  ExpectSameAddress(area4.view(0).data(), pointer1);
  EXPECT_EQ(area4.view(0).size(), 5);

  area4 = MutableAlignedArea();  // reinterprets type
  EXPECT_EQ(area4.num_views(), 0);
  EXPECT_EQ(area4.view_size(), 0);
  EXPECT_TRUE(area4.empty());
}

}  // namespace
}  // namespace runtime
}  // namespace dragnn
}  // namespace syntaxnet
