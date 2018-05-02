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

#include "dragnn/runtime/mmap.h"

#include <stddef.h>
#include <sys/mman.h>
#include <string>
#include <utility>

#include "dragnn/core/test/generic.h"
#include "syntaxnet/base.h"
#include <gmock/gmock.h>
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/platform/test.h"

namespace syntaxnet {
namespace dragnn {
namespace runtime {
namespace {

using ::testing::Return;

// A mockable set of system calls.
class MockSyscalls : public UniqueAlignedMmap::Syscalls {
 public:
  MOCK_CONST_METHOD1(Open, int(const string &path));
  MOCK_CONST_METHOD1(Close, int(int file_descriptor));
  MOCK_CONST_METHOD2(Mmap, void *(int file_descriptor, size_t size));
  MOCK_CONST_METHOD2(Munmap, int(void *, size_t size));
};

class UniqueAlignedMmapTest : public ::testing::Test {
 protected:
  const string kInvalidFile = "/some/invalid/path";
  const string kEmptyFile = tensorflow::io::JoinPath(
      test::GetTestDataPrefix(), "dragnn/runtime/testdata/empty_file");
  const string kTenBytes = tensorflow::io::JoinPath(
      test::GetTestDataPrefix(), "dragnn/runtime/testdata/ten_bytes");

  std::unique_ptr<MockSyscalls> syscalls_{new MockSyscalls()};
};

// Tests that the mapped region is empty by default.
TEST_F(UniqueAlignedMmapTest, EmptyByDefault) {
  UniqueAlignedMmap data;
  EXPECT_TRUE(data.view().empty());
}

// Tests that an empty file can be mapped.
TEST_F(UniqueAlignedMmapTest, EmptyFile) {
  UniqueAlignedMmap data;
  TF_ASSERT_OK(data.Reset(kEmptyFile));
  EXPECT_TRUE(data.view().empty());
}

// Tests that a non-empty file can be mapped.
TEST_F(UniqueAlignedMmapTest, TenBytes) {
  UniqueAlignedMmap data;
  TF_ASSERT_OK(data.Reset(kTenBytes));
  ASSERT_EQ(data.view().size(), 10);
  EXPECT_STREQ(data.view().data(), "0123456789");
}

// Tests that the mapped files can be move-constructed and move-assigned.
TEST_F(UniqueAlignedMmapTest, Movement) {
  UniqueAlignedMmap data1;
  TF_ASSERT_OK(data1.Reset(kTenBytes));

  UniqueAlignedMmap data2(std::move(data1));
  ASSERT_EQ(data2.view().size(), 10);
  EXPECT_STREQ(data2.view().data(), "0123456789");

  UniqueAlignedMmap data3;
  data3 = std::move(data2);
  ASSERT_EQ(data3.view().size(), 10);
  EXPECT_STREQ(data3.view().data(), "0123456789");
}

// Tests that the mapping fails if the file is invalid.
TEST_F(UniqueAlignedMmapTest, InvalidFile) {
  UniqueAlignedMmap data;
  EXPECT_FALSE(data.Reset(kInvalidFile).ok());
}

// Tests that the mapping fails if the file cannot be open()ed.
TEST_F(UniqueAlignedMmapTest, FailToOpen) {
  EXPECT_CALL(*syscalls_, Open(kTenBytes)).WillOnce(Return(-1));

  UniqueAlignedMmap data(std::move(syscalls_));
  EXPECT_THAT(data.Reset(kTenBytes), test::IsErrorWithSubstr("Failed to open"));
}

// Tests that the mapping fails if the file cannot be mmap()ed.
TEST_F(UniqueAlignedMmapTest, FailToMmap) {
  const int kFileDescriptor = 5;
  EXPECT_CALL(*syscalls_, Open(kTenBytes)).WillOnce(Return(kFileDescriptor));
  EXPECT_CALL(*syscalls_, Mmap(kFileDescriptor, 10))
      .WillOnce(Return(MAP_FAILED));
  EXPECT_CALL(*syscalls_, Close(kFileDescriptor)).WillOnce(Return(0));

  UniqueAlignedMmap data(std::move(syscalls_));
  EXPECT_THAT(data.Reset(kTenBytes), test::IsErrorWithSubstr("Failed to mmap"));
}

// As above, but also fails to close.
TEST_F(UniqueAlignedMmapTest, FailToMmapAndClose) {
  const int kFileDescriptor = 5;
  EXPECT_CALL(*syscalls_, Open(kTenBytes)).WillOnce(Return(kFileDescriptor));
  EXPECT_CALL(*syscalls_, Mmap(kFileDescriptor, 10))
      .WillOnce(Return(MAP_FAILED));
  EXPECT_CALL(*syscalls_, Close(kFileDescriptor)).WillOnce(Return(-1));

  UniqueAlignedMmap data(std::move(syscalls_));
  EXPECT_THAT(data.Reset(kTenBytes), test::IsErrorWithSubstr("Failed to mmap"));
}

// Tests that the mapping fails if the file cannot be close()ed.
TEST_F(UniqueAlignedMmapTest, FailToClose) {
  const int kFileDescriptor = 5;
  EXPECT_CALL(*syscalls_, Open(kTenBytes)).WillOnce(Return(kFileDescriptor));
  EXPECT_CALL(*syscalls_, Mmap(kFileDescriptor, 10)).WillOnce(Return(nullptr));
  EXPECT_CALL(*syscalls_, Close(kFileDescriptor)).WillOnce(Return(-1));
  EXPECT_CALL(*syscalls_, Munmap(nullptr, 10)).WillOnce(Return(0));

  UniqueAlignedMmap data(std::move(syscalls_));
  EXPECT_THAT(data.Reset(kTenBytes),
              test::IsErrorWithSubstr("Failed to close"));
}

// As above, but also fails to munmap().
TEST_F(UniqueAlignedMmapTest, FailToCloseAndMunmap) {
  const int kFileDescriptor = 5;
  EXPECT_CALL(*syscalls_, Open(kTenBytes)).WillOnce(Return(kFileDescriptor));
  EXPECT_CALL(*syscalls_, Mmap(kFileDescriptor, 10)).WillOnce(Return(nullptr));
  EXPECT_CALL(*syscalls_, Close(kFileDescriptor)).WillOnce(Return(-1));
  EXPECT_CALL(*syscalls_, Munmap(nullptr, 10)).WillOnce(Return(-1));

  UniqueAlignedMmap data(std::move(syscalls_));
  EXPECT_THAT(data.Reset(kTenBytes),
              test::IsErrorWithSubstr("Failed to close"));
}

// Tests that the mapping fails if the mapped region is misaligned.
TEST_F(UniqueAlignedMmapTest, Misaligned) {
  char *ptr = nullptr;
  ++ptr;
  const int kFileDescriptor = 5;
  EXPECT_CALL(*syscalls_, Open(kTenBytes)).WillOnce(Return(kFileDescriptor));
  EXPECT_CALL(*syscalls_, Mmap(kFileDescriptor, 10)).WillOnce(Return(ptr));
  EXPECT_CALL(*syscalls_, Close(kFileDescriptor)).WillOnce(Return(0));
  EXPECT_CALL(*syscalls_, Munmap(ptr, 10)).WillOnce(Return(0));

  UniqueAlignedMmap data(std::move(syscalls_));
  EXPECT_THAT(data.Reset(kTenBytes),
              test::IsErrorWithSubstr("Pointer fails alignment requirement"));
}

}  // namespace
}  // namespace runtime
}  // namespace dragnn
}  // namespace syntaxnet
