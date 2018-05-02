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

#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <utility>

#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/gtl/cleanup.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/logging.h"

namespace syntaxnet {
namespace dragnn {
namespace runtime {

int UniqueAlignedMmap::Syscalls::Open(const string &path) const {
  return open(path.c_str(), O_RDONLY);
}

int UniqueAlignedMmap::Syscalls::Close(int file_descriptor) const {
  return close(file_descriptor);
}

void *UniqueAlignedMmap::Syscalls::Mmap(int file_descriptor,
                                        size_t size) const {
  return mmap(nullptr, size, PROT_READ, MAP_SHARED, file_descriptor, 0);
}

int UniqueAlignedMmap::Syscalls::Munmap(void *data, size_t size) const {
  return munmap(data, size);
}

UniqueAlignedMmap::UniqueAlignedMmap(std::unique_ptr<Syscalls> syscalls)
    : syscalls_(std::move(syscalls)) {}

UniqueAlignedMmap::UniqueAlignedMmap(UniqueAlignedMmap &&that)
    : syscalls_(std::move(that.syscalls_)) {
  view_ = that.view_;
  path_ = that.path_;
  that.view_ = MutableAlignedView();
  that.path_.clear();
}

UniqueAlignedMmap &UniqueAlignedMmap::operator=(UniqueAlignedMmap &&that) {
  syscalls_ = std::move(that.syscalls_);
  view_ = that.view_;
  path_ = that.path_;
  that.view_ = MutableAlignedView();
  that.path_.clear();
  return *this;
}

UniqueAlignedMmap::~UniqueAlignedMmap() {
  UnmapIfNonEmpty(view_.data(), view_.size(), path_);
}

tensorflow::Status UniqueAlignedMmap::Reset(const string &path) {
  uint64 size = 0;
  TF_RETURN_IF_ERROR(tensorflow::Env::Default()->GetFileSize(path, &size));

  // Since mmap() cannot map 0 bytes, we skip the call on empty files.  This is
  // OK because UnmapIfNonEmpty() also skips munmap() on an empty region.
  if (size == 0) {
    view_ = MutableAlignedView();
    path_ = path;
    return tensorflow::Status::OK();
  }

  const int file_descriptor = syscalls_->Open(path);
  if (file_descriptor == -1) {
    // TODO(googleuser): Use strerror_r() to export the system error message.
    return tensorflow::errors::Unknown("Failed to open '", path, "'");
  }

  // In case we error out.
  auto ensure_closed = tensorflow::gtl::MakeCleanup([&] {
    if (syscalls_->Close(file_descriptor) != 0) {
      LOG(ERROR) << "Failed to close '" << path << "'";
    }
  });

  void *mmap_result = syscalls_->Mmap(file_descriptor, size);
  if (mmap_result == MAP_FAILED) {
    return tensorflow::errors::Unknown("Failed to mmap '", path, "'");
  }

  // In case we error out.
  auto ensure_unmapped = tensorflow::gtl::MakeCleanup(
      [&] { UnmapIfNonEmpty(mmap_result, size, path); });

  // Since mmap() increments the refcount of the |file_descriptor|, it must be
  // closed to prevent a leak.
  ensure_closed.release();  // going to close it manually
  if (syscalls_->Close(file_descriptor) != 0) {
    return tensorflow::errors::Unknown("Failed to close '", path, "'");
  }

  // Most implementations of mmap() place the mapped region on a page boundary,
  // which is plenty of alignment.  Since this is so unlikely to fail, we don't
  // try to recover if the address is misaligned.  A potential recovery method
  // is to allocate a UniqueAlignedArray and read the file normally.
  MutableAlignedView data;
  TF_RETURN_IF_ERROR(data.Reset(reinterpret_cast<char *>(mmap_result), size));

  // Success; make modifications.
  view_ = data;
  path_ = path;
  ensure_unmapped.release();  // this has taken ownership of the mapped file
  return tensorflow::Status::OK();
}

void UniqueAlignedMmap::UnmapIfNonEmpty(void *data, size_t size,
                                        const string &path) const {
  if (size == 0) return;
  if (syscalls_->Munmap(data, size) != 0) {
    LOG(ERROR) << "Failed to munmap() file '" << path << "'";
  }
}

}  // namespace runtime
}  // namespace dragnn
}  // namespace syntaxnet
