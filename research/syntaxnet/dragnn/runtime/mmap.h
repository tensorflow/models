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

// Utils for establishing and managing memory-mapped files.

#ifndef DRAGNN_RUNTIME_MMAP_H_
#define DRAGNN_RUNTIME_MMAP_H_

#include <stddef.h>
#include <memory>
#include <string>

#include "dragnn/runtime/alignment.h"
#include "syntaxnet/base.h"
#include "tensorflow/core/lib/core/status.h"

namespace syntaxnet {
namespace dragnn {
namespace runtime {

// A uniquely-owned aligned memory-mapped file.  This has virtual methods only
// for mocking in tests; do not derive from class.
class UniqueAlignedMmap {
 public:
  // A mockable wrapper around the system calls used by this class.
  class Syscalls {
   public:
    virtual ~Syscalls() = default;

    // Each method below forwards to the similarly-named syscall.  Some methods
    // have been simplified by omitting arguments that are never varied.
    virtual int Open(const string &path) const;
    virtual int Close(int file_descriptor) const;
    virtual void *Mmap(int file_descriptor, size_t size) const;
    virtual int Munmap(void *data, size_t size) const;
  };

  // Creates an empty, unmapped memory region.
  UniqueAlignedMmap() = default;

  // FOR TESTS ONLY.  As above, but injects the |syscalls|.
  explicit UniqueAlignedMmap(std::unique_ptr<Syscalls> syscalls);

  // Supports movement only.
  UniqueAlignedMmap(UniqueAlignedMmap &&that);
  UniqueAlignedMmap &operator=(UniqueAlignedMmap &&that);
  UniqueAlignedMmap(const UniqueAlignedMmap &that) = delete;
  UniqueAlignedMmap &operator=(const UniqueAlignedMmap &that) = delete;

  // Unmaps the current memory-mapped file, if any.
  ~UniqueAlignedMmap();

  // Resets this to a memory-mapping of the |path|.  On error, returns non-OK
  // and modifies nothing.
  tensorflow::Status Reset(const string &path);

  // Returns the mapped memory region.
  AlignedView view() const { return AlignedView(view_); }

 private:
  // Unmaps [|data|,|data|+|size|), if non-empty.  Uses the |path| for error
  // logging.  Does not return a status because none of the call sites could
  // pass it along; they'd log it anyways.
  void UnmapIfNonEmpty(void *data, size_t size, const string &path) const;

  // The system calls used to perform the memory-mapping.
  std::unique_ptr<Syscalls> syscalls_{new Syscalls()};

  // The current memory-mapped file, or empty if unmapped.  Mutable to satisfy
  // munmap(), which requires a non-const pointer---contents are not modified.
  MutableAlignedView view_;

  // The path to the current memory-mapped file, if any, for debug logging.
  string path_;
};

}  // namespace runtime
}  // namespace dragnn
}  // namespace syntaxnet

#endif  // DRAGNN_RUNTIME_MMAP_H_
