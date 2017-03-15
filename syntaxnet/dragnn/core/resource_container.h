#ifndef NLP_SAFT_OPENSOURCE_DRAGNN_CORE_RESOURCE_CONTAINER_H_
#define NLP_SAFT_OPENSOURCE_DRAGNN_CORE_RESOURCE_CONTAINER_H_

#include <memory>

#include "syntaxnet/base.h"
#include "tensorflow/core/framework/resource_mgr.h"

namespace syntaxnet {
namespace dragnn {

using tensorflow::strings::StrCat;

// Wrapper to store a data type T in the ResourceMgr. There should be one per
// Session->Run() call that may happen concurrently.
template <class T>
class ResourceContainer : public tensorflow::ResourceBase {
 public:
  explicit ResourceContainer(std::unique_ptr<T> data)
      : data_(std::move(data)) {}

  ~ResourceContainer() override {}

  T *get() { return data_.get(); }
  std::unique_ptr<T> release() { return std::move(data_); }

  string DebugString() override { return "ResourceContainer"; }

 private:
  std::unique_ptr<T> data_;
};

}  // namespace dragnn
}  // namespace syntaxnet

#endif  // NLP_SAFT_OPENSOURCE_DRAGNN_CORE_RESOURCE_CONTAINER_H_
