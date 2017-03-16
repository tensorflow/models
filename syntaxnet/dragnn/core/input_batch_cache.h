#ifndef NLP_SAFT_OPENSOURCE_DRAGNN_CORE_INPUT_BATCH_CACHE_H_
#define NLP_SAFT_OPENSOURCE_DRAGNN_CORE_INPUT_BATCH_CACHE_H_

#include <memory>
#include <string>
#include <typeindex>

#include "dragnn/core/interfaces/input_batch.h"
#include "tensorflow/core/platform/logging.h"

namespace syntaxnet {
namespace dragnn {

// A InputBatchCache holds data converted to a DRAGNN internal representation.
// It performs the conversion lazily via Data objects and caches the result.

class InputBatchCache {
 public:
  // Create an empty cache..
  InputBatchCache() : stored_type_(std::type_index(typeid(void))) {}

  // Create a InputBatchCache from a single example. This copies the string.
  explicit InputBatchCache(const string &data)
      : stored_type_(std::type_index(typeid(void))), source_data_({data}) {}

  // Create a InputBatchCache from a vector of examples. The vector is copied.
  explicit InputBatchCache(const std::vector<string> &data)
      : stored_type_(std::type_index(typeid(void))), source_data_(data) {}

  // Adds a single string to the cache. Only useable before GetAs() has been
  // called.
  void AddData(const string &data) {
    CHECK(stored_type_ == std::type_index(typeid(void)))
        << "You may not add data to an InputBatchCache after the cache has "
           "been converted via GetAs().";
    source_data_.emplace_back(data);
  }

  // Convert the stored strings into protos and return them in a specific
  // InputBatch subclass. T should always be of type InputBatch. After this
  // method is called once, all further calls must be of the same data type.
  template <class T>
  T *GetAs() {
    if (!converted_data_) {
      stored_type_ = std::type_index(typeid(T));
      converted_data_.reset(new T());
      converted_data_->SetData(source_data_);
    }
    CHECK(std::type_index(typeid(T)) == stored_type_)
        << "Attempted to convert to two object types! Existing object type was "
        << stored_type_.name() << ", new object type was "
        << std::type_index(typeid(T)).name();

    return dynamic_cast<T *>(converted_data_.get());
  }

  // Return the serialized representation of the data held in the input batch
  // object within this cache.
  const std::vector<string> SerializedData() const {
    CHECK(converted_data_) << "Cannot return batch without data.";
    return converted_data_->GetSerializedData();
  }

 private:
  // The typeid of the stored data.
  std::type_index stored_type_;

  // The raw data.
  std::vector<string> source_data_;

  // The converted data, contained in an InputBatch object.
  std::unique_ptr<InputBatch> converted_data_;
};

}  // namespace dragnn
}  // namespace syntaxnet

#endif  // NLP_SAFT_OPENSOURCE_DRAGNN_CORE_INPUT_BATCH_CACHE_H_
