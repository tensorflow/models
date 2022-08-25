/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef TENSORFLOW_MODELS_SEQ_FLOW_LITE_TFLITE_OPS_TFLITE_DECODER_CACHE_H_
#define TENSORFLOW_MODELS_SEQ_FLOW_LITE_TFLITE_OPS_TFLITE_DECODER_CACHE_H_

#include <memory>

#include "tensorflow/lite/c/common.h"

namespace seq_flow_lite {
namespace ops {
namespace custom {

namespace tflite_decoder_base {

// Base decoder op that can be derived to implement different decoding schemes.
template <typename T>
class BaseDecoderOp {
 public:
  explicit BaseDecoderOp(int feature_size, int beam_size)
      : feature_size_(feature_size),
        beam_size_(beam_size),
        cache1_(new T[feature_size * beam_size]),
        cache2_(new T[feature_size * beam_size]) {}

  virtual ~BaseDecoderOp() {}

  int BeamSize() const { return beam_size_; }
  int FeatureSize() const { return feature_size_; }

  virtual void InitCache(TfLiteTensor* cache = nullptr) {
    memset(cache1_.get(), 0, beam_size_ * feature_size_ * sizeof(T));
  }

  T* CurrentCache(int step) const {
    return (step & 0x1) == 0x1 ? cache1_.get() : cache2_.get();
  }

  T* NextCache(int step) const {
    return (step & 0x1) == 0x1 ? cache2_.get() : cache1_.get();
  }

 private:
  const int feature_size_;
  const int beam_size_;
  const std::unique_ptr<T[]> cache1_;
  const std::unique_ptr<T[]> cache2_;
};

// DynamicCacheOp stores caches of different timesteps. It supports reallocate
// memory for past timestep when beam size is dynamically added.
template <typename T>
class DynamicCacheOp {
 public:
  explicit DynamicCacheOp(int feature_size) : feature_size_(feature_size) {}

  virtual ~DynamicCacheOp() {}

  int FeatureSize() const { return feature_size_; }

  virtual void InitCache(TfLiteTensor* cache = nullptr) { cache_list_.clear(); }

  // GetCache is called by the new step in UnifromAttn. The caller wants to add
  // a new cache or dynamically appends attn value to an existing cache.
  std::vector<T>* GetCache(int step, int beam_size) {
    // If the wanted cache is larger than cache_list_.size(), will return a
    // invalid pointer. There may be an error of the step, and the caller should
    // stop using cache.
    if (step - 1 > cache_list_.size()) {
      return nullptr;
    } else if (step - 1 == cache_list_.size()) {
      // The caller wants to add a new cache if the wanted step equals the size
      // of cache_list_.
      cache_list_.push_back(
          std::move(std::vector<T>(feature_size_ * beam_size)));
    } else {
      // Allocates new memory in previous cache to store new uniform attention.
      cache_list_[step - 1].resize(cache_list_[step - 1].size() +
                                   beam_size * feature_size_);
    }
    return &cache_list_[step - 1];
  }

  // GetStaticCache will return the cached attention which is readonly.
  std::vector<T>* GetStaticCache(int step) {
    // No previous cache for the initial step.
    if (step == 0) {
      return nullptr;
    } else {
      // Gets the previous cache.
      return &cache_list_[step - 1];
    }
  }

 private:
  const int feature_size_;
  std::vector<std::vector<T>> cache_list_;
};
}  // namespace tflite_decoder_base

}  // namespace custom
}  // namespace ops
}  // namespace seq_flow_lite
#endif  // TENSORFLOW_MODELS_SEQ_FLOW_LITE_TFLITE_OPS_TFLITE_DECODER_CACHE_H_
