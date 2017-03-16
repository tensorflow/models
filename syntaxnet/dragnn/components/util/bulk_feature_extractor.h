#ifndef NLP_SAFT_OPENSOURCE_DRAGNN_COMPONENTS_UTIL_BULK_FEATURE_EXTRACTOR_H_
#define NLP_SAFT_OPENSOURCE_DRAGNN_COMPONENTS_UTIL_BULK_FEATURE_EXTRACTOR_H_

#include <functional>
#include <utility>
#include "tensorflow/core/platform/types.h"

namespace syntaxnet {
namespace dragnn {

// Provides a wrapper for allocator functions and padding data for the Bulk
// ExtractFixedFeatures operation.
class BulkFeatureExtractor {
 public:
  // Create a BulkFeatureExtractor with the given allocator functions and
  // padding. The allocator functions should take a channel and an element
  // count and return a contigous block of memory that is associated with that
  // channel (the caller can decide what that means). If use_padding is true,
  // the provided pad_to_step and pad_to_element will be used to calculate
  // the ID size.
  BulkFeatureExtractor(
      std::function<tensorflow::int32 *(int channel, int num_elements)>
          allocate_indices_by_channel,
      std::function<tensorflow::int64 *(int channel, int num_elements)>
          allocate_ids_by_channel,
      std::function<float *(int channel, int num_elements)>
          allocate_weights_by_channel,
      bool use_padding, int pad_to_step, int pad_to_element)
      : use_padding_(use_padding),
        pad_to_step_(pad_to_step),
        pad_to_element_(pad_to_element),
        allocate_indices_by_channel_(std::move(allocate_indices_by_channel)),
        allocate_ids_by_channel_(std::move(allocate_ids_by_channel)),
        allocate_weights_by_channel_(std::move(allocate_weights_by_channel)) {}

  // Create a BulkFeatureExtractor with allocator functions as above, but with
  // use_padding set to False. Useful when you know your caller will never
  // need to pad.
  BulkFeatureExtractor(
      std::function<tensorflow::int32 *(int channel, int num_elements)>
          allocate_indices_by_channel,
      std::function<tensorflow::int64 *(int channel, int num_elements)>
          allocate_ids_by_channel,
      std::function<float *(int channel, int num_elements)>
          allocate_weights_by_channel)
      : use_padding_(false),
        pad_to_step_(-1),
        pad_to_element_(-1),
        allocate_indices_by_channel_(std::move(allocate_indices_by_channel)),
        allocate_ids_by_channel_(std::move(allocate_ids_by_channel)),
        allocate_weights_by_channel_(std::move(allocate_weights_by_channel)) {}

  // Invoke the index memory allocator.
  tensorflow::int32 *AllocateIndexMemory(int channel, int num_elements) const {
    return allocate_indices_by_channel_(channel, num_elements);
  }

  // Invoke the ID memory allocator.
  tensorflow::int64 *AllocateIdMemory(int channel, int num_elements) const {
    return allocate_ids_by_channel_(channel, num_elements);
  }

  // Invoke the weight memory allocator.
  float *AllocateWeightMemory(int channel, int num_elements) const {
    return allocate_weights_by_channel_(channel, num_elements);
  }

  // Given the total number of steps and total number of elements for a given
  // feature, calculate the index (not ID) of that feature. Based on how the
  // BulkFeatureExtractor was constructed, it may use the given number of steps
  // and number of elements, or it may use the passed padded number.
  int GetIndex(int total_steps, int num_elements, int feature_idx,
               int element_idx, int step_idx) const {
    const int steps = (use_padding_) ? pad_to_step_ : total_steps;
    const int elements = (use_padding_) ? pad_to_element_ : num_elements;
    const int feature_offset = elements * steps;
    const int element_offset = steps;
    return (feature_idx * feature_offset) + (element_idx * element_offset) +
           step_idx;
  }

 private:
  const bool use_padding_;
  const int pad_to_step_;
  const int pad_to_element_;
  const std::function<tensorflow::int32 *(int, int)>
      allocate_indices_by_channel_;
  const std::function<tensorflow::int64 *(int, int)> allocate_ids_by_channel_;
  const std::function<float *(int, int)> allocate_weights_by_channel_;
};

}  // namespace dragnn
}  // namespace syntaxnet

#endif  // NLP_SAFT_OPENSOURCE_DRAGNN_COMPONENTS_UTIL_BULK_FEATURE_EXTRACTOR_H_
