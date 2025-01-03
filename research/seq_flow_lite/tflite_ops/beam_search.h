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

#ifndef TENSORFLOW_MODELS_SEQ_FLOW_LITE_TFLITE_OPS_BEAM_SEARCH_H_
#define TENSORFLOW_MODELS_SEQ_FLOW_LITE_TFLITE_OPS_BEAM_SEARCH_H_

#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <set>
#include <vector>

#include "tensorflow/lite/c/common.h"

namespace seq_flow_lite {
namespace ops {
namespace custom {

class BeamSearchTestPeer;

// Implements Beam search util for decoding operations. The derived class
// should implement the Decode method to complete the actual decoding
// operation which outputs the probabilities for each beam and class.
class BeamSearch {
 public:
  BeamSearch(int beam_size, int num_classes, int sos_id, int eos_id,
             float alpha = 0.6, bool use_logtis = false)
      : beam_size_(beam_size),
        num_classes_(num_classes),
        sos_id_(sos_id),
        eos_id_(eos_id),
        alpha_(alpha),
        beam_log_probabilities_(beam_size, 0.0f),
        logits_mask_(num_classes, true),
        compute_topk_with_logits_(use_logtis),
        debug_log_(false) {
    topk_heap_.reserve(2 * beam_size_);
  }
  // Virtual method that should be overridden to perform decode operations.
  virtual TfLiteTensor* Decode(int timestep,
                               std::vector<int32_t>& selected_beams,
                               std::vector<int32_t>& input_indices) = 0;
  virtual ~BeamSearch() {}
  // Runs decoding process for num_steps.
  std::vector<std::vector<int32_t>> Process(int num_steps);

  int NumBeams() { return beam_size_; }
  int NumClasses() { return num_classes_; }

  void SetNumClasses(int num_classes) { num_classes_ = num_classes; }

  // Sets boolean mask of size num_classes to process only valid logit indices.
  // Example mask: {true, true, false, true, false} would result in processing
  // logits at indices 0, 1 and 3.
  void SetMaskForLogits(const std::vector<bool>& mask);

 private:
  friend class BeamSearchTestPeer;
  // Floating point version of finding top_k classes from decoder output.
  void FindTopKFloat(const TfLiteTensor& tensor, int valid_beams, int K);
  // Quantized version of finding top_k classes from decoder output probs.
  void FindTopKQuantized(const TfLiteTensor& tensor, int valid_beams, int K);
  // Quantized version of finding top_k classes from decoder output logits.
  void FindTopKQuantizedFromLogits(const TfLiteTensor& tensor, int valid_beams,
                                   int topk_k);
  // Optimized version for FindTopKQuantizedFromLogits.
  void FindTopKQuantizedFromLogitsV1(const TfLiteTensor& tensor,
                                     int valid_beams, int topk_k);
  // Length penalty is given by = (5+len(decode)/6) ^ -\alpha.
  // Pls refer to  https://arxiv.org/abs/1609.08144.
  float InverseLengthPenalty(int step);
  // Populates log probabilities for int values 0-255.
  void PopulateLogLookupTable(const TfLiteTensor& tensor);
  // Populates exp probabilities for int values 0-255.
  void PopulateSoftmaxLookupTable(const TfLiteTensor& tensor);
  std::vector<std::pair<float, int32_t>> topk_heap_;
  const int beam_size_;
  int num_classes_;
  // Start of sequence ID.
  const int sos_id_;
  // End of sequence ID.
  const int eos_id_;
  // Alpha to be used in length penality computation.
  const float alpha_;
  std::vector<float> beam_log_probabilities_;
  // Mask for valid logits. Used when computing TopK with logits.
  std::vector<bool> logits_mask_;
  // Computes TopK using logits instead of probabilities.
  bool compute_topk_with_logits_ = false;
  float log_lookup_table_[256];
  bool log_lookup_table_populated_ = false;
  float exp_lookup_table_[256];
  bool exp_lookup_table_populated_ = false;
  bool debug_log_;
};

}  // namespace custom
}  // namespace ops
}  // namespace seq_flow_lite
#endif  // TENSORFLOW_MODELS_SEQ_FLOW_LITE_TFLITE_OPS_BEAM_SEARCH_H_
