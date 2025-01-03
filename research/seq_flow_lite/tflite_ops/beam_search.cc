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

#include "tflite_ops/beam_search.h"  // seq_flow_lite

#include <algorithm>
#include <cstdint>
#include <numeric>
#include <vector>

#include <glog/logging.h>
#include "absl/strings/str_join.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/internal/types.h"
#include "tflite_ops/quantization_util.h"  // seq_flow_lite

namespace seq_flow_lite {
namespace ops {
namespace custom {

namespace {

constexpr int kLeftShiftNumBits = 24;
constexpr int kClassIndexMask = (1 << kLeftShiftNumBits) - 1;

// Tracks finished sequences within the beams.
class SequenceTracker {
 public:
  explicit SequenceTracker(int beam_size, int eos_id)
      : beam_size_(beam_size),
        eos_id_(eos_id),
        min_terminated_scores_(-kInfinite) {}
  void AddSequence(const int32_t *begin, const int32_t *end, float score);
  int NumSequences() { return terminated_topk_.size(); }
  std::vector<std::vector<int32_t>> GetTopBeams();
  float MinTrackedScore() { return min_terminated_scores_; }
  float MaxTrackedScore() {
    return terminated_topk_.empty() ? -kInfinite
                                    : terminated_topk_.begin()->first;
  }

 private:
  static constexpr float kInfinite = 1e7;
  const int beam_size_;
  const int eos_id_;
  // TODO(akandoor): Consider using std::vector and heap accessors instead.
  std::map<float, std::vector<int32_t>, std::greater<float>> terminated_topk_;
  float min_terminated_scores_;
};

void PrintBeam(const int32 *array_new, int cur_step) {
  LOG(INFO) << absl::StrJoin(array_new, array_new + cur_step, ", ");
}

bool HeapCompare(std::pair<float, int> &a, std::pair<float, int> &b) {
  return a.first > b.first;
}
}  // namespace

void SequenceTracker::AddSequence(const int32_t *begin, const int32_t *end,
                                  float score) {
  if (NumSequences() < beam_size_ || score > min_terminated_scores_) {
    // TODO(akandoor): Handle duplicate scores.
    if (NumSequences() >= beam_size_) {
      terminated_topk_.erase(std::prev(terminated_topk_.end()));
    }
    // TODO(prabhumk): This can potentially slow things down. Fix this.
    terminated_topk_[score] = std::vector<int32_t>(begin, end);
    // Pushing EOS_ID to terminate the sequence.
    terminated_topk_[score].push_back(eos_id_);
    min_terminated_scores_ = terminated_topk_.rbegin()->first;
  }
}

std::vector<std::vector<int32_t>> SequenceTracker::GetTopBeams() {
  std::vector<std::vector<int32_t>> return_value;
  return_value.reserve(terminated_topk_.size());
  for (const auto &v : terminated_topk_) {
    return_value.push_back(v.second);
  }
  return return_value;
}

void BeamSearch::PopulateLogLookupTable(const TfLiteTensor &tensor) {
  if (!log_lookup_table_populated_) {
    for (int value = 0; value < 256; ++value) {
      log_lookup_table_[value] =
          logf(::seq_flow_lite::PodDequantizeValue<uint8_t>(tensor, value));
    }
    log_lookup_table_populated_ = true;
  }
}

void BeamSearch::PopulateSoftmaxLookupTable(const TfLiteTensor &tensor) {
  if (!exp_lookup_table_populated_) {
    const int32_t max_uint8 = std::numeric_limits<uint8_t>::max();
    for (int32_t val = 0; val <= max_uint8; ++val) {
      exp_lookup_table_[max_uint8 - val] = expf(-tensor.params.scale * val);
    }
    exp_lookup_table_populated_ = true;
  }
}

float BeamSearch::InverseLengthPenalty(int step) {
  return 1.0f / std::pow((5.f + step) / 6.f, alpha_);
}

void BeamSearch::FindTopKFloat(const TfLiteTensor &tensor, int valid_beams,
                               int K) {
  topk_heap_.clear();
  const float *probabilities = ::tflite::GetTensorData<float>(&tensor);
  for (int j = 0; j < valid_beams; ++j) {
    for (int k = 0; k < num_classes_; ++k) {
      const int index = j * num_classes_ + k;
      float log_probs =
          (beam_log_probabilities_[j] + logf(probabilities[index]));
      topk_heap_.push_back(std::pair<float, int>(log_probs, index));
      std::push_heap(topk_heap_.begin(), topk_heap_.end(), HeapCompare);
      if (topk_heap_.size() > K) {
        std::pop_heap(topk_heap_.begin(), topk_heap_.end(), HeapCompare);
        topk_heap_.pop_back();
      }
    }
  }
  std::sort(topk_heap_.begin(), topk_heap_.end(), HeapCompare);
}

void BeamSearch::FindTopKQuantized(const TfLiteTensor &tensor, int valid_beams,
                                   int K) {
  PopulateLogLookupTable(tensor);
  topk_heap_.clear();
  const uint8 *probabilities = ::tflite::GetTensorData<uint8_t>(&tensor);
  for (int j = 0; j < valid_beams; ++j) {
    for (int k = 0; k < num_classes_; ++k) {
      const int index = j * num_classes_ + k;
      const float log_probs = (beam_log_probabilities_[j] +
                               log_lookup_table_[probabilities[index]]);
      topk_heap_.push_back(std::pair<float, int>(log_probs, index));
      std::push_heap(topk_heap_.begin(), topk_heap_.end(), HeapCompare);
      if (topk_heap_.size() > K) {
        std::pop_heap(topk_heap_.begin(), topk_heap_.end(), HeapCompare);
        topk_heap_.pop_back();
      }
    }
  }
  std::sort(topk_heap_.begin(), topk_heap_.end(), HeapCompare);
}

void BeamSearch::SetMaskForLogits(const std::vector<bool> &mask) {
  logits_mask_.assign(mask.begin(), mask.end());
  CHECK_EQ(logits_mask_.size(), num_classes_)
      << "Mask size should be same as num_classes";
}

void BeamSearch::FindTopKQuantizedFromLogits(const TfLiteTensor &tensor,
                                             int valid_beams, int topk_k) {
  PopulateSoftmaxLookupTable(tensor);
  topk_heap_.clear();
  const uint8_t *logits = ::tflite::GetTensorData<uint8_t>(&tensor);
  for (int j = 0; j < valid_beams; ++j) {
    const uint8_t *beam_logits = logits + j * num_classes_;
    uint8_t max_val = std::numeric_limits<uint8_t>::min();
    // Finding max quantized value in the current beam.
    for (int k = 0; k < num_classes_; ++k) {
      if (!logits_mask_[k]) continue;
      max_val = std::max(max_val, beam_logits[k]);
    }

    float sum_exp = 0.0f;
    const int32_t max_uint8 = std::numeric_limits<uint8>::max();
    // Offset into table to compute exp(scale*(x - xmax)) instead of
    // exp(scale*(x)) to prevent overflow.
    const float *table_offset = &exp_lookup_table_[max_uint8 - max_val];
    // Calculate sum(exp(scale*(x - x_max))).
    for (int k = 0; k < num_classes_; ++k) {
      if (!logits_mask_[k]) continue;
      sum_exp += table_offset[beam_logits[k]];
    }
    CHECK(sum_exp) << "Invalid logits or Mask provided.";
    const float log_sum_exp = std::log(sum_exp);
    const float precomputed = (tensor.params.scale * max_val + log_sum_exp);
    for (int k = 0; k < num_classes_; ++k) {
      if (!logits_mask_[k]) continue;
      const int index = j * num_classes_ + k;
      const float log_prob = tensor.params.scale * beam_logits[k] - precomputed;
      const float beam_log_prob = (beam_log_probabilities_[j] + log_prob);
      topk_heap_.push_back(std::pair<float, int>(beam_log_prob, index));
      std::push_heap(topk_heap_.begin(), topk_heap_.end(), HeapCompare);
      if (topk_heap_.size() > topk_k) {
        std::pop_heap(topk_heap_.begin(), topk_heap_.end(), HeapCompare);
        topk_heap_.pop_back();
      }
    }
  }
  std::sort(topk_heap_.begin(), topk_heap_.end(), HeapCompare);
}

void BeamSearch::FindTopKQuantizedFromLogitsV1(const TfLiteTensor &tensor,
                                               int valid_beams, int topk_k) {
  PopulateSoftmaxLookupTable(tensor);
  topk_heap_.clear();

  std::vector<uint32_t> curr_beam_topk(topk_k);

  const uint8 *logits = ::tflite::GetTensorData<uint8_t>(&tensor);
  for (int j = 0; j < valid_beams; ++j) {
    // Resetting the topk logits vector for each beam.
    curr_beam_topk.clear();
    const uint8_t *beam_logits = logits + j * num_classes_;
    uint8_t max_val = std::numeric_limits<uint8_t>::min();
    // Finding max quantized value in the current beam.
    for (int k = 0; k < num_classes_; ++k) {
      if (!logits_mask_[k]) continue;
      max_val = std::max(max_val, beam_logits[k]);
    }

    float sum_exp = 0.0f;
    const int32_t max_uint8 = std::numeric_limits<uint8>::max();
    // Offset into table to compute exp(scale*(x - xmax)) instead of
    // exp(scale*(x)) to prevent overflow.
    const float *table_offset = &exp_lookup_table_[max_uint8 - max_val];
    // Calculate sum(exp(scale*(x - x_max))).
    for (int k = 0; k < num_classes_; ++k) {
      if (!logits_mask_[k]) continue;
      sum_exp += table_offset[beam_logits[k]];
    }
    CHECK(sum_exp) << "Invalid logits or mask provided.";
    const float log_sum_exp = std::log(sum_exp);
    const float precomputed = (tensor.params.scale * max_val + log_sum_exp);
    // Computing indices for topk logits in the current beam.

    for (uint32_t k = 0; k < num_classes_; ++k) {
      if (!logits_mask_[k]) continue;
      // Pushing logits uint8 value to MSB and storing index in the 24 LSB.
      const uint32_t val =
          (beam_logits[k] << kLeftShiftNumBits) | (k & kClassIndexMask);
      curr_beam_topk.push_back(val);
      std::push_heap(curr_beam_topk.begin(), curr_beam_topk.end(),
                     std::greater<>());
      if (curr_beam_topk.size() > topk_k) {
        std::pop_heap(curr_beam_topk.begin(), curr_beam_topk.end(),
                      std::greater<>());
        curr_beam_topk.pop_back();
      }
    }
    // Updating topk across all beams.
    for (uint32_t curr_beam : curr_beam_topk) {
      const uint32_t curr_beam_index = curr_beam & kClassIndexMask;
      const uint32_t index = j * num_classes_ + curr_beam_index;
      const float log_prob =
          tensor.params.scale * beam_logits[curr_beam_index] - precomputed;
      const float beam_log_prob = (beam_log_probabilities_[j] + log_prob);
      topk_heap_.push_back(std::pair<float, int>(beam_log_prob, index));
      std::push_heap(topk_heap_.begin(), topk_heap_.end(), HeapCompare);
      if (topk_heap_.size() > topk_k) {
        std::pop_heap(topk_heap_.begin(), topk_heap_.end(), HeapCompare);
        topk_heap_.pop_back();
      }
    }
  }
  std::sort(topk_heap_.begin(), topk_heap_.end(), HeapCompare);
}

std::vector<std::vector<int32_t>> BeamSearch::Process(int num_steps) {
  // Encode();
  std::vector<int32> input_indices(beam_size_, sos_id_);
  // Favor beam index 0 for the first sos input.
  beam_log_probabilities_[0] = 0.0f;
  SequenceTracker sequence_tracker(beam_size_, eos_id_);
  std::vector<int32_t> selected_beam(beam_size_, 0);
  std::vector<std::vector<int32_t>> arrays;
  arrays.emplace_back(num_steps * beam_size_);
  arrays.emplace_back(num_steps * beam_size_);
  int32_t *array_new = nullptr;
  int valid_beam_entries = 1;
  const float inverse_max_length_penalty = InverseLengthPenalty(num_steps);
  for (int i = 0; i < num_steps; ++i) {
    TfLiteTensor *decoder_output = Decode(i + 1, selected_beam, input_indices);
    CHECK_EQ(decoder_output->dims->size, 3);
    CHECK_EQ(decoder_output->dims->data[0], beam_size_);
    CHECK_EQ(decoder_output->dims->data[1], 1);
    CHECK_EQ(decoder_output->dims->data[2], num_classes_);
    const float inverse_length_penalty = InverseLengthPenalty(i + 1);
    if (decoder_output->type == kTfLiteUInt8) {
      if (compute_topk_with_logits_) {
        FindTopKQuantizedFromLogitsV1(*decoder_output, valid_beam_entries,
                                      beam_size_ * 2);
      } else {
        FindTopKQuantized(*decoder_output, valid_beam_entries, beam_size_ * 2);
      }
    } else if (decoder_output->type == kTfLiteFloat32) {
      LOG(ERROR) << "TopK is not optimized in this path.";
      CHECK_EQ(compute_topk_with_logits_, false)
          << "TopK with logits for Float is not supported";
      FindTopKFloat(*decoder_output, valid_beam_entries, beam_size_ * 2);
    } else {
      CHECK(false) << "Invalid data type: " << decoder_output->type;
    }

    const int32_t offset = i & 0x1;
    const int32_t *array_old = arrays[1 - offset].data();
    array_new = arrays[offset].data();

    valid_beam_entries = 0;
    for (int src = 0; src < beam_size_ * 2; ++src) {
      const int new_class = (topk_heap_[src].second % num_classes_);
      if (new_class == eos_id_) {
        const int old_beam = topk_heap_[src].second / num_classes_;
        sequence_tracker.AddSequence(
            array_old + old_beam * num_steps,
            array_old + old_beam * num_steps + i,
            topk_heap_[src].first * inverse_length_penalty);
      } else if (valid_beam_entries < beam_size_) {
        if (valid_beam_entries != src) {
          topk_heap_[valid_beam_entries] = topk_heap_[src];
        }
        valid_beam_entries++;
      }
    }

    if (valid_beam_entries == 0) {
      break;
    }
    const float max_alive_score =
        topk_heap_[0].first * inverse_max_length_penalty;
    if (max_alive_score < sequence_tracker.MaxTrackedScore()) {
      break;
    }
    for (int j = 0; j < valid_beam_entries; ++j) {
      beam_log_probabilities_[j] = topk_heap_[j].first;
      const int new_class = (topk_heap_[j].second % num_classes_);
      input_indices[j] = new_class;
      const int old_beam = topk_heap_[j].second / num_classes_;
      memcpy(array_new + j * num_steps, array_old + old_beam * num_steps,
             i * sizeof(int32));
      array_new[j * num_steps + i] = new_class;
      if (debug_log_) PrintBeam(array_new + j * num_steps, i + 1);
      selected_beam[j] = old_beam;
    }
  }

  if (sequence_tracker.NumSequences() == 0) {
    // No terminated sequence, the best alive sequence is the optimal one.
    sequence_tracker.AddSequence(array_new, array_new + num_steps, 0.0f);
  }
  return sequence_tracker.GetTopBeams();
}

}  // namespace custom
}  // namespace ops
}  // namespace seq_flow_lite
