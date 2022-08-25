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

#include <cstdint>
#include <functional>
#include <iostream>
#include <memory>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/strings/str_join.h"
#include "tensorflow/lite/c/c_api_types.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/optimized/optimized_ops.h"
#include "tensorflow/lite/kernels/internal/reference/dequantize.h"
#include "tensorflow/lite/kernels/internal/reference/reference_ops.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/internal/types.h"
#include "tflite_ops/quantization_util.h"  // seq_flow_lite

namespace seq_flow_lite {
namespace ops {
namespace custom {

void CheckOutputData(const float* test_output, const float* reference_output,
                     const ::tflite::RuntimeShape& shape_common) {
  const int buffer_size = shape_common.FlatSize();
  double sum_abs_diff = 0;
  float max_abs_val = 0;

  for (int i = 0; i < buffer_size; i++) {
    sum_abs_diff += std::abs(test_output[i] - reference_output[i]);
    max_abs_val = std::max(max_abs_val, std::abs(reference_output[i]));
  }

  if (sum_abs_diff != 0.f) {
    const float mean_diff = static_cast<float>(sum_abs_diff / buffer_size);
    const float relative_error = std::abs(mean_diff) / max_abs_val;
    ASSERT_LT(relative_error, 1e-5f);
  }
}

class BeamSearchImpl : public BeamSearch {
 public:
  BeamSearchImpl(int beam_size, int num_classes, int sos_id, int eos_id,
                 bool use_logits = false, bool quantize = false)
      : BeamSearch(beam_size, num_classes, sos_id, eos_id,
                   /*alpha=*/0.6, /*use_logtis=*/use_logits) {
    CreateDecoderOutputTensor({beam_size, 1, num_classes}, quantize);
    InitializeCache();
  }
  TfLiteTensor* Decode(int timestep, std::vector<int32_t>& selected_beams,
                       std::vector<int32_t>& indices) override {
    const float* cur_cache = CurrentCache(timestep);
    float* next_cache = NextCache(timestep);

    if (decoder_output_->type == kTfLiteUInt8) {
      auto data_ptr = ::tflite::GetTensorData<uint8_t>(decoder_output_.get());
      for (int beam = 0, index = 0; beam < NumBeams(); ++beam) {
        const float* selected =
            cur_cache + (selected_beams[beam] * NumClasses());
        for (int j = 0; j < NumClasses(); ++j, index++) {
          next_cache[index] = (selected[j] + next_cache[index]) / 2;
          data_ptr[index] = PodQuantize(
              next_cache[index], decoder_output_->params.zero_point,
              1.0f / decoder_output_->params.scale);
        }
      }
    } else {
      auto data_ptr = ::tflite::GetTensorData<float>(decoder_output_.get());
      for (int beam = 0, index = 0; beam < NumBeams(); ++beam) {
        const float* selected =
            cur_cache + (selected_beams[beam] * NumClasses());
        for (int j = 0; j < NumClasses(); ++j, index++) {
          next_cache[index] = (selected[j] + next_cache[index]) / 2;
          data_ptr[index] = next_cache[index];
        }
      }
    }
    return decoder_output_.get();
  }

 private:
  void CreateDecoderOutputTensor(const std::vector<int>& dims,
                                 bool quantize = false) {
    decoder_output_.reset(new TfLiteTensor);
    decoder_output_->dims = TfLiteIntArrayCreate(dims.size());
    int tensor_size = 1;
    for (int i = 0; i < dims.size(); ++i) {
      decoder_output_->dims->data[i] = dims[i];
      tensor_size *= dims[i];
    }
    if (quantize) {
      decoder_output_->type = kTfLiteUInt8;
      decoder_output_->bytes = tensor_size * sizeof(uint8_t);
      decoder_output_->params.scale = 1.0 / 255.0;
      decoder_output_->params.zero_point = 0;
    } else {
      decoder_output_->type = kTfLiteFloat32;
      decoder_output_->bytes = tensor_size * sizeof(float);
    }

    decoder_output_->data.raw = new char[decoder_output_->bytes];
  }

  struct DeleteTensor {
    void operator()(TfLiteTensor* t) const {
      TfLiteIntArrayFree(t->dims);
      delete[] t->data.raw;
      delete t;
    }
  };

  float* CurrentCache(int step) {
    return (step & 0x1) == 0x1 ? cache1_.data() : cache2_.data();
  }

  float* NextCache(int step) {
    return (step & 0x1) == 0x1 ? cache2_.data() : cache1_.data();
  }

  void InitializeCache() {
    cache1_ = {/* 0: */ 0.6, 0.8, 0.3, 0.7, 0.2,
               /* 1: */ 0.5, 0.2, 0.1, 0.3, 0.4};

    cache2_ = {/* 0: */ 0.6, 0.9, 0.8, 0.2, 0.8,
               /* 1: */ 0.5, 0.8, 0.5, 0.7, 0.9};
  }

  std::unique_ptr<TfLiteTensor, DeleteTensor> decoder_output_;
  std::vector<float> cache1_{20, 0.0};
  std::vector<float> cache2_{20, 0.0};
};

class BeamSearchTestPeer {
 public:
  BeamSearchTestPeer(int beam_size, int num_classes, int sos_id, int eos_id,
                     bool use_logits = false, bool quantize = false)
      : beam_size_(beam_size),
        num_classes_(num_classes),
        sos_id_(sos_id),
        eos_id_(eos_id),
        use_logits_(use_logits),
        quantize_(quantize) {}
  std::vector<std::vector<int32_t>> Process(int num_steps) {
    BeamSearchImpl bs(beam_size_, num_classes_, sos_id_, eos_id_, use_logits_,
                      quantize_);
    return bs.Process(num_steps);
  }

  std::vector<float> InvokeFindTopKQuantizedWithLogits(
      const TfLiteTensor& logits, const std::vector<bool>& mask,
      int valid_beams, int topk_k, bool optimized = false) {
    BeamSearchImpl bs(beam_size_, num_classes_, sos_id_, eos_id_, use_logits_,
                      quantize_);
    bs.SetMaskForLogits(mask);
    if (optimized) {
      bs.FindTopKQuantizedFromLogitsV1(logits, valid_beams, topk_k);
    } else {
      bs.FindTopKQuantizedFromLogits(logits, valid_beams, topk_k);
    }
    std::vector<float> result;
    for (int i = 0; i < topk_k; ++i) {
      result.push_back(bs.topk_heap_[i].first);
    }
    return result;
  }

 private:
  int beam_size_;
  int num_classes_;
  int sos_id_;
  int eos_id_;
  bool use_logits_;
  bool quantize_;
};

TEST(BeamSearch, BasicTest) {
  BeamSearchTestPeer bst(2, 5, 0, 2);
  auto beams = bst.Process(4);
  EXPECT_EQ(absl::StrJoin(beams[0], ","), "2");
  EXPECT_EQ(absl::StrJoin(beams[1], ","), "1,2");
}

TEST(BeamSearch, BasicTestQuantized) {
  BeamSearchTestPeer bst(2, 5, 0, 2, /*use_logits*/ false, /*quantize=*/true);
  auto beams = bst.Process(4);
  EXPECT_EQ(absl::StrJoin(beams[0], ","), "2");
  EXPECT_EQ(absl::StrJoin(beams[1], ","), "1,2");
}

TEST(BeamSearch, TestFindTopKQuantizedFromLogits) {
  int beam_size = 2;
  int num_classes = 5;
  BeamSearchImpl bs(beam_size, num_classes, 0, 2, /*use_logits=*/true,
                    /*quantize=*/true);
  std::vector<int32_t> selected_beams = {0, 1};
  std::vector<int32> input_indices(2, 0);
  auto* logits_tensor = bs.Decode(1, selected_beams, input_indices);
  BeamSearchTestPeer bst(beam_size, num_classes, 0, 2, /*use_logits=*/true,
                         /*quantize=*/true);
  std::vector<bool> mask(num_classes, true);
  auto topk_output = bst.InvokeFindTopKQuantizedWithLogits(
      *logits_tensor, mask, beam_size, beam_size * num_classes);

  auto shape_common = ::tflite::RuntimeShape({beam_size, 1, num_classes});

  const int buffer_size = shape_common.FlatSize();
  std::vector<float> reference_dequant_data(buffer_size);
  std::vector<float> reference_output_float_data(buffer_size);

  ::tflite::DequantizationParams dq_params;
  dq_params.zero_point = logits_tensor->params.zero_point;
  dq_params.scale = logits_tensor->params.scale;
  ::tflite::reference_ops::Dequantize(dq_params, shape_common,
                                      logits_tensor->data.uint8, shape_common,
                                      reference_dequant_data.data());
  ::tflite::SoftmaxParams sm_params;
  ::tflite::optimized_ops::LogSoftmax(
      sm_params, shape_common, reference_dequant_data.data(), shape_common,
      reference_output_float_data.data());

  std::sort(reference_output_float_data.begin(),
            reference_output_float_data.end(), std::greater<float>());
  CheckOutputData(topk_output.data(), reference_output_float_data.data(),
                  shape_common);
}

TEST(BeamSearch, TestFindTopKQuantizedFromLogitsV1) {
  int beam_size = 2;
  int num_classes = 5;
  BeamSearchImpl bs(beam_size, num_classes, 0, 2, /*use_logits=*/true,
                    /*quantize=*/true);
  std::vector<int32_t> selected_beams = {0, 1};
  std::vector<int32> input_indices(2, 0);
  auto* logits_tensor = bs.Decode(1, selected_beams, input_indices);
  BeamSearchTestPeer bst(beam_size, num_classes, 0, 2, /*use_logits=*/true,
                         /*quantize=*/true);
  int topk_k = beam_size * 2;
  std::vector<bool> mask = {true, true, false, true, false};
  auto topk_output = bst.InvokeFindTopKQuantizedWithLogits(*logits_tensor, mask,
                                                           beam_size, topk_k);
  auto topk_output_v1 = bst.InvokeFindTopKQuantizedWithLogits(
      *logits_tensor, mask, beam_size, topk_k, /*optimized=*/true);
  auto shape_common = ::tflite::RuntimeShape({beam_size, 1, 1});
  CheckOutputData(topk_output_v1.data(), topk_output.data(), shape_common);
}

}  // namespace custom
}  // namespace ops
}  // namespace seq_flow_lite
