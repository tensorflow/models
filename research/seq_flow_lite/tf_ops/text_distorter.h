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
#ifndef TENSORFLOW_MODELS_SEQ_FLOW_LITE_TF_OPS_TEXT_DISTORTER_H_
#define TENSORFLOW_MODELS_SEQ_FLOW_LITE_TF_OPS_TEXT_DISTORTER_H_

#include <assert.h>

#include "icu4c/source/common/unicode/unistr.h"
#include "tensorflow/core/lib/random/simple_philox.h"

// A class that can be used to distort text randomly.
class TextDistorter {
 public:
  // Add a random seed for PhiloxRandom constructor
  explicit TextDistorter(float distortion_probability)
      : philox_(171),
        generator_(&philox_),
        distortion_probability_(distortion_probability) {
    assert(distortion_probability_ >= 0.0);
    assert(distortion_probability_ <= 1.0);
  }
  std::string DistortText(icu::UnicodeString* uword);

 private:
  tensorflow::random::PhiloxRandom philox_;
  tensorflow::random::SimplePhilox generator_;
  float distortion_probability_;
  UChar32 random_char_ = 0;
};

#endif  // TENSORFLOW_MODELS_SEQ_FLOW_LITE_TF_OPS_TEXT_DISTORTER_H_
