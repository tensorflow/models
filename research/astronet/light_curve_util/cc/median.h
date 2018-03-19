/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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

#ifndef TENSORFLOW_MODELS_ASTRONET_LIGHT_CURVE_UTIL_CC_MEDIAN_H_
#define TENSORFLOW_MODELS_ASTRONET_LIGHT_CURVE_UTIL_CC_MEDIAN_H_

#include <algorithm>
#include <iterator>
#include <vector>

namespace astronet {

// Computes the median value in the range [first, last).
//
// After calling this function, the elements in [first, last) will be rearranged
// such that, if middle = first + distance(first, last) / 2:
//   1. The element pointed at by middle is changed to whatever element would
//     occur in that position if [first, last) was sorted.
//   2. All of the elements before this new middle element are less than or
//      equal to the elements after the new nth element.
template <class RandomIt>
typename std::iterator_traits<RandomIt>::value_type InPlaceMedian(
    RandomIt first, RandomIt last) {
  // If n is odd, 'middle' points to the middle element. If n is even, 'middle'
  // points to the upper middle element.
  const auto n = std::distance(first, last);
  const auto middle = first + (n / 2);

  // Partially sort such that 'middle' in its place.
  std::nth_element(first, middle, last);

  // n is odd: the median is simply the middle element.
  if (n & 1) {
    return *middle;
  }

  // The maximum value lower than *middle is located in [first, middle) as a
  // a post condition of nth_element.
  const auto lower_middle = std::max_element(first, middle);

  // Prevent overflow. We know that *lower_middle <= *middle. If both are on
  // opposite sides of zero, the sum won't overflow, otherwise the difference
  // won't overflow.
  if (*lower_middle <= 0 && *middle >= 0) {
    return (*lower_middle + *middle) / 2;
  }
  return *lower_middle + (*middle - *lower_middle) / 2;
}

// Computes the median value in the range [first, last) without modifying the
// input.
template <class ForwardIterator>
typename std::iterator_traits<ForwardIterator>::value_type Median(
    ForwardIterator first, ForwardIterator last) {
  std::vector<typename std::iterator_traits<ForwardIterator>::value_type>
      values(first, last);
  return InPlaceMedian(values.begin(), values.end());
}

}  // namespace astronet

#endif  // TENSORFLOW_MODELS_ASTRONET_LIGHT_CURVE_UTIL_CC_MEDIAN_H_
