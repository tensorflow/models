// @file bnorm.hpp
// @brief Batch Normalization block implementation
// @author Sebastien Ehrhardt

/*
Copyright (C) 2015-16 Sebastien Ehrhardt.
All rights reserved.

This file is part of the VLFeat library and is made available under
the terms of the BSD license (see the COPYING file).
*/

#ifndef __vl__bnorm__
#define __vl__bnorm__

#include "../data.hpp"
#include <cstddef>

namespace vl { namespace impl {

  template<vl::DeviceType dev, typename type>
  struct bnorm
  {
    static vl::ErrorCode
    forward(Context& context,
            type* output,
            type* moments, // can be null and it will be allocated size_ternally
            type const* data,
            type const* multipliers,
            type const* biases,
            size_t height, size_t width, size_t depth, size_t size,
            type epsilon) ;

    static vl::ErrorCode
    forward_given_moments(Context& context,
                          type* output,
                          type const* moments,
                          type const* data,
                          type const* multipliers,
                          type const* biases,
                          size_t height, size_t width, size_t depth, size_t size) ;

    static vl::ErrorCode
    backward(Context& context,
             type* derData,
             type* derMultipliers,
             type* derBiases,
             type* moments, // can be null and it will be allocated size_ternally
             type const* data,
             type const* multipliers,
             type const* biases,
             type const* derOutput,
             size_t height, size_t width, size_t depth, size_t size,
             type epsilon) ;

    static vl::ErrorCode
    backward_given_moments(Context& context,
                           type* derData,
                           type* derMultipliers,
                           type* derBiases,
                           type const* moments,
                           type const* data,
                           type const* multipliers,
                           type const* biases,
                           type const* derOutput,
                           size_t height, size_t width, size_t depth, size_t size,
                           type epsilon) ;
  } ;

} }
#endif /* __vl__bnorm__ */
