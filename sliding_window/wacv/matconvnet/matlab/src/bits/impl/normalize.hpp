// @file normalize.hpp
// @brief Normalize block implementation
// @author Andrea Vedaldi

/*
Copyright (C) 2014-16 Andrea Vedaldi.
All rights reserved.

This file is part of the VLFeat library and is made available under
the terms of the BSD license (see the COPYING file).
*/

#ifndef __vl__normalize__
#define __vl__normalize__

#include "../data.hpp"
#include <cstddef>

namespace vl { namespace impl {

  template<vl::DeviceType dev, typename type>
  struct lrn
  {
    static vl::ErrorCode
    forward(type* output,
            type const* data,
            size_t height, size_t width, size_t depth, size_t size,
            size_t normDepth,
            type  kappa, type  alpha, type  beta) ;

    static vl::ErrorCode
    backward(type* derData,
             type const* data,
             type const* derOutput,
             size_t height, size_t width, size_t depth, size_t size,
             size_t normDepth,
             type  kappa, type  alpha, type  beta) ;
  } ;

} }

#endif /* __vl__normalize__ */
