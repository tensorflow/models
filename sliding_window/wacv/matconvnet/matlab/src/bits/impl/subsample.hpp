// @file subsampling.hpp
// @brief Subsampling block implementation
// @author Andrea Vedaldi
// @author Karel Lenc

/*
Copyright (C) 2014-16 Andrea Vedaldi and Karel Lenc.
All rights reserved.

This file is part of the VLFeat library and is made available under
the terms of the BSD license (see the COPYING file).
*/

#ifndef VL_NNSUBSAMPLE_H
#define VL_NNSUBSAMPLE_H

#include "../data.hpp"
#include <stddef.h>

namespace vl { namespace impl {

  template<vl::DeviceType dev, typename type>
  struct subsample {

    static vl::ErrorCode
    forward(vl::Context& context,
            type* output,
            type const* data,
            size_t height, size_t width, size_t depth,
            size_t strideY, size_t strideX,
            size_t padTop, size_t padBottom, size_t padLeft, size_t padRight) ;

    static vl::ErrorCode
    backward(vl::Context& context,
             type* derData,
             type const* derOutput,
             size_t height, size_t width, size_t depth,
             size_t strideY, size_t strideX,
             size_t padTop, size_t padBottom, size_t padLeft, size_t padRight) ;
  } ;

} }

#endif /* defined(VL_NNSUBSAMPLE_H) */
