// @file bilinearsampler.hpp
// @brief Bilinear sampler implementation
// @author Ankush Gupta
// @author Andrea Vedaldi

/*
Copyright (C) 2016- Ankush Gupta and Andrea Vedaldi.
All rights reserved.

This file is part of the VLFeat library and is made available under
the terms of the BSD license (see the COPYING file).
*/

#ifndef VL_BILINEARSAMPLER_H
#define VL_BILINEARSAMPLER_H

#include "../data.hpp"
#include <cstddef>

// defines the dispatcher for CUDA kernels:
namespace vl { namespace impl {

  template<vl::DeviceType dev, typename type>
  struct bilinearsampler {

    static vl::ErrorCode
    forward(Context& context,
            type* output,
            type const* data,
            type const* grid,
            size_t outHeight, size_t outWidth, size_t outDepth, size_t outCardinality,
            size_t inHeight, size_t inWidth, size_t inCardinality) ;


    static vl::ErrorCode
    backward(Context& context,
             type* derData,
             type* derGrid,
             type const* data,
             type const* grid,
             type const* derOutput,
             size_t outHeight, size_t outWidth, size_t outDepth, size_t outCardinality,
             size_t inHeight, size_t inWidth, size_t inCardinality) ;
  } ;

} }

#endif /* defined(VL_BILINEARSAMPLER_H) */
