// @file subsampling_cpu.cpp
// @brief Subsampling block implementation (CPU)
// @author Andrea Vedaldi
// @author Karel Lenc

/*
Copyright (C) 2014-16 Andrea Vedaldi and Karel Lenc.
All rights reserved.

This file is part of the VLFeat library and is made available under
the terms of the BSD license (see the COPYING file).
*/

#include "subsample.hpp"
#include <cstring>
#include <iostream>


namespace vl { namespace impl {

  template <typename type>
  struct subsample<vl::VLDT_CPU, type>
  {

    static vl::ErrorCode
    forward(vl::Context& context,
            type* output,
            type const* data,
            size_t height, size_t width, size_t depth,
            size_t strideY, size_t strideX,
            size_t padTop, size_t padBottom, size_t padLeft, size_t padRight)
    {
      int outputWidth = (width + (padLeft + padRight) - 1)/strideX + 1 ;
      int outputHeight = (height + (padTop + padBottom) - 1)/strideY + 1 ;
      for (int z = 0; z < depth; ++z) {
        for (int x = 0; x < outputWidth; ++x) {
          for (int y = 0; y < outputHeight; ++y) {
            int x1 = x * (signed)strideX - (signed)padLeft ;
            int y1 = y * (signed)strideY - (signed)padTop ;
            type value = 0 ;
            if (x1 >= 0 && x1 < width && y1 >= 0 && y1 < height) {
              value = data[x1 * height + y1] ;
            }
            output[x * outputHeight + y] = value ;
          }
        }
        data += width*height ;
        output += outputWidth*outputHeight ;
      }
      return VLE_Success ;
    }

    static vl::ErrorCode
    backward(vl::Context& context,
             type* derData,
             type const* derOutput,
             size_t height, size_t width, size_t depth,
             size_t strideY, size_t strideX,
             size_t padTop, size_t padBottom, size_t padLeft, size_t padRight)
    {
      int outputWidth = (width + (padLeft + padRight) - 1)/strideX + 1 ;
      int outputHeight = (height + (padTop + padBottom) - 1)/strideY + 1 ;

      memset(derData, 0, sizeof(type) * width * height * depth) ;

      for (int z = 0; z < depth; ++z) {
        for (int px = 0; px < outputWidth; ++px) {
          for (int py = 0; py < outputHeight; ++py) {
            int x1 = px * (int)strideX - (int)padLeft ;
            int y1 = py * (int)strideY - (int)padTop ;
            if (x1 >= 0 && x1 < width && y1 >= 0 && y1 < height) {
              derData[x1 * height + y1] = derOutput[px * outputHeight + py] ;
            }
          }
        }
        derData += width*height ;
        derOutput += outputWidth*outputHeight ;
      }
      return VLE_Success ;
    }
  } ;

} }

// Instantiations
template struct vl::impl::subsample<vl::VLDT_CPU, float> ;

#ifdef ENABLE_DOUBLE
template struct vl::impl::subsample<vl::VLDT_CPU, double> ;
#endif
