// @file nnpooling.hpp
// @brief Pooling block
// @author Andrea Vedaldi

/*
Copyright (C) 2014-16 Andrea Vedaldi and Karel Lenc.
All rights reserved.

This file is part of the VLFeat library and is made available under
the terms of the BSD license (see the COPYING file).
*/

#ifndef __vl__nnpooling__
#define __vl__nnpooling__

#include "data.hpp"
#include <stdio.h>

namespace vl {

  enum PoolingMethod { vlPoolingMax, vlPoolingAverage } ;

  vl::ErrorCode
  nnpooling_forward(vl::Context& context,
                    vl::Tensor output,
                    vl::Tensor data,
                    PoolingMethod method,
                    int poolHeight, int poolWidth,
                    int strideY, int strideX,
                    int padTop, int padBottom,
                    int padLeft, int padRight) ;

  vl::ErrorCode
  nnpooling_backward(vl::Context& context,
                     vl::Tensor derData,
                     vl::Tensor data,
                     vl::Tensor derOutput,
                     PoolingMethod method,
                     int poolHeight, int poolWidth,
                     int strideY, int strideX,
                     int padTop, int padBottom,
                     int padLeft, int padRight) ;
}

#endif /* defined(__vl__nnpooling__) */
