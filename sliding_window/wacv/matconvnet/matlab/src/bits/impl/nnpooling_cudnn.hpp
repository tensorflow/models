// @file nnpooling_blas.hpp
// @brief Pooling block CuDNN-based implementation.
// @author Andrea Vedaldi

/*
Copyright (C) 2015-16 Andrea Vedaldi.
All rights reserved.

This file is part of the VLFeat library and is made available under
the terms of the BSD license (see the COPYING file).
*/

#ifndef __vl__nnpooling_cudnn__
#define __vl__nnpooling_cudnn__

#include "../nnpooling.hpp"
#include "../data.hpp"
#include "cudnn.h"


namespace vl { namespace impl {

  // todo: data type should be handled internally?

  template<vl::DataType dataType>
  struct nnpooling_cudnn
  {
    static vl::ErrorCode
    forward(Context& context,
            Tensor output,
            Tensor data,
            vl::PoolingMethod method,
            int poolHeight, int poolWidth,
            int strideY, int strideX,
            int padTop, int padBottom,
            int padLeft, int padRight) ;

    static vl::ErrorCode
    backward(Context& context,
             Tensor derData,
             Tensor data,
             Tensor output,
             Tensor derOutput,
             vl::PoolingMethod method,
             int poolHeight, int poolWidth,
             int strideY, int strideX,
             int padTop, int padBottom,
             int padLeft, int padRight) ;
  };

} }

#endif /* defined(__vl__nnpooling_cudnn__) */
