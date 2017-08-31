// @file nnconv_blas.hpp
// @brief Convolution block CuDNN-based implementation.
// @author Andrea Vedaldi

/*
Copyright (C) 2015-16 Andrea Vedaldi.
All rights reserved.

This file is part of the VLFeat library and is made available under
the terms of the BSD license (see the COPYING file).
*/

#ifndef __vl__nnconv_cudnn__
#define __vl__nnconv_cudnn__

#include "../data.hpp"
#include "cudnn.h"

namespace vl { namespace impl {

  template<vl::DataType dataType>
  struct nnconv_cudnn
  {
    static vl::ErrorCode
    forward(Context& context,
            Tensor output, double outputMult,
            Tensor data, double dataMult,
            Tensor filters,
            Tensor biases,
            int strideX, int strideY,
            int padLeft, int padRight,
            int padTop, int padBottom,
            int dilateX, int dilateY) ;

    static vl::ErrorCode
    backward(Context& context,
             Tensor derData,
             Tensor derFilters,
             Tensor derBiases,
             Tensor data,
             Tensor filters,
             Tensor derOutput,
             int strideX, int strideY,
             int padLeft, int padRight,
             int padTop, int padBottom,
             int dilateX, int dilateY) ;
  } ;

} }
#endif /* defined(__vl__nnconv_cudnn__) */
