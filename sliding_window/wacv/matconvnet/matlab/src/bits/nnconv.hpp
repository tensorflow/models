// @file nnconv.cu
// @brief Convolution block
// @author Andrea Vedaldi
// @author Max Jaderberg

/*
Copyright (C) 2014 Andrea Vedaldi and Max Jaderberg
Copyright (C) 2015-16 Andrea Vedaldi.

All rights reserved.

This file is part of the VLFeat library and is made available under
the terms of the BSD license (see the COPYING file).
*/

#ifndef __vl__nnconv__
#define __vl__nnconv__

#include "data.hpp"

namespace vl {

  vl::ErrorCode
  nnconv_forward(vl::Context& context,
                 vl::Tensor output, double outputMult,
                 vl::Tensor data, double dataMult,
                 vl::Tensor filters,
                 vl::Tensor biases,
                 int strideY, int strideX,
                 int padTop, int padBottom,
                 int padLeft, int padRight,
                 int dilateY, int dilateX) ;

  vl::ErrorCode
  nnconv_backward(vl::Context& context,
                  vl::Tensor derData,
                  vl::Tensor derFilters,
                  vl::Tensor derBiases,
                  vl::Tensor data,
                  vl::Tensor filters,
                  vl::Tensor derOutput,
                  int strideY, int strideX,
                  int padTop, int padBottom,
                  int padLeft, int padRight,
                  int dilateY, int dilateX) ;

  vl::ErrorCode
  nnconvt_forward(vl::Context& context,
                  vl::Tensor output,
                  vl::Tensor data,
                  vl::Tensor filters,
                  vl::Tensor biases,
                  int upsampleY, int upsampleX,
                  int cropTop, int cropBottom,
                  int cropLeft, int cropRight) ;

  vl::ErrorCode
  nnconvt_backward(vl::Context& context,
                   vl::Tensor derData,
                   vl::Tensor derFilters,
                   vl::Tensor derBiases,
                   vl::Tensor data,
                   vl::Tensor filters,
                   vl::Tensor derOutput,
                   int upsampleY, int upsampleX,
                   int cropTop, int cropBottom,
                   int cropLeft, int cropRight) ;
}


#endif /* defined(__vl__nnconv__) */
