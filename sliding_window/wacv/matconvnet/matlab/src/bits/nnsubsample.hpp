// @file nnsubsample.hpp
// @brief Subsamping block
// @author Andrea Vedaldi

/*
Copyright (C) 2014-16 Andrea Vedaldi and Karel Lenc.
All rights reserved.

This file is part of the VLFeat library and is made available under
the terms of the BSD license (see the COPYING file).
*/

#ifndef __vl__nnsubsample__
#define __vl__nnsubsample__

#include "data.hpp"

namespace vl {

  vl::ErrorCode
  nnsubsample_forward(vl::Context& context,
                      vl::Tensor output,
                      vl::Tensor data,
                      vl::Tensor biases,
                      int strideY, int strideX,
                      int padTop, int padBottom,
                      int padLeft, int padRight) ;

  vl::ErrorCode
  nnsubsample_backward(vl::Context& context,
                       vl::Tensor derData,
                       vl::Tensor derBiases,
                       vl::Tensor derOutput,
                       int strideY, int strideX,
                       int padTop, int padBottom,
                       int padLeft, int padRight) ;
}

#endif /* defined(__vl__nnsubsample__) */
