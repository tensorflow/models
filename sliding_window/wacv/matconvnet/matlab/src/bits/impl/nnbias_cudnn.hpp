// @file nnbias_blas.hpp
// @brief biasolution block CuDNN-based implementation.
// @author Andrea Vedaldi

/*
Copyright (C) 2015 Andrea Vedaldi.
All rights reserved.

This file is part of the VLFeat library and is made available under
the terms of the BSD license (see the COPYING file).
*/

#ifndef __vl__nnbias_cudnn__
#define __vl__nnbias_cudnn__

#include "../data.hpp"
#include "cudnn.h"

namespace vl { namespace impl {

  // todo: data type should be handled internally?

  template<vl::DataType dataType>
  struct nnbias_cudnn
  {
    static vl::ErrorCode
    forward(vl::Context& context,
            vl::Tensor output, double outputMult,
            vl::Tensor data, double dataMult,
            vl::Tensor biases, double biasesMult) ;

    static vl::ErrorCode
    backward(vl::Context& context,
             vl::Tensor derData, double derDataMult,
             vl::Tensor derBiases, double derBiasesMult,
             vl::Tensor derOutput, double derOutputMult) ;
  } ;

} }

#endif /* defined(__vl__nnbias_cudnn__) */
