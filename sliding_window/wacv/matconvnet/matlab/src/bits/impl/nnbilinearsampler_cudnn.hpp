// @file nnbilinearsampler_cudnn.hpp
// @brief BilinearSampler CuDNN-based implementation.
// @author Ankush Gupta, Andrea Vedaldi

/*
Copyright (C) 2016 Ankush Gupta and Andrea Vedaldi.
All rights reserved.

This file is part of the VLFeat library and is made available under
the terms of the BSD license (see the COPYING file).
*/

#ifndef __vl__bilinearsampler_cudnn__
#define __vl__bilinearsampler_cudnn__

#include "../data.hpp"
#include "cudnn.h"

namespace vl { namespace impl {

  template<vl::DataType dataType>
  struct nnbilinearsampler_cudnn
  {
    static vl::ErrorCode
    forward(Context& context,
            Tensor output,
            Tensor data,
            Tensor grid) ;

    static vl::ErrorCode
    backward(Context& context,
             Tensor derData,
             Tensor derGrid,
             Tensor data,
             Tensor grid,
             Tensor derOutput) ;
  } ;

} }

#endif /* defined(__vl__nnbilinearsampler_cudnn__) */
