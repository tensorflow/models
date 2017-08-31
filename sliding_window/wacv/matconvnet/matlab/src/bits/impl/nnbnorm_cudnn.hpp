// @file nnbnorm_cudnn.hpp
// @brief bnorm CuDNN-based implementation.
// @author Ankush Gupta, Andrea Vedaldi

/*
Copyright (C) 2016 Ankush Gupta and Andrea Vedaldi.
All rights reserved.

This file is part of the VLFeat library and is made available under
the terms of the BSD license (see the COPYING file).
*/

#ifndef __vl__bnorm_cudnn__
#define __vl__bnorm_cudnn__

#include "../data.hpp"
#include "cudnn.h"

namespace vl { namespace impl {

  template<vl::DataType dataType>
  struct nnbnorm_cudnn
  {
    static vl::ErrorCode
    forward(vl::Context& context,
            vl::Tensor output,
            vl::Tensor moments,
            vl::Tensor data,
            vl::Tensor multipliers,
            vl::Tensor biases,
            double epsilon) ;

    static vl::ErrorCode
    forward_given_moments(vl::Context& context,
                          vl::Tensor output,
                          vl::Tensor moments,
                          vl::Tensor data,
                          vl::Tensor multipliers,
                          vl::Tensor biases) ;

    static vl::ErrorCode
    backward(Context& context,
             vl::Tensor derData,
             vl::Tensor derMultipliers,
             vl::Tensor derBiases,
             vl::Tensor moments,
             vl::Tensor data,
             vl::Tensor multipliers,
             vl::Tensor biases,
             vl::Tensor derOutput,
             double epsilon) ;

    static vl::ErrorCode
    backward_given_moments(Context& context,
                           vl::Tensor derData,
                           vl::Tensor derMultipliers,
                           vl::Tensor derBiases,
                           vl::Tensor moments,
                           vl::Tensor data,
                           vl::Tensor multipliers,
                           vl::Tensor biases,
                           vl::Tensor derOutput,
                           double epsilon) ;
  } ;

} }

#endif /* defined(__vl__nnbnorm_cudnn__) */
