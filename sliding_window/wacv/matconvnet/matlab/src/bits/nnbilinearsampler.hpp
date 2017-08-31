// @file nnbilinearsampler.hpp
// @brief Bilinear sampler block
// @author Ankush Gupta
// @author Andrea Vedaldi

/*
Copyright (C) 2016- Ankush Gupta and Andrea Vedaldi.
All rights reserved.
This file is part of the VLFeat library and is made available under
the terms of the BSD license (see the COPYING file).
*/

#ifndef __vl__nnbilinearsampler__
#define __vl__nnbilinearsampler__

#include "data.hpp"
#include <stdio.h>

namespace vl {
  vl::ErrorCode
  nnbilinearsampler_forward(vl::Context& context,
                            vl::Tensor output,
                            vl::Tensor data,
                            vl::Tensor grid) ;

  vl::ErrorCode
  nnbilinearsampler_backward(vl::Context& context,
                             vl::Tensor derData,
                             vl::Tensor derGrid,
                             vl::Tensor data,
                             vl::Tensor grid,
                             vl::Tensor derOutput) ;
}

#endif /* defined(__vl__nnbilinearsampler__) */
