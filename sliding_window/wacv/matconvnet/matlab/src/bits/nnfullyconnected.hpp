// @file nnfullyconnected.hpp
// @brief Fully-connected block
// @author Andrea Vedaldi

/*
Copyright (C) 2014-16 Andrea Vedaldi.
All rights reserved.

This file is part of the VLFeat library and is made available under
the terms of the BSD license (see the COPYING file).
*/


#ifndef __vl__nnfullyconnected__
#define __vl__nnfullyconnected__

#include "data.hpp"

namespace vl {

  vl::ErrorCode
  nnfullyconnected_forward(vl::Context& context,
                           vl::Tensor output,
                           vl::Tensor data,
                           vl::Tensor filters,
                           vl::Tensor biases) ;

  vl::ErrorCode
  nnfullyconnected_backward(vl::Context& context,
                            vl::Tensor derData,
                            vl::Tensor derFilters,
                            vl::Tensor derBiases,
                            vl::Tensor data,
                            vl::Tensor filters,
                            vl::Tensor derOutput) ;
}


#endif /* defined(__vl__nnfullyconnected__) */
