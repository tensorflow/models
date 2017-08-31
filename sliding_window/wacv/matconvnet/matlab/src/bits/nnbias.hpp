// @file nnbias.hpp
// @brief Bias block
// @author Andrea Vedaldi

/*
Copyright (C) 2015 Andrea Vedaldi.
All rights reserved.

This file is part of the VLFeat library and is made available under
the terms of the BSD license (see the COPYING file).
*/

#ifndef __vl__nnbias__
#define __vl__nnbias__

#include "data.hpp"

namespace vl {

  vl::ErrorCode
  nnbias_forward(vl::Context& context,
                 vl::Tensor output, double outputMult,
                 vl::Tensor data, double dataMult,
                 vl::Tensor biases, double biasesMult) ;

  vl::ErrorCode
  nnbias_backward(vl::Context& context,
                  vl::Tensor derData, double derDataMult,
                  vl::Tensor derBiases, double derBiasesMult,
                  vl::Tensor derOutput, double derOutputMult) ;
}

#endif /* defined(__vl__nnbias__) */
