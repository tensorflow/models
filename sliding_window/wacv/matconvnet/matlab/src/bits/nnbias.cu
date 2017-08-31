// @file nnbias.cu
// @brief Bias block
// @author Andrea Vedaldi

/*
Copyright (C) 2015-16 Andrea Vedaldi.
All rights reserved.

This file is part of the VLFeat library and is made available under
the terms of the BSD license (see the COPYING file).
*/

#include "nnbias.hpp"
#include "impl/nnbias_blas.hpp"
#if ENABLE_CUDNN
#include "impl/nnbias_cudnn.hpp"
#endif
#include <assert.h>

using namespace vl ;

/* ---------------------------------------------------------------- */
/* Forward                                                          */
/* ---------------------------------------------------------------- */

#define DISPATCH(deviceType,dataType) \
status = vl::impl::nnbias_forward_blas<deviceType,dataType> \
(context, output, outputMult, data, dataMult, biases, biasesMult) ;

#define DISPATCH2(deviceType) \
switch (dataType) { \
case VLDT_Float : DISPATCH(deviceType,VLDT_Float) ; break ; \
IF_DOUBLE(case VLDT_Double : DISPATCH(deviceType,VLDT_Double) ; break ;) \
default: assert(false) ; return VLE_Unknown ; \
}

#define DISPATCHCUDNN(dataType) \
status = vl::impl::nnbias_cudnn<dataType>::forward \
(context, output, outputMult, data, dataMult, biases, biasesMult) ;

#define DISPATCHCUDNN2() \
switch (dataType) { \
case VLDT_Float : DISPATCHCUDNN(VLDT_Float) ; break ; \
IF_DOUBLE(case VLDT_Double : DISPATCHCUDNN(VLDT_Double) ; break ;) \
default: assert(false) ; return VLE_Unknown ; \
}

vl::ErrorCode
vl::nnbias_forward(vl::Context& context,
                   vl::Tensor output, double outputMult,
                   vl::Tensor data, double dataMult,
                   vl::Tensor biases, double biasesMult)
{
  vl::ErrorCode status = VLE_Success ;
  vl::DataType dataType = output.getDataType() ;

  switch (output.getDeviceType()) {
    default:
      assert(false) ;
      status = vl::VLE_Unknown ;
      break ;

    case vl::VLDT_CPU:
      DISPATCH2(vl::VLDT_CPU) ;
      break ;

#if ENABLE_GPU
    case vl::VLDT_GPU:
#if ENABLE_CUDNN
      if (context.getCudaHelper().getCudnnEnabled()) {
        DISPATCHCUDNN2() ;
        if (status == vl::VLE_Success) { return status ; }
        if (status != vl::VLE_Unsupported) { goto done ; }
        /* this case was not supported by CUDNN -- fallback */
      }
#endif
      DISPATCH2(vl::VLDT_GPU) ;
      break ;
#endif
  }
#if ENABLE_CUDNN
done:
#endif
  return context.passError(status, __func__) ;
}

/* ---------------------------------------------------------------- */
/* Backward                                                         */
/* ---------------------------------------------------------------- */

#undef DISPATCH
#define DISPATCH(deviceType,dataType) \
status = vl::impl::nnbias_backward_blas<deviceType,dataType> \
(context, derData, derDataMult, derBiases, derBiasesMult, derOutput, derOutputMult) ;

#undef DISPATCHCUDNN
#define DISPATCHCUDNN(dataType) \
status = vl::impl::nnbias_cudnn<dataType>::backward \
(context, derData, derDataMult, derBiases, derBiasesMult, derOutput, derOutputMult) ;

vl::ErrorCode
vl::nnbias_backward(vl::Context& context,
                    vl::Tensor derData, double derDataMult,
                    vl::Tensor derBiases, double derBiasesMult,
                    vl::Tensor derOutput, double derOutputMult)
{
  vl::ErrorCode status = VLE_Success ;
  vl::DataType dataType = derOutput.getDataType() ;

  switch (derOutput.getDeviceType()) {
    default:
      assert(false) ;
      status = vl::VLE_Unknown ;
      break ;

    case vl::VLDT_CPU:
      DISPATCH2(vl::VLDT_CPU) ;
      break ;

#if ENABLE_GPU
    case vl::VLDT_GPU:
#if ENABLE_CUDNN
      if (context.getCudaHelper().getCudnnEnabled()) {
        DISPATCHCUDNN2() ;
        if (status == vl::VLE_Success) { return status ; }
        if (status != vl::VLE_Unsupported) { goto done ; }
        /* this case was not supported by CUDNN -- fallback */
      }
#endif
      DISPATCH2(vl::VLDT_GPU) ;
      break ;
#endif
  }
#if ENABLE_CUDNN
done:
#endif
  return context.passError(status, __func__) ;
}

